"""`bgm extract` (SPEC M3): cache every feature once, per video and per music file, in data/features/.

Groups (`--what`) and the keys they write (SPEC §4.2):
  sem_video  video: sem_siglip_1hz [T1, 1152] f16, sem_clip_1hz [T1, 768] f16 (when store_clip)
  sem_music  music: sem_clap_1hz [T1, 512] f16, sem_mert_1hz [T1, 1024] f16 (+ sem_mert_layers_1hz when enabled)
  attr       both:  attr_logits_1hz [T1, M] f32 (raw mood logits, from the stored semantic embeddings)
  bridge     both:  bridge_logits_1hz [T1, O] f32 (raw occasion logits)
  imagebind  both:  imagebind_1hz [T1, 1024] f16 (baseline only)
  dyn_video  video: dyn_25hz [T25, 5] f32 raw, cut_times_s
  dyn_music  music: dyn_25hz [T25, 5] f32 raw (novelty from sem_mert_1hz), beat_times_s, tempo_bpm
  key        music: key_pc, key_mode, tonic_pc (int8)

Each file's `meta` (JSON) records a config hash per group; a group is recomputed only when its hash (or, for
dyn_video, the pair's cut times) changed. Writes are atomic, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from bgm.data import manifest
from bgm.data.media import ffprobe
from bgm.data.prep import interim_music_path, interim_video_path
from bgm.features import dynamics, keys
from bgm.features.timing import n_1hz, n_25hz

log = logging.getLogger(__name__)

GROUPS = ["sem_video", "sem_music", "attr", "bridge", "imagebind", "dyn_video", "dyn_music", "key"]   # run order
SIDES = {"sem_video": ("video",), "sem_music": ("music",), "attr": ("video", "music"), "bridge": ("video", "music"),
         "imagebind": ("video", "music"), "dyn_video": ("video",), "dyn_music": ("music",), "key": ("music",)}
VERSION = {g: 1 for g in GROUPS} | {"dyn_video": 2}   # v2: duplicate-frame motion repair (PROGRESS.md D27)
SR = {"clap": 48000, "mert": 24000, "librosa": 22050, "imagebind": 16000}


# ---------------------------------------------------------------- config hashes and files
def group_hash(cfg: DictConfig, group: str) -> str:
    f = cfg.features
    parts = {"sem_video": [f.sem_video], "sem_music": [f.sem_music, cfg.media.music_sr],
             "attr": [f.attributes, f.sem_video.model, f.sem_music.clap_model],
             "bridge": [f.bridge, f.sem_video.model, f.sem_music.clap_model], "imagebind": [f.imagebind],
             "dyn_video": [f.dynamics, cfg.shots], "dyn_music": [f.dynamics, f.sem_music], "key": [f.tonic]}[group]
    blob = json.dumps({"v": VERSION[group], "cfg": [OmegaConf.to_container(p, resolve=True) if isinstance(p, DictConfig) else p
                                                    for p in parts]}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def feature_path(cfg: DictConfig, side: str, item_id: str) -> Path:
    return Path(cfg.paths.features) / side / f"{item_id}.npz"


def load_features(path: Path) -> dict:
    """All arrays of a feature file, with `meta` parsed to a dict ({} if the file does not exist)."""
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"])) if "meta" in d else {}
    return d


def save_features(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in arrays.items() if k != "meta"}
    out["meta"] = np.array(json.dumps(arrays.get("meta", {}), sort_keys=True))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)


# ---------------------------------------------------------------- media access
def frames_1hz(video_path: Path, T1: int) -> np.ndarray:
    """[T1, H, W, 3] uint8: the frame containing k + 0.5 s (frame 25k + 12 of the 25 fps interim video)."""
    info = ffprobe(video_path)
    assert info["audio"] is None, f"video extractor refuses {video_path}: it has an audio stream"
    w, h = info["video"]["width"], info["video"]["height"]
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-an", "-map", "0:v:0",
                          "-vf", r"select=eq(mod(n\,25)\,12)", "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    assert len(frames) >= T1, f"{video_path}: {len(frames)} frames for T1 = {T1}"
    return frames[:T1]


def load_music(path: Path, sr: int) -> np.ndarray:
    """Mono float32 at `sr` from the 48 kHz interim FLAC (soxr resampling)."""
    import soundfile as sf
    import soxr
    y, file_sr = sf.read(path, dtype="float32")
    assert file_sr == SR["clap"], f"{path}: interim music must be 48 kHz, is {file_sr}"
    return y if sr == file_sr else soxr.resample(y, file_sr, sr).astype(np.float32)


def centred_windows(y: np.ndarray, sr: int, T1: int, window_s: float, pad: bool) -> list[np.ndarray]:
    """For each second k, y over [k + 0.5 - w/2, k + 0.5 + w/2] clipped to the file (zero-padded to w if `pad`)."""
    half, out = window_s / 2, []
    for k in range(T1):
        a, b = int(round((k + 0.5 - half) * sr)), int(round((k + 0.5 + half) * sr))
        seg = y[max(a, 0):min(b, len(y))]
        if pad:
            seg = np.concatenate([np.zeros(max(0, -a), np.float32), seg, np.zeros(max(0, b - len(y)), np.float32)])
        out.append(seg)
    return out


# ---------------------------------------------------------------- encoders (real or fake), cached per run
class EncoderBank:
    def __init__(self, cfg: DictConfig, fake: bool):
        self.cfg, self.fake, self._c = cfg, fake, {}
        self.device = cfg.features.device if not fake else "cpu"

    def get(self, name: str):
        if name not in self._c:
            self._c[name] = self._make(name)
        return self._c[name]

    def _make(self, name: str):
        f, dev = self.cfg.features, self.device
        if self.fake:
            from bgm.features import encoders as E
            fake = {"siglip": (E.FakeFrameEncoder(1152, 10), E.FakeTextEncoder(1152, 11)), "clip": E.FakeFrameEncoder(768, 17),
                    "clap": (E.FakeAudioEncoder(512, 48000, 12), E.FakeTextEncoder(512, 13)), "mert": E.FakeAudioEncoder(1024, 24000, 14),
                    "imagebind": (E.FakeFrameEncoder(1024, 15), E.FakeAudioEncoder(1024, 16000, 16))}[name]
            return fake
        from bgm.features import hf
        if name == "siglip":
            return hf.SiglipEncoder(f.sem_video.model, dev, f.sem_video.batch)
        if name == "clip":
            return hf.ClipEncoder(f.sem_video.clip_model, dev, f.sem_video.batch)
        if name == "clap":
            return hf.ClapEncoder(f.sem_music.clap_model, dev)
        if name == "mert":
            return hf.MertEncoder(f.sem_music.mert_model, dev, f.sem_music.mert_chunk_s, f.sem_music.mert_context_s)
        if name == "imagebind":
            from bgm.tools import REPO_ROOT
            return hf.ImageBindEncoder(str(REPO_ROOT / ".checkpoints" / "imagebind_huge.pth"), dev)
        raise KeyError(name)

    # uniform calls over real / fake
    def frames(self, name, frames):
        e = self.get(name)
        return (e[0] if isinstance(e, tuple) else e).encode_frames(frames)

    def audio(self, name, windows, sr):
        e = self.get(name)
        e = e[1] if (isinstance(e, tuple) and name == "imagebind") else (e[0] if isinstance(e, tuple) else e)
        if self.fake:
            n = max(len(w) for w in windows)
            windows = np.stack([np.pad(np.asarray(w, np.float32), (0, n - len(w))) for w in windows])
        return e.encode_audio(windows, sr)

    def text(self, name, texts):
        e = self.get(name)
        return (e[1] if isinstance(e, tuple) else e).encode_text(texts)

    def mert_track(self, y):
        e = self.get("mert")
        if self.fake:
            T1 = n_1hz(len(y) / SR["mert"])
            return e.encode_audio(np.stack(centred_windows(y, SR["mert"], T1, 1.0, pad=True)), SR["mert"])
        return e.encode_track(y, SR["mert"], keep_layers=self.cfg.features.sem_music.store_all_mert_layers)

    def logit_params(self, name):
        if self.fake:
            return 10.0, 0.0
        e = self.get(name)
        return e.logit_scale, getattr(e, "logit_bias", 0.0)


def word_embeddings(bank: EncoderBank, name: str, words, templates) -> np.ndarray:
    """[W, D]: per word, the mean of its L2-normalized template embeddings, renormalized (SPEC §5.3)."""
    out = []
    for w in words:
        e = bank.text(name, [t.format(w) for t in templates]).mean(0)
        out.append(e / (np.linalg.norm(e) + 1e-12))
    return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------- CPU workers (process pool)
def cpu_pool(workers: int) -> ProcessPoolExecutor:
    """Spawned (the parent may hold a CUDA context), one BLAS / numba thread per worker (else they oversubscribe)."""
    import multiprocessing as mp
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in thread_vars}
    os.environ.update({k: "1" for k in thread_vars})
    try:
        return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
    finally:   # children read the variables when they start; the parent keeps its own
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _dyn_video_job(args):
    path, cuts, sigma, short_side, grid = args
    return dynamics.video_curves(path, cuts, sigma, short_side, grid)


def _dyn_music_job(args):
    path, sigma, floor = args
    y = load_music(Path(path), SR["librosa"])
    return dynamics.music_curves(y, SR["librosa"], sigma, floor)


def _key_job(path):
    y = load_music(Path(path), SR["librosa"])
    return keys.estimate_key(y, SR["librosa"])


# ---------------------------------------------------------------- the run
class Extractor:
    def __init__(self, cfg: DictConfig, fake: bool = False, workers: int = 16, limit: int | None = None):
        self.cfg, self.fake, self.workers = cfg, fake, workers
        pairs = manifest.read_pairs(cfg)
        if limit:
            pairs = pairs.head(limit)
        self.videos = {p: interim_video_path(cfg, p) for p in pairs["pair_id"]}
        self.music = {m: interim_music_path(cfg, m) for m in dict.fromkeys(pairs["music_id"])}
        shots = manifest.read_shots(cfg) if (Path(cfg.paths.manifests) / "shots.csv").exists() else None
        self.cuts = {p: manifest.cut_times(shots, p) for p in self.videos} if shots is not None else {}
        self.bank = EncoderBank(cfg, fake)
        self._dur: dict[tuple[str, str], float] = {}

    def items(self, side):
        return self.videos if side == "video" else self.music

    def duration(self, side, item_id) -> float:
        key = (side, item_id)
        if key not in self._dur:
            path = self.items(side)[item_id]
            if side == "video":
                v = ffprobe(path)["video"]
                self._dur[key] = v["duration_s"]
            else:
                import soundfile as sf
                self._dur[key] = sf.info(path).duration
        return self._dur[key]

    def stale(self, side, item_id, group) -> bool:
        d = load_features(feature_path(self.cfg, side, item_id))
        if d.get("meta", {}).get("hashes", {}).get(group) != group_hash(self.cfg, group):
            return True
        if group == "dyn_video":
            return not np.allclose(d.get("cut_times_s", np.zeros(0)), np.array(self.cuts.get(item_id, []), np.float32), atol=1e-3) \
                or len(d.get("cut_times_s", [])) != len(self.cuts.get(item_id, []))
        return False

    def update(self, side, item_id, group, arrays: dict) -> None:
        path = feature_path(self.cfg, side, item_id)
        d = load_features(path)
        meta = d.get("meta") or {}
        dur = self.duration(side, item_id)
        meta.update({"id": item_id, "side": side, "duration_s": dur, "T1": n_1hz(dur), "T25": n_25hz(dur), "fps": 25,
                     "sr": SR, "fake": self.fake})
        meta.setdefault("hashes", {})[group] = group_hash(self.cfg, group)
        meta.setdefault("versions", {}).update(self._versions())
        d.update(arrays)
        d["meta"] = meta
        save_features(path, d)

    @staticmethod
    def _versions():
        import librosa
        import transformers
        return {"transformers": transformers.__version__, "librosa": librosa.__version__}

    def run(self, groups: list[str]) -> None:
        unknown = set(groups) - set(GROUPS)
        assert not unknown, f"unknown feature groups {unknown}; known: {GROUPS}"
        for g in [g for g in GROUPS if g in groups]:
            for side in SIDES[g]:
                todo = [i for i in self.items(side) if self.stale(side, i, g)]
                log.info("extract %s/%s: %d of %d stale", g, side, len(todo), len(self.items(side)))
                if todo:
                    t0 = time.time()
                    getattr(self, f"_{g}")(side, todo)
                    dt = time.time() - t0
                    log.info("extract %s/%s: %d items in %.1f s (%.2f s/item)", g, side, len(todo), dt, dt / len(todo))

    # ---- groups
    def _prefetch(self, fn, ids, threads=8):
        with ThreadPoolExecutor(threads) as ex:
            yield from zip(ids, ex.map(fn, ids))

    def _sem_video(self, side, ids):
        store_clip = self.cfg.features.sem_video.store_clip
        load = lambda i: frames_1hz(self.videos[i], n_1hz(self.duration("video", i)))  # noqa: E731
        for i, fr in self._prefetch(load, ids):
            arr = {"sem_siglip_1hz": self.bank.frames("siglip", fr).astype(np.float16)}
            if store_clip:
                arr["sem_clip_1hz"] = self.bank.frames("clip", fr).astype(np.float16)
            self.update("video", i, "sem_video", arr)

    def _sem_music(self, side, ids):
        sm = self.cfg.features.sem_music
        assert sm.clap_sr == SR["clap"] and sm.mert_sr == SR["mert"]

        def load(i):
            y48 = load_music(self.music[i], SR["clap"])
            return y48, load_music(self.music[i], SR["mert"])
        for i, (y48, y24) in self._prefetch(load, ids, threads=4):
            T1 = n_1hz(len(y48) / SR["clap"])
            clap = self.bank.audio("clap", centred_windows(y48, SR["clap"], T1, sm.clap_window_s, pad=False), SR["clap"])
            mert = self.bank.mert_track(y24)
            arr = {"sem_clap_1hz": clap.astype(np.float16)}
            if isinstance(mert, tuple):
                arr["sem_mert_1hz"], arr["sem_mert_layers_1hz"] = mert[0].astype(np.float16), mert[1].astype(np.float16)
            else:
                arr["sem_mert_1hz"] = mert.astype(np.float16)
            # The 48 -> 24 kHz resample can round a clip cut just short of a whole second up by one sample, and MERT
            # then counts one more second than CLAP (M8 pool clip u01126_00: 20 vs 19). Drop that partial last second.
            if len(arr["sem_mert_1hz"]) == T1 + 1:
                arr["sem_mert_1hz"] = arr["sem_mert_1hz"][:T1]
                if "sem_mert_layers_1hz" in arr:
                    arr["sem_mert_layers_1hz"] = arr["sem_mert_layers_1hz"][:, :T1]
            assert len(arr["sem_mert_1hz"]) == T1 == len(clap), (i, T1, len(clap), len(arr["sem_mert_1hz"]))
            self.update("music", i, "sem_music", arr)

    def _logits(self, side, ids, group, words, v_tpl, m_tpl, key):
        name, src = ("siglip", "sem_siglip_1hz") if side == "video" else ("clap", "sem_clap_1hz")
        W = word_embeddings(self.bank, name, words, v_tpl if side == "video" else m_tpl)
        scale, bias = self.bank.logit_params(name)
        for i in ids:
            e = load_features(feature_path(self.cfg, side, i)).get(src)
            assert e is not None, f"{group} needs {src} for {i}: run --what {'sem_video' if side == 'video' else 'sem_music'} first"
            self.update(side, i, group, {key: (scale * (e.astype(np.float32) @ W.T) + bias).astype(np.float32)})

    def _attr(self, side, ids):
        a = self.cfg.features.attributes
        self._logits(side, ids, "attr", list(a.moods), list(a.video_templates), list(a.music_templates), "attr_logits_1hz")

    def _bridge(self, side, ids):
        b = self.cfg.features.bridge
        self._logits(side, ids, "bridge", list(b.occasions), list(b.video_templates), list(b.music_templates), "bridge_logits_1hz")

    def _imagebind(self, side, ids):
        if not self.cfg.features.imagebind.enabled:
            return
        if side == "video":
            load = lambda i: frames_1hz(self.videos[i], n_1hz(self.duration("video", i)))  # noqa: E731
            for i, fr in self._prefetch(load, ids):
                self.update("video", i, "imagebind", {"imagebind_1hz": self.bank.frames("imagebind", fr).astype(np.float16)})
        else:
            sr = self.cfg.features.imagebind.sr
            assert sr == SR["imagebind"]
            for i, y in self._prefetch(lambda i: load_music(self.music[i], sr), ids, threads=4):
                wins = centred_windows(y, sr, n_1hz(len(y) / sr), self.cfg.features.imagebind.window_s, pad=True)
                self.update("music", i, "imagebind", {"imagebind_1hz": self.bank.audio("imagebind", wins, sr).astype(np.float16)})

    def _dyn_video(self, side, ids):
        assert self.cuts, "dyn_video needs data/manifests/shots.csv: run `bgm shots` first"
        d = self.cfg.features.dynamics
        jobs = [(str(self.videos[i]), self.cuts[i], d.impulse_sigma_s, d.flow_short_side, d.affine_grid) for i in ids]
        assert d.flow == "farneback", "only Farneback flow is built"
        with cpu_pool(self.workers) as ex:
            for n, (i, dyn) in enumerate(zip(ids, ex.map(_dyn_video_job, jobs, chunksize=4))):
                self.update("video", i, "dyn_video", {"dyn_25hz": dyn, "cut_times_s": np.array(self.cuts[i], np.float32)})
                if (n + 1) % 200 == 0:
                    log.info("  dyn_video %d/%d", n + 1, len(ids))

    def _dyn_music(self, side, ids):
        d = self.cfg.features.dynamics
        assert d.librosa_sr == SR["librosa"]
        jobs = [(str(self.music[i]), d.impulse_sigma_s, d.centroid_floor_hz) for i in ids]
        with cpu_pool(self.workers) as ex:
            for i, (dyn, beats, tempo) in zip(ids, ex.map(_dyn_music_job, jobs, chunksize=4)):
                mert = load_features(feature_path(self.cfg, "music", i)).get("sem_mert_1hz")
                assert mert is not None, f"dyn_music needs sem_mert_1hz for {i} (novelty): run --what sem_music first"
                dyn = dynamics.add_novelty(dyn, mert.astype(np.float32), int(d.novelty_half_width_s))
                self.update("music", i, "dyn_music", {"dyn_25hz": dyn, "beat_times_s": beats, "tempo_bpm": np.float32(tempo)})

    def _key(self, side, ids):
        with cpu_pool(self.workers) as ex:
            for i, (pc, mode) in zip(ids, ex.map(_key_job, [str(self.music[i]) for i in ids], chunksize=4)):
                self.update("music", i, "key", {"key_pc": np.int8(pc), "key_mode": np.int8(mode), "tonic_pc": np.int8(-1)})


def run_extract(cfg: DictConfig, what: str, fake: bool = False, limit: int | None = None, workers: int = 16) -> None:
    groups = [w.strip() for w in what.split(",") if w.strip()]
    Extractor(cfg, fake=fake, workers=workers, limit=limit).run(groups)
