"""How much of the dataset's music track is actually background music? PANNs CNN14 per-second audit (D114).

Input: the cached PANNs AudioSet tags (data/features/music_panns, one 527-class sigmoid row per second, 4 s context;
src/bgm/features/panns_tags.py) and the per-second RMS level of the interim 48 kHz music file. Each second gets
one category, checked in this order:

  silent        RMS < silent_db (dBFS)
  speech        P(Speech) >= thr and P(Music) < thr          (dialogue, no music)
  music+speech  P(Speech) >= thr and P(Music) >= thr         (music with dialogue leaking in)
  music         P(Music) >= thr
  weak music    weak <= P(Music) < thr                       (quiet / sparse music)
  other sound   everything else; the top non-music class is kept (drones, rumble, effects, ambience)

Two views. **raw**: the cached tags of the music as delivered, level = RMS of the file. **hum removed** (the main
view): PANNs re-run on the music high-passed at 150 Hz and set to -20 dBFS (panns_tags.extract_hp). Konjum salangai's raw audio
carries a +0.06 DC offset (the whole file shifted; ~ -21 dBFS of 0 Hz energy) and a 100 Hz hum over music ~27 dB
quieter than the other sources, so the raw tags hear 'silence'), level = RMS above 150 Hz, silent below -60 dBFS (hum only).

Per file: BGM share = (music + music+speech) seconds / all seconds. Verdict: clean BGM (share >= 0.9 and no
speech seconds), mostly BGM (share >= 0.5), little BGM (share < 0.5). Cross-check: data/manifests/song_labels.csv
(Demucs vocal share + CLAP, analysis D-song labels), for singing that PANNs' "Music" class does not separate.

Run: python -m bgm.analysis.bgm_presence [key=value ...] -> <paths.results>/bgm_presence.md, bgm_presence_files.csv
(cleaned dataset, D117: paths.manifests=cleaned_dataset/manifests paths.interim=cleaned_dataset/interim
paths.features=cleaned_dataset/features paths.results=results/cleaned)
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

log = logging.getLogger(__name__)

CATS = ("music", "music+speech", "weak music", "speech", "silent", "other sound")
BGM_CATS = ("music", "music+speech")
MUSIC_LIKE = ("Music", "Musical instrument")       # left out when naming what an 'other sound' second is


def rms_db_1hz(y: np.ndarray, sr: int, n_seconds: int) -> np.ndarray:
    """[n_seconds] RMS level (dBFS) of each 1 s block of mono audio y (sample rate sr); missing tail = -inf-ish."""
    out = np.full(n_seconds, -120.0)
    for k in range(n_seconds):
        b = y[k * sr:(k + 1) * sr]
        if len(b):
            out[k] = 20 * np.log10(np.sqrt(np.mean(b.astype(np.float64) ** 2)) + 1e-12)
    return out


def second_categories(p_music: np.ndarray, p_speech: np.ndarray, rms_db: np.ndarray, thr: float = 0.5,
                      weak: float = 0.2, silent_db: float = -50.0) -> np.ndarray:
    """[T] category names (CATS) per second from P(Music), P(Speech) (both [T], 0..1) and RMS level [T] (dBFS)."""
    m, s = p_music >= thr, p_speech >= thr
    cat = np.full(len(p_music), "other sound", dtype=object)
    cat[(p_music >= weak) & ~m] = "weak music"
    cat[m] = "music"
    cat[m & s] = "music+speech"
    cat[s & ~m] = "speech"
    cat[rms_db < silent_db] = "silent"
    return cat


def file_verdict(cats: np.ndarray) -> tuple[float, str]:
    """(BGM share, verdict) for one file's per-second categories."""
    if len(cats) == 0:
        return 0.0, "little BGM"
    share = float(np.isin(cats, BGM_CATS).mean())
    if share >= 0.9 and not np.any(cats == "speech"):
        return share, "clean BGM"
    return share, "mostly BGM" if share >= 0.5 else "little BGM"


def _load(cfg) -> list[dict]:
    from scipy.signal import butter, sosfiltfilt
    from panns_inference import labels
    from bgm.data import manifest
    from bgm.data.prep import interim_music_path
    from bgm.features.panns_tags import HP_CONFIG, load_tags, load_tags_hp
    L = list(labels)
    iM, iS = L.index("Music"), L.index("Speech")
    skip = [L.index(x) for x in MUSIC_LIKE]
    pairs = manifest.read_pairs(cfg).drop_duplicates("music_id")
    sos = butter(HP_CONFIG["highpass_order"], HP_CONFIG["highpass_hz"], "highpass", fs=48000, output="sos")
    rows = []
    for r in pairs.itertuples():
        P = load_tags(cfg, r.music_id)
        y, sr = sf.read(interim_music_path(cfg, r.music_id), dtype="float32")
        assert sr == 48000, f"{r.music_id}: interim music must be 48 kHz, got {sr}"
        y = y.mean(1) if y.ndim == 2 else y
        H = load_tags_hp(cfg, r.music_id)
        Q = H.copy()
        Q[:, skip] = 0
        rows.append({"music_id": r.music_id, "film": r.meta_film, "source": r.meta_source,
                     "raw": {"p_music": P[:, iM], "p_speech": P[:, iS], "rms_db": rms_db_1hz(y, sr, len(P)),
                             "silent_db": -50.0},
                     "hp": {"p_music": H[:, iM], "p_speech": H[:, iS],
                            "rms_db": rms_db_1hz(sosfiltfilt(sos, y.astype(np.float64)), sr, len(P)), "silent_db": -60.0},
                     "top_other": np.array([L[i] for i in Q.argmax(1)], dtype=object)})
    return rows


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _cats(v: dict, thr: float = 0.5) -> np.ndarray:
    return second_categories(v["p_music"], v["p_speech"], v["rms_db"], thr=thr, silent_db=v["silent_db"])


def _group(r: dict) -> str:
    return "Konjum salangai" if "Konjum" in str(r["film"]) else str(r["source"])


def build_report(rows: list[dict], songs: pd.DataFrame | None, dataset: str = "data/manifests") -> tuple[str, pd.DataFrame]:
    n_sec = sum(len(r["hp"]["p_music"]) for r in rows)
    views = {"raw": "raw audio", "hp": "hum removed"}
    C = {k: [_cats(r[k]) for r in rows] for k in views}
    per_file, other_top = [], Counter()
    for r, c, c_raw in zip(rows, C["hp"], C["raw"]):
        share, verdict = file_verdict(c)
        share_raw, verdict_raw = file_verdict(c_raw)
        other_top.update(r["top_other"][c == "other sound"])
        cc = Counter(c)
        per_file.append({"music_id": r["music_id"], "film": r["film"], "source": r["source"], "group": _group(r),
                         "seconds": len(c), "bgm_share": round(share, 3), "verdict": verdict,
                         "bgm_share_raw": round(share_raw, 3), "verdict_raw": verdict_raw,
                         **{f"s_{k}": cc.get(k, 0) for k in CATS},
                         "mean_p_music": round(float(r["hp"]["p_music"].mean()), 3),
                         "mean_p_music_raw": round(float(r["raw"]["p_music"].mean()), 3),
                         "median_db_above_150hz": round(float(np.median(r["hp"]["rms_db"])), 1)})
    df = pd.DataFrame(per_file)
    if songs is not None:
        df = df.merge(songs[["music_id", "label", "vocal_seconds"]].rename(columns={"label": "song_label"}),
                      on="music_id", how="left")
    VERD = ("clean BGM", "mostly BGM", "little BGM")

    out = [f"# How much real background music is in the dataset? (PANNs audit, D114; {dataset})", "",
           "Generated by `python -m bgm.analysis.bgm_presence` (src/bgm/analysis/bgm_presence.py). "
           f"PANNs CNN14 (AudioSet, 527 classes, frozen, fp32, 4 s context per second) on every second of the "
           f"{len(rows):,} music files ({n_sec:,} s = {n_sec / 3600:.2f} h). Two runs: **raw audio** (the music as "
           "delivered) and **hum removed** (high-passed at 150 Hz, file set to −20 dBFS; panns_tags.extract_hp), "
           "because Konjum salangai's raw audio carries a +0.06 DC offset and a 100 Hz hum over very quiet music (its "
           "sound is ~27 dB quieter than the other sources), which the raw run hears as silence. Per second, in this order: **silent** (raw: RMS < −50 "
           "dBFS; hum removed: RMS above 150 Hz < −60 dBFS, i.e. hum or silence only), **speech** (P(Speech) ≥ 0.5, "
           "P(Music) < 0.5), **music+speech** (both ≥ 0.5), **music** (P(Music) ≥ 0.5), **weak music** (0.2 ≤ "
           "P(Music) < 0.5), **other sound** (the rest). BGM seconds = music + music+speech. File verdict: **clean "
           "BGM** (BGM share ≥ 90% and no speech second), **mostly BGM** (≥ 50%), **little BGM** (< 50%). "
           "Sections 2–7: the verdicts use the hum-removed run. Per-file table: bgm_presence_files.csv in the same folder.", "",
           "## 1. Seconds", "", "| category | raw audio | hum removed |", "| --- | --- | --- |"]
    cnt = {k: Counter(np.concatenate(C[k])) for k in views}
    out += [f"| {c} | {cnt['raw'].get(c, 0):,} ({_pct(cnt['raw'].get(c, 0) / n_sec)}) | "
            f"{cnt['hp'].get(c, 0):,} ({_pct(cnt['hp'].get(c, 0) / n_sec)}) |" for c in CATS]
    bgm = {k: sum(cnt[k].get(c, 0) for c in BGM_CATS) for k in views}
    out.append(f"| **BGM seconds** | **{bgm['raw']:,} ({_pct(bgm['raw'] / n_sec)}) = {bgm['raw'] / 3600:.2f} h** | "
               f"**{bgm['hp']:,} ({_pct(bgm['hp'] / n_sec)}) = {bgm['hp'] / 3600:.2f} h** |")

    out += ["", "## 2. Files", "", "| verdict | raw audio | hum removed |", "| --- | --- | --- |"]
    vr, vh = df.verdict_raw.value_counts(), df.verdict.value_counts()
    out += [f"| {v} | {vr.get(v, 0):,} ({_pct(vr.get(v, 0) / len(df))}) | {vh.get(v, 0):,} "
            f"({_pct(vh.get(v, 0) / len(df))}) |" for v in VERD]
    out += ["", f"Hum removed: BGM share per file median {_pct(df.bgm_share.median())}, mean "
            f"{_pct(df.bgm_share.mean())}; files with any speech second: "
            f"{(df['s_speech'] + df['s_music+speech'] > 0).sum()}; with ≥ 3 speech-only seconds: "
            f"{(df['s_speech'] >= 3).sum()}; with ≥ 3 silent (hum-only) seconds: {(df['s_silent'] >= 3).sum()}.", ""]

    out += ["## 3. By source", "", f"Konjum salangai (one 1950s film, {(df.group == 'Konjum salangai').sum()} clips, train-only) is split out of "
            "full_bgm_dataset because of its DC offset, hum and low level.", "",
            "| group | files | hours | BGM seconds, raw | BGM seconds, hum removed | clean BGM | mostly BGM | little BGM |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rest = df[df.group != "Konjum salangai"]
    rb = (rest["s_music"] + rest["s_music+speech"]).sum()
    rvc = rest.verdict.value_counts()
    summary = (f"**Without Konjum salangai** (the {len(rest):,} clips the test folds draw from): BGM seconds "
               f"{_pct(rb / rest.seconds.sum())} ({rb / 3600:.2f} of {rest.seconds.sum() / 3600:.2f} h); files: clean BGM "
               f"{rvc.get('clean BGM', 0)} ({_pct(rvc.get('clean BGM', 0) / len(rest))}), mostly BGM "
               f"{rvc.get('mostly BGM', 0)} ({_pct(rvc.get('mostly BGM', 0) / len(rest))}), little BGM "
               f"{rvc.get('little BGM', 0)} ({_pct(rvc.get('little BGM', 0) / len(rest))}).")
    for g, d in df.groupby("group"):
        sec = d.seconds.sum()
        vc = d.verdict.value_counts()
        raw_b = sum(np.isin(c, BGM_CATS).sum() for c, gg in zip(C["raw"], df.group) if gg == g)
        out.append(f"| {g} | {len(d)} | {sec / 3600:.2f} | {_pct(raw_b / sec)} | "
                   f"{_pct((d['s_music'] + d['s_music+speech']).sum() / sec)} | {vc.get('clean BGM', 0)} | "
                   f"{vc.get('mostly BGM', 0)} | {vc.get('little BGM', 0)} |")

    out += ["", summary]
    out += ["", "## 4. Sensitivity to the threshold (hum removed)", "",
            "| threshold | BGM seconds | speech-only seconds | clean BGM files | mostly BGM | little BGM |",
            "| --- | --- | --- | --- | --- | --- |"]
    for thr in (0.3, 0.5, 0.7):
        cs = [_cats(r["hp"], thr) for r in rows]
        c = np.concatenate(cs)
        v = Counter(file_verdict(x)[1] for x in cs)
        out.append(f"| {thr} | {_pct(np.isin(c, BGM_CATS).mean())} | {_pct((c == 'speech').mean())} | "
                   f"{v.get('clean BGM', 0)} | {v.get('mostly BGM', 0)} | {v.get('little BGM', 0)} |")

    n_other = cnt["hp"].get("other sound", 0)
    out += ["", "## 5. What the 'other sound' seconds are (hum removed)", "",
            f"Top AudioSet class of the {n_other:,} other-sound seconds, Music and Musical instrument left out.", "",
            "| class | seconds | share of other sound |", "| --- | --- | --- |"]
    out += [f"| {k} | {v:,} | {_pct(v / max(n_other, 1))} |" for k, v in other_top.most_common(12)]

    if songs is not None and "song_label" in df:
        out += ["", "## 6. Singing (cross-check with data/manifests/song_labels.csv)", "",
                "PANNs' Music class does not separate a song from background score. song_labels.csv: Demucs vocal "
                "share + CLAP zero-shot (song = CLAP singing wins and ≥ 30% vocal seconds).", "",
                "| song_labels | files | clean BGM | mostly BGM | little BGM |", "| --- | --- | --- | --- | --- |"]
        for lab, g in df.groupby("song_label", dropna=False):
            vc = g.verdict.value_counts()
            out.append(f"| {lab} | {len(g)} | {vc.get('clean BGM', 0)} | {vc.get('mostly BGM', 0)} | "
                       f"{vc.get('little BGM', 0)} |")
        out += ["", f"Files with Demucs vocal seconds ≥ 30%: {(df.vocal_seconds >= 0.3).sum()}."]

    worst = df.sort_values(["bgm_share", "mean_p_music"]).head(15)
    out += ["", "## 7. Files with the least music (listen to these first)", "",
            "| music_id | seconds | BGM share (raw → hum removed) | speech s | silent s | other s | weak s | mean P(Music) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    out += [f"| {r.music_id} | {r.seconds} | {_pct(r.bgm_share_raw)} → {_pct(r.bgm_share)} | {r.s_speech} | "
            f"{r.s_silent} | {r['s_other sound']} | {r['s_weak music']} | {r.mean_p_music} |"
            for _, r in worst.iterrows()]
    return "\n".join(out) + "\n", df


def main(overrides: list[str] = ()) -> None:
    """Overrides as for bgm commands; the report goes to <paths.results>/bgm_presence.md (+ _files.csv)."""
    from bgm.config import load_config
    cfg = load_config("configs/default.yaml", overrides)
    rows = _load(cfg)
    sl = Path(cfg.paths.manifests) / "song_labels.csv"
    md, df = build_report(rows, pd.read_csv(sl) if sl.exists() else None, str(cfg.paths.manifests))
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bgm_presence.md").write_text(md)
    df.to_csv(out / "bgm_presence_files.csv", index=False)
    log.info("wrote %s/bgm_presence.md, bgm_presence_files.csv (%d files)", out, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    import sys
    main(sys.argv[1:])
