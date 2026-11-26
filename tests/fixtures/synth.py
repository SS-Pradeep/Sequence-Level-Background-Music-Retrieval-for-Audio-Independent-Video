"""Synthetic video-music pairs with known ground truth (SPEC M0).

Each pair is 12-30 s of 25 fps, 320 x 240 video with a music track built on a beat grid:

* Music: a click on every beat at a known tempo (70-180 BPM), and a sustained pad chord.
* Cuts fall on random beats 2-5 s apart. Each cut brings a new background colour and a moving square whose
  speed changes, and the music plays an accented click with a different timbre and switches pad chords.
  The music at a cut therefore differs from a whole-beat shift of itself, so grounding is identifiable.
* The pad's pitch-class set follows the shot's colour palette, so the semantic branch has something to learn.
* "Embedded" pairs carry the music in the video's own audio stream, starting `audio_offset_s` (0.12 s)
  after the first video frame; the other pairs have a separate music file and a silent video.

Ground truth goes to `truth.json` next to the media. Everything is seeded; nothing touches the real dataset.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

FPS = 25
SIZE = (320, 240)          # width, height
MUSIC_SR = 44100           # like the real dataset; prep resamples to 48 kHz
EMBED_OFFSET_S = 0.12
SQUARE = 40

# (background RGB, square RGB, pad chord as pitch classes)
PALETTES: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, ...]]] = [
    ((200, 40, 40), (250, 230, 60), (0, 4, 7)),      # red -> C major
    ((40, 160, 60), (240, 240, 240), (9, 0, 4)),     # green -> A minor
    ((40, 70, 200), (250, 150, 30), (7, 11, 2)),     # blue -> G major
    ((220, 200, 40), (60, 20, 120), (2, 5, 9)),      # yellow -> D minor
    ((150, 50, 180), (40, 220, 220), (5, 9, 0)),     # purple -> F major
    ((30, 30, 30), (230, 90, 150), (4, 7, 11)),      # dark -> E minor
]


@dataclass
class SynthPair:
    pair_id: str
    video_path: str
    music_path: str | None           # None when the music is embedded in the video
    duration_s: float                # requested (and container) duration, a whole number of frames
    fps: int
    tempo_bpm: float
    beat_times_s: list[float]        # on the video timeline
    cut_times_s: list[float]         # time of the first frame of each new shot (frame-snapped)
    cut_beat_times_s: list[float]    # the beat each cut sits on (where the accent is)
    shot_palettes: list[int]
    shot_speeds: list[float]         # square speed, px / frame
    embedded: bool
    audio_offset_s: float            # audio stream start relative to the first video frame
    music_sr: int = MUSIC_SR
    extra: dict = field(default_factory=dict)


def _tone(freq: float, dur: float, sr: int, decay: float) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)


def _render_music(p: SynthPair, rng: np.random.Generator) -> np.ndarray:
    """Mono music on the video timeline (sample 0 = first video frame), peak 0.9."""
    sr, n = p.music_sr, int(round(p.duration_s * p.music_sr))
    y = np.zeros(n, np.float64)
    click = 0.3 * _tone(1500.0, 0.03, sr, 0.008)
    accent = 0.6 * (_tone(3200.0, 0.06, sr, 0.02) + 0.5 * rng.standard_normal(int(0.06 * sr)) * np.exp(-np.arange(int(0.06 * sr)) / (0.015 * sr)))
    cut_beats = set(np.round(p.cut_beat_times_s, 6))
    for b in p.beat_times_s:
        s = int(round(b * sr))
        w = accent if round(b, 6) in cut_beats else click
        e = min(n, s + len(w))
        y[s:e] += w[: e - s]
    bounds = [0.0] + list(p.cut_beat_times_s) + [p.duration_s]
    t = np.arange(n) / sr
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        i0, i1 = int(round(a * sr)), int(round(b * sr))
        seg = t[i0:i1]
        pad = sum(np.sin(2 * np.pi * 261.63 * 2 ** (pc / 12) * seg) for pc in PALETTES[p.shot_palettes[k]][2]) * 0.06
        ramp = min(len(seg) // 2, int(0.01 * sr))
        if ramp:
            env = np.ones(len(seg))
            env[:ramp], env[-ramp:] = np.linspace(0, 1, ramp), np.linspace(1, 0, ramp)
            pad = pad * env
        y[i0:i1] += pad
    return (0.9 * y / max(np.abs(y).max(), 1e-9)).astype(np.float32)


def _frames(p: SynthPair, rng: np.random.Generator):
    """Yield RGB frames [H, W, 3] uint8."""
    w, h = SIZE
    n = int(round(p.duration_s * p.fps))
    cuts = np.array(p.cut_times_s)
    ys = rng.integers(10, h - SQUARE - 10, len(p.shot_palettes))
    x, direction = float(rng.integers(0, w - SQUARE)), 1.0
    for f in range(n):
        k = int(np.searchsorted(cuts, f / p.fps + 1e-9, side="right"))
        bg, fg, _ = PALETTES[p.shot_palettes[k]]
        x += direction * p.shot_speeds[k]
        if x < 0 or x > w - SQUARE:
            direction = -direction
            x = float(np.clip(x, 0, w - SQUARE))
        img = np.empty((h, w, 3), np.uint8)
        img[:] = bg
        xi, yi = int(x), int(ys[k])
        img[yi:yi + SQUARE, xi:xi + SQUARE] = fg
        yield img


def make_pair(out_dir: Path, pair_id: str, rng: np.random.Generator, duration_s: float,
              embedded: bool, audio_offset_s: float = EMBED_OFFSET_S) -> SynthPair:
    duration_s = round(duration_s * FPS) / FPS
    tempo = float(rng.uniform(70, 180))
    period = 60.0 / tempo
    beats = np.arange(rng.uniform(0.1, period), duration_s, period)
    cut_beats, cur = [], 0.0
    while True:
        ok = beats[(beats - cur >= 2.0) & (beats - cur <= 5.0) & (beats <= duration_s - 1.0)]
        if not len(ok):
            break
        cur = float(rng.choice(ok))
        cut_beats.append(cur)
    palettes = [int(rng.integers(len(PALETTES)))]
    speeds = [float(rng.uniform(1, 10))]
    for _ in cut_beats:
        palettes.append(int(rng.choice([i for i in range(len(PALETTES)) if i != palettes[-1]])))
        speeds.append(float(rng.choice([s for s in np.arange(1.0, 10.5, 0.5) if abs(s - speeds[-1]) >= 3.0])))
    cut_frames = [int(np.ceil(b * FPS - 1e-9)) for b in cut_beats]
    p = SynthPair(pair_id=pair_id, video_path=str(out_dir / f"{pair_id}.mp4"),
                  music_path=None if embedded else str(out_dir / f"{pair_id}.wav"), duration_s=duration_s, fps=FPS,
                  tempo_bpm=tempo, beat_times_s=[float(b) for b in beats], cut_times_s=[f / FPS for f in cut_frames],
                  cut_beat_times_s=cut_beats, shot_palettes=palettes, shot_speeds=speeds, embedded=embedded,
                  audio_offset_s=audio_offset_s if embedded else 0.0)
    music = _render_music(p, rng)
    w, h = SIZE
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-"]
    with tempfile.TemporaryDirectory() as tmp:
        if embedded:
            wav = Path(tmp) / "a.wav"
            # the audio stream starts audio_offset_s after the first frame: drop that much from the start
            sf.write(wav, music[int(round(audio_offset_s * MUSIC_SR)):], MUSIC_SR)
            cmd += ["-itsoffset", f"{audio_offset_s}", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
        else:
            sf.write(p.music_path, np.stack([music, music], 1), MUSIC_SR)
        cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", p.video_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for img in _frames(p, rng):
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        assert proc.wait() == 0, f"ffmpeg failed for {pair_id}"
    return p


def make_pairs(out_dir: str | Path, n: int = 6, seed: int = 0, durations: tuple[float, float] = (12.0, 30.0),
               embed_every: int = 3) -> list[SynthPair]:
    """Write n pairs to out_dir (every `embed_every`-th one embedded, with a 0.12 s audio offset) plus truth.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        embedded = embed_every > 0 and i % embed_every == embed_every - 1
        pairs.append(make_pair(out_dir, f"synth_{seed}_{i:03d}", rng, float(rng.uniform(*durations)), embedded))
    (out_dir / "truth.json").write_text(json.dumps([asdict(p) for p in pairs], indent=1))
    return pairs


def load_truth(out_dir: str | Path) -> list[SynthPair]:
    return [SynthPair(**d) for d in json.loads((Path(out_dir) / "truth.json").read_text())]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for p in make_pairs(a.out_dir, a.n, a.seed):
        print(p.pair_id, f"{p.duration_s:.2f}s", f"{p.tempo_bpm:.0f}bpm", f"{len(p.cut_times_s)} cuts", "embedded" if p.embedded else "")
