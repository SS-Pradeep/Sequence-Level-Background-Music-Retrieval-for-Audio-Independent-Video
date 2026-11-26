"""Rendering (SPEC §5.11): runs -> crossfades -> loudness -> fade-out -> limiter -> (optional) mux.

A run is a maximal stretch of segments whose windows are contiguous in one file; runs are the units of rendering.
Between runs: a crossfade (equal-power by default, linear with `crossfade_shape: linear`) of 4 beats at the incoming tempo, clamped to [1, 2] s, centred on the
boundary, shortened when the audio before / after is not there. Each run is normalized to `target_lufs` (runs
shorter than `min_run_for_loudness_s` use their file's overall loudness), gain capped at +/- `gain_cap_db`. Optionally
(`gain_ramp_s` > 0) linear gain ramps in dB meet the two sides' local loudness at each seam (`seam_gain_ramps`).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000


@dataclass
class Run:
    music_id: str
    offset: float        # file time at the run's start, s
    start: float         # video time where the run starts, s
    end: float           # video time where it ends, s
    tempo: float


def runs_from_path(chosen, segments, tol_s: float = 0.04) -> list[Run]:
    """chosen: Candidate per segment; segments: [(a, b)] on the video timeline."""
    runs = []
    for c, (a, b) in zip(chosen, segments):
        if runs and runs[-1].music_id == c.music_id and abs(c.offset - (runs[-1].offset + runs[-1].end - runs[-1].start)) <= tol_s:
            runs[-1].end = b
        else:
            runs.append(Run(c.music_id, c.offset, a, b, c.tempo))
    return runs


def beat_sync_runs(runs: list[Run], cuts_s, beats_of, file_dur_of, max_shift_s: float = 2.0,
                   tol_s: float = 0.07) -> tuple[list[Run], list[dict]]:
    """Render-time beat sync (no model involved): move each run's file offset onto a nearby beat, so the music
    enters on a beat exactly at the run's start (a cut), choosing among the beats within `max_shift_s` of the retrieved
    offset the one that puts the most of the run's internal cuts within `tol_s` of a beat (ties: the smallest shift).
    Runs whose file has no beats in reach keep their offset. Returns new runs and a per-run report."""
    cuts = np.sort(np.asarray(cuts_s, float))
    out, rep = [], []
    for run in runs:
        beats = np.sort(np.asarray(beats_of(run.music_id), float))
        L = run.end - run.start
        inner = cuts[(cuts > run.start + 1e-6) & (cuts < run.end - 1e-6)] - run.start           # cut times inside the run
        cand = beats[(np.abs(beats - run.offset) <= max_shift_s) & (beats + L <= file_dur_of(run.music_id) + 1e-6)]

        def hits(o: float) -> int:
            if not len(inner) or not len(beats):
                return 0
            t = o + inner
            return int((np.min(np.abs(t[:, None] - beats[None, :]), 1) <= tol_s).sum())
        before = hits(run.offset)
        if len(cand):
            best = max(cand, key=lambda b: (hits(b), -abs(b - run.offset)))
            new = Run(run.music_id, float(best), run.start, run.end, run.tempo)
        else:
            new = run
        out.append(new)
        rep.append({"music_id": run.music_id, "start": run.start, "offset_before": run.offset, "offset_after": new.offset,
                    "shift_s": new.offset - run.offset, "inner_cuts": int(len(inner)), "on_beat_before": before,
                    "on_beat_after": hits(new.offset), "enters_on_beat": bool(len(cand))})
    return out, rep


def fill_short_runs(runs: list[Run], beats_of, file_dur_of, tail_s: float = 1.0) -> tuple[list[Run], list[dict]]:
    """Runs longer than the audio left in their file used to render silence after the file ended. Split such a run at
    the last beat that leaves `tail_s` of audio for the crossfade, and continue from the beat nearest the run's own
    offset (a musical repeat; file start when it has no beats), until the run's span is filled. The splits are ordinary
    run boundaries, so the renderer crossfades them. Returns new runs and one report entry per filled run."""
    out, rep = [], []
    for run in runs:
        dur = file_dur_of(run.music_id)
        if run.offset + (run.end - run.start) <= dur + 1e-3 or dur < 2 * tail_s:
            out.append(run)
            continue
        beats = np.sort(np.asarray(beats_of(run.music_id), float))
        loop_at = float(beats[np.argmin(np.abs(beats - run.offset))]) if len(beats) else 0.0
        t, off, parts = run.start, run.offset, 0
        while run.end - t > 1e-3:
            ok = beats[(beats > off + 0.5) & (beats <= dur - tail_s)]
            cut = float(ok[-1]) if len(ok) else dur
            span = min(cut - off, run.end - t)
            if span <= 1e-3:                                  # no audio left from this offset: restart at the loop point
                off = 0.0 if off <= loop_at + 1e-6 else loop_at
                continue
            out.append(Run(run.music_id, off, t, t + span, run.tempo))
            t, off, parts = t + span, loop_at, parts + 1
        rep.append({"music_id": run.music_id, "start": run.start, "end": run.end, "file_s": dur, "parts": parts, "loop_at": loop_at})
    return out, rep


def equal_power(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(fade-out, fade-in) gains with cos^2 + sin^2 = 1 (constant power)."""
    t = (np.arange(n) + 0.5) / max(n, 1)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def linear(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(fade-out, fade-in) gains that ramp linearly and sum to 1 (constant amplitude; dips ~3 dB mid-fade on
    uncorrelated music, where equal power does not)."""
    t = (np.arange(n) + 0.5) / max(n, 1)
    return 1.0 - t, t


CROSSFADE_SHAPES = ("equal_power", "linear", "sqrt", "smoothstep", "smoothstep_power", "quadratic", "slow", "sinc",
                    "smootherstep", "smootherstep_power")
SINC_LOBES = 3                       # "sinc": half-width of the band-limited step, in sinc lobes (D51)


def crossfade_gains(n: int, shape: str = "equal_power") -> tuple[np.ndarray, np.ndarray]:
    """(fade-out, fade-in) gains of length n. Mid-fade power on uncorrelated material (out^2 + in^2, dB at t = 0.5):
    equal_power 0; linear -3 (constant amplitude); sqrt 0 (constant power, steep ends); smoothstep -3 (constant
    amplitude, zero-slope ends); smoothstep_power 0 (constant power, zero-slope ends); quadratic -9 (a DAW
    "exponential" pair, dips); slow +0.5 (a DAW "logarithmic" pair, bumps); sinc -3 plus Gibbs ripple (the
    Whittaker-Shannon band-limited step: gains overshoot 1 and dip below 0 near the boundary, D51);
    smootherstep -3 and smootherstep_power 0 (the quintic C2 Hermite 6t^5 - 15t^4 + 10t^3, amplitude / power form).
    Equivalences: the cubic Hermite step with flat tangents IS smoothstep; a natural cubic spline on two knots IS
    linear (D51)."""
    if shape == "equal_power":
        return equal_power(n)
    if shape == "linear":
        return linear(n)
    t = (np.arange(n) + 0.5) / max(n, 1)
    if shape == "sqrt":
        return np.sqrt(1.0 - t), np.sqrt(t)
    if shape == "smoothstep":
        s = t * t * (3.0 - 2.0 * t)
        return 1.0 - s, s
    if shape == "smoothstep_power":
        s = t * t * (3.0 - 2.0 * t)
        return np.cos(s * np.pi / 2), np.sin(s * np.pi / 2)
    if shape == "quadratic":
        return (1.0 - t) ** 2, t ** 2
    if shape == "slow":
        return 1.0 - t ** 2, 1.0 - (1.0 - t) ** 2
    if shape == "smootherstep":
        s = t ** 3 * (t * (6.0 * t - 15.0) + 10.0)
        return 1.0 - s, s
    if shape == "smootherstep_power":
        s = t ** 3 * (t * (6.0 * t - 15.0) + 10.0)
        return np.cos(s * np.pi / 2), np.sin(s * np.pi / 2)
    if shape == "sinc":              # Whittaker-Shannon: the band-limited unit step, fade-out = its complement.
        from scipy.special import sici
        si_edge = sici(np.pi * SINC_LOBES)[0]
        g = (sici(np.pi * SINC_LOBES * (2.0 * t - 1.0))[0] + si_edge) / (2.0 * si_edge)
        return 1.0 - g, g            # constant amplitude; Gibbs ripple puts g outside [0, 1] near the seam (~5% at L = 3)
    raise ValueError(f"unknown crossfade shape {shape!r}; one of {CROSSFADE_SHAPES}")


def _loud(meter, x: np.ndarray) -> float:
    if len(x) < int(0.4 * SR) or not np.any(np.abs(x) > 1e-7):
        return -np.inf
    return float(meter.integrated_loudness(x))


def crossfade_lengths(runs: list[Run], cfg_render) -> list[float]:
    """Crossfade length (s) at each of the len(runs) - 1 seams: `crossfade_beats` at the incoming tempo, clamped."""
    r = cfg_render
    return [float(np.clip(r.crossfade_beats * 60.0 / max(run.tempo, 1e-3), r.crossfade_min_s, r.crossfade_max_s))
            if run.tempo > 0 else r.crossfade_min_s for run in runs[1:]]


def seam_gain_ramps(parts, runs: list[Run], meter, ramp_s: float, window_s: float = 3.0, max_db: float = 6.0) -> list[np.ndarray]:
    """Per-run gain envelopes in dB (one value per sample of the run's extended audio) that meet in level at each seam.

    parts: (audio at its normalization gain, pre s, post s) per run, the audio spanning video time [start - pre, end + post].
    At seam k, d = loudness(first `window_s` of the incoming run) - loudness(last `window_s` of the outgoing run), LU,
    clipped to +/- max_db. The outgoing run ramps linearly (in dB) from 0 to +d/2 over its last R s and holds that through
    its fade-out; the incoming run holds -d/2 through its fade-in and ramps back to 0 over its first R s, with
    R = min(ramp_s, half the run) so ramps from a run's two seams never overlap. d = 0 when either side is silent."""
    env = [np.zeros(len(seg)) for seg, _, _ in parts]
    t = [run.start - pre + np.arange(len(seg)) / SR for run, (seg, pre, _) in zip(runs, parts)]   # video time per sample

    def level(k: int, a: float, b: float) -> float:
        seg, pre, _ = parts[k]
        i0 = max(0, int(round((a - runs[k].start + pre) * SR)))
        return _loud(meter, seg[i0:i0 + int(round((b - a) * SR))])
    for k in range(1, len(runs)):
        o, i = runs[k - 1], runs[k]
        l_out = level(k - 1, o.end - min(window_s, o.end - o.start), o.end)
        l_in = level(k, i.start, i.start + min(window_s, i.end - i.start))
        if not (np.isfinite(l_out) and np.isfinite(l_in)):
            continue
        d = float(np.clip(l_in - l_out, -max_db, max_db))
        r_out, r_in = min(ramp_s, (o.end - o.start) / 2), min(ramp_s, (i.end - i.start) / 2)
        env[k - 1] += d / 2 * np.clip(1 - (o.end - t[k - 1]) / max(r_out, 1e-6), 0, 1)
        env[k] += -d / 2 * np.clip(1 - (t[k] - i.start) / max(r_in, 1e-6), 0, 1)
    return env


def render_soundtrack(runs: list[Run], duration_s: float, music_path, cfg_render) -> tuple[np.ndarray, list[float]]:
    """Mono float32 soundtrack at 48 kHz of exactly `duration_s`, plus the seam times (video s)."""
    import pyloudnorm as pyln
    r = cfg_render
    assert r.sr == SR
    meter = pyln.Meter(SR)
    shape = r.get("crossfade_shape", "equal_power")
    N = int(round(duration_s * SR))
    mix = np.zeros(N, np.float64)
    audio = {}
    for run in runs:
        if run.music_id not in audio:
            y, sr = sf.read(music_path(run.music_id), dtype="float32")
            assert sr == SR, f"interim music must be {SR} Hz"
            audio[run.music_id] = y
    xf = [0.0] + crossfade_lengths(runs, r) + [0.0]
    seams = [run.start for run in runs[1:]]
    parts = []                                                # (audio at its normalization gain, pre s, post s) per run
    for k, run in enumerate(runs):
        y = audio[run.music_id]
        # extend by half a crossfade on each side, shortened when the file has no audio there
        pre = min(xf[k] / 2, run.offset)
        post_avail = len(y) / SR - (run.offset + run.end - run.start)
        post = max(0.0, min(xf[k + 1] / 2, post_avail))
        s0 = int(round((run.offset - pre) * SR))
        seg = y[s0:s0 + int(round((run.end - run.start + pre + post) * SR))].astype(np.float64)
        loud = _loud(meter, seg) if run.end - run.start >= r.min_run_for_loudness_s else _loud(meter, y.astype(np.float64))
        if np.isfinite(loud):
            seg = seg * 10 ** (np.clip(r.target_lufs - loud, -r.gain_cap_db, r.gain_cap_db) / 20)
        parts.append((seg, pre, post))
    if r.get("gain_ramp_s", 0.0) > 0 and len(runs) > 1:
        env = seam_gain_ramps(parts, runs, meter, r.gain_ramp_s, r.gain_ramp_window_s, r.gain_ramp_max_db)
        parts = [(seg * 10 ** (e / 20), pre, post) for (seg, pre, post), e in zip(parts, env)]
    for k, (run, (seg, pre, post)) in enumerate(zip(runs, parts)):
        n_in, n_out = int(round(2 * pre * SR)) if k else 0, int(round(2 * post * SR)) if k < len(runs) - 1 else 0
        n_in, n_out = min(n_in, len(seg)), min(n_out, len(seg))
        if n_in:
            seg[:n_in] *= crossfade_gains(n_in, shape)[1]
        if n_out:
            seg[len(seg) - n_out:] *= crossfade_gains(n_out, shape)[0]
        t0 = int(round((run.start - pre) * SR))
        lo, hi = max(0, t0), min(N, t0 + len(seg))
        if hi > lo:
            mix[lo:hi] += seg[lo - t0:hi - t0]
    fade = int(min(r.fade_out_s, r.fade_out_max_frac * duration_s) * SR)
    if fade:
        mix[N - fade:] *= np.linspace(1, 0, fade)
    return lookahead_limiter(mix, r.limiter_dbfs), seams


def lookahead_limiter(x: np.ndarray, ceiling_dbfs: float = -1.0, lookahead_s: float = 0.005, release_s: float = 0.05) -> np.ndarray:
    """Peak limiter that never adds gain: gain = min(1, ceiling / max|x| over the next `lookahead_s`), with instant attack
    and an exponential release. (pedalboard's JUCE Limiter was tried first; it adds makeup gain, ~+5 dB here.)"""
    from scipy.ndimage import maximum_filter1d
    ceil = 10 ** (ceiling_dbfs / 20)
    n = max(1, int(lookahead_s * SR))
    peak = maximum_filter1d(np.abs(x), size=2 * n + 1, mode="nearest")
    want = np.minimum(1.0, ceil / np.maximum(peak, 1e-12))
    if want.min() >= 1.0:
        return x.astype(np.float32)
    return (x * _release(want, float(np.exp(-1.0 / (release_s * SR))))).astype(np.float32)


def _release_py(want: np.ndarray, a: float) -> np.ndarray:
    """Attack: follow the wanted gain down instantly; release: recover exponentially."""
    g = np.empty_like(want)
    cur = 1.0
    for i in range(len(want)):
        w = want[i]
        cur = w if w < cur else a * cur + (1 - a) * w
        g[i] = cur
    return g


try:
    from numba import njit
    _release = njit(cache=True)(_release_py)
except ImportError:  # pragma: no cover
    _release = _release_py


def seam_loudness_jumps(mix: np.ndarray, seams: list[float], window_s: float = 3.0) -> list[float]:
    """|short-term loudness over window_s after - before| at each seam, LU (NaN when either side is silent / too short)."""
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    out = []
    for t in seams:
        a, b = int((t - window_s) * SR), int((t + window_s) * SR)
        if a < 0 or b > len(mix):
            out.append(float("nan"))
            continue
        la, lb = _loud(meter, mix[a:int(t * SR)].astype(np.float64)), _loud(meter, mix[int(t * SR):b].astype(np.float64))
        out.append(abs(lb - la) if np.isfinite(la) and np.isfinite(lb) else float("nan"))
    return out


def mux(video: str | Path, bgm_wav: str | Path, out: str | Path, video_start_s: float = 0.0, bitrate: str = "192k") -> None:
    """Mux onto the ORIGINAL video; assert the two stream start times differ by < 1 ms.

    No -itsoffset up front (SPEC §5.11 passes the video's start time): the stream-copied video is usually rebased to
    start at 0, so offsetting the audio by the source's video start (0.041 s on 1.mp4) put the music 19 ms late
    (PROGRESS.md D23). A source whose own audio starts before its first frame keeps the gap (12.mp4: B-frame delay
    without an edit list puts the first frame at 0.04 s, its audio at 0; D86): then the soundtrack is muxed again with
    that many seconds of silence in front, so music sample 0 still meets the first frame. (Not -itsoffset: the AAC
    encoder's 1024 priming samples are then played instead of skipped, which shifts the stream start time by 21 ms.)
    `video_start_s` is kept for the record only."""
    def run(wav: str | Path) -> dict[str, float]:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate, "-shortest", str(out)], check=True)
        j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(out)],
                                      capture_output=True, text=True, check=True).stdout)
        return {s["codec_type"]: float(s["start_time"]) for s in j["streams"]}
    st, lead = run(bgm_wav), 0.0
    if st["video"] - st["audio"] >= 1e-3:
        info = sf.info(str(bgm_wav))
        y, _ = sf.read(bgm_wav, dtype="float32", always_2d=True)
        n = int(round((st["video"] - st["audio"]) * info.samplerate))
        padded = Path(out).with_suffix(".lead.wav")
        sf.write(padded, np.concatenate([np.zeros((n, y.shape[1]), np.float32), y]), info.samplerate, subtype=info.subtype)
        st, lead = run(padded), n / info.samplerate
        padded.unlink()
    assert abs(st["video"] - (st["audio"] + lead)) < 1e-3, f"music sample 0 and the first video frame differ: {st}, lead {lead:.4f} s"


def silence_envelope(n: int, spans: list[tuple[float, float]], fade_s: float, sr: int = SR) -> np.ndarray:
    """[n] gain: 0 inside each silent span, an equal-power fade over fade_s before it (out) and after it (in), 1 elsewhere."""
    g = np.ones(n)
    f = int(round(fade_s * sr))
    out_g, in_g = crossfade_gains(max(f, 1), "equal_power")
    for a, b in spans:
        i, j = int(round(a * sr)), int(round(b * sr))
        g[max(0, i):min(n, j)] = 0.0
        if f:
            lo = max(0, i - f)
            g[lo:i] = np.minimum(g[lo:i], out_g[f - (i - lo):])
            hi = min(n, j + f)
            g[j:hi] = np.minimum(g[j:hi], in_g[:hi - j])
    return g
