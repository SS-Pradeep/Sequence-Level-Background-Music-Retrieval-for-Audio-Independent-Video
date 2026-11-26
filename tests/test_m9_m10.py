"""M9 / M10: fusion z-normalization, prefix means, transition costs, Viterbi vs brute force, continuations."""
import itertools

import numpy as np
import pytest

from bgm.retrieval.index import Prefix
from bgm.retrieval.pipeline import fuse, zstats
from bgm.retrieval.stitch import Candidate, add_continuations, greedy, transition_cost, viterbi


def test_fuse_standardizes_with_sigma_floor():
    raw = {"sem": np.array([1.0, 2.0, 3.0]), "dyn": np.array([5.0, 5.0, 5.0])}   # constant branch: sigma floored
    st = zstats(raw)
    assert st["dyn"][1] == 1e-3
    s = fuse(raw, st, {"sem": 0.5, "dyn": 0.5})
    assert np.allclose(s, 0.5 * (raw["sem"] - 2) / raw["sem"].std())
    big = {"sem": 1000 * raw["sem"], "dyn": raw["sem"][::-1].copy()}
    assert np.allclose(fuse(big, zstats(big), {"sem": 0.5, "dyn": 0.5}), 0)          # scale-free


def test_prefix_means_match_brute_force():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    p = Prefix(x)
    a = rng.uniform(0, 39, 200)
    b = a + rng.uniform(0.1, 15, 200)
    got = p.means(a, b)
    c = np.arange(40) + 0.5
    for i in range(200):
        inside = (c >= a[i]) & (c < b[i])
        want = x[inside].mean(0) if inside.any() else x[int(np.clip(np.floor((a[i] + b[i]) / 2), 0, 39))]
        assert np.allclose(got[i], want)


def cand(m, off, dur, cost, key=0, tempo=120.0):
    return Candidate(m, off, dur, key, tempo, cost)


def test_transition_cost_cases():
    a = cand("x", 10.0, 5.0, 0)
    assert transition_cost(a, cand("x", 15.02, 5, 0), 1.0, 2.0, 0.5) == 0.0             # contiguous
    assert transition_cost(a, cand("x", 3.0, 5, 0), 1.0, 2.0, 0.5) == 1.0               # jump
    assert transition_cost(a, cand("y", 0.0, 5, 0, key=0, tempo=240.0), 1.0, 2.0, 0.5) == pytest.approx(2.0)   # double time, same key
    assert transition_cost(a, cand("y", 0.0, 5, 0, key=6, tempo=170.0), 1.0, 2.0, 0.5) > 2.5                   # tritone + tempo clash


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = rng.integers(2, 5)
        cands = [[cand(str(rng.integers(3)), float(rng.choice([0, 5, 10])), 5.0, float(rng.normal()), key=int(rng.integers(12)),
                       tempo=float(rng.uniform(70, 180))) for _ in range(rng.integers(1, 4))] for _ in range(n)]
        tc = dict(lam_jump=0.7, lam_switch=1.3, mu=0.5)

        def total(path):
            return sum(cands[i][j].cost for i, j in enumerate(path)) + sum(
                transition_cost(cands[i][path[i]], cands[i + 1][path[i + 1]], **tc) for i in range(n - 1))
        best = min(itertools.product(*[range(len(c)) for c in cands]), key=total)
        assert total(viterbi(cands, **tc)) == pytest.approx(total(best))


def test_viterbi_rejects_infinite_penalties():
    cands = [[cand("x", 0, 5, 0)], [cand("y", 0, 5, 0)]]
    with pytest.raises(AssertionError):
        viterbi(cands, lam_jump=1.0, lam_switch=float("inf"), mu=0.0)


def test_continuations_and_greedy():
    prev = [cand("x", 0.0, 5.0, 0.0), cand("y", 20.0, 5.0, 0.0)]
    cur = [cand("z", 0.0, 4.0, -1.0)]
    out = add_continuations(prev, cur, 4.0, {"x": 30.0, "y": 27.0}, lambda m, o, d: 0.5)
    assert [(c.music_id, c.offset) for c in out] == [("z", 0.0), ("x", 5.0)]           # y would run past its end
    assert greedy([out]) == [0]


# ---------------------------------------------------------------- vectorized stitching == reference
from bgm.retrieval.stitch_eval import Layer, decode_lattice, switch_f1, transition_matrix  # noqa: E402


def random_layers(rng, n_seg, n_top=3, n_files=3):
    layers = []
    for i in range(n_seg):
        m = rng.integers(0, n_files, n_top)
        off = rng.choice([0.0, 5.0, 10.0], n_top)
        layers.append(Layer(m, off, np.full(n_top, 5.0), rng.integers(-1, 12, n_top), rng.choice([-1.0, 90.0, 120.0, 175.0], n_top),
                            rng.normal(size=n_top), n_top, None if i == 0 else np.full(len(layers[-1].mid), -1)))
    return layers


def to_cands(L):
    return [Candidate(str(int(m)), float(o), float(d), int(k), float(t), float(c)) for m, o, d, k, t, c in
            zip(L.mid, L.off, L.dur, L.key, L.tempo, L.cost)]


def test_transition_matrix_equals_reference():
    rng = np.random.default_rng(1)
    pen = dict(lam_jump=0.7, lam_switch=1.9, mu=0.5)
    for _ in range(20):
        P, C = random_layers(rng, 2, 5)
        T = transition_matrix(P, C, pen)
        ref = np.array([[transition_cost(a, b, **pen) for b in to_cands(C)] for a in to_cands(P)])
        assert np.allclose(T, ref)


def test_decode_lattice_without_beam_equals_viterbi():
    rng = np.random.default_rng(2)
    pen = dict(lam_jump=0.4, lam_switch=1.1, mu=0.3)
    for _ in range(30):
        layers = random_layers(rng, int(rng.integers(2, 6)), int(rng.integers(1, 5)))
        cands = [to_cands(L) for L in layers]
        got = decode_lattice(layers, pen, beam=None)
        ref = viterbi(cands, **pen)

        def total(path):
            return sum(cands[i][j].cost for i, j in enumerate(path)) + sum(
                transition_cost(cands[i][path[i]], cands[i + 1][path[i + 1]], **pen) for i in range(len(path) - 1))
        assert total(got) == pytest.approx(total(ref))


def test_switch_f1():
    assert switch_f1([10.2, 30.0], [10.0, 20.0]) == pytest.approx(0.5)
    assert switch_f1([], []) == 1.0 and switch_f1([5.0], []) == 0.0
    assert switch_f1([9.5, 10.4], [10.0]) == pytest.approx(2 * 0.5 * 1 / 1.5)          # one-to-one matching


# ---------------------------------------------------------------- rendering (SPEC M10 acceptance)
def test_equal_power_crossfade_is_constant_power():
    from bgm.render.render import equal_power
    out, inn = equal_power(1000)
    assert np.allclose(out ** 2 + inn ** 2, 1.0)


def test_linear_crossfade_is_constant_amplitude():
    from bgm.render.render import crossfade_gains
    out, inn = crossfade_gains(1000, "linear")
    assert np.allclose(out + inn, 1.0) and out[0] > 0.99 and inn[-1] > 0.99
    assert np.all(np.diff(out) < 0) and np.allclose(np.diff(inn), np.diff(inn)[0])      # monotone, straight ramp


@pytest.mark.parametrize("shape", ["equal_power", "linear", "sqrt", "smoothstep", "smoothstep_power", "quadratic", "slow",
                                   "smootherstep", "smootherstep_power"])
def test_crossfade_gain_family(shape):
    import numpy as np

    from bgm.render.render import crossfade_gains
    out, inn = crossfade_gains(2001, shape)
    assert out.shape == inn.shape == (2001,)
    assert np.all((out >= 0) & (out <= 1)) and np.all((inn >= 0) & (inn <= 1))
    assert np.all(np.diff(out) <= 1e-12) and np.all(np.diff(inn) >= -1e-12)          # monotone
    assert out[0] > 0.98 and inn[0] < 0.02 and out[-1] < 0.02 and inn[-1] > 0.98    # full fade
    assert abs(out[1000] - inn[1000]) < 1e-6                                         # symmetric at the midpoint
    mid_db = 10 * np.log10(out[1000] ** 2 + inn[1000] ** 2)
    theory = {"equal_power": 0.0, "linear": -3.01, "sqrt": 0.0, "smoothstep": -3.01,
              "smoothstep_power": 0.0, "quadratic": -9.03, "slow": 0.51,
              "smootherstep": -3.01, "smootherstep_power": 0.0}[shape]
    assert abs(mid_db - theory) < 0.05


def test_sinc_crossfade_is_the_bandlimited_step():
    """Whittaker-Shannon (D51): exact endpoints, complementary gains, -3 dB mid like any constant-amplitude pair,
    and the Gibbs ripple of a band-limited step (~5.3% over/undershoot at SINC_LOBES = 3) - deliberately NOT monotone."""
    import numpy as np

    from bgm.render.render import crossfade_gains
    out, inn = crossfade_gains(48001, "sinc")
    assert np.allclose(out + inn, 1.0)                                   # complementary (constant amplitude)
    assert abs(inn[0]) < 1e-6 and abs(out[-1]) < 1e-6                    # exact endpoints (normalized Si)
    mid = 10 * np.log10(out[24000] ** 2 + inn[24000] ** 2)
    assert abs(mid - (-3.01)) < 0.02
    assert abs(inn.max() - 1.0529) < 0.004 and abs(inn.min() + 0.0529) < 0.004   # Gibbs over/undershoot
    assert np.any(np.diff(inn) < 0)                                      # ripple: not monotone


def test_crossfade_gains_rejects_unknown_shape():
    import pytest as _pytest

    from bgm.render.render import crossfade_gains
    with _pytest.raises(ValueError, match="unknown crossfade shape"):
        crossfade_gains(100, "bezier")


@pytest.mark.parametrize("shape", ["equal_power", "linear"])
def test_render_crossfade_shape_at_seam(tmp_path, shape):
    """Two runs of constant DC: across the seam the mix is the sum of the two faded pieces, so a linear fade stays flat
    at the (equal) level while equal power bumps up to sqrt(2) mid-fade."""
    import soundfile as sf
    from omegaconf import OmegaConf

    from bgm.render.render import SR, Run, render_soundtrack
    for name in ("a", "b"):
        sf.write(tmp_path / f"{name}.flac", np.full(30 * SR, 0.1, np.float32), SR, subtype="PCM_24")
    cfg = OmegaConf.load("configs/default.yaml").render
    cfg.crossfade_shape, cfg.fade_out_s = shape, 0.0
    runs = [Run("a", 5.0, 0.0, 10.0, 120.0), Run("b", 5.0, 10.0, 20.0, 120.0)]
    mix, _ = render_soundtrack(runs, 20.0, lambda m: tmp_path / f"{m}.flac", cfg)
    mid, flat = abs(mix[10 * SR]), abs(mix[5 * SR])
    assert np.isclose(mid / flat, 1.0 if shape == "linear" else np.sqrt(2), atol=0.02)


def test_render_duration_and_loudness(tmp_path):
    import soundfile as sf
    from omegaconf import OmegaConf

    from bgm.render.render import SR, Run, render_soundtrack
    rng = np.random.default_rng(0)
    t = np.arange(40 * SR) / SR
    for name, f in (("a", 220.0), ("b", 330.0)):
        y = 0.25 * np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(len(t))
        sf.write(tmp_path / f"{name}.flac", y.astype(np.float32), SR, subtype="PCM_24")
    cfg = OmegaConf.load("configs/default.yaml").render
    runs = [Run("a", 3.0, 0.0, 12.0, 120.0), Run("b", 10.0, 12.0, 25.0, 100.0)]
    mix, seams = render_soundtrack(runs, 25.0, lambda m: tmp_path / f"{m}.flac", cfg)
    assert len(mix) == 25 * SR and seams == [12.0]
    import pyloudnorm as pyln
    body = mix[: int(20 * SR)].astype(np.float64)                               # before the fade-out
    assert abs(pyln.Meter(SR).integrated_loudness(body) - cfg.target_lufs) < 1.0
    assert np.abs(mix).max() <= 10 ** (cfg.limiter_dbfs / 20) + 1e-3


def test_gain_ramps_meet_at_the_seam_and_return_to_zero():
    import pyloudnorm as pyln

    from bgm.render.render import SR, Run, seam_gain_ramps
    rng = np.random.default_rng(0)
    quiet, loud = 0.05 * rng.standard_normal(12 * SR), 0.2 * rng.standard_normal(12 * SR)   # ~12 dB apart
    runs = [Run("a", 1.0, 0.0, 10.0, 120.0), Run("b", 1.0, 10.0, 20.0, 120.0)]
    parts = [(quiet[: 11 * SR], 0.0, 1.0), (loud[: 11 * SR], 1.0, 0.0)]
    env = seam_gain_ramps(parts, runs, pyln.Meter(SR), ramp_s=4.0, window_s=3.0, max_db=20.0)
    d = env[0][-1] - env[1][0]                                   # outgoing +d/2, incoming -d/2
    assert 11.0 < d < 13.0 and np.isclose(env[0][-1], -env[1][0])
    assert env[0][: 5 * SR].max() == 0 and env[1][6 * SR:].max() == 0      # back to the normalization gain beyond R = 4 s
    assert np.all(np.diff(env[0]) >= -1e-12) and np.all(np.diff(env[1]) >= -1e-12)   # monotone linear ramps
    capped = seam_gain_ramps(parts, runs, pyln.Meter(SR), ramp_s=4.0, window_s=3.0, max_db=6.0)
    assert np.isclose(capped[0][-1] - capped[1][0], 6.0)


@pytest.mark.parametrize("edit_list", [1, 0])
def test_mux_starts_the_music_at_the_first_frame(tmp_path, edit_list):
    """B-frame reordering puts the first frame at pts 0.04 s. Without an MP4 edit list (12.mp4) and with the source's
    own audio at 0, the stream copy keeps that gap: the music must start with the first frame, not 40 ms before it."""
    import json
    import subprocess

    import soundfile as sf

    from bgm.render.render import SR, mux

    def starts(path):
        j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(path)],
                                      capture_output=True, text=True, check=True).stdout)
        return {s["codec_type"]: float(s["start_time"]) for s in j["streams"]}
    src, wav, out = tmp_path / "src.mp4", tmp_path / "bgm.wav", tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:d=2", "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=stereo", "-t", "2", "-c:v", "libx264", "-bf", "1", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-use_editlist", str(edit_list), str(src)], check=True)
    first = starts(src)["video"]
    assert np.isclose(first, 0.0 if edit_list else 0.04, atol=1e-3)
    y = np.zeros((2 * SR, 2), np.float32)
    y[SR // 2:SR // 2 + 48] = 0.9                                   # a click 0.5 s into the soundtrack
    sf.write(wav, y, SR)
    mux(src, wav, out, bitrate="320k")
    st = starts(out)
    pcm = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-map", "0:a", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                       capture_output=True, check=True).stdout, np.float32)
    click_s = st["audio"] + int(np.argmax(np.abs(pcm) > 0.3)) / SR   # where the click plays on the file's timeline
    assert abs(st["video"] - first) < 1e-3 and abs(click_s - (first + 0.5)) < 1e-3, (st, click_s)


def test_lookahead_limiter_caps_peaks_without_adding_gain():
    from bgm.render.render import SR, lookahead_limiter
    x = np.full(SR, 0.1, np.float32)
    x[SR // 2] = 2.0                                       # one spike
    y = lookahead_limiter(x, -1.0)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    assert np.allclose(y[: SR // 4], 0.1) and np.allclose(y[-SR // 4:], 0.1, atol=1e-3)   # untouched away from the spike


def test_beat_sync_enters_on_a_beat_and_puts_inner_cuts_on_beats():
    """A run retrieved 0.2 s off the beat grid: beat sync moves it onto a beat (entry at the cut) and picks the beat
    that lines the run's inner cuts up with beats; a file without beats in reach keeps its offset."""
    from bgm.render.render import Run, beat_sync_runs
    beats = {"m": np.arange(0.0, 60.0, 0.5), "flat": np.array([50.0])}
    cuts = [10.0, 11.5, 13.0, 14.4, 20.0]                   # run 1 spans 10-20 s; inner cuts at +1.5, +3.0, +4.4 s
    runs = [Run("m", 5.2, 10.0, 20.0, 120.0), Run("flat", 3.0, 20.0, 25.0, 100.0)]
    out, rep = beat_sync_runs(runs, cuts, lambda m: beats[m], lambda m: 60.0, max_shift_s=2.0, tol_s=0.07)
    assert out[0].offset in beats["m"] and abs(out[0].offset - 5.2) <= 2.0
    assert rep[0]["on_beat_before"] == 0 and rep[0]["on_beat_after"] == 2 and rep[0]["enters_on_beat"]  # +4.4 s never fits the grid
    assert out[0].offset == 5.0                              # ties broken by the smallest shift
    assert out[1] == runs[1] and not rep[1]["enters_on_beat"]  # no beat within 2 s of offset 3.0
    assert [(r.start, r.end) for r in out] == [(r.start, r.end) for r in runs]


def test_fill_short_runs_loops_at_beats_without_gaps():
    """A 30 s run on a 10 s file: split at beats, continued from the beat nearest its offset, no gaps, never past the file."""
    from bgm.render.render import Run, fill_short_runs
    beats = {"short": np.arange(0.25, 10.0, 0.5), "long": np.arange(0.0, 100.0, 0.5)}
    dur = {"short": 10.0, "long": 100.0}
    runs = [Run("long", 20.0, 0.0, 12.0, 120.0), Run("short", 2.26, 12.0, 42.0, 120.0)]
    out, rep = fill_short_runs(runs, lambda m: beats[m], lambda m: dur[m], tail_s=1.0)
    assert out[0] == runs[0] and len(rep) == 1 and rep[0]["parts"] == len(out) - 1 >= 3
    assert out[1].start == 12.0 and out[-1].end == 42.0
    assert all(abs(a.end - b.start) < 1e-9 for a, b in zip(out, out[1:]))                     # contiguous: no silence
    for r in out[1:]:
        assert r.offset + (r.end - r.start) <= dur["short"] - 1.0 + 1e-9 or r is out[-1]      # leaves the crossfade tail
        assert r.offset + (r.end - r.start) <= dur["short"] + 1e-9
    assert all(abs(r.offset - 2.25) < 1e-9 for r in out[2:])                                  # loops restart on the beat at 2.25 s
