"""`bgm check --plan renders/<name>.plan.json` (SPEC §5.13): a reference-free report for a render of a silent video.

Every check is compared with the beat-snapped control: random library windows with the same piece boundaries and
beat-snapped starts, 20 draws. Writes reports/check_<name>.md and reports/check_<name>.png.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy.stats import spearmanr

from bgm.eval.metrics import cut_on_beat_rate
from bgm.eval.store import FeatureStore, l2
from bgm.features.dynamics import M_CHANNELS, V_CHANNELS
from bgm.features.keys import fifths_distance
from bgm.retrieval.stitch_eval import within_piece_cuts

log = logging.getLogger(__name__)


def _excess(cuts, beats, dur, rng, tol, draws=20):
    if not len(cuts):
        return np.nan
    return cut_on_beat_rate(cuts, beats, tol) - np.mean([cut_on_beat_rate(cuts, (beats + rng.uniform(0, dur)) % dur, tol) for _ in range(draws)])


def soundtrack_curves(store, windows, dur):
    """windows: [(start, end, music_id, offset)] -> (beats on the video timeline, loudness rms_db at 25 Hz)."""
    n = int(round(dur * 25))
    loud = np.full(n, np.nan)
    beats = []
    for a, b, m, off in windows:
        md = store.music(m)
        bt = np.asarray(md["beat_times_s"], float)
        bt = bt[(bt >= off) & (bt < off + (b - a))]
        beats.append(a + bt - off)
        rms = md["dyn_25hz"][:, M_CHANNELS.index("rms_db")]
        i0, i1 = int(round(a * 25)), min(n, int(round(b * 25)))
        src = np.clip(int(round(off * 25)) + np.arange(i1 - i0), 0, len(rms) - 1)
        loud[i0:i1] = rms[src]
    return (np.concatenate(beats) if beats else np.zeros(0)), loud


def control_windows(store, pieces, rng):
    lib = list(dict.fromkeys(store.pairs.music_id))
    out = []
    for p in pieces:
        m = lib[rng.integers(len(lib))]
        d = p["end"] - p["start"]
        bt = np.asarray(store.music(m)["beat_times_s"], float)
        ok = bt[bt + d <= store.duration("music", m)]
        out.append((p["start"], p["end"], m, float(rng.choice(ok)) if len(ok) else 0.0))
    return out


def synced_windows(store, windows, cuts_s, cfg):
    """Apply the render's beat sync (bgm.render.render.beat_sync_runs) to [(start, end, music_id, offset)] windows."""
    from bgm.render.render import Run, beat_sync_runs
    runs, _ = beat_sync_runs([Run(m, off, a, b, 0.0) for a, b, m, off in windows], cuts_s, lambda m: store.music(m)["beat_times_s"],
                             lambda m: store.duration("music", m), cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
    return [(r.start, r.end, r.music_id, r.offset) for r in runs]


def energy_corr(motion_25hz, loud_25hz, win_s=2.0) -> float:
    k = int(win_s * 25)
    n = min(len(motion_25hz), len(loud_25hz)) // k
    if n < 3:
        return np.nan
    a = np.nanmean(motion_25hz[: n * k].reshape(n, k), 1)
    b = np.nanmean(loud_25hz[: n * k].reshape(n, k), 1)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= 3 else np.nan


def run_check(cfg: DictConfig, plan_path: str) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    name = Path(plan_path).name.removesuffix(".plan.json")
    work = Path(cfg.paths.renders) / name
    with np.load(work / "query.npz") as z:
        feat = {k: z[k] for k in z.files}
    store = FeatureStore(cfg)
    rng = np.random.default_rng(0)
    dur = plan["duration_s"]
    tol, draws = cfg.eval.cut_beat_tol_s, cfg.eval.circular_shift_draws
    segs, pieces = plan["segments"], plan["pieces"]
    ours = [(p["start"], p["end"], p["music_id"], p["offset"]) for p in pieces]        # what was rendered (after beat sync)
    all_cuts = np.asarray(feat["cut_times_s"], float)
    cuts = within_piece_cuts(all_cuts, dur, [p["start"] for p in pieces[1:]])
    beats, loud = soundtrack_curves(store, ours, dur)
    motion = feat["dyn_25hz"][:, V_CHANNELS.index("log_motion")]
    res = {"within_piece_cuts": int(len(cuts)), "sync_excess": _excess(cuts, beats, dur, rng, tol, draws),
           "energy_spearman": energy_corr(motion, loud)}
    synced = "beat_sync" in plan
    ctrl_sync, ctrl_energy, ctrl_synced = [], [], []
    for _ in range(draws):
        cw = control_windows(store, pieces, rng)
        cb, cl = soundtrack_curves(store, cw, dur)
        ctrl_sync.append(_excess(cuts, cb, dur, rng, tol, 5))
        ctrl_energy.append(energy_corr(motion, cl))
        if synced:                                          # the same beat sync applied to the random windows
            ctrl_synced.append(_excess(cuts, soundtrack_curves(store, synced_windows(store, cw, all_cuts, cfg), dur)[0], dur, rng, tol, 5))
    res["sync_excess_control"] = float(np.nanmean(ctrl_sync)) if len(cuts) else np.nan
    res["sync_excess_control_beat_synced"] = float(np.nanmean(ctrl_synced)) if synced and len(cuts) else np.nan
    res["energy_spearman_control"] = float(np.nanmean(ctrl_energy))
    # switch placement: switches vs the biggest scene changes (top quartile of adjacent-segment SigLIP distance)
    sig = feat["sem_siglip_1hz"].astype(np.float32)
    bounds, dist = [], []
    for s0, s1 in zip(segs, segs[1:]):
        def emb(a, b):
            k = np.arange(len(sig))
            sel = (k + 0.5 >= a) & (k + 0.5 < b)
            return l2(sig[sel].mean(0) if sel.any() else sig[min(int((a + b) / 2), len(sig) - 1)])
        bounds.append(s1["start_s"])
        dist.append(1 - float(emb(s0["start_s"], s0["end_s"]) @ emb(s1["start_s"], s1["end_s"])))
    sw = [i for i, (s0, s1) in enumerate(zip(segs, segs[1:])) if s0["music_id"] != s1["music_id"]]
    top = set(np.where(np.array(dist) >= np.quantile(dist, 0.75))[0]) if dist else set()
    res.update({"switches": len(sw), "switches_per_min": len(sw) / (dur / 60),
                "switch_at_top_quartile_scene_change": float(np.mean([i in top for i in sw])) if sw else np.nan,
                "top_quartile_share_by_chance": 0.25})
    # seams
    jumps = [x for x in plan.get("seam_loudness_jump_LU", []) if x is not None and np.isfinite(x)]
    res["seam_jump_LU_median"] = float(np.median(jumps)) if jumps else np.nan
    res["seam_jumps_over_2LU"] = int(sum(j > 2 for j in jumps))
    res["switch_fifths_distance_mean"] = float(np.mean([fifths_distance(segs[i]["key"], segs[i + 1]["key"]) for i in sw])) if sw else np.nan
    tr = [abs(np.log2(segs[i + 1]["tempo"] / segs[i]["tempo"])) for i in sw if segs[i]["tempo"] > 0 and segs[i + 1]["tempo"] > 0]
    res["switch_folded_tempo_mean"] = float(np.mean([2 * abs(r - round(r)) for r in tr])) if tr else np.nan
    # independent content check: ImageBind never takes part in selection
    ib_v = feat["imagebind_1hz"].astype(np.float32)
    lib = list(dict.fromkeys(store.pairs.music_id))
    pct = []
    for s in segs:
        a, b, d = s["start_s"], s["end_s"], s["end_s"] - s["start_s"]
        k = np.arange(len(ib_v))
        sel = (k + 0.5 >= a) & (k + 0.5 < b)
        v = l2(ib_v[sel].mean(0) if sel.any() else ib_v[min(int((a + b) / 2), len(ib_v) - 1)])
        own = l2(store.pooled("music", s["music_id"], "imagebind_1hz", s["offset_s"], s["offset_s"] + d)) @ v
        rand = []
        for _ in range(200):
            m = lib[rng.integers(len(lib))]
            o = rng.uniform(0, max(0.0, store.duration("music", m) - d))
            rand.append(l2(store.pooled("music", m, "imagebind_1hz", o, o + d)) @ v)
        pct.append(100 * float(np.mean(np.array(rand) < own)))
    res["imagebind_percentile_median"] = float(np.median(pct))
    # model agreement across the fold's 3 seeds
    other = [json.loads(p.read_text()) for p in sorted(Path(cfg.paths.renders).glob(f"{name}.seed*.plan.json"))]
    if other:
        sets = [set(s["track_id"] for s in plan["segments"])] + [set(s["track_id"] for s in o["segments"]) for o in other]
        jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        res["seed_track_jaccard_mean"] = float(np.mean(jac))
    _timeline(plan, feat, beats, loud, motion, Path(cfg.paths.reports) / f"check_{name}.png")
    md = Path(cfg.paths.reports) / f"check_{name}.md"
    md.write_text(_markdown(name, plan, res, pct))
    log.info("check %s: %s", name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})
    return md


def _markdown(name, plan, r, pct) -> str:
    f = lambda x: "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    L = [f"# Check: {name}", "", f"Video {plan['video']} ({plan['duration_s']:.1f} s), run {plan['run']}, penalties {plan['penalties']}. "
         f"{len(plan['segments'])} segments, {len(plan['pieces'])} pieces. Reference-free: the video is silent, so no R@k.", "",
         "| check | ours | beat-snapped control | good looks like |", "| --- | --- | --- | --- |",
         f"| sync: within-piece excess cut-on-beat ({r['within_piece_cuts']} cuts) | {f(r['sync_excess'])} | {f(r['sync_excess_control'])} | above the control |",
         *([f"| … vs random music put through the same beat sync (render-time snapping, D35) | {f(r['sync_excess'])} | "
            f"{f(r['sync_excess_control_beat_synced'])} | above this control = the chosen music fits the cuts beyond snapping |"]
           if "beat_sync" in plan else []),
         *([f"| cuts inside pieces within ±70 ms of a beat: retrieved offsets → after beat sync | "
            f"{sum(s['on_beat_before'] for s in plan['beat_sync'])} → {sum(s['on_beat_after'] for s in plan['beat_sync'])} of "
            f"{sum(s['inner_cuts'] for s in plan['beat_sync'])} | | pieces entering on a beat at their cut: "
            f"{sum(s['enters_on_beat'] for s in plan['beat_sync'])} of {len(plan['beat_sync'])} |"] if "beat_sync" in plan else []),
         f"| energy match: Spearman(motion, loudness), 2 s windows | {f(r['energy_spearman'])} | {f(r['energy_spearman_control'])} | positive, above the control |",
         f"| switch placement: share of switches at top-quartile scene changes | {f(r['switch_at_top_quartile_scene_change'])} | 0.25 by chance | most switches |",
         f"| switches per minute | {f(r['switches_per_min'])} ({r['switches']} switches) | | |",
         f"| seams: median loudness jump LU (> 2 LU: {r['seam_jumps_over_2LU']}) | {f(r['seam_jump_LU_median'])} | | under 2 LU |",
         f"| key clash at switches: mean fifths distance (0 same, 1 tritone) | {f(r['switch_fifths_distance_mean'])} | | low |",
         f"| tempo clash at switches: mean folded tempo distance (0-1) | {f(r['switch_folded_tempo_mean'])} | | low |",
         f"| independent content: ImageBind percentile of chosen window among 200 random, median | {f(r['imagebind_percentile_median'])} | 50 by chance | well above 50 |"]
    if "seed_track_jaccard_mean" in r:
        L.append(f"| model agreement: Jaccard of chosen tracks across the fold's 3 seeds | {f(r['seed_track_jaccard_mean'])} | | high |")
    L += ["", f"![timeline](check_{name}.png)", "", "## Segments", "", "| start | end | music | offset | fused | z sem | z dyn | z attr | ImageBind pct |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s, p in zip(plan["segments"], pct):
        L.append(f"| {s['start_s']:.1f} | {s['end_s']:.1f} | {s['music_id'][-40:]} | {s['offset_s']:.1f} | {s['fused']:.2f} | "
                 f"{s['z']['sem']:+.2f} | {s['z']['dyn']:+.2f} | {s['z']['attr']:+.2f} | {p:.0f} |")
    L += ["", "The branch scores come from the system's own models; they explain the choices but are not evidence of quality."]
    return "\n".join(L) + "\n"


def _timeline(plan, feat, beats, loud, motion, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dur = plan["duration_s"]
    fig, ax = plt.subplots(3, 1, figsize=(16, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.4]})
    colors = {}
    for p in plan["pieces"]:
        c = colors.setdefault(p["music_id"], plt.cm.tab20(len(colors) % 20))
        ax[0].axvspan(p["start"], p["end"], color=c, alpha=0.7)
    for c in feat["cut_times_s"]:
        ax[0].axvline(c, color="k", lw=0.4)
    for s in plan["segments"][1:]:
        ax[0].axvline(s["start_s"], color="r", lw=1)
    ax[0].set_ylabel("pieces / cuts")
    ax[0].set_yticks([])
    ax[1].vlines(beats, 0, 1, color="gray", lw=0.4)
    ax[1].set_ylabel("beats")
    ax[1].set_yticks([])
    t = np.arange(len(motion)) / 25
    ax[2].plot(t, (motion - np.nanmean(motion)) / (np.nanstd(motion) + 1e-6), lw=0.6, label="video motion (z)")
    tl = np.arange(len(loud)) / 25
    ax[2].plot(tl, (loud - np.nanmean(loud)) / (np.nanstd(loud) + 1e-6), lw=0.6, label="music loudness (z)")
    ax[2].legend(loc="upper right")
    ax[2].set_xlabel("time (s)")
    ax[2].set_xlim(0, dur)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
