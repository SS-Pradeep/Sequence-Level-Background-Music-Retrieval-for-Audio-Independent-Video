"""Evaluation metrics (SPEC §5.12). M4 needs the excess cut-on-beat rate; M5 adds the rest."""
from __future__ import annotations

import numpy as np


def cut_on_beat_rate(cuts_s: np.ndarray, beats_s: np.ndarray, tol_s: float = 0.07) -> float:
    """Share of cuts with a beat within +/- tol_s (0 if there are no beats)."""
    cuts_s, beats_s = np.asarray(cuts_s, float), np.asarray(beats_s, float)
    if not len(cuts_s):
        return float("nan")
    if not len(beats_s):
        return 0.0
    return float((np.abs(cuts_s[:, None] - beats_s[None, :]).min(1) <= tol_s).mean())


def eligible_cuts(cuts_s, duration_s: float, margin_s: float = 0.5) -> np.ndarray:
    """Cuts more than margin_s from either end."""
    c = np.asarray(cuts_s, float)
    return c[(c > margin_s) & (c < duration_s - margin_s)]


def excess_cut_on_beat(cuts_s, beats_s, duration_s: float, rng: np.random.Generator, tol_s: float = 0.07,
                       draws: int = 20) -> tuple[float, float, float]:
    """(rate, mean rate over `draws` circular shifts of the beats, excess = rate - shifted).
    Shifts are uniform over the soundtrack's duration; NaN when no cut is eligible."""
    cuts = eligible_cuts(cuts_s, duration_s)
    if not len(cuts):
        return float("nan"), float("nan"), float("nan")
    beats = np.asarray(beats_s, float)
    rate = cut_on_beat_rate(cuts, beats, tol_s)
    shifted = float(np.mean([cut_on_beat_rate(cuts, (beats + rng.uniform(0, duration_s)) % duration_s, tol_s)
                             for _ in range(draws)]))
    return rate, shifted, rate - shifted


# ---------------------------------------------------------------- retrieval ranks and chance
from math import comb  # noqa: E402


def rank_of_first_hit(scores: np.ndarray, hits: np.ndarray) -> int:
    """1-based rank of the best-scored hit among the gallery: 1 + number of non-hits scored above it."""
    scores, hits = np.asarray(scores, float), np.asarray(hits, bool)
    assert hits.any(), "query has no relevant item in the gallery"
    assert np.isfinite(scores).all(), "non-finite retrieval scores"
    return 1 + int((scores[~hits] > scores[hits].max()).sum())


def chance_recall(N: int, h: int, k: int) -> float:
    """Exact probability that a random ranking of N items puts at least one of h relevant ones in the top k."""
    if k >= N:
        return 1.0
    return 1.0 - comb(N - h, k) / comb(N, k)


def chance_mrr(N: int, h: int) -> float:
    """Expected reciprocal rank of the first relevant item under a random ranking: sum_r P(first at r) / r."""
    return sum(comb(N - r, h - 1) / comb(N, h) / r for r in range(1, N - h + 2))


def recall_summary(ranks, ks=(1, 5, 10)) -> dict:
    r = np.asarray(ranks, float)
    out = {f"R@{k}": float((r <= k).mean()) for k in ks}
    out.update({"MedR": float(np.median(r)), "MRR": float((1 / r).mean()), "n": int(len(r))})
    return out


def chance_summary(N_list, h_list, ks=(1, 5, 10)) -> dict:
    """Chance level, averaged over queries (each with its own gallery size N and relevant count h)."""
    out = {f"R@{k}": float(np.mean([chance_recall(N, h, k) for N, h in zip(N_list, h_list)])) for k in ks}
    out["MRR"] = float(np.mean([chance_mrr(N, h) for N, h in zip(N_list, h_list)]))
    out["n"] = len(N_list)
    return out


# ---------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap(values, clusters, n_resamples: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean of per-query `values` with a CI from resampling whole clusters (track_id groups) with replacement.
    For a system comparison pass per-query differences (already averaged over seeds)."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), (n_resamples, len(uniq)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    a = (1 - ci) / 2
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a)),
            "n_queries": int(len(values)), "n_clusters": int(len(uniq))}


def cluster_bootstrap_ratio(num, den, clusters, n_resamples: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """sum(num) / sum(den) over queries, with a CI from resampling whole clusters with replacement. For conditional
    rates, e.g. offset hits among the queries whose top-1 is in the right track group (den = that hit, possibly
    fractional when ties share it). Resamples whose denominator is 0 are dropped."""
    num, den, clusters = np.asarray(num, float), np.asarray(den, float), np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(uniq))
    sd = np.bincount(inv, weights=den, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), (n_resamples, len(uniq)))
    dn, dd = sn[draws].sum(1), sd[draws].sum(1)
    r = dn[dd > 0] / dd[dd > 0]
    a = (1 - ci) / 2
    return {"mean": float(num.sum() / den.sum()) if den.sum() > 0 else float("nan"), "lo": float(np.quantile(r, a)),
            "hi": float(np.quantile(r, 1 - a)), "n_queries": int(len(num)), "n_clusters": int(len(uniq))}


# ---------------------------------------------------------------- grounding chance levels
# lines_up and grounding_rules: tested reference code (grounding_chance.py), ported unchanged.
# lines_up reuses the guards of sample_negative_shifts without changing the sampler, so already-trained dynamics
# models stay valid.

def lines_up(onset, beats, start, T, d, fps=25, guard_s=0.1, max_ac=0.5, ctx=25):
    """(tested) Does the music window [start, start + T) (frames), shifted by d frames, still line up with
    itself? True when either guard of sample_negative_shifts fires, so the sampler rejects exactly the
    in-range shifts for which this is True. The sampler itself is unchanged. The shifted window must lie
    inside the file; near the file edges the autocorrelation uses the context that exists. d = 0 is True."""
    n = len(onset)
    assert 0 <= start + d and start + d + T <= n, "the shifted window must lie inside the file"
    if d == 0:
        return True
    local = beats[(beats >= start / fps) & (beats <= (start + T) / fps)]
    if len(local) >= 2:
        near = np.min(np.abs((local + d / fps)[:, None] - beats[None, :]), axis=1)
        if np.median(near) < guard_s:
            return True
    lo, hi = max(0, start - ctx, -d), min(n, start + T + ctx, n - d)
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.corrcoef(onset[lo:hi], onset[lo + d:hi + d])[0, 1]
    return not (np.isfinite(ac) and ac < max_ac)


def grounding_rules(onset, beats, start, T, fps=25, exact_s=0.1, strict_s=0.5, **guards):
    """(tested) One grounding query: the true window starts at frame `start` and lasts T frames, in a music
    file of len(onset) frames. Returns a boolean array per rule over every candidate offset o = 0 .. n - T:
    would a prediction at o count as correct? A rule's hit is its array at the predicted offset; its chance
    level for this query is the array's mean. guards: keyword arguments for lines_up."""
    d = np.arange(len(onset) - T + 1) - start
    err = np.abs(d) / fps
    exact = err <= exact_s + 1e-9
    strict = err <= strict_s + 1e-9
    sync = exact | np.array([lines_up(onset, beats, start, T, int(x), fps, **guards) for x in d])
    return {"exact": exact, "strict": strict, "sync": sync}


GROUNDING_RULES = ("exact", "strict", "sync")
RULE_LABELS = {"exact": "exact (±0.1 s)", "strict": "within 0.5 s", "sync": "sync (exact, or the shift still lines up)"}


def rule_hits(rules: dict[str, np.ndarray], pred) -> dict[str, tuple[float, float]]:
    """{rule: (hit, chance)} for one query: hit = the rule's array at the predicted offset, chance = its mean.
    pred may be an index array (offsets tied at the top score): the hit is then the rule's mean over them, i.e. the
    expected hit when the tie is broken uniformly at random (the same assumption as the chance level)."""
    return {k: (float(np.mean(rules[k][pred])), float(rules[k].mean())) for k in GROUNDING_RULES}


def top_offsets(scores: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Offsets whose score ties the maximum (within rel_tol of the score range; all of them when the scores are flat)."""
    s = np.asarray(scores, float)
    return np.flatnonzero(s >= s.max() - rel_tol * max(float(np.ptp(s)), 1e-12))
