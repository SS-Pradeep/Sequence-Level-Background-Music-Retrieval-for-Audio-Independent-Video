"""D99 film-split final report: disjoint ensembles, seed filtering, the weights label and the ledger rows."""
from __future__ import annotations

import numpy as np

from bgm.eval.ensemble import SPECS, EnsembleSpec
from bgm.eval.final_film import _weights, rank_runs, section_ledger
from bgm.person.evaluate import MeanScorer


def test_ensemble_run_seeds_are_disjoint_member_sets():
    spec = SPECS["siglip-both-ens10"]
    sets = [set(spec.members(e)) for e in spec.run_seeds]
    assert [len(s) for s in sets] == [10, 10, 10]
    assert set.union(*sets) == set(range(30)) and sum(len(s) for s in sets) == 30
    assert spec.members(0) == list(range(10))           # run seed 0 = the D65 ensemble (heads 0-9)
    assert EnsembleSpec("baselines", "cfd", 3, (0,)).members(0) == [0, 1, 2]


def test_mean_scorer_averages_member_scores():
    class Const:
        def __init__(self, v):
            self.v = np.asarray(v, float)

        def track(self, q, gallery):
            return self.v

        def window(self, q, a, b, windows):
            return self.v

    sc = MeanScorer([Const([1.0, 0.0]), Const([0.0, 1.0]), Const([2.0, 2.0])])
    assert np.allclose(sc.track("q", ["a", "b"]), [1.0, 1.0])
    assert np.allclose(sc.window("q", 0, 1, [None, None]), [1.0, 1.0])


def test_rank_runs_keeps_only_the_requested_seeds(tmp_path):
    secs = ("track_test_fold", "track_all", "segment_all", "segment_other")
    for k in (0, 1):
        for s in (0, 1, 2, 5, 12):
            d = tmp_path / "semantic" / "v" / f"fold{k}" / f"seed{s}"
            d.mkdir(parents=True)
            np.savez(d / "perquery.npz", **{sec: np.full(3, s) for sec in secs})
    r = rank_runs(tmp_path, "semantic", "v", (0, 1, 2))
    assert sorted(r) == [0, 1] and all(len(v) == 3 for v in r.values())
    assert sorted(int(x["track_test_fold"][0]) for x in r[0]) == [0, 1, 2]
    assert sum(len(v) for v in rank_runs(tmp_path, "semantic", "v").values()) == 10


def test_weights_label_fixed_and_tuned():
    fixed = [{"sem": 0.8, "dyn": 0.2}, {"sem": 0.8, "dyn": 0.2}]
    assert _weights(fixed, ("sem", "dyn", "attr")) == "sem 0.8 / dyn 0.2 / attr 0"
    tuned = [{"sem": 0.6, "attr": 0.4}, {"sem": 0.4, "attr": 0.6}]
    assert _weights(tuned, ("sem", "attr")) == "sem 0.50 (tuned) / attr 0.50 (tuned)"


def test_ledger_keeps_film_rows_only(tmp_path):
    p = tmp_path / "ledger.md"
    p.write_text("\n".join([
        "# ledger", "", "| date | change | split | gallery | R@1 | R@10 | MRR | Δ | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| 2026-10-02 | 1a mix | film | 196 | 0.055 | 0.290 | 0.132 | **+0.002** [-0.000, +0.004] | NULL |",
        "| 2026-10-06 | M9 full | extended (4,246) | 781 | a | b | c | d | info |",
        "| 2026-10-06 | ensemble | film | 196 | 0.061 | 0.314 | 0.148 | **+0.025** [+0.012, +0.040] | KEEP |"]) + "\n")
    L = section_ledger(p)
    rows = [ln for ln in L if ln.startswith("| 2026")]
    assert rows == ["| 2026-10-02 | 1a mix | 0.290 | **+0.002** [-0.000, +0.004] | NULL |",
                    "| 2026-10-06 | ensemble | 0.314 | **+0.025** [+0.012, +0.040] | KEEP |"]
    assert section_ledger(tmp_path / "missing.md") == []
