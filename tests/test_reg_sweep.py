"""D138 Step 3: combination picking and variant overrides."""
from bgm.analysis.reg_sweep import FAMILY, NEW, SINGLE, overrides, pick_combo


def test_pick_combo_takes_the_best_of_each_family_above_baseline_top3():
    val = {v: 0.30 for v in SINGLE}
    val |= {"d138-tau01": 0.36, "d138-tau005": 0.35, "d135-do05": 0.357, "d138-wd03": 0.34, "d138-noise005": 0.33, "d138-h256": 0.31}
    assert pick_combo(val, 0.32) == ["d138-tau01", "d135-do05", "d138-wd03"]
    assert pick_combo(val, 0.355) == ["d138-tau01", "d135-do05"]
    assert pick_combo(val, 0.5) == []


def test_every_single_variant_is_in_one_family_and_has_overrides():
    fam = [v for vs in FAMILY.values() for v in vs]
    assert sorted(fam) == sorted(SINGLE) and len(fam) == len(set(fam))
    assert all(overrides(v) and all(o.startswith("train.semantic.") for o in overrides(v)) for v in NEW)
    assert overrides("d134-base") == []
