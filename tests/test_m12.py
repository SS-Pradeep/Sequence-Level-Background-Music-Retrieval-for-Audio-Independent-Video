"""M12: sessions are balanced and randomized; the analysis recovers a known effect from fake ratings."""
import numpy as np
import pandas as pd

from bgm.listening import analysis, kit


def test_sessions_and_dry_run_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(kit, "KIT", tmp_path)
    monkeypatch.setattr(analysis, "KIT", tmp_path)
    (tmp_path / "sessions").mkdir()
    rng = np.random.default_rng(0)
    key = pd.DataFrame([{"video": f"v{n:02d}", "letter": L, "condition": c}
                        for n in range(8) for c, L in zip(kit.CONDITIONS, rng.permutation(list("ABCD")))])
    key.to_csv(tmp_path / "key.csv", index=False)
    kit.write_sessions(key, n_raters=12, seed=0)
    s0, s1 = pd.read_csv(tmp_path / "sessions" / "rater_00.csv"), pd.read_csv(tmp_path / "sessions" / "rater_01.csv")
    assert len(s0) == 8 * 3 and list(s0.video) != list(s1.video)                     # every comparison once, own order
    cond = {f"{r.video}_{r.letter}.mp4": r.condition for r in key.itertuples()}
    assert {tuple(sorted((cond[a], cond[b]))) for a, b in zip(s0.left, s0.right)} == \
           {tuple(sorted(c)) for c in kit.COMPARISONS}
    left_is_ours = np.mean([cond[x] == "ours" for x in pd.concat([s0, s1]).left])
    assert 0.3 < left_is_ours < 0.7                                                   # left / right placement randomized
    fake = analysis.fake_ratings(tmp_path / "key.csv")
    fake.to_csv(tmp_path / "fake.csv", index=False)
    text = analysis.analyze(tmp_path / "fake.csv", tmp_path / "key.csv", n_boot=500)
    rows = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if " vs " in ln and ln.startswith("|")}
    won = lambda k: float(rows[k].split("|")[2].strip().rstrip("%")) / 100  # noqa: E731
    assert won("ours vs greedy") > 0.55 and won("ours vs baseline") > 0.55 and won("ours vs original") < 0.45
    assert "ours − greedy" in text
