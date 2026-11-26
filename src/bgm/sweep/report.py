"""Stage 5 (TEST, once) and the final report of the loss sweep.

Stage 5: infonce, BEST_MAIN, BEST_COMBO, FINAL and the top-2 single-aux Stage 2 runs, retrained on every non-TEST
clip with seeds 0, 1, 2 for a fixed number of epochs (median early-stopping epoch count over that config's 5 folds),
scored once on TEST. Paired bootstrap over TEST queries (1,000 resamples) of per-query hit@10 / reciprocal rank,
averaged over the 3 seeds, vs infonce.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.losses.registry import AUX_DEFAULT
from bgm.sweep.run import (CONF_DIR, DROPPED, N_FOLDS, OUT, REF_NAME, Ctx, append_row, config_hash, fmt_table,
                           is_tag_based, make_spec, read_runs, run_one, sigma_table, state, summarise, write_config)
import yaml

SEEDS = (0, 1, 2)
B_RES = 1000


def _spec(name: str) -> dict:
    return yaml.safe_load((CONF_DIR / f"{name}.yaml").read_text())


def stage5_configs() -> list[tuple[str, dict]]:
    st = state()
    s2 = summarise([make_spec("soft" if k == "tag_teacher" else st["best_main"], {k: AUX_DEFAULT[k]})["name"]
                    for k in st["stage2"]])
    s2 = s2[s2.folds_ok == N_FOLDS]
    out = [("baseline", make_spec("infonce")), ("BEST_MAIN", _spec(st["best_main"])), ("BEST_COMBO", st["best_combo"]),
           ("FINAL", st["final"])] + [(f"Stage 2 #{i + 1}", _spec(n)) for i, n in enumerate(s2.name[:2])]
    seen, uniq = {}, []
    for role, s in out:
        h = config_hash(s)
        if h in seen:
            seen[h][0].append(role)
        else:
            seen[h] = ([role], s)
            uniq.append(h)
    return [(" / ".join(seen[h][0]), seen[h][1]) for h in uniq]


def _epochs(name: str) -> int:
    d = read_runs()
    d = d[(d.name == name) & (d.status == "ok") & (d.seed == 0) & d.fold.astype(str).isin([str(k) for k in range(N_FOLDS)])]
    return int(np.median(d.best_epoch.astype(float) + 1))


def _ranks(name: str, seed: int) -> dict:
    p = Path("runs/loss_sweep/loss_sweep") / name / "foldfinal" / f"seed{seed}" / "perquery.npz"
    z = np.load(p)
    return {"v2m": z["v2m"], "m2v": z["m2v"]}


def stage5(ctx: Ctx) -> None:
    cfgs = stage5_configs()
    done = read_runs()
    for role, s in cfgs:
        write_config(s)
        E = _epochs(s["name"])
        for seed in SEEDS:
            h = config_hash(s)
            if len(done[(done.config_hash == h) & (done.fold.astype(str) == "final") & (done.seed == seed)]):
                continue
            run_one(ctx, s, "5", "final", seed, mode="final", fixed_epochs=E)
    (OUT / "stage5_configs.json").write_text(json.dumps([{"role": r, "spec": s, "epochs": _epochs(s["name"])} for r, s in cfgs], indent=1))
    test_tables(cfgs)


def _boot(delta: np.ndarray, rng) -> tuple[float, float, float]:
    idx = rng.integers(0, len(delta), (B_RES, len(delta)))
    bs = delta[idx].mean(1)
    return float(delta.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def test_tables(cfgs) -> pd.DataFrame:
    d = read_runs()
    d = d[(d.fold.astype(str) == "final") & (d.status == "ok")]
    base = np.mean([_ranks("infonce", s)["v2m"] for s in SEEDS], 0) if False else None
    per = {}
    for _, s in cfgs:
        rk = [_ranks(s["name"], sd) for sd in SEEDS]
        per[s["name"]] = {"hit10": np.mean([(r["v2m"] <= 10).astype(float) for r in rk], 0),
                          "rr": np.mean([1.0 / r["v2m"] for r in rk], 0)}
    b = per["infonce"]
    rows = []
    for role, s in cfgs:
        x = d[d.name == s["name"]].drop_duplicates("seed", keep="last")
        r = {"role": role, "name": s["name"], "epochs": _epochs(s["name"]), "seeds": len(x),
             "tag_based": "yes" if is_tag_based(s) else ""}
        for c in ("R1", "R5", "R10", "MedR", "MRR", "m2v_R10", "m2v_MRR"):
            v = x[c].astype(float)
            r[c], r[c + "_std"] = v.mean(), v.std(ddof=1)
        rng = np.random.default_rng(0)
        r["dR10"], r["dR10_lo"], r["dR10_hi"] = _boot(per[s["name"]]["hit10"] - b["hit10"], rng)
        r["dMRR"], r["dMRR_lo"], r["dMRR_hi"] = _boot(per[s["name"]]["rr"] - b["rr"], np.random.default_rng(1))
        r["chance_R10"] = float(x.chance_R10.astype(float).mean())
        r["n"] = len(b["hit10"])
        rows.append(r)
    t = pd.DataFrame(rows)
    _write_test_md(t)
    _write_test_tex(t)
    return t


def _write_test_md(t: pd.DataFrame) -> None:
    n = int(t.n.iloc[0])
    lines = [f"# Stage 5: TEST (scored once)", "",
             f"TEST = 22 held-out title film groups ({n} segments; gallery = every TEST segment). Each config retrained on "
             "every non-TEST clip with seeds 0, 1, 2 for the median early-stopping epoch count of its 5 folds. Mean ± std "
             "over the 3 seeds. Δ = paired bootstrap over TEST queries (1,000 resamples) of per-query hit@10 / reciprocal "
             "rank averaged over the seeds, vs `infonce`; * = 95% CI excludes 0.", "",
             "| role | config | epochs | R@1 | R@5 | R@10 | MedR ↓ | MRR | m→v R@10 | Δ R@10 [95% CI] | Δ MRR [95% CI] | tag-based |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pm = lambda r, c, p=3: f"{r[c]:.{p}f} ± {r[c + '_std']:.{p}f}"
    ci = lambda r, c: ("—" if r["name"] == "infonce" else
                       f"{r[c]:+.3f} [{r[c + '_lo']:+.3f}, {r[c + '_hi']:+.3f}]" + ("*" if r[c + "_lo"] > 0 or r[c + "_hi"] < 0 else ""))
    for _, r in t.iterrows():
        lines.append(f"| {r.role} | {r['name']} | {r.epochs} | {pm(r, 'R1')} | {pm(r, 'R5')} | {pm(r, 'R10')} | "
                     f"{pm(r, 'MedR', 1)} | {pm(r, 'MRR')} | {pm(r, 'm2v_R10')} | {ci(r, 'dR10')} | {ci(r, 'dMRR')} | {r.tag_based} |")
    lines += ["", f"Random ranking: R@10 {t.chance_R10.mean():.3f}."]
    (OUT / "test.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("#", r"\#").replace("−", "$-$")


def _write_test_tex(t: pd.DataFrame) -> None:
    cols = [("R1", "R@1", True), ("R5", "R@5", True), ("R10", "R@10", True), ("MedR", "MedR$\\downarrow$", False), ("MRR", "MRR", True)]
    best = {c: (t[c].max() if hi else t[c].min()) for c, _, hi in cols}
    out = [r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
           "Loss & " + " & ".join(h for _, h, _ in cols) + r" & $\Delta$R@10 [95\% CI] \\", r"\midrule"]
    for _, r in t.iterrows():
        cells = []
        for c, _, _ in cols:
            p = 1 if c == "MedR" else 3
            s = f"{r[c]:.{p}f} $\\pm$ {r[c + '_std']:.{p}f}"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(r[c], best[c]) else s)
        dci = "--" if r["name"] == "infonce" else f"{r.dR10:+.3f} [{r.dR10_lo:+.3f}, {r.dR10_hi:+.3f}]"
        out.append(_tex_escape(r["name"]) + " & " + " & ".join(cells) + f" & {dci} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "test.tex").write_text("\n".join(out) + "\n")


def dot_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = read_runs()
    d = d[d.fold.astype(str).isin([str(k) for k in range(N_FOLDS)]) & (d.seed == 0)]
    t = summarise(list(dict.fromkeys(d.name)))
    t = t[t.folds_ok == N_FOLDS].sort_values("R10")
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(t) + 1.2), dpi=150)
    y = np.arange(len(t))
    base = t[t.name == "infonce"].R10.iloc[0] if (t.name == "infonce").any() else None
    col = ["#E45756" if n == "infonce" else "#888888" if n == REF_NAME else "#4C78A8" for n in t.name]
    ax.errorbar(t.R10, y, xerr=t.R10_std, fmt="none", ecolor="#BBBBBB", elinewidth=1, capsize=2)
    ax.scatter(t.R10, y, c=col, s=18, zorder=3)
    if base is not None:
        ax.axvline(base, color="#E45756", lw=0.8, ls="--", label="infonce (baseline)")
    ax.set_yticks(y, [n + ("  [tag]" if is_tag_based(n) else "") for n in t.name], fontsize=6.5)
    ax.set_xlabel("v→m segment R@10, mean ± std over 5 folds (seed 0)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "r10_by_config.png")
    plt.close(fig)


def final_report(ctx: Ctx | None = None) -> None:
    st = state()
    dot_plot()
    d = read_runs()
    lines = ["# Loss-function sweep: summary", ""]
    lines += ["## 1. Final table (TEST, scored once)", "", *(OUT / "test.md").read_text().splitlines()[2:], ""]
    lines += ["## 2. Stage by stage (5 folds × seed 0, reported films; TEST untouched)", ""]
    s1 = summarise(["infonce", "siglip", "soft", "vsepp", "hardneg", "debiased", "hgm", "pq_tau", REF_NAME])
    w = s1[s1.name != REF_NAME].iloc[0]
    ref = s1[s1.name == REF_NAME].iloc[0]
    inf = s1[s1.name == "infonce"].iloc[0]
    lines.append(f"- **Reference / Stage 1:** current trainer {ref.R10:.3f} ± {ref.R10_std:.3f}; sweep infonce "
                 f"{inf.R10:.3f} ± {inf.R10_std:.3f}. Best main loss `{w['name']}` {w.R10:.3f} ± {w.R10_std:.3f} "
                 f"(Δ vs infonce {w.dR10:+.3f}); BEST_MAIN = `{st['best_main']}`.")
    s2 = st["stage2"]
    lines.append("- **Stage 2:** " + "; ".join(f"`{k}` {v}" for k, v in s2.items()) + ". Not run: "
                 + "; ".join(f"`{k}` ({v})" for k, v in DROPPED.items()) + ".")
    bc = st["best_combo"]
    bcr = summarise([bc["name"]]).iloc[0]
    lines.append(f"- **Stage 3:** FULL = {st['best_main']} + {', '.join(st['full_terms']) or 'nothing'}; BEST_COMBO = "
                 f"`{bc['name']}` {bcr.R10:.3f} ± {bcr.R10_std:.3f} (Δ vs infonce {bcr.dR10:+.3f}).")
    fr = summarise([st["final"]["name"]]).iloc[0]
    lines.append(f"- **Stage 4:** FINAL = `{st['final']['name']}` {fr.R10:.3f} ± {fr.R10_std:.3f}.")
    lines += ["", "## 3. Leave-one-out (which terms matter)", ""]
    if st.get("loo") and len(st["loo"]) > 1:
        t = summarise(st["loo"])
        full = [n for n in st["loo"] if n.startswith("FULL (")][0]
        fv = t[t.name == full].iloc[0]
        lines += ["| config | R@10 | Δ vs FULL |", "|---|---|---|"]
        for _, r in t.iterrows():
            lines.append(f"| {r['name']} | {r.R10:.3f} ± {r.R10_std:.3f} | {r.R10 - fv.R10:+.3f} |")
        lines.append("\nA term matters when removing it lowers R@10 (negative Δ) by more than FULL's fold-std "
                     f"({fv.R10_std:.3f}).")
    else:
        lines.append("FULL had fewer than two terms, so there is no leave-one-out table.")
    lines += ["", "## 4. Learned σ_k (uncert)", ""]
    if st["final"]["aux"] or bc["aux"]:
        lines.append(sigma_table(dict(bc, weighting="uncert", name=f"{bc['name']}_uncert")))
    else:
        lines.append("Stage 4 skipped (BEST_COMBO has no auxiliary terms).")
    lines += ["", "## 5. Verdict on the tags", "", *tag_verdict(st), "", "## 6. Diverged runs", ""]
    dv = d[d.status == "diverged"]
    lines += [f"- `{r['name']}` stage {r.stage} fold {r.fold} seed {r.seed}" for _, r in dv.iterrows()] or ["None."]
    lines += ["", "Figures: `r10_by_config.png` (all fold-level configs), `curves/<name>_f<k>.png`."]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def tag_verdict(st) -> list[str]:
    best = st["best_main"]
    out = []
    items = [("tag_teacher", "soft"), ("tag_heads", best), ("supcon", best), ("center", best)]
    names = {k: make_spec("soft" if k == "tag_teacher" else best, {k: AUX_DEFAULT[k]})["name"] for k, _ in items}
    t = summarise(list(names.values()) + [best, "soft"]).set_index("name")
    for k, b in items:
        if names[k] not in t.index or b not in t.index:
            continue
        r, br = t.loc[names[k]], t.loc[b]
        dd = _paired(names[k], b)
        out.append(f"- **`{k}`** ({'tag teacher in the soft targets' if k == 'tag_teacher' else 'tag-prediction heads' if k == 'tag_heads' else 'tag genre labels'}): "
                   f"{st['stage2'].get(k, '—')} — R@10 {r.R10:.3f} ± {r.R10_std:.3f} vs `{b}` {br.R10:.3f} ± {br.R10_std:.3f}; "
                   f"paired Δ {dd[0]:+.3f} ± {dd[1]:.3f} over 5 folds.")
    return out


def _paired(a: str, b: str) -> tuple[float, float]:
    d = read_runs()
    d = d[(d.seed == 0) & (d.status == "ok") & d.fold.astype(str).isin([str(k) for k in range(N_FOLDS)])]
    x = d[d.name == a].drop_duplicates("fold", keep="last").set_index(d[d.name == a].drop_duplicates("fold", keep="last").fold.astype(str)).R10
    y = d[d.name == b].drop_duplicates("fold", keep="last").set_index(d[d.name == b].drop_duplicates("fold", keep="last").fold.astype(str)).R10
    common = sorted(set(x.index) & set(y.index))
    dd = np.array([float(x[f]) - float(y[f]) for f in common])
    return (dd.mean(), dd.std(ddof=1)) if len(dd) > 1 else (float("nan"), float("nan"))
