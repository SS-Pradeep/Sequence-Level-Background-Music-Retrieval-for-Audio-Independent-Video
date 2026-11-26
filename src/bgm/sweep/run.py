"""Loss-function sweep driver (results/loss_sweep). python -m bgm.sweep.run <command>

    split        write data/manifests/folds_losssweep.json (decision 1)
    pairs        within-video consecutive-pair statistics for `trans` (decision 2)
    smoke        2-epoch fold-0 runs: bit-identity with train_one (infonce, drop_last off), then every term
    reference    the current trainer (`bgm train semantic`, siglip-both settings) on the 5 new folds, seed 0
    stage1 .. stage4, stage5, report

Every (config, fold, seed) is one row of results/loss_sweep/runs.csv; a row already there (same config hash, fold,
seed, stage group) is never rerun. Stop rules raise StopSweep with the reason.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgm.losses.registry import AUX_DEFAULT, TAG_BASED

log = logging.getLogger("bgm.sweep")
OUT = Path("results/loss_sweep")
CONF_DIR = Path(__file__).with_name("configs")
RUNS_CSV = OUT / "runs.csv"
FIELDS = ["config_hash", "name", "stage", "fold", "seed", "epochs", "best_epoch", "status", "R1", "R5", "R10", "MedR",
          "MRR", "m2v_R1", "m2v_R5", "m2v_R10", "m2v_MedR", "m2v_MRR", "chance_R10", "m2v_chance_R10", "es_R10",
          "train_R10", "wall_s"]
N_FOLDS = 5
FOLD_LIMIT_S = 15 * 60
SIMPLE_ORDER = ["infonce", "siglip", "soft", "pq_tau"]
DROP_LAST = False      # user 2026-11-16: drop_last off (on, it moved infonce +0.026 vs the reference, early-stopping noise)
DROPPED = {"trans": "within-video consecutive pairs: 1,875, change class 1.2% at cos 0.8 (< 15%); decision 2",
           "barlow": "its own guard `assert D <= B` fails: the current trainer's heads output D = 256 > B = 128 "
                     "(and with drop_last off the last batch is smaller still)"}
REF_NAME = "reference (current trainer)"


class StopSweep(Exception):
    pass


# ---------------------------------------------------------------- config, data
def cfg_overrides() -> list[str]:
    """The shipped siglip-both trainer (runs/titlesplit/semantic/siglip-both) on the loss-sweep folds."""
    return ["splits.folds_file=folds_losssweep.json", "train.semantic.music_feature=both",
            "train.semantic.max_clips_per_film_per_epoch=30", "paths.runs=runs/loss_sweep"]


def load_cfg():
    from bgm.config import load_config
    return load_config("configs/default.yaml", cfg_overrides())


def spec_yaml(spec: dict) -> str:
    s = {"name": spec["name"], "main": spec["main"], "aux": dict(sorted(spec.get("aux", {}).items())),
         "weighting": spec.get("weighting", "fixed")}
    return yaml.safe_dump(s, sort_keys=False)


def config_hash(spec: dict) -> str:
    return hashlib.sha1(spec_yaml(spec).encode()).hexdigest()[:12]


def write_config(spec: dict) -> str:
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    (CONF_DIR / f"{spec['name']}.yaml").write_text(spec_yaml(spec))
    return config_hash(spec)


def make_spec(main: str, aux: dict | None = None, weighting: str = "fixed", name: str | None = None) -> dict:
    aux = dict(aux or {})
    if name is None:
        name = "+".join([main] + sorted(aux)) + ("" if weighting == "fixed" else f"_{weighting}")
    return {"name": name, "main": main, "aux": aux, "weighting": weighting}


def is_tag_based(spec_or_name) -> bool:
    if isinstance(spec_or_name, dict):
        aux = spec_or_name["aux"]
    else:
        f = CONF_DIR / f"{spec_or_name}.yaml"
        aux = yaml.safe_load(f.read_text())["aux"] if f.exists() else {}
    return bool(set(aux or {}) & TAG_BASED)


def read_runs() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(RUNS_CSV)


def append_row(row: dict) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


class Ctx:
    """Store, split and (lazily) per-fold tag vectors, shared by all runs of a process."""

    def __init__(self):
        import torch
        from bgm.eval.store import FeatureStore
        self.cfg = load_cfg()
        self.store = FeatureStore(self.cfg)
        self.split = json.loads(Path("data/manifests/folds_losssweep.json").read_text())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tags, self._raw = {}, None

    def tags(self, key, train_ids):
        if key not in self._tags:
            from bgm.losses.tag_data import TagVectors, raw_long
            if self._raw is None:
                self._raw = (raw_long("scene"), raw_long("music"))
            self._tags[key] = TagVectors(self.store, train_ids, *self._raw)
        return self._tags[key]

    def train_ids(self, fold) -> list[str]:
        if fold == "final":
            test = set(self.split["heldout_test"])
            return sorted(p for p in self.split["film_group"] if p not in test)
        return self.split["folds"][str(fold)]["train"]


def run_one(ctx: Ctx, spec: dict, stage: str, fold, seed: int = 0, mode: str = "fold", fixed_epochs=None,
            max_epochs=None, drop_last=False, record=True) -> dict:
    """Train + evaluate one (config, fold, seed); append to runs.csv (unless record=False) and write runs/loss_sweep/."""
    from bgm.runs import run_dir, write_run
    from bgm.sweep.train import run
    h = write_config(spec) if record else config_hash(spec)
    tags = None
    if set(spec.get("aux", {})) & TAG_BASED:
        tags = ctx.tags(fold, ctx.train_ids(fold))
    png = OUT / "curves" / f"{spec['name']}_f{fold}.png" if (record and mode == "fold") else None
    res = run(ctx.cfg, ctx.store, spec, ctx.split, 0 if fold == "final" else int(fold), seed, ctx.device, tags, mode,
              fixed_epochs=fixed_epochs, max_epochs=max_epochs, drop_last=drop_last, curve_png=png)
    row = {"config_hash": h, "name": spec["name"], "stage": stage, "fold": fold, "seed": seed,
           "epochs": res["epochs"], "best_epoch": res.get("best_epoch", ""), "status": res["status"],
           "wall_s": res["wall_s"]}
    if res["status"] == "ok":
        a, b = res["v2m"], res["m2v"]
        row |= {k: a[k] for k in ("R1", "R5", "R10", "MedR", "MRR")} | {f"m2v_{k}": b[k] for k in ("R1", "R5", "R10", "MedR", "MRR")}
        row |= {"chance_R10": a["chance_R10"], "m2v_chance_R10": b["chance_R10"], "es_R10": res.get("es_R10", ""),
                "train_R10": res["train_R10"]}
    if record:
        out = run_dir(ctx.cfg, f"loss_sweep/{spec['name']}", fold, seed)
        metrics = {k: v for k, v in res.items() if not k.startswith("ranks_")} | {"spec": spec, "config_hash": h, "stage": stage}
        write_run(out, ctx.cfg, metrics, seed)
        if res["status"] == "ok":
            np.savez(out / "perquery.npz", v2m=res["ranks_v2m"], m2v=res["ranks_m2v"])
        append_row(row)
    print(f"  {spec['name']:<34} fold {fold} seed {seed}: {res['status']:<8} R@10 {row.get('R10', float('nan')):.3f} "
          f"(m→v {row.get('m2v_R10', float('nan')):.3f}) epochs {res['epochs']:>3} [{res['wall_s']:.0f} s]", flush=True)
    if res["wall_s"] > FOLD_LIMIT_S:
        raise StopSweep(f"stop rule: {spec['name']} fold {fold} took {res['wall_s'] / 60:.1f} min (> 15 min)")
    return res | {"row": row}


def run_config(ctx: Ctx, spec: dict, stage: str, folds=range(N_FOLDS), seed: int = 0) -> list[dict]:
    done = read_runs()
    h = config_hash(spec)
    rows = []
    for k in folds:
        hit = done[(done.config_hash == h) & (done.fold.astype(str) == str(k)) & (done.seed == seed)]
        if len(hit):
            rows.append(hit.iloc[-1].to_dict())
            continue
        rows.append(run_one(ctx, spec, stage, k, seed)["row"])
    return rows


# ---------------------------------------------------------------- tables
def summarise(names: list[str], ref: str = "infonce") -> pd.DataFrame:
    """Mean ± std over the 5 folds per config name (status ok only), paired Δ R@10 vs `ref` on the same folds."""
    d = read_runs()
    d = d[(d.seed == 0) & d.fold.astype(str).isin([str(k) for k in range(N_FOLDS)])]
    base = d[(d.name == ref) & (d.status == "ok")].set_index(d[(d.name == ref) & (d.status == "ok")].fold.astype(str))
    out = []
    for n in dict.fromkeys(names):
        x = d[d.name == n].drop_duplicates(["fold"], keep="last")
        ok = x[x.status == "ok"]
        r = {"name": n, "folds_ok": len(ok), "diverged": int((x.status == "diverged").sum()),
             "tag_based": "yes" if is_tag_based(n) else ""}
        for col in ("R1", "R5", "R10", "MedR", "MRR", "m2v_R10", "m2v_MRR", "train_R10", "chance_R10"):
            v = ok[col].astype(float)
            r[col], r[col + "_std"] = v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan")
        r["best_epoch_median"] = float(np.median(ok.best_epoch.astype(float))) if len(ok) else float("nan")
        common = [f for f in ok.fold.astype(str) if f in base.index]
        dd = np.array([float(ok[ok.fold.astype(str) == f].R10.iloc[0]) - float(base.loc[f, "R10"]) for f in common])
        r["dR10"], r["dR10_std"] = (dd.mean(), dd.std(ddof=1)) if len(dd) > 1 else (float("nan"), float("nan"))
        out.append(r)
    return pd.DataFrame(out).sort_values("R10", ascending=False, na_position="last").reset_index(drop=True)


def fmt_table(t: pd.DataFrame, extra: dict[str, list] | None = None) -> str:
    cols = ["config", "R@1", "R@5", "R@10", "MedR ↓", "MRR", "m→v R@10", "m→v MRR", "train R@10", "Δ R@10 vs infonce (paired)", "tag-based"]
    if extra:
        cols += list(extra)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    pm = lambda r, c, p=3: f"{r[c]:.{p}f} ± {r[c + '_std']:.{p}f}" if np.isfinite(r[c]) else "—"
    for i, r in t.iterrows():
        cells = [r["name"] + ("" if r["folds_ok"] == N_FOLDS else f" ({r['folds_ok']}/5 ok)"), pm(r, "R1"), pm(r, "R5"),
                 pm(r, "R10"), pm(r, "MedR", 1), pm(r, "MRR"), pm(r, "m2v_R10"), pm(r, "m2v_MRR"), pm(r, "train_R10"),
                 (f"{r['dR10']:+.3f} ± {r['dR10_std']:.3f}" if np.isfinite(r["dR10"]) else "—") if r["name"] != "infonce" else "—",
                 r["tag_based"]]
        if extra:
            cells += [str(extra[k][i]) for k in extra]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def chance_line(t: pd.DataFrame) -> str:
    c = t.chance_R10.dropna()
    return f"Random ranking (K/N): v→m R@10 {c.mean():.3f} (gallery = every segment of the fold's reported films)." if len(c) else ""


def write_stage(n: int, title: str, t: pd.DataFrame, notes: list[str], extra=None) -> str:
    md = [f"# Stage {n}: {title}", "",
          "Single-fold-split development numbers: 5 folds × seed 0, reported films of each fold (TEST untouched). "
          "Mean ± std over folds; Δ = paired per-fold difference vs `infonce`. v→m segment retrieval is primary. "
          "A config *helps* only if its mean R@10 beats the comparison by more than the comparison's fold-std.", "",
          fmt_table(t, extra), "", chance_line(t), ""] + [f"- {x}" for x in notes]
    text = "\n".join(md) + "\n"
    (OUT / f"stage{n}.md").write_text(text)
    print("\n" + text, flush=True)
    return text


def diverged_in(names) -> int:
    d = read_runs()
    return int(((d.name.isin(list(names))) & (d.status == "diverged") & (d.seed == 0)).sum())


def verdict(row, base) -> str:
    if not np.isfinite(row["R10"]):
        return "—"
    if row["R10"] > base["R10"] + base["R10_std"]:
        return "helps"
    if row["R10"] < base["R10"] - base["R10_std"]:
        return "hurts"
    return "neutral"


def state() -> dict:
    p = OUT / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(**kw) -> None:
    s = state() | kw
    (OUT / "state.json").write_text(json.dumps(s, indent=1))


# ---------------------------------------------------------------- commands
def cmd_pairs(ctx: Ctx) -> dict:
    from bgm.train.semantic import SemanticInputs
    st, c = ctx.store, ctx.cfg.train.semantic
    inp = SemanticInputs(st, c.video_feature, c.music_feature)
    test = set(ctx.split["heldout_test"])
    cos = []
    for p, segs in st.segments.items():
        if p in test or len(segs) < 2:
            continue
        M = [inp.music(st.music_of[p], a, b) for a, b in segs]
        M = [x / np.linalg.norm(x) for x in M]
        cos += [float(M[i] @ M[i + 1]) for i in range(len(M) - 1)]
    cos = np.array(cos)
    r = {"pairs": int(len(cos)), "continue_0.8": float((cos > 0.8).mean()), "change_0.8": float((cos <= 0.8).mean())}
    print(json.dumps(r))
    return r


def cmd_smoke(ctx: Ctx) -> None:
    """(1) infonce with drop_last off vs `bgm train semantic` (train_one), 2 epochs, fold 0: identical per-epoch
    early-stopping R@10 and loss. (2) the three requested smoke runs + every main / aux term for 2 epochs."""
    import torch
    from bgm.train.semantic import train_one
    print("smoke 1: sweep `infonce` (drop_last off) vs the current trainer, fold 0, 2 epochs", flush=True)
    import copy
    cfg2 = copy.deepcopy(ctx.cfg)
    cfg2.train.semantic.max_epochs = 2
    from omegaconf import OmegaConf
    OmegaConf.update(cfg2, "paths.runs", "runs/loss_sweep/smoke")
    torch.backends.cudnn.deterministic = True
    m = train_one(cfg2, ctx.store, 0, 0, "ref-smoke", ctx.device)
    ref_log = pd.read_csv(Path("runs/loss_sweep/smoke/semantic/ref-smoke/fold0/seed0/train_log.csv"))
    from bgm.sweep.train import run
    ours = run(cfg2, ctx.store, make_spec("infonce"), ctx.split, 0, 0, ctx.device, None, "fold", max_epochs=2,
               drop_last=False)
    for e, row in ref_log.iterrows():
        mine = ours["curve"][e]
        print(f"  epoch {e}: trainer loss {row['loss']:.6f} es-R@10 {row['val_seg_R@10']:.4f} | sweep loss "
              f"{mine[1]:.6f} es-R@10 {mine[3]:.4f}")
    print(f"  reported-films v→m R@10: trainer {m['test']['segment_all']['R@10']:.4f} | sweep {ours['v2m']['R10']:.4f}")
    print("smoke 2: 2 epochs, fold 0 (not recorded)", flush=True)
    specs = [make_spec("infonce"), make_spec("soft"), make_spec("infonce", {"tag_heads": 0.3})]
    specs += [make_spec(mn) for mn in ("siglip", "vsepp", "hardneg", "debiased", "hgm", "pq_tau")]
    specs += [make_spec("infonce", {k: AUX_DEFAULT[k]}) for k in AUX_DEFAULT if k not in ("tag_teacher", "tag_heads") and k not in DROPPED]
    specs += [make_spec("soft", {"tag_teacher": 0.0})]
    specs += [make_spec("infonce", {"intra": 0.5, "supcon": 0.1}, w) for w in ("uncert", "dwa", "gradnorm")]
    for s in specs:
        run_one(ctx, s, "smoke", 0, max_epochs=2, record=False)


def cmd_reference(ctx: Ctx) -> pd.DataFrame:
    """`bgm train semantic` unchanged (train_one) on the 5 folds, seed 0; then its checkpoints scored by the sweep's
    evaluation (both directions). Recorded in runs.csv as REF_NAME."""
    import torch
    from bgm.sweep import eval as sweep_eval
    from bgm.train.semantic import SemanticInputs, SemanticModel, train_one
    done = read_runs()
    h = hashlib.sha1(b"reference: bgm train semantic, siglip-both settings").hexdigest()[:12]
    c = ctx.cfg.train.semantic
    for k in range(N_FOLDS):
        if len(done[(done.config_hash == h) & (done.fold.astype(str) == str(k))]):
            continue
        t0 = time.time()
        m = train_one(ctx.cfg, ctx.store, k, 0, "reference", ctx.device)
        out = Path(ctx.cfg.paths.runs) / "semantic" / "reference" / f"fold{k}" / "seed0"
        ck = torch.load(out / "ckpt.pt", map_location=ctx.device)
        inp = SemanticInputs(ctx.store, c.video_feature, c.music_feature)
        model = SemanticModel(ck["dv"], ck["dm"], ctx.cfg, ctx.device)
        model.load(ck["state"])
        model.eval()
        ev = sweep_eval.evaluate(ctx.store, model, inp, ctx.split["folds"][str(k)]["test"])
        assert abs(ev["v2m"]["R10"] - m["test"]["segment_all"]["R@10"]) < 1e-9, "sweep eval != trainer's segment_all"
        log_ = pd.read_csv(out / "train_log.csv")
        diag = sorted(np.random.default_rng([k, 134]).choice(ctx.split["folds"][str(k)]["train"],
                                                            len(ctx.split["folds"][str(k)]["val"]), replace=False))
        row = {"config_hash": h, "name": REF_NAME, "stage": "reference", "fold": k, "seed": 0, "epochs": len(log_),
               "best_epoch": m["val"]["best_epoch"], "status": "ok", "es_R10": m["val"]["segment_R@10"],
               "train_R10": sweep_eval.r10_v2m(ctx.store, model, inp, diag), "wall_s": round(time.time() - t0, 1)}
        row |= {x: ev["v2m"][x] for x in ("R1", "R5", "R10", "MedR", "MRR")} | {f"m2v_{x}": ev["m2v"][x] for x in ("R1", "R5", "R10", "MedR", "MRR")}
        row |= {"chance_R10": ev["v2m"]["chance_R10"], "m2v_chance_R10": ev["m2v"]["chance_R10"]}
        np.savez(out / "perquery_sweep_eval.npz", v2m=ev["ranks_v2m"], m2v=ev["ranks_m2v"])
        append_row(row)
        print(f"  {REF_NAME:<34} fold {k}: R@10 {row['R10']:.3f} (m→v {row['m2v_R10']:.3f}) epochs {row['epochs']} "
              f"[{row['wall_s']:.0f} s]", flush=True)
        if row["wall_s"] > FOLD_LIMIT_S:
            raise StopSweep(f"stop rule: reference fold {k} took {row['wall_s'] / 60:.1f} min")
    t = summarise([REF_NAME], ref=REF_NAME)
    r = t.iloc[0]
    print(f"\nreference: v→m R@10 {r.R10:.3f} ± {r.R10_std:.3f}, MRR {r.MRR:.3f} ± {r.MRR_std:.3f}; "
          f"m→v R@10 {r.m2v_R10:.3f} ± {r.m2v_R10_std:.3f}", flush=True)
    return t


def cmd_stage1(ctx: Ctx) -> str:
    mains = ["infonce", "siglip", "soft", "vsepp", "hardneg", "debiased", "hgm", "pq_tau"]
    specs = [make_spec(m) for m in mains]
    # infonce first: the reference stop rule is checked before spending time on the rest
    run_config(ctx, specs[0], "1")
    ref = summarise([REF_NAME], ref=REF_NAME).iloc[0]
    inf = summarise(["infonce"]).iloc[0]
    gap = inf.R10 - ref.R10
    print(f"infonce {inf.R10:.3f} vs reference {ref.R10:.3f}: Δ {gap:+.3f}", flush=True)
    if abs(gap) > 0.01:
        raise StopSweep(f"stop rule: sweep infonce R@10 {inf.R10:.3f} differs from the reference {ref.R10:.3f} by "
                        f"{gap:+.3f} (> 0.01)")
    for s in specs[1:]:
        run_config(ctx, s, "1")
        if diverged_in(mains) > 3:
            raise StopSweep("stop rule: more than 3 diverged runs in Stage 1")
    t = summarise(mains + [REF_NAME])
    win = t[t.name.isin(mains)].iloc[0]
    close = t[t.name.isin(mains) & (t.R10 >= win.R10 - win.R10_std)]
    order = SIMPLE_ORDER + list(t[t.name.isin(mains)].name)
    best = min(close.name, key=order.index)
    save_state(best_main=best)
    notes = [f"Winner on mean R@10: `{win['name']}` {win.R10:.3f} ± {win.R10_std:.3f}. Within one of its fold-stds: "
             + ", ".join(f"`{n}`" for n in close.name) + f". BEST_MAIN = `{best}` (simplest of those: "
             "infonce < siglip < soft < pq_tau < others).",
             f"`{REF_NAME}` = `bgm train semantic` with the shipped siglip-both settings, unchanged; `infonce` = the sweep "
             f"trainer with the same settings (Δ {gap:+.3f}, stop rule 0.01; drop_last was switched off after it moved infonce by +0.026, see stop_rule_diagnosis.json).",
             "Same trainer for every row (decision 5): heads LayerNorm → 512 → GELU → Dropout 0.2 → 256, AdamW 1e-3 / "
             "wd 0.05, cosine warm-up, batch 128 (last partial batch kept: drop_last off, user 2026-11-16), ≤ 30 clips per film per epoch, 25% frame drop, ±1 s "
             "music jitter, same-track masking, early stopping on the fold's early-stopping films (patience 20)."]
    return write_stage(1, "main losses alone", t, notes)


def cmd_stage2(ctx: Ctx) -> str:
    best = state()["best_main"]
    terms = [k for k in AUX_DEFAULT if k not in DROPPED]
    specs = {k: make_spec("soft" if k == "tag_teacher" else best, {k: AUX_DEFAULT[k]}) for k in terms}
    for k, s in specs.items():
        run_config(ctx, s, "2")
        if diverged_in([x["name"] for x in specs.values()]) > 3:
            raise StopSweep("stop rule: more than 3 diverged runs in Stage 2")
    names = [s["name"] for s in specs.values()]
    t = summarise(names + [best] + (["soft"] if best != "soft" else []))
    s1 = summarise([best, "soft"]).set_index("name")
    marks, base_of = {}, {}
    for k, s in specs.items():
        b = "soft" if k == "tag_teacher" else best
        r = t[t.name == s["name"]].iloc[0]
        marks[s["name"]], base_of[s["name"]] = verdict(r, s1.loc[b]), b
    t2 = t.copy()
    extra = {"vs": [base_of.get(n, "—") for n in t2.name], "verdict": [marks.get(n, "(comparison row)") for n in t2.name]}
    term_of = {s["name"]: k for k, s in specs.items()}
    ranked = [term_of[n] for n in t2.name if n in term_of]
    save_state(stage2={term_of[n]: marks[n] for n in marks}, stage2_rank=ranked)
    notes = [f"Each term at its default λ, fixed weighting, on BEST_MAIN = `{best}`; `tag_teacher` is only defined "
             "with main = `soft`, so it is compared with the Stage 1 `soft` row.",
             *(f"`{k}` not run: {v}." for k, v in DROPPED.items()),
             "`supcon` and `center` use the music tags' genre argmax as labels (decision 6), so they are marked tag-based."]
    return write_stage(2, f"each auxiliary term alone on {best}", t2, notes, extra)


def _full_terms(best: str) -> list[str]:
    s2 = state()["stage2"]
    keep = [k for k, v in s2.items() if v in ("helps", "neutral")]
    if best != "soft":
        keep = [k for k in keep if k != "tag_teacher"]
    return keep


def cmd_stage3(ctx: Ctx) -> str:
    best = state()["best_main"]
    full = _full_terms(best)
    specs = []
    if full:
        full_spec = make_spec(best, {k: AUX_DEFAULT[k] for k in full}, name=f"FULL ({best}+{len(full)} terms)")
        specs.append(("full", full_spec))
        if len(full) > 1:
            for k in full:
                rest = [x for x in full if x != k]
                specs.append(("loo", make_spec(best, {x: AUX_DEFAULT[x] for x in rest}, name=f"FULL − {k}")))
    rank = [k for k in state()["stage2_rank"] if not (k == "tag_teacher" and best != "soft")][:4]
    for a, b in itertools.combinations(rank, 2):
        specs.append(("pair", make_spec(best, {a: AUX_DEFAULT[a], b: AUX_DEFAULT[b]})))
    seen, uniq = set(), []
    for kind, s in specs:
        h = config_hash(s)
        if h not in seen:
            seen.add(h), uniq.append((kind, s))
    for kind, s in uniq:
        run_config(ctx, s, "3")
        if diverged_in([x["name"] for _, x in uniq]) > 3:
            raise StopSweep("stop rule: more than 3 diverged runs in Stage 3")
    names = [s["name"] for _, s in uniq]
    t = summarise(names + [best])
    # BEST_COMBO = the best config so far (Stages 1-3)
    s1_names = ["infonce", "siglip", "soft", "vsepp", "hardneg", "debiased", "hgm", "pq_tau"]
    s2_names = [make_spec("soft" if k == "tag_teacher" else best, {k: AUX_DEFAULT[k]})["name"] for k in state()["stage2"]]
    allt = summarise(s1_names + s2_names + names)
    allt = allt[allt.folds_ok == N_FOLDS]
    bc = allt.iloc[0]["name"]
    specs_all = {s["name"]: s for _, s in uniq} | {n: make_spec(n) for n in s1_names}
    specs_all |= {make_spec("soft" if k == "tag_teacher" else best, {k: AUX_DEFAULT[k]})["name"]:
                  make_spec("soft" if k == "tag_teacher" else best, {k: AUX_DEFAULT[k]}) for k in state()["stage2"]}
    save_state(full_terms=full, best_combo=specs_all[bc], loo=[s["name"] for k, s in uniq if k in ("full", "loo")])
    kinds = {s["name"]: k for k, s in uniq}
    extra = {"kind": [kinds.get(n, "BEST_MAIN") for n in t.name]}
    notes = [f"FULL = `{best}` + every Stage 2 term marked helps or neutral: {', '.join(f'`{k}`' for k in full) or 'none'}.",
             "Leave-one-out rows drop one FULL term each; pairs are every 2-term combination of the top 4 Stage 2 terms "
             f"by mean R@10: {', '.join(f'`{k}`' for k in rank)}.",
             f"BEST_COMBO (best mean R@10 over Stages 1–3, 5/5 folds ok) = `{bc}` "
             f"({allt.iloc[0].R10:.3f} ± {allt.iloc[0].R10_std:.3f})."]
    return write_stage(3, "combinations", t, notes, extra)


def cmd_stage4(ctx: Ctx) -> str:
    bc = state()["best_combo"]
    if not bc["aux"]:
        note = f"BEST_COMBO `{bc['name']}` has no auxiliary terms, so there is nothing to weight; Stage 4 skipped, FINAL = BEST_COMBO."
        save_state(final=bc)
        text = f"# Stage 4: weighting schemes\n\n{note}\n"
        (OUT / "stage4.md").write_text(text)
        print(text)
        return text
    specs = {w: dict(bc, weighting=w, name=bc["name"] if w == "fixed" else f"{bc['name']}_{w}") for w in ("fixed", "uncert", "dwa", "gradnorm")}
    for s in specs.values():
        run_config(ctx, s, "4")
    names = [s["name"] for s in specs.values()]
    if diverged_in(names) > 3:
        raise StopSweep("stop rule: more than 3 diverged runs in Stage 4")
    t = summarise(names)
    tt = t[t.folds_ok == N_FOLDS]
    win = tt.iloc[0]
    close = set(tt[tt.R10 >= win.R10 - win.R10_std].name)
    final = next(specs[w] for w in ("fixed", "uncert", "dwa", "gradnorm") if specs[w]["name"] in close)
    save_state(final=final)
    sig = sigma_table(specs["uncert"])
    extra = {"weighting": [next(w for w, s in specs.items() if s["name"] == n) for n in t.name]}
    notes = [f"FINAL = `{final['name']}` (best within one std of the winner `{win['name']}`, preferring fixed, then uncert).",
             "uncert: L_k/(2σ_k²) + log σ_k with σ_k learned, σ_k initialised so that 1/(2σ_k²) = λ_k; dwa and gradnorm "
             "scale the default λ_k (multipliers start at 1, sum to K).", "", "Learned σ_k (uncert), mean over folds "
             "at the early-stopping epoch's run end; effective weight 1/(2σ²) vs default λ:", "", sig]
    return write_stage(4, f"weighting schemes on {bc['name']}", t, notes, extra)


def sigma_table(spec) -> str:
    rows = []
    for k in range(N_FOLDS):
        p = Path("runs/loss_sweep/loss_sweep") / spec["name"] / f"fold{k}" / "seed0" / "metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text()).get("sigmas_final", {}))
    if not rows:
        return "(no uncert runs)"
    terms = list(rows[0])
    lines = ["| term | default λ | σ_k (mean ± std) | 1/(2σ_k²) |", "|---|---|---|---|"]
    for t in terms:
        s = np.array([r[t] for r in rows if t in r])
        lines.append(f"| {t} | {AUX_DEFAULT[t]:g} | {s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f} | "
                     f"{np.mean(1 / (2 * s ** 2)):.4g} |")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cmd = argv[0]
    if cmd == "split":
        from bgm.sweep import split
        p = split.build(load_cfg())
        print(split.describe(json.loads(p.read_text())))
        return
    ctx = Ctx()
    try:
        if cmd == "pairs":
            cmd_pairs(ctx)
        elif cmd == "smoke":
            cmd_smoke(ctx)
        elif cmd == "reference":
            cmd_reference(ctx)
        elif cmd in ("stage1", "stage2", "stage3", "stage4"):
            {"stage1": cmd_stage1, "stage2": cmd_stage2, "stage3": cmd_stage3, "stage4": cmd_stage4}[cmd](ctx)
        elif cmd == "all":
            for f in (cmd_reference, cmd_stage1, cmd_stage2, cmd_stage3, cmd_stage4):
                f(ctx)
        elif cmd in ("stage5", "report"):
            from bgm.sweep import report
            (report.stage5 if cmd == "stage5" else report.final_report)(ctx)
        else:
            raise SystemExit(f"unknown command {cmd}")
    except StopSweep as e:
        print(f"\nSTOPPED — {e}", flush=True)
        (OUT / "STOPPED.txt").write_text(str(e) + "\n")
        sys.exit(3)


if __name__ == "__main__":
    main(sys.argv[1:])
