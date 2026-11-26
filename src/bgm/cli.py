"""`bgm` command line. Every command takes `--config configs/default.yaml` plus dotted overrides,
e.g. `bgm train semantic fold=2 seed=1 train.semantic.lr_finetune=3e-4`.

Commands not built yet exit with status 2 and name the milestone that builds them (docs/SPEC.md §7).
"""
from __future__ import annotations

import logging

import typer

from bgm.config import load_config
from bgm.log import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Video-to-BGM retrieval. The build spec is docs/SPEC.md.")
log = logging.getLogger("bgm")

EXTRA = {"allow_extra_args": True, "ignore_unknown_options": True}
CONFIG_HELP = "YAML config; dotted key=value overrides may follow the command."


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")):
    setup_logging(logging.DEBUG if verbose else logging.INFO)


def _cfg(ctx: typer.Context, config: str):
    return load_config(config, ctx.args)


def _todo(command: str, milestone: str) -> None:
    log.error("`bgm %s` is not implemented yet; it is built in milestone %s (docs/SPEC.md §7).", command, milestone)
    raise typer.Exit(2)


@app.command(context_settings=EXTRA)
def doctor(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Report GPU, ffmpeg, ffprobe, fpcalc and package versions."""
    from bgm.doctor import report
    _cfg(ctx, config)
    for line in report():
        typer.echo(line)


@app.command(context_settings=EXTRA)
def smoke(ctx: typer.Context, fake: bool = typer.Option(False, "--fake", help="Fake encoders on CPU (required for now)."),
          n: int = typer.Option(3, help="Number of synthetic pairs."),
          config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Run synthetic clips through everything built so far."""
    from bgm.smoke import run_smoke
    _cfg(ctx, config)
    if not fake:
        log.error("only `bgm smoke --fake` exists so far")
        raise typer.Exit(2)
    run_smoke(n_pairs=n)


@app.command(context_settings=EXTRA)
def audit(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Write data/manifests/pairs.csv, tracks.csv and reports/audit.md."""
    from bgm.data.audit import run_audit
    df = run_audit(_cfg(ctx, config))
    log.info("pairs.csv: %d pairs, %d track groups -> reports/audit.md", len(df), df.track_id.nunique())


@app.command(context_settings=EXTRA)
def folds(ctx: typer.Context, film: bool = typer.Option(False, "--film", help="Film-level split -> folds_film.json (PROGRESS.md D36)."),
          extend: str = typer.Option(None, help="With --film: keep this film split's assignments and add the new pairs (D49)."),
          out: str = typer.Option("folds_film.json", help="With --film: output file name in data/manifests (D102: folds_title.json)."),
          links: str = typer.Option(None, help="With --film: CSV of film_key_a,film_key_b judged one film, added to extra_links (D102)."),
          config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Write data/manifests/folds.json (SPEC §3.3), or folds_film.json with --film."""
    from bgm.data import manifest
    from bgm.data.folds import check_folds, extend_film_folds, fold_summary, folds_document, make_film_folds, make_folds
    cfg = _cfg(ctx, config)
    pairs = manifest.read_pairs(cfg)
    if film:
        from bgm.data.films import film_groups
        cfg.splits.folds_file = out
        extra_links = [tuple(x) for x in cfg.splits.film.extra_links]
        if links:
            import pandas as pd
            extra_links += list(pd.read_csv(links)[["film_key_a", "film_key_b"]].itertuples(index=False, name=None))
        groups = film_groups(pairs, extra_links)
        train_only = {groups[p] for p, k in zip(pairs.pair_id, pairs.meta_source + ":" + pairs.meta_film) if k in set(cfg.splits.film.train_only)}
        extra = {}
        if extend:
            import json
            from pathlib import Path
            base = json.loads(Path(extend).read_text())
            moved = {}
            folds = extend_film_folds(base["folds"], pairs, groups, train_only, cfg.splits.n_folds, cfg.splits.seed, moved)
            extra = {"extends": extend, "base_manifest_hash": base["manifest_hash"],
                     "moved_from_base": {p: {"old": list(a), "new": list(b)} for p, (a, b) in sorted(moved.items())}}
            log.info("extended %s: %d base pairs moved by merged film groups", extend, len(moved))
        else:
            folds = make_film_folds(pairs, groups, train_only, cfg.splits.n_folds, cfg.splits.val_fraction, cfg.splits.seed)
        check_folds(folds, pairs, groups, train_only)
        if links:
            extra["links_file"] = links
        doc = folds_document(folds, manifest.manifest_hash(cfg), split="film", film_group=groups, train_only=sorted(train_only), **extra)
        log.info("film split: %d film groups, train-only %s", len(set(groups.values())), sorted(train_only))
    else:
        folds = make_folds(pairs, cfg.splits.n_folds, cfg.splits.val_fraction, cfg.splits.seed)
        check_folds(folds, pairs)
        doc = folds_document(folds, manifest.manifest_hash(cfg))
    manifest.write_folds(doc, cfg)
    typer.echo(fold_summary(folds, pairs).to_string(index=False))


@app.command(context_settings=EXTRA)
def prep(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Silent 25 fps interim videos and A/V-aligned 48 kHz music (SPEC §3.4)."""
    from bgm.data.prep import run_prep
    run_prep(_cfg(ctx, config))


@app.command(context_settings=EXTRA)
def shots(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Shot detection into data/manifests/shots.csv (SPEC §3.5)."""
    from bgm.data.shots import run_shots
    run_shots(_cfg(ctx, config))


@app.command(context_settings=EXTRA)
def peek(ctx: typer.Context, pair_id: str = typer.Argument(..., help="Pair to inspect."),
         config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Contact sheet of a pair's shot first frames."""
    from bgm.data.shots import peek as make_sheet
    typer.echo(str(make_sheet(_cfg(ctx, config), pair_id)))


@app.command(context_settings=EXTRA)
def extract(ctx: typer.Context, what: str = typer.Option("sem_video,sem_music,dyn_video,dyn_music,attr,bridge,imagebind,key",
                                                         help="Comma-separated extractors."),
            fake: bool = typer.Option(False, "--fake", help="Fake encoders (tests / smoke)."),
            limit: int = typer.Option(None, help="Only the first N pairs."),
            workers: int = typer.Option(16, help="CPU worker processes."),
            config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Cache features in data/features/ (SPEC §5.1-5.4)."""
    from bgm.features.extract import run_extract
    run_extract(_cfg(ctx, config), what, fake=fake, limit=limit, workers=workers)


@app.command("check-features", context_settings=EXTRA)
def check_features(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Validate cached features against SPEC §4.2."""
    from bgm.features.check import run_check
    bad = run_check(_cfg(ctx, config))
    if bad:
        raise typer.Exit(1)


@app.command(context_settings=EXTRA)
def segment(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Shots -> segments into data/manifests/segments.csv (SPEC §3.5), then the M4 alignment audit."""
    from bgm.data.alignment import run_alignment_audit
    from bgm.data.segments import run_segment
    cfg = _cfg(ctx, config)
    run_segment(cfg)
    run_alignment_audit(cfg)


@app.command(context_settings=EXTRA)
def train(ctx: typer.Context, what: str = typer.Argument(..., help="semantic | dynamics | baseline"),
          all_pairs: bool = typer.Option(False, "--all", help="Deployment models on all pairs."),
          grid: bool = typer.Option(False, "--grid", help="All folds x train.seeds (default: config fold / seed)."),
          variant: str = typer.Option(None, help="Run name under runs/semantic/ (default: from the features)."),
          init: str = typer.Option(None, help="Fine-tune from pretrained checkpoints: runs/pretrain/<semantic|dynamics>."),
          config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Train the semantic heads (M6), dynamics encoders (M7), or a published-method baseline on our features
    (`baseline --variant cfd|utcmvmr|mvpt`, PROGRESS.md D43)."""
    if what not in ("semantic", "dynamics", "baseline"):
        raise typer.BadParameter("expected `semantic`, `dynamics` or `baseline`")
    if all_pairs:
        _todo(f"train {what} --all", "M10")
    cfg = _cfg(ctx, config)
    folds = None if grid else [cfg.fold]
    seeds = None if grid else [cfg.seed]
    if what == "baseline":
        from bgm.train.baselines import run_train_baseline
        run_train_baseline(cfg, variant, folds, seeds)
    elif what == "semantic":
        from bgm.train.semantic import run_train_semantic
        run_train_semantic(cfg, folds, seeds, variant, init)
    else:
        from bgm.train.dynamics import run_train_dynamics
        run_train_dynamics(cfg, folds, seeds, variant, init)


@app.command(context_settings=EXTRA)
def pretrain(ctx: typer.Context, stage: str = typer.Option(..., help="data | extract | train"),
             config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """M8: unlabeled set from Data/video_bgm_pairs (data), its features + CLAP dedupe (extract), pretraining (train)."""
    cfg = _cfg(ctx, config)
    if stage == "data":
        from bgm.pretrain.unlabeled import build_unlabeled
        build_unlabeled(cfg)
    elif stage == "extract":
        from bgm.data.segments import run_segment
        from bgm.data.shots import run_shots
        from bgm.features.extract import run_extract
        from bgm.pretrain.unlabeled import clap_dedupe, unlabeled_config
        ucfg = unlabeled_config(cfg)
        run_shots(ucfg)
        run_extract(ucfg, "sem_video,sem_music,dyn_video,dyn_music")
        clap_dedupe(cfg)
        run_segment(ucfg)
    elif stage == "train":
        import torch
        from bgm.pretrain.train import pretrain_dynamics, pretrain_semantic
        dev = cfg.features.device if torch.cuda.is_available() else "cpu"
        for s in cfg.train.seeds:
            pretrain_semantic(cfg, s, dev)
            pretrain_dynamics(cfg, s, dev)
    else:
        raise typer.BadParameter("stage must be data, extract or train")


@app.command(context_settings=EXTRA)
def fuse(ctx: typer.Context, grid: bool = typer.Option(False, "--grid", help="All folds x train.seeds (default: config fold / seed)."),
         config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Grid-search fusion weights on val (SPEC §5.8), then test pipeline accuracy (M9)."""
    from bgm.retrieval.fuse import run_fuse
    cfg = _cfg(ctx, config)
    run_fuse(cfg, None if grid else [cfg.fold], None if grid else [cfg.seed])


@app.command("tune-stitch", context_settings=EXTRA)
def tune_stitch(ctx: typer.Context, grid: bool = typer.Option(False, "--grid", help="All folds x train.seeds (default: config fold / seed)."),
                config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Tune stitching penalties on val concatenations (SPEC §5.10), evaluate Viterbi vs greedy on test (M10)."""
    from bgm.retrieval.tune_stitch import run_tune_stitch
    cfg = _cfg(ctx, config)
    run_tune_stitch(cfg, None if grid else [cfg.fold], None if grid else [cfg.seed])


@app.command("eval", context_settings=EXTRA)
def eval_(ctx: typer.Context, exp: str = typer.Option("zeroshot", help="Experiment name."),
          config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Evaluate an experiment over the folds (SPEC §5.12). `--exp grounding`: re-score the saved grounding
    predictions with chance levels (no training)."""
    if exp == "grounding":
        from bgm.eval.rescore_grounding import run_grounding_rescore
        run_grounding_rescore(_cfg(ctx, config))
        return
    from bgm.eval.run import run_eval
    run_eval(_cfg(ctx, config), exp)


@app.command(context_settings=EXTRA)
def render(ctx: typer.Context, video: str = typer.Option(..., help="Input video."),
           out: str = typer.Option(..., help="Output video."),
           run: str = typer.Option(None, help="Run directory to use instead of the deployment models."),
           config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Segment, retrieve, stitch and mux BGM onto a video (SPEC §5.9-5.11)."""
    from bgm.render.run import render_video
    typer.echo(str(render_video(_cfg(ctx, config), video, out, run)))


@app.command(context_settings=EXTRA)
def brief(ctx: typer.Context, video: str = typer.Option(..., help="Input video."),
          out: str = typer.Option(..., help="The output video you will render (names renders/<out stem>.brief.yaml)."),
          config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Draft the editable scene brief of a video (D101) without rendering."""
    from pathlib import Path
    from bgm.render.query import featurize
    from bgm.render.scene import brief_path, ensure_brief
    cfg = _cfg(ctx, config)
    work = Path(cfg.paths.renders) / Path(out).stem
    work.mkdir(parents=True, exist_ok=True)
    feat = featurize(video, work, cfg)
    path = brief_path(cfg, Path(out))
    ensure_brief(cfg, path, work / "silent25.mp4", feat["segments"], video)
    typer.echo(str(path))


@app.command(context_settings=EXTRA)
def check(ctx: typer.Context, plan: str = typer.Option(..., help="renders/<name>.plan.json"),
          config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Reference-free check of a render, for silent videos (SPEC §5.13)."""
    from bgm.eval.check import run_check
    typer.echo(str(run_check(_cfg(ctx, config), plan)))


@app.command(context_settings=EXTRA)
def report(ctx: typer.Context, final: bool = typer.Option(False, "--final", help="All paper tables."),
           filmsplit: bool = typer.Option(False, "--filmsplit", help="results/semantic_filmsplit.md and learning_curve.md (D36)."),
           split: str = typer.Option("film", "--split", help="--final: film (results/final.md, the paper) | trackid (results/final_trackid.md)."),
           config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Aggregate results tables from runs/ (`--final`: results/final.md)."""
    if filmsplit:
        from bgm.eval.report_filmsplit import write_baselines_filmsplit, write_filmsplit_report, write_learning_curve
        cfg = _cfg(ctx, config)
        typer.echo(str(write_filmsplit_report(cfg)))
        typer.echo(str(write_baselines_filmsplit(cfg)))
        typer.echo(str(write_learning_curve(cfg)))
        return
    if not final:
        _todo("report", "M11")
    if split == "trackid":
        from bgm.eval.final import write_final
        typer.echo(str(write_final(_cfg(ctx, config))))
        return
    from bgm.eval.final_film import write_final_film
    typer.echo(str(write_final_film(load_config(config, ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", *ctx.args]))))


# ---------------------------------------------------------------- cue-switch predictor (opt-in; PROGRESS.md D78+)
cue_app = typer.Typer(no_args_is_help=True, help="Learned music-change predictor for Viterbi stitching (src/bgm/cue/).")
app.add_typer(cue_app, name="cue")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x != ""]


@cue_app.command("build", context_settings=EXTRA)
def cue_build(ctx: typer.Context, workers: int = typer.Option(16), config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Segment tokens, ContentDetector score cache (R1) and the frozen val / test sequence sets."""
    from bgm.cue.scenes import build_scene_cache
    from bgm.cue.sequences import write_sets
    from bgm.cue.tokens import build_tokens
    from bgm.eval.store import FeatureStore
    cfg = _cfg(ctx, config)
    build_tokens(FeatureStore(cfg))
    build_scene_cache(cfg, workers)
    sets = write_sets(cfg)
    for k, s in sets.items():
        log.info("fold %s: %s", k, {f"{sp}/{m}": len(q) for sp, ms in s.items() for m, q in ms.items()})


@cue_app.command("fit-cosine", context_settings=EXTRA)
def cue_fit_cosine(ctx: typer.Context, folds: str = typer.Option("0,1,2,3,4"), config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """R2: fit the cosine rule's a, b on training sequences per fold."""
    from bgm.cue.baselines import run_fit_cosine
    run_fit_cosine(_cfg(ctx, config), _ints(folds))


@cue_app.command("train", context_settings=EXTRA)
def cue_train(ctx: typer.Context, variants: str = typer.Option("pair,ctx,ctx_cross"), folds: str = typer.Option("0,1,2,3,4"),
              seeds: str = typer.Option("0,1,2"), config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Train cue-switch predictors (pair = R3, ctx = R4, ctx_cross = R5)."""
    from bgm.cue.train import run_train
    run_train(_cfg(ctx, config), variants.split(","), _ints(folds), _ints(seeds))


@cue_app.command("eval", context_settings=EXTRA)
def cue_eval(ctx: typer.Context, fold: int = typer.Option(...), seeds: str = typer.Option("0,1,2"),
             variants: str = typer.Option("pair,ctx,ctx_cross,ctx_nodur", help="Learned variants to score (D94: + pair_splice,ctx_splice)."),
             no_render: bool = typer.Option(False, "--no-render", help="Skip seam-loudness rendering."),
             hazard: str = typer.Option("", help="Comma-separated L (s) for the constant duration prior p = 1 - exp(-d / L)."),
             config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """R0-R6 on the frozen sequences of one fold (needs the D75 emission overrides)."""
    from bgm.cue.evaluate import eval_fold
    hz = tuple(float(x) for x in hazard.split(",") if x)
    typer.echo(str(eval_fold(_cfg(ctx, config), fold, _ints(seeds), tuple(variants.split(",")), render=not no_render, hazard=hz)))


@cue_app.command("probe", context_settings=EXTRA)
def cue_probe(ctx: typer.Context, folds: str = typer.Option("0,1,2,3,4"), seeds: str = typer.Option("0,1,2"),
              variants: str = typer.Option("pair,ctx,ctx_cross,ctx_nodur", help="Learned variants to probe (D94: + pair_splice,ctx_splice)."),
              config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Clip joins or music changes (D85): same-music test sequences, the segment-length rule Rdur, probe logits."""
    from bgm.cue.probes import run_probe
    for d in run_probe(_cfg(ctx, config), _ints(folds), _ints(seeds), variants.split(",")):
        typer.echo(str(d))


@cue_app.command("report", context_settings=EXTRA)
def cue_report(ctx: typer.Context, config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """results/cue_switch.md from runs/<paths.runs>/<cue.exp>/eval/fold*/eval.json."""
    from bgm.cue.report import write_report
    typer.echo(str(write_report(_cfg(ctx, config))))


@cue_app.command("real-eval", context_settings=EXTRA)
def cue_real_eval(ctx: typer.Context, csv: str = typer.Option("data/annotations/cue_changes.csv"),
                  media: str = typer.Option("media/annotated", help="Local videos <media>/<youtube_id>.mp4 (nothing is downloaded)."),
                  out: str = typer.Option("results/cue_real.md"), config: str = typer.Option("configs/default.yaml", help=CONFIG_HELP)):
    """Step 5: boundary metrics on real annotated excerpts (youtube_id,start,end,cue_change_times[,film_key])."""
    from bgm.cue.real_eval import run_real_eval
    typer.echo(str(run_real_eval(_cfg(ctx, config), csv, media, out)))


if __name__ == "__main__":
    app()
