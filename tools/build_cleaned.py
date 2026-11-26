"""Build `cleaned_dataset/` from the pair trimmer's output (tools/pair_trimmer.py, dataset_1248/trimmed/).

The trimmer's pairs.csv points at two places: trimmed copies under `trimmed/` and untouched originals. This makes
one self-contained folder, and the pipeline manifests for it:

* `cleaned_dataset/video/`, `audio/`: every kept pair's files, copied (originals and trimmed parts alike).
* `cleaned_dataset/pairs.csv`: the trimmer's manifest with paths relative to `cleaned_dataset/`.
* `cleaned_dataset/manifests/`: `pairs.csv` (paths relative to the repo root, `trim_*` kept as `meta_trim_*`),
  `tracks.csv` and `folds_title.json` for `paths.manifests=cleaned_dataset/manifests`.

Folds are carried over, not redrawn: every kept pair, and every part of a split pair, takes its source pair's
train / val / test role in each fold of `data/manifests/folds_title.json` (D102), and its film group. Excluded pairs
are dropped, and so are parts under 1 s (no 1 Hz feature frame). `check_folds` runs on the result.

    .venv/bin/python tools/build_cleaned.py
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from bgm.data import manifest
from bgm.data.folds import check_folds, folds_document

log = logging.getLogger(__name__)
TRIM_COLS = ["trim_start_s", "trim_end_s", "trim_source_id"]
MIN_DURATION_S = 1.0           # T1 = floor(duration_s) 1 Hz frames: a shorter clip has none


def source_id(row: dict) -> str:
    """The original pair a cleaned pair comes from (itself if it was not split)."""
    return row.get("trim_source_id") or row["pair_id"]


def carry_folds(old: dict, pair_ids: list[str], sources: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """New folds and film groups: each pair takes its source pair's split in every fold.

    old: a folds document ({"folds": {k: {"train"|"val"|"test": [pair_id]}}, "film_group": {pair_id: group}}).
    sources: {new pair_id: old pair_id}. Returns (folds, film_group) over `pair_ids`."""
    role = {k: {p: s for s in ("train", "val", "test") for p in f[s]} for k, f in old["folds"].items()}
    folds = {k: {s: sorted(p for p in pair_ids if role[k][sources[p]] == s) for s in ("train", "val", "test")}
             for k in old["folds"]}
    return folds, {p: old["film_group"][sources[p]] for p in pair_ids}


def carry_tracks(old: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """tracks.csv for the kept pairs: their track groups, music_ids rewritten (split parts have new ids)."""
    ids = pairs.groupby("track_id").music_id.apply(lambda s: ";".join(sorted(set(s))))
    out = old[old.track_id.isin(ids.index)].copy()
    out["music_ids"] = out.track_id.map(ids)
    assert len(out) == len(ids), "a kept pair's track_id is missing from tracks.csv"
    return out


def build(src_root: Path, out_root: Path, old_manifests: Path, folds_name: str = "folds_title.json") -> pd.DataFrame:
    trimmed = pd.read_csv(src_root / "trimmed" / "pairs.csv", dtype=str, keep_default_na=False)
    for c in TRIM_COLS:
        trimmed[c] = trimmed.get(c, "")
    short = trimmed.video_duration_s.astype(float) < MIN_DURATION_S
    for pid in trimmed.pair_id[short]:
        log.warning("dropped %s: shorter than %.0f s, so it has no 1 Hz feature frame (T1 = floor(duration), SPEC §4.2)",
                    pid, MIN_DURATION_S)
    trimmed = trimmed[~short].reset_index(drop=True)
    for kind, col in (("video", "video_path"), ("audio", "music_path")):
        names = trimmed[col].map(lambda p: Path(p).name)
        assert names.is_unique, f"two {kind} files share a name"
        (out_root / kind).mkdir(parents=True, exist_ok=True)
        for rel, name in zip(trimmed[col], names):
            src, dst = src_root / rel, out_root / kind / name
            if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
                shutil.copy2(src, dst)
        trimmed[col] = [f"{kind}/{n}" for n in names]
    trimmed.to_csv(out_root / "pairs.csv", index=False)

    pairs = trimmed.copy()
    for col in ("video_path", "music_path"):
        pairs[col] = [str(out_root / p) for p in pairs[col]]
    pairs = pairs.rename(columns={c: f"meta_{c}" for c in TRIM_COLS})
    cfg = OmegaConf.create({"paths": {"manifests": str(out_root / "manifests")}, "splits": {"folds_file": folds_name}})
    old_cfg = OmegaConf.create({"paths": {"manifests": str(old_manifests)}, "splits": {"folds_file": folds_name}})
    manifest.write_pairs(pairs, cfg)
    manifest.write_tracks(carry_tracks(manifest.read_tracks(old_cfg), pairs), cfg)

    old = manifest.read_folds(old_cfg)
    pairs = manifest.read_pairs(cfg)
    sources = {r["pair_id"]: r["meta_trim_source_id"] or r["pair_id"] for r in pairs.to_dict("records")}
    folds, film_group = carry_folds(old, list(pairs.pair_id), sources)
    train_only = set(old.get("train_only", []))
    check_folds(folds, pairs, film_group, train_only)
    extra = {k: old[k] for k in ("split", "links_file") if k in old}
    doc = folds_document(folds, manifest.manifest_hash(cfg), film_group=film_group, train_only=sorted(train_only),
                         carried_from=str(old_manifests / folds_name), **extra)
    manifest.write_folds(doc, cfg)
    return pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("dataset_1248"))
    ap.add_argument("--out", type=Path, default=Path("cleaned_dataset"))
    ap.add_argument("--old-manifests", type=Path, default=Path("data/manifests"))
    a = ap.parse_args()
    pairs = build(a.src, a.out, a.old_manifests)
    log.info("%d pairs, %d tracks, %d films -> %s", len(pairs), pairs.track_id.nunique(), pairs.meta_film.nunique(), a.out)


if __name__ == "__main__":
    main()
