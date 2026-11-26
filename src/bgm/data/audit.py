"""`bgm audit` (SPEC M1): pairs.csv, tracks.csv and reports/audit.md from data/raw/.

Layout of data/raw (a read-only symlink to dataset_merged/clips, PROGRESS.md M1):
  video/<stem>.mp4, stem = <pair_id> or <pair_id>_no_vocals
  audio/<pair_id>.wav                 the music (music_source = file)
  audio/<pair_id>_no_vocals.wav       digitally silent; never used
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.data import manifest, tracks
from bgm.data.media import ffprobe

log = logging.getLogger(__name__)


def parse_meta(pair_id: str) -> dict:
    """meta_source / meta_film from the file naming, plus the sunnxt clip key."""
    if pair_id.startswith("sunnxt_"):
        parts = pair_id.split("_")
        return {"meta_source": "sunnxt", "meta_film": parts[1], "meta_clip_key": "_".join(parts[2:])}
    parts = pair_id.split("__")
    if len(parts) >= 3:
        return {"meta_source": parts[0].removeprefix("combined_"), "meta_film": parts[1], "meta_clip_key": ""}
    return {"meta_source": "unknown", "meta_film": "", "meta_clip_key": ""}


def sunnxt_positions(root: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """(movie id, clip_id) -> (start_s, end_s) in the source movie, from each movie's manifest.csv."""
    out = {}
    for m in sorted(root.glob("*/manifest.csv")):
        df = pd.read_csv(m, dtype={"clip_id": str})
        for r in df.itertuples():
            out[(m.parent.name, r.clip_id)] = (float(r.start_sec), float(r.end_sec))
    return out


def discover(raw: Path) -> tuple[list[dict], list[str]]:
    """Pairs found in data/raw, and music files that belong to no video (reported, not used)."""
    rows, seen = [], {}
    for v in sorted((raw / "video").glob("*.mp4")):
        pid = v.stem.removesuffix("_no_vocals")
        # a few ids exist as both <id>.mp4 and <id>_no_vocals.mp4; keep the _no_vocals copy like the rest
        if pid in seen and not v.stem.endswith("_no_vocals"):
            continue
        row = {"pair_id": pid, "video_path": str(raw / "video" / v.name), "music_path": str(raw / "audio" / f"{pid}.wav")}
        if pid in seen:
            rows[seen[pid]] = row
        else:
            seen[pid] = len(rows)
            rows.append(row)
    used = {Path(r["music_path"]).name for r in rows}
    orphans = sorted(a.name for a in (raw / "audio").glob("*.wav")
                     if a.name not in used and not a.stem.endswith("_no_vocals"))
    return rows, orphans


def probe_row(row: dict) -> dict:
    """Fill the ffprobe columns of SPEC §3.1; raises on unreadable media. An empty `music_path` means the
    music is embedded in the video's own audio stream (music_source = embedded)."""
    v = ffprobe(row["video_path"])
    embedded = not row.get("music_path")
    m = {"duration_s": v["audio"]["duration_s"] if v["audio"] else np.nan, "audio": v["audio"]} if embedded else ffprobe(row["music_path"])
    assert v["video"] is not None, f"no video stream in {row['video_path']}"
    assert m["audio"] is not None, f"no audio stream for the music of {row['pair_id']}"
    return {**row, "music_path": "" if embedded else row["music_path"], "music_source": "embedded" if embedded else "file",
            "music_id": row.get("music_id") or row["pair_id"],
            "video_duration_s": v["video"]["duration_s"] if np.isfinite(v["video"]["duration_s"]) else v["duration_s"],
            "music_duration_s": m["duration_s"], "video_start_s": v["video"]["start_s"],
            "audio_start_s": v["audio"]["start_s"] if v["audio"] else np.nan,
            "fps": v["video"]["fps"], "width": v["video"]["width"], "height": v["video"]["height"],
            "video_has_audio": v["audio"] is not None,
            "music_sr": m["audio"]["sr"], "music_channels": m["audio"]["channels"]}


def run_audit(cfg: DictConfig) -> pd.DataFrame:
    raw = Path(cfg.paths.raw)
    rows, orphans = discover(raw)
    log.info("data/raw: %d videos, %d orphan music files", len(rows), len(orphans))

    excluded, keep = [], []
    for r in rows:
        rule = next((e for e in cfg.audit.exclude if re.search(e.pattern, r["pair_id"])), None)
        (excluded if rule else keep).append((r, rule.reason if rule else None))
    keep = [r for r, _ in keep]

    failures, probed = [], []
    with ThreadPoolExecutor(cfg.audit.workers) as ex:
        futs = {ex.submit(probe_row, r): r for r in keep}
        for f, r in futs.items():
            try:
                probed.append(f.result())
            except Exception as e:  # noqa: BLE001 -- every failure is reported (never silently skipped)
                failures.append((r["pair_id"], f"{type(e).__name__}: {e}"))
    df = pd.DataFrame(probed)
    meta = pd.DataFrame([parse_meta(p) for p in df["pair_id"]])
    df = pd.concat([df.reset_index(drop=True), meta], axis=1)
    pos = sunnxt_positions(Path(cfg.audit.sunnxt_manifests))
    starts = [pos.get((f, k), (np.nan, np.nan)) for f, k in zip(df["meta_film"], df["meta_clip_key"])]
    df["meta_movie_start_s"] = [s for s, _ in starts]
    df["meta_movie_end_s"] = [e for _, e in starts]
    df = df.drop(columns="meta_clip_key").sort_values("pair_id").reset_index(drop=True)

    # track groups (§3.2 step 2); metadata grouping (step 1) has no song titles to use here
    fps = tracks.fingerprint_all(df["music_path"].tolist(), cfg.audit.workers, cfg.tracks.highpass_hz)
    labels, checks, links = tracks.group_tracks(fps, cfg.tracks.ber_max, cfg.tracks.min_overlap_s, cfg.tracks.min_votes,
                                                df["music_path"].tolist(), cfg.tracks.spec_min)
    # user-confirmed embedding links (§3.2 step 3) join the fingerprint links
    from bgm.data.near_dups import read_confirmed
    idx = {m: k for k, m in enumerate(df["music_id"])}
    confirmed = [(idx[a], idx[b]) for a, b in zip(*[read_confirmed(cfg)[c] for c in ("music_id_a", "music_id_b")]) if a in idx and b in idx]
    if confirmed:
        labels = tracks.union_find(len(df), [(c.i, c.j) for c in links] + confirmed)
    order = {lab: k for k, lab in enumerate(sorted(set(labels)))}
    df["track_id"] = [f"trk_{order[lab]:04d}" for lab in labels]
    linked = {i for c in links for i in (c.i, c.j)}
    trk = (df.assign(_linked=[i in linked for i in range(len(df))])
             .groupby("track_id").agg(music_ids=("music_id", ";".join), _linked=("_linked", "any")).reset_index())
    emb_linked = {i for pair in confirmed for i in pair}
    trk_emb = set(df.track_id.iloc[sorted(emb_linked)]) if emb_linked else set()
    trk["evidence"] = [("fingerprint" if l else "none") if t not in trk_emb else ("fingerprint;embedding-confirmed" if l else "embedding-confirmed")
                       for t, l in zip(trk.track_id, trk._linked)]

    manifest.write_pairs(df, cfg)
    manifest.write_tracks(trk, cfg)
    write_report(cfg, df, excluded, failures, orphans, checks, links, fps)
    return df


def write_report(cfg, df, excluded, failures, orphans, checks, links, fps) -> None:
    rep = Path(cfg.paths.reports)
    rep.mkdir(parents=True, exist_ok=True)
    mism = df[(df.video_duration_s - df.music_duration_s).abs() > cfg.media.max_duration_mismatch_s]
    av = (df.audio_start_s - df.video_start_s).dropna()
    groups = df.groupby("track_id").size()
    ber_max = cfg.tracks.ber_max
    spec_min = cfg.tracks.spec_min
    near = sorted([c for c in checks if c.ber <= ber_max], key=lambda c: abs(c.spec_corr - spec_min))
    above = [c for c in near if c.spec_corr >= spec_min][: cfg.tracks.borderline_report]
    below = [c for c in near if c.spec_corr < spec_min][: cfg.tracks.borderline_report]
    ids = df["pair_id"].tolist()
    L = ["# Audit (`bgm audit`)", "",
         f"- Videos in data/raw: {len(df) + len(excluded) + len(failures)}; **kept {len(df)}**, "
         f"excluded {len(excluded)}, unreadable {len(failures)}.",
         f"- Orphan music files (no video): {len(orphans)}" + (f", e.g. {orphans[:3]}" if orphans else ""),
         "", "## Durations (s)", "", "| | min | median | max |", "| --- | --- | --- | --- |",
         *[f"| {c} | {df[c].min():.2f} | {df[c].median():.2f} | {df[c].max():.2f} |" for c in ("video_duration_s", "music_duration_s")],
         "", f"- Video vs music duration differs by > {cfg.media.max_duration_mismatch_s} s: **{len(mism)}** pairs"
         + (f", e.g. {mism.pair_id.head(5).tolist()}" if len(mism) else ""),
         "", "## Streams", "",
         f"- fps: {df.fps.round(3).value_counts().sort_index().to_dict()}",
         f"- Music sample rates: {df.music_sr.value_counts().to_dict()}; channels: {df.music_channels.value_counts().to_dict()}",
         f"- Videos with their own audio stream: {int(df.video_has_audio.sum())}/{len(df)}",
         f"- Video stream start_s: {df.video_start_s.value_counts().head(5).to_dict()}",
         f"- Embedded audio start minus video start (s): min {av.min():.3f}, median {av.median():.3f}, max {av.max():.3f}; "
         f"nonzero in {int((av.abs() > 1e-3).sum())} videos" if len(av) else "- No embedded audio",
         "", "## Excluded pairs", ""]
    by_reason = {}
    for r, reason in excluded:
        by_reason.setdefault(reason, []).append(r["pair_id"])
    L += [f"- {len(v)} pairs: {k}" for k, v in by_reason.items()] or ["- none"]
    L += ["", "## Unreadable files", ""] + ([f"- `{p}`: {e}" for p, e in failures] or ["- none"])
    L += ["", "## Track groups (Chromaprint, SPEC §3.2)", "",
          f"- {len(checks)} candidate pairs checked; **{len(links)} linked**: BER <= {ber_max} over >= {cfg.tracks.min_overlap_s} s "
          f"after a {cfg.tracks.highpass_hz:.0f} Hz high-pass, and log-mel correlation >= {spec_min} at the matched offset.",
          f"- Candidates passing BER but failing the log-mel check (rejected): {sum(c.ber <= ber_max and c.spec_corr < spec_min for c in checks)}.",
          f"- {groups.size} track groups; {int((groups > 1).sum())} with more than one pair; largest {int(groups.max())}.",
          f"- Fingerprint rate: {np.median([f.rate_hz for f in fps]):.2f} sub-fingerprints / s.",
          "", f"### Borderline candidates for listening (log-mel correlation closest to {spec_min})", "",
          "| pair A | pair B | BER | log-mel corr | overlap s | linked |", "| --- | --- | --- | --- | --- | --- |",
          *[f"| {ids[c.i]} | {ids[c.j]} | {c.ber:.3f} | {c.spec_corr:.2f} | {c.overlap_s:.1f} | {'yes' if c in above else 'no'} |"
            for c in sorted(below + above, key=lambda c: -c.spec_corr)],
          "", "### Groups with more than one pair", ""]
    multi = groups[groups > 1].sort_values(ascending=False)
    L += [f"- {t} ({n}): {', '.join(df.pair_id[df.track_id == t].head(4))}{' ...' if n > 4 else ''}"
          for t, n in multi.items()] or ["- none"]
    (rep / "audit.md").write_text("\n".join(L) + "\n")
