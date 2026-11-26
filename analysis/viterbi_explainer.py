"""Data for the interactive Viterbi explainer: the worked example's real lattice (D125), exported as JSON.

python -m bgm.analysis.viterbi_explainer         -> outputs/viterbi_explainer/lattice.json (needs the GPU models)
python -m bgm.analysis.viterbi_explainer build   -> outputs/viterbi_explainer/keep_or_switch.html (template + JSON)

Rebuilds the lattice of outputs/worked_example (title split fold 1, blocked test sequence 21, fold-1 models, cached
query features in outputs/worked_example/work/query.npz) with the same calls as bgm.analysis.worked_example.main, and
stores every candidate (file, offset, duration, key, tempo, emission z) plus the change predictor's logits. A plain
full Viterbi over every candidate (no beam: at most 37 candidates per segment, beam 100) is the algorithm the page
runs in the browser; it is asserted to reproduce the greedy, fixed-penalty and final paths saved in example.json.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
OUT = Path("outputs/viterbi_explainer")
EX = Path("outputs/worked_example")


def full_viterbi(layers: list[dict], lam_switch: np.ndarray, lam_jump: np.ndarray, mu: float, tol: float = 0.04) -> list[int]:
    """Viterbi over every candidate. layers[i]: dict of arrays mid, off, dur, key, tempo, z. lam_*[i]: boundary i."""
    from bgm.retrieval.stitch_eval import fifths_matrix, tempo_matrix
    D = -layers[0]["z"]
    backs = []
    for i in range(1, len(layers)):
        P, C = layers[i - 1], layers[i]
        same = P["mid"][:, None] == C["mid"][None, :]
        contiguous = np.abs(C["off"][None, :] - (P["off"] + P["dur"])[:, None]) <= tol
        switch = lam_switch[i - 1] + mu * (fifths_matrix(P["key"], C["key"]) + tempo_matrix(P["tempo"], C["tempo"]))
        tot = D[:, None] + np.where(same, np.where(contiguous, 0.0, lam_jump[i - 1]), switch)
        backs.append(tot.argmin(0))
        D = tot.min(0) - C["z"]
    path = [int(D.argmin())]
    for b in reversed(backs):
        path.append(int(b[path[-1]]))
    return path[::-1]


def export() -> None:
    from bgm.analysis.worked_example import FOLD, OVERRIDES, RUN, tags_for
    from bgm.config import load_config
    from bgm.cue.hook import cue_logits, video_tokens
    from bgm.data.manifest import read_folds
    from bgm.eval.store import FeatureStore
    from bgm.render.query import as_video_query, featurize
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.retrieval.pipeline import coarse_shortlist, fine
    from bgm.retrieval.stitch_eval import build_lattice

    ex = json.loads((EX / "example.json").read_text())
    cfg = load_config("configs/default.yaml", OVERRIDES)
    store = FeatureStore(cfg)
    assert (EX / "work" / "query.npz").exists(), "run python -m bgm.analysis.worked_example first"
    feat = featurize(EX / "input_silent.mp4", EX / "work", cfg)
    fusion = json.loads((RUN / "fusion.json").read_text())
    stitch = json.loads((RUN / "stitch.json").read_text())
    models = FoldModels.load(cfg, store, FOLD, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant)
    gallery = list(dict.fromkeys(store.music_of[p] for p in read_folds(cfg)["folds"][str(FOLD)]["test"]))
    index = MusicIndex(store, models, gallery)
    vq = as_video_query(feat)
    vq.prepare(models, store.min_span)
    shortlist = coarse_shortlist(index, vq, fusion["w_coarse"], fusion["K"])
    assert shortlist == ex["coarse"]["shortlist"]
    results = fine(index, vq, shortlist, fusion["window_hop_s"])
    lat = build_lattice(index, vq, results, fusion["w_fine"], fusion["J"])
    z, prm = cue_logits(cfg, video_tokens(feat), cfg.cue.variant, [FOLD], list(cfg.cue.seeds))
    z = np.asarray(z, float)
    assert np.allclose(1 / (1 + np.exp(-z)), ex["predictor"]["p"], atol=1e-6)

    layers = [{"mid": L.mid.astype(int), "off": L.off.astype(float), "dur": L.dur.astype(float), "key": L.key.astype(int),
               "tempo": L.tempo.astype(float), "z": -L.cost.astype(float), "n_top": int(L.n_top)} for L in lat]
    nb = len(layers) - 1
    pen, tol = stitch["penalties"], stitch["tol_s"]
    picks = {"greedy": [int(np.argmax(L["z"][:L["n_top"]])) for L in layers],
             "viterbi_fixed": full_viterbi(layers, np.full(nb, pen["lam_switch"]), np.full(nb, pen["lam_jump"]), pen["mu"], tol)}
    ls = -prm["alpha"] * z
    picks["final"] = full_viterbi(layers, ls, ls + prm["lam_jump"], prm["mu"], tol)
    for name, pk in picks.items():                     # same windows as the saved decoders
        saved = ex["paths"][name]["segments"]
        got = [(index.ids[layers[i]["mid"][j]], round(float(layers[i]["off"][j]), 3)) for i, j in enumerate(pk)]
        assert got == [(s["music_id"], round(s["offset_s"], 3)) for s in saved], (name, got)
        log.info("%s reproduced: %s", name, pk)

    tag, _ = tags_for(ex)
    used = sorted({int(m) for L in layers for m in L["mid"]})
    n = sum(1 for t in tag if not tag[t][0].startswith("S"))
    files = []
    for m in used:                                     # label every lattice file by its song (track) group
        mid = index.ids[m]
        trk = store.music_track[mid]
        if trk not in tag:
            tag[trk] = (chr(ord("a") + n) if n < 26 else f"z{n}", None)
            n += 1
        files.append({"code": m, "music_id": mid, "track_id": trk, "label": tag[trk][0], "duration_s": float(index.items[m].duration)})
    thumbs = [base64.b64encode(Path(p).read_bytes()).decode() for p in ex["keyframes"]]
    out = {
        "case": ex["case"], "segments": ex["segments"], "boundary_times_s": ex["predictor"]["boundary_times_s"],
        "tol_s": tol, "fixed": pen, "cue": {"alpha": prm["alpha"], "lam_jump": prm["lam_jump"], "mu": prm["mu"],
                                            "logits": z.tolist(), "p": ex["predictor"]["p"]},
        "files": files, "true_tracks": [c["track_id"] for c in ex["case"]["clips"]],
        "layers": [{"n_top": L["n_top"], "c": [[int(m), round(float(o), 3), round(float(d), 3), int(k), round(float(t), 2), float(zz)]
                                              for m, o, d, k, t, zz in zip(L["mid"], L["off"], L["dur"], L["key"], L["tempo"], L["z"])]}
                   for L in layers],
        "saved_picks": picks,
        "saved_metrics": {k: {m: ex["paths"][k][m] for m in ("switch_f1_1s", "part_track_accuracy", "switches_per_min")} for k in picks},
        "thumbs_jpg_b64": thumbs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lattice.json").write_text(json.dumps(out))
    log.info("wrote %s (%d segments, %d-%d candidates, %d files)", OUT / "lattice.json", len(layers),
             min(len(L["z"]) for L in layers), max(len(L["z"]) for L in layers), len(files))


def build() -> None:
    """Inline lattice.json into the page template (src/bgm/analysis/viterbi_explainer.html)."""
    page = Path(__file__).with_suffix(".html").read_text()
    data = (OUT / "lattice.json").read_text().replace("</", "<\\/")
    page = page.replace("/*DATA*/null", data)
    (OUT / "keep_or_switch.html").write_text(page)    # artifact body: the host adds doctype and charset
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n')
    page = page.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    (OUT / "keep_or_switch_standalone.html").write_text(head + page + "\n</body>\n</html>\n")   # supplementary file
    log.info("wrote %s and the standalone copy", OUT / "keep_or_switch.html")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    {"export": export, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
