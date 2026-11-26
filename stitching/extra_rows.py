"""Extra rows decoded on the same cached episodes: the segmental (semi-Markov) Viterbi.
`.venv/bin/python -m stitching.extra_rows --fold k` -> stitching/results/fold<k>_semi.json"""
import argparse, json, logging, time
from pathlib import Path

import numpy as np

from stitching.data import load_fold
from stitching.metrics import evaluate_path, fold_entropy
from stitching.semi_viterbi import SemiViterbi

log = logging.getLogger(__name__)
SEMI_ROWS = [("semi_viterbi", "Segmental Viterbi (semi-Markov)")]


def main(k):
    t0 = time.time()
    F = load_fold(k)
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    st = SemiViterbi()
    params = st.fit(tr, va)
    paths = [st.timed(e) for e in te]
    rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
    log.info("fold %d semi: %s | test acc %.3f F1 %.3f", k, params, np.mean([r["track_acc"] for r in rows]), np.mean([r["switch_f1"] for r in rows]))
    out = {"fold": k, "methods": {"semi_viterbi": {"name": SEMI_ROWS[0][1], "params": params, "rows": rows,
                                                   "entropy": fold_entropy(te, paths, F["gallery_test"])}},
           "timelines": {"semi_viterbi": [[e.files[m] for m in p.mid] for e, p in zip(te, paths)]}}
    Path(f"stitching/results/fold{k}_semi.json").write_text(json.dumps(out, default=float))
    log.info("fold %d done [%.0f s]", k, time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True)
    main(ap.parse_args().fold)
