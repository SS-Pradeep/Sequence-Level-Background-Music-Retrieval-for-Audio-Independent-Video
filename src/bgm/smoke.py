"""`bgm smoke --fake`: synthetic clips through everything built so far, on CPU with fake encoders.

M0 fixtures -> M1 manifest -> M2 prep + shots -> M3 extract (fake encoders) + check-features -> M4 segments + alignment.
"""
from __future__ import annotations

import logging
import sys
import tempfile

from bgm.tools import REPO_ROOT

log = logging.getLogger(__name__)


def run_smoke(n_pairs: int = 3, seed: int = 0) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from tests.fixtures.pipeline import run_through_extract
    from tests.fixtures.synth import make_pairs

    from bgm.data.alignment import run_alignment_audit
    from bgm.data.segments import run_segment
    from bgm.features.check import run_check

    with tempfile.TemporaryDirectory(prefix="bgm_smoke_") as tmp:
        pairs = make_pairs(f"{tmp}/raw", n=n_pairs, seed=seed, durations=(12.0, 14.0))
        cfg, df = run_through_extract(tmp, pairs)
        bad = run_check(cfg)
        assert not bad, bad
        segs = run_segment(cfg)
        res = run_alignment_audit(cfg)
        assert res["mean_excess"] > 0, res            # fixture cuts sit on beats by construction
        if n_pairs >= 20:                              # M1 folds (5 per duration quartile) + M5 zero-shot + M6 semantic heads
            from omegaconf import OmegaConf

            from bgm.data import manifest
            from bgm.data.folds import check_folds, folds_document, make_folds
            from bgm.eval.run import run_eval
            from bgm.train.semantic import run_train_semantic
            pairs = manifest.read_pairs(cfg)
            folds = make_folds(pairs)
            check_folds(folds, pairs)
            manifest.write_folds(folds_document(folds, manifest.manifest_hash(cfg)), cfg)
            zs = run_eval(cfg, "zeroshot")
            cfg2 = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["train.semantic.max_epochs=3", "train.semantic.batch=4"]))
            m = run_train_semantic(cfg2, folds=[0], seeds=[0])[0]
            log.info("M5 zero-shot: %d result rows; M6 semantic: test track R@10 %.2f", len(zs), m["test"]["track_test_fold"]["R@10"])
        log.info("smoke --fake passed: %d pairs, %d segments, alignment excess %+.2f", len(df), len(segs), res["mean_excess"])
