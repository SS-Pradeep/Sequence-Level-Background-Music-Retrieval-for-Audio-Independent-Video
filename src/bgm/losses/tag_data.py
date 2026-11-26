"""Tag features for the loss sweep: full option probabilities from data/features/tags/raw/ (D141), via the provided
`load_tag_probs` (prompts A and B averaged, groups normalised, prior-corrected) and `pool_windows`.

The prior correction is estimated on the run's TRAINING items only (scene segments of training videos; music windows
of their music files), so no reported, early-stopping or TEST item shapes the training targets. Tags are only used in
training losses, so only training items need vectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.losses.tag_aware_loss import load_tag_probs, pool_windows
from bgm.tagging.vocab import load_vocab

RAW = Path("data/features/tags/raw")
WIN_S = 10.0


def _raw_file(side: str) -> Path:
    files = sorted(RAW.glob(f"{side}_*.jsonl"))
    assert len(files) == 1, f"expected one {side} tag file in {RAW}, found {files}"
    return files[0]


def raw_long(side: str) -> pd.DataFrame:
    """Long table: id, k (music: window offset s; scene: seg_idx), prompt, group, tag, logp — every option present."""
    vocab = load_vocab()
    rows = []
    with open(_raw_file(side)) as f:
        for line in f:
            r = json.loads(line)
            (i, k), pr = r["key"], r["prompt_id"]
            for g, lp in r["logp"].items():
                assert len(lp) == len(vocab[g]), (side, g, len(lp))
                rows += [(i, k, pr, g, t, v) for t, v in zip(vocab[g], lp)]
    return pd.DataFrame(rows, columns=["id", "k", "prompt", "group", "tag", "logp"])


class TagVectors:
    """Per-segment scene and music tag vectors [n_tags] for training items, prior from the given training videos."""

    def __init__(self, store, train_ids: list[str], scene_df: pd.DataFrame, music_df: pd.DataFrame):
        train = set(train_ids)
        musics = {store.music_of[p] for p in train}
        sd = scene_df[scene_df.id.isin(train)].assign(item=lambda d: d.id + "|" + d.k.astype(int).astype(str))
        md = music_df[music_df.id.isin(musics)].assign(item=lambda d: d.id + "|" + d.k.astype(float).map("{:.3f}".format))
        kw = dict(item_col="item", prompt_col="prompt", group_col="group", tag_col="tag", value_col="logp",
                  value_is_logprob=True, prior_correct=True)
        s_items, s_vocab, s_slices, S = load_tag_probs(sd, **kw)
        m_items, m_vocab, m_slices, M = load_tag_probs(md, **kw)
        assert s_vocab == m_vocab and s_slices == m_slices, "scene and music tag vocabularies differ"
        self.vocab, self.group_slices = s_vocab, s_slices
        self.groups = sorted({g for g, _ in s_vocab})
        self._scene = {it: S[i] for i, it in enumerate(s_items)}
        self._win: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        by_m: dict[str, list] = {}
        for i, it in enumerate(m_items):
            mid, off = it.rsplit("|", 1)
            by_m.setdefault(mid, []).append((float(off), i))
        for mid, lst in by_m.items():
            lst.sort()
            starts = np.array([o for o, _ in lst])
            dur = store.duration("music", mid)
            self._win[mid] = (M[[i for _, i in lst]], starts, np.minimum(starts + WIN_S, max(dur, WIN_S)))
        self.store = store
        self.genre_slice = self.group_slices[self.groups.index("genre")]

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def scene(self, pair_id: str, seg_idx: int) -> np.ndarray:
        return self._scene[f"{pair_id}|{int(seg_idx)}"]

    def music(self, music_id: str, a: float, b: float) -> np.ndarray:
        P, s, e = self._win[music_id]
        return pool_windows(P, s, e, a, b, self.group_slices).astype(np.float32)

    def genre(self, t_music: np.ndarray) -> int:
        """Genre label (decision 6): argmax of the prior-corrected music genre group."""
        return int(np.argmax(t_music[self.genre_slice]))
