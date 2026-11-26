"""Zero-shot references (SPEC M5): ImageBind, text bridge, hand-crafted dynamics. Random is exact chance."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bgm.eval.attributes import fit_calibration, s_attr
from bgm.eval.store import FeatureStore, l2
from bgm.features.dynamics import M_CHANNELS, V_CHANNELS


def zscore_over(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), floor)


class ZeroShot:
    def __init__(self, store: FeatureStore, cfg, train_ids: list[str], tag: str = ""):
        self.store, self.cfg = store, cfg
        temps = list(cfg.features.attributes.temps)
        self.cal = {"attr": fit_calibration(store, train_ids, "attr_logits_1hz", temps, tag),
                    "bridge": fit_calibration(store, train_ids, "bridge_logits_1hz", temps, tag)}

    # ---- ImageBind: cosine between imagebind_1hz span means
    def imagebind_track(self, q, gallery):
        v = l2(self.store.whole("video", q, "imagebind_1hz"))
        return np.array([l2(self.store.whole("music", m, "imagebind_1hz")) @ v for m in gallery])

    def imagebind_window(self, q, a, b, windows):
        v = l2(self.store.pooled("video", q, "imagebind_1hz", a, b))
        return np.array([l2(self.store.pooled("music", m, "imagebind_1hz", s, s + d)) @ v for m, s, d in windows])

    # ---- attributes and text bridge: mean of s_attr over calibrated attr and bridge distributions
    def _dist(self, side, item, kind, a=None, b=None):
        cal = self.cal[kind]
        tf = cal.transform(side)
        if a is None:
            return self.store.whole(side, item, cal.key, tf)
        return self.store.pooled(side, item, cal.key, a, b, tf)

    def attr_track(self, q, gallery, kinds=("attr",)):
        return np.mean([[s_attr(self._dist("video", q, k), self._dist("music", m, k)) for m in gallery] for k in kinds], 0)

    def bridge_track(self, q, gallery):
        return self.attr_track(q, gallery, ("attr", "bridge"))

    def attr_window(self, q, a, b, windows, kinds=("attr",)):
        return np.mean([[s_attr(self._dist("video", q, k, a, b), self._dist("music", m, k, s, s + d)) for m, s, d in windows]
                        for k in kinds], 0)

    def bridge_window(self, q, a, b, windows):
        return self.attr_window(q, a, b, windows, ("attr", "bridge"))

    # ---- hand-crafted dynamics over every 25 Hz offset (SPEC §5.12 baseline)
    def hand_offsets(self, q, a, b, m):
        """z(cut-on-beat rate of the segment's internal cuts vs the window's beats) + z(Pearson r of z-scored
        log_motion and rms_db); the correlation alone when the segment has no internal cuts."""
        st = self.store
        v, mu = st.dyn("video", q), st.dyn("music", m)
        f0, L = int(round(a * 25)), int(round((b - a) * 25))
        motion = v[f0:f0 + L, V_CHANNELS.index("log_motion")]
        L = len(motion)
        W = sliding_window_view(mu[:, M_CHANNELS.index("rms_db")], L)           # [n_off, L]
        mc = motion - motion.mean()
        Wc = W - W.mean(1, keepdims=True)
        r = (Wc @ mc) / (np.sqrt((Wc ** 2).sum(1) * (mc ** 2).sum()) + 1e-9)
        score = zscore_over(r)
        cuts = np.asarray(st.video(q)["cut_times_s"], float)
        rel = cuts[(cuts > a) & (cuts < b)] - a
        beats = np.sort(np.asarray(st.music(m)["beat_times_s"], float))
        if len(rel) and len(beats):
            t = np.arange(len(W))[:, None] / 25 + rel[None, :]                   # [n_off, n_cuts] music times
            i = np.clip(np.searchsorted(beats, t), 1, len(beats) - 1)
            near = np.minimum(np.abs(t - beats[i - 1]), np.abs(t - beats[i]))
            rate = (near <= self.cfg.eval.cut_beat_tol_s).mean(1)
            score = score + zscore_over(rate)
        return score
