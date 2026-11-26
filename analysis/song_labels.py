"""Song/dance vs background-score labels for every music file, from two independent signals.

1. Demucs (htdemucs, 44.1 kHz) vocal share: vocal-stem energy / total energy, and the share of seconds where the
   vocal stem carries >= 30% of the energy.
2. CLAP zero-shot (the stored 1 Hz CLAP embeddings, music_and_speech checkpoint): softmax over
   {singing, speech/dialogue, instrumental score} prompt groups, averaged over the clip.
Label: song = CLAP 'singing' wins AND vocal seconds >= 0.3; background = CLAP 'instrumental' wins AND vocal
seconds < 0.3; everything else (e.g. dialogue over score, disagreement) = uncertain.
Run: python -m bgm.analysis.song_labels -> data/manifests/song_labels.csv (analysis only; not used by training).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch

from bgm.config import load_config
from bgm.data import manifest
from bgm.eval.store import FeatureStore

log = logging.getLogger(__name__)
PROMPTS = {"singing": ["a song with a singer singing", "vocal music with singing", "an Indian film song with vocals",
                       "people singing and dancing to a song"],
           "speech": ["people talking", "dialogue over background music", "a person speaking"],
           "instrumental": ["instrumental background score", "orchestral film background music without vocals",
                            "instrumental music with no singing"]}


def clap_zero_shot(cfg, store: FeatureStore, music_ids) -> pd.DataFrame:
    from bgm.features.hf import ClapEncoder
    enc = ClapEncoder(cfg.features.sem_music.clap_model, "cuda")
    groups = list(PROMPTS)
    T = np.stack([(lambda e: e / np.linalg.norm(e))(enc.encode_text(PROMPTS[g]).mean(0)) for g in groups])
    rows = []
    for m in music_ids:
        e = store.music(m)["sem_clap_1hz"].astype(np.float32)
        z = enc.logit_scale * e @ T.T
        p = np.exp(z - z.max(1, keepdims=True))
        p = (p / p.sum(1, keepdims=True)).mean(0)
        rows.append({"music_id": m, **{f"clap_{g}": float(x) for g, x in zip(groups, p)}})
    del enc
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


@torch.no_grad()
def demucs_vocals(cfg, music_ids) -> pd.DataFrame:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs").cuda().eval()
    sr = model.samplerate
    vi = model.sources.index("vocals")
    rows = []
    for n, m in enumerate(music_ids):
        y, fsr = sf.read(Path(cfg.paths.interim) / "music" / f"{m}.flac", dtype="float32")
        assert fsr == 48000
        y = soxr.resample(y, fsr, sr).astype(np.float32)
        x = torch.from_numpy(np.stack([y, y]))[None].cuda()
        ref = x.mean(1, keepdim=True)
        xs = (x - ref.mean()) / (ref.std() + 1e-8)
        src = apply_model(model, xs, split=True, overlap=0.1, progress=False)[0]        # [S, 2, N]
        e = (src ** 2).mean(1)                                                         # [S, N]
        tot = e.sum(0)
        share = float(e[vi].sum() / (tot.sum() + 1e-12))
        sec = int(sr)
        k = tot.shape[0] // sec
        es = e[:, : k * sec].reshape(e.shape[0], k, sec).sum(-1)
        voc_secs = float(((es[vi] / (es.sum(0) + 1e-12)) >= 0.3).float().mean()) if k else 0.0
        rows.append({"music_id": m, "vocal_share": share, "vocal_seconds": voc_secs})
        if (n + 1) % 200 == 0:
            log.info("demucs %d/%d", n + 1, len(music_ids))
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    store = FeatureStore(cfg)
    ids = list(dict.fromkeys(manifest.read_pairs(cfg).music_id))
    df = clap_zero_shot(cfg, store, ids).merge(demucs_vocals(cfg, ids), on="music_id")
    top = df[["clap_singing", "clap_speech", "clap_instrumental"]].idxmax(1).str.replace("clap_", "")
    df["clap_top"] = top
    df["label"] = np.where((top == "singing") & (df.vocal_seconds >= 0.3), "song",
                           np.where((top == "instrumental") & (df.vocal_seconds < 0.3), "background", "uncertain"))
    df.to_csv(manifest.manifests_dir(cfg) / "song_labels.csv", index=False)
    log.info("labels: %s", df.label.value_counts().to_dict())
    log.info("CLAP top class: %s", df.clap_top.value_counts().to_dict())
    log.info("vocal seconds >= 0.3: %d / %d", int((df.vocal_seconds >= 0.3).sum()), len(df))


if __name__ == "__main__":
    main()
