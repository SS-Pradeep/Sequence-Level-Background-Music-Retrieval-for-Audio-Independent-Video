"""bgm.tagging.parse / jobs / prompts: calibration, group rules, A/B agreement, resumable jobs, JSON pilot parsing, leakage."""
from __future__ import annotations

import numpy as np

from bgm.tagging import jobs, prompts
from bgm.tagging.models import FakeTagger, first_token_ids
from bgm.tagging.parse import agree, calibrated_conf, clean, extract_json, select
from bgm.tagging.pilot_json import summarize
from bgm.tagging.vectors import agreed_vectors, to_vector
from bgm.tagging.vocab import GROUPS, index, load_vocab

V = load_vocab()


def test_calibration_removes_a_shared_prior():
    rng = np.random.default_rng(0)
    lp = rng.normal(0, 1, (500, 4))
    lp[:, 0] += 3.0                                                           # option 0 wins almost everywhere raw
    raw, cal = calibrated_conf(lp, False), calibrated_conf(lp, True)
    assert (raw.argmax(1) == 0).mean() > 0.8
    assert 0.15 < (cal.argmax(1) == 0).mean() < 0.35                          # back near 1/4
    assert np.allclose(cal.sum(1), 1.0) and np.allclose(raw.sum(1), 1.0)


def test_select_single_and_multi():
    c = np.array([0.1, 0.5, 0.3, 0.1])
    assert select(c, ("a", "b", "c", "d"), "energy", 0.2) == {"b": 0.5}
    assert select(c, ("a", "b", "c", "d"), "mood", 0.2) == {"b": 0.5, "c": 0.3}
    assert select(np.full(6, 1 / 6), tuple("abcdef"), "genre", 0.2) == {}      # flat: nothing passes tau
    assert len(select(np.array([0.3, 0.25, 0.25, 0.2]), tuple("abcd"), "genre", 0.2)) == 3


def test_agree_keeps_shared_tags_with_mean_confidence():
    a = {"mood": {"sad": 0.8, "calm": 0.6}, "energy": {"low": 0.9}, "tempo": {"slow": 0.5}}
    b = {"mood": {"sad": 0.4}, "energy": {"medium": 0.9}, "tempo": {"slow": 0.7}}
    g = agree(a, b)
    assert abs(g["mood"]["sad"] - 0.6) < 1e-9 and len(g["mood"]) == 1
    assert g["energy"] == {} and abs(g["tempo"]["slow"] - 0.6) < 1e-9
    v = to_vector(g, V)
    assert v.shape == (len(index(V)),) and np.count_nonzero(v) == 2


def test_jobs_resume_and_table(tmp_path):
    items = [((f"m{i}", 0.0), lambda i=i: np.full(160, i, np.float32)) for i in range(6)]
    out = tmp_path / "raw.jsonl"
    fake = FakeTagger()
    r1 = jobs.run("music", items[:4], fake, out, batch=3)
    assert r1["n_new"] == 4 and fake.calls == 4 * 2 * len(GROUPS)
    r2 = jobs.run("music", items, fake, out, batch=3)                          # only the 2 new items
    assert r2["n_new"] == 2 and fake.calls == 6 * 2 * len(GROUPS)
    recs = jobs.read_raw(out)
    assert len(recs) == 12 and set(recs[0]["logp"]) == set(GROUPS) and len(recs[0]["logp"]["mood"]) == len(V["mood"])
    df, st = jobs.build_table(recs, "music", tau=0.2)
    assert set(df.columns) == {"music_id", "offset_s", "prompt_id", "group", "tag", "conf"}
    assert (df[df.group == "energy"].groupby(["music_id", "prompt_id"]).size() == 1).all()
    assert (df.groupby(["music_id", "prompt_id", "group"]).size() <= 3).all()
    assert (df[~df.group.isin(["energy", "tempo"])].conf >= 0.2).all()
    vecs, ag = agreed_vectors(df, "music_id", "offset_s", [k for k, _ in items] + [("missing", 0.0)])
    assert ag["n_with_both"] == 6 and not vecs[("missing", 0.0)].any()


def test_questions_list_every_option_and_reverse_for_b():
    for side in ("music", "scene"):
        for g in GROUPS:
            qa, qb = prompts.question(side, "A", g, V[g]), prompts.question(side, "B", g, V[g])
            labels = [prompts.label(t) for t in V[g]]
            assert all(lab in qa and lab in qb for lab in labels)
            assert qa.index(labels[0]) < qa.index(labels[-1]) and qb.index(labels[0]) > qb.index(labels[-1])


def test_scene_questions_carry_no_names():
    for p in prompts.PROMPT_IDS:
        for g in GROUPS:
            txt = prompts.question("scene", p, g, V[g])
            assert ".mp4" not in txt and "combined_" not in txt and "bgm_" not in txt and "sunnxt" not in txt


class _Tok:
    """Whitespace tokenizer stand-in: first token = the word up to 3 letters, so 'S' style tokens can be tested."""

    def __init__(self, short=()):
        self.short, self.vocab = set(short), {}

    def encode(self, text, add_special_tokens=False):
        w = text.strip()
        piece = (" " if text.startswith(" ") else "") + (w[:1] if w.lower() in self.short else w[:4])
        return [self.vocab.setdefault(piece, len(self.vocab))]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return inv[ids[0]]


def test_first_token_ids_skip_short_tokens():
    tok = _Tok(short={"sitar"})
    ids = first_token_ids(tok, "sitar")
    assert len(ids) == 1 and tok.decode(ids) in ("S", "s", " S", " s")       # only short spellings: longest kept
    tok2 = _Tok()
    assert len(first_token_ids(tok2, "tabla")) == 4


def test_pilot_summary_counts_failures_and_copying():
    recs = [{"key": ["m", 0.0], "prompt_id": "A", "attempt": 0, "ok": True,
             "text": '{"mood": ["sad"], "instruments": ["tabla", "sitar", "veena", "flute"], "energy": "low"}'},
            {"key": ["m", 0.0], "prompt_id": "B", "attempt": 0, "ok": False, "text": "Sure"},
            {"key": ["m", 0.0], "prompt_id": "B", "attempt": 1, "ok": True, "text": '{"mood": {"tag": "sad"}}'}]
    s = summarize(recs)
    assert s["json_fail_first"] == 0.5 and s["json_fail_after_retry"] == 0
    assert s["windows_with_any_agreed_tag"] == 0 and s["instruments_in_vocab_order"] == 1 and s["groups_with_more_than_3"] == 1
    assert s["oov_top"] == {"mood:tag": 1}


def test_extract_json_and_clean():
    assert extract_json('```json\n{"mood": {"sad": 0.8}}\n```') == {"mood": {"sad": 0.8}}
    assert extract_json("no json") is None and extract_json('{"a": 1') is None
    t, oov = clean({"mood": {"sad": 0.9, "nostalgic": 0.8}, "Instruments": ["Tabla", "vocals male"], "energy": {"low": 0.2, "high": 0.7}}, V)
    assert t["mood"] == {"sad": 0.9} and t["instruments"] == {"tabla": 1.0, "vocals_male": 1.0} and t["energy"] == {"high": 0.7}
    assert oov == {"mood:nostalgic": 1}
