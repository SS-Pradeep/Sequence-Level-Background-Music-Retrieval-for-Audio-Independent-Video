from pathlib import Path

import numpy as np

from bgm.render.scene import (EMOTIONS, FakeDescriber, brief_text, draft_brief, parse_reply, read_brief, sample_times,
                              span_weights, write_brief)
from tests.fixtures.synth import make_pairs


def test_sample_times_inside_span():
    t = sample_times(10.0, 18.0)
    assert len(t) == 4 and t.min() > 10.0 and t.max() < 18.0
    assert len(sample_times(0.0, 1.0)) == 1


def test_parse_reply_tolerant():
    d = parse_reply('```json\n{"scene": "a fight", "emotion": "Action", "music": "fast drums"}\n```')
    assert d == {"scene": "a fight", "emotion": "action", "music": "fast drums", "needs_music": True}
    assert parse_reply("no json here")["emotion"] in EMOTIONS


def test_span_weights_overlap_and_nearest():
    brief = [{"start_s": 0.0, "end_s": 10.0}, {"start_s": 10.0, "end_s": 20.0}]
    np.testing.assert_allclose(span_weights(brief, 5.0, 15.0), [0.5, 0.5])
    np.testing.assert_allclose(span_weights(brief, 12.0, 14.0), [0.0, 1.0])
    np.testing.assert_allclose(span_weights(brief, 30.0, 31.0), [0.0, 1.0])


def test_draft_write_read_roundtrip(tmp_path: Path):
    make_pairs(tmp_path / "synth", n=1, seed=0)
    video = next((tmp_path / "synth").rglob("*.mp4"))
    summary, brief = draft_brief(FakeDescriber(), video, [(0.0, 4.0), (4.0, 9.0)])
    assert summary.endswith("over 9 s")
    assert [e["start_s"] for e in brief] == [0.0, 4.0] and all(e["scene"].startswith(("1", "2")) for e in brief)
    write_brief(tmp_path / "x.brief.yaml", brief, str(video), summary)
    back = read_brief(tmp_path / "x.brief.yaml")
    assert back == brief and brief_text(back[0]).startswith(back[0]["emotion"])


def test_replay_penalty_moves_off_a_repeated_passage():
    from bgm.render.replay import covered, decode_without_replays
    from bgm.retrieval.stitch_eval import Layer
    assert covered(0.0, 10.0, [(2.0, 4.0), (3.0, 6.0), (8.0, 20.0)]) == 6.0
    assert covered(0.0, 10.0, [(2.0, 4.0), (3.0, 6.0), (8.0, 20.0)], times=2) == 1.0
    # file 0 at offset 0 is best in every segment; file 1 is slightly worse
    L = [Layer(np.array([0, 1]), np.array([0.0, 0.0]), np.array([5.0, 5.0]), np.zeros(2, int), np.zeros(2), np.array([-1.0, -0.8]), 2)
         for _ in range(3)]
    greedy = lambda layers: [int(np.argmin(x.cost)) for x in layers]
    picks, _, info = decode_without_replays(greedy, L, penalty=1.0)
    assert info["replays_before"] == 2 and picks[:2] == [0, 1]
    picks2, _, info2 = decode_without_replays(greedy, L, penalty=1.0, max_plays=2)
    assert info2["replays_before"] == 1 and picks2[:2] == [0, 0]


def test_needs_music_parsing_and_silent_spans():
    from bgm.render.scene import silent_spans
    assert parse_reply('{"scene": "x", "emotion": "sad", "music": "m", "needs_music": "no"}')["needs_music"] is False
    assert parse_reply('{"scene": "x", "emotion": "sad", "music": "m"}')["needs_music"] is True
    brief = [{"start_s": 0.0, "end_s": 5.0, "needs_music": True, "music": "a"},
             {"start_s": 5.0, "end_s": 7.0, "needs_music": False, "music": "a"},
             {"start_s": 7.0, "end_s": 9.0, "needs_music": True, "music": "none"},       # music: none also counts
             {"start_s": 9.0, "end_s": 12.0, "needs_music": True, "music": "a"},
             {"start_s": 12.0, "end_s": 13.0, "needs_music": False, "music": "a"}]       # 1 s: too short, keeps music
    assert silent_spans(brief, 3.0) == [(5.0, 9.0)]


def test_silence_envelope_fades():
    from bgm.render.render import silence_envelope
    g = silence_envelope(1000, [(4.0, 6.0)], 1.0, sr=100)
    assert g[0] == 1.0 and g[450] == 0.0 and g[999] == 1.0
    assert 0.0 < g[350] < 1.0 and 0.0 < g[650] < 1.0                               # fading out before, in after
