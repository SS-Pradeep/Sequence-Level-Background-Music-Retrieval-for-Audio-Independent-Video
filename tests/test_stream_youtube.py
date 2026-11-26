"""tools/stream_youtube.py: which downloads are picked up, and the ImageBind compaction."""
import json

import numpy as np

from tools.stream_youtube import compact_imagebind, last_status, pending_ids


def _write_jsonl(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs) + "{broken\n")


def _media(src, vid, audio=True):
    (src / "video").mkdir(exist_ok=True)
    (src / "audio").mkdir(exist_ok=True)
    (src / "video" / f"{vid}.mp4").write_bytes(b"x")
    if audio:
        (src / "audio" / f"{vid}.m4a").write_bytes(b"x")


def test_last_status_keeps_last_line(tmp_path):
    _write_jsonl(tmp_path / "m.jsonl", [{"id": "a", "status": "retry"}, {"id": "a", "status": "ok"}])
    assert last_status(tmp_path / "m.jsonl") == {"a": {"id": "a", "status": "ok"}}
    assert last_status(tmp_path / "missing.jsonl") == {}


def test_pending_ids(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for v in ("ok1", "ok2", "done", "bad", "noaudio"):
        _media(src, v, audio=v != "noaudio")
    (src / "audio" / "part.m4a.part").write_bytes(b"x")
    _write_jsonl(src / "manifest.jsonl", [{"id": v, "status": "ok"} for v in ("ok2", "ok1", "done", "bad", "noaudio")]
                 + [{"id": "dead", "status": "dead"}, {"id": "part", "status": "ok"}])
    _write_jsonl(out / "stream_manifest.jsonl", [{"id": "done", "status": "ok"}, {"id": "bad", "status": "failed"}])
    assert pending_ids(src, out, retry_failed=False) == ["ok2", "ok1"]            # download order, finished skipped
    assert pending_ids(src, out, retry_failed=True) == ["ok2", "ok1", "bad"]


def test_compact_imagebind():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 1024)).astype(np.float16)
    d = compact_imagebind({"imagebind_1hz": x, "other": 1})
    assert "imagebind_1hz" not in d and d["other"] == 1
    m = d["imagebind_mean"]
    assert m.shape == (1024,) and m.dtype == np.float16
    np.testing.assert_allclose(np.linalg.norm(m.astype(np.float32)), 1.0, atol=1e-2)
    np.testing.assert_allclose(m.astype(np.float32), x.astype(np.float32).mean(0) / np.linalg.norm(x.astype(np.float32).mean(0)), atol=2e-3)
    assert compact_imagebind({"a": 1}) == {"a": 1}
