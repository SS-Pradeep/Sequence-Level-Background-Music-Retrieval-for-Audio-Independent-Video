"""tools/pair_trimmer.py: frame-snapped cuts that keep video and music in sync, manifest rewrite, HTTP serving."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.pair_trimmer import Store, build_pairs, clamp_cut, clamp_ranges, make_handler, sample_range, snap_to_frame

FIELDS = ["pair_id", "video_path", "music_path", "music_source", "music_id", "video_duration_s", "music_duration_s",
          "fps", "meta_film", "meta_movie_start_s", "meta_movie_end_s", "meta_source"]


def test_snap_and_clamp():
    assert snap_to_frame(1.013, 25) == pytest.approx(1.0)
    assert snap_to_frame(1.03, 25) == pytest.approx(1.04)
    s, e = clamp_cut(-1.0, 99.0, 10.01, 25)              # clipped to [0, last frame boundary inside the clip]
    assert (s, e) == (0.0, pytest.approx(10.0))
    s, e = clamp_cut(2.0, 5.0, 10.0, 30000 / 1001)        # 29.97 fps: snapped onto that frame grid
    assert round(s * 30000 / 1001, 6) == int(round(s * 30000 / 1001)) and round(e * 30000 / 1001, 6) == int(round(e * 30000 / 1001))
    with pytest.raises(ValueError):
        clamp_cut(3.0, 3.2, 10.0, 25)


def test_clamp_ranges_sorts_and_rejects_overlap():
    assert clamp_ranges([[53, 59], [0, 30], [35, 45]], 60.0, 25) == [(0.0, 30.0), (35.0, 45.0), (53.0, 59.0)]
    assert clamp_ranges([[0, 30], [30, 40]], 60.0, 25) == [(0.0, 30.0), (30.0, 40.0)]   # touching is fine
    with pytest.raises(ValueError):
        clamp_ranges([[0, 30], [29, 40]], 60.0, 25)
    with pytest.raises(ValueError):
        clamp_ranges([], 60.0, 25)


def test_sample_range_clips_to_short_music():
    assert sample_range(1.0, 2.0, 48000, 10**6) == (48000, 96000)
    assert sample_range(1.0, 30.0, 48000, 100000) == (48000, 100000)


def test_build_pairs_drops_excluded_and_repoints_trimmed():
    rows = [{"pair_id": "a", "video_path": "video/a.mp4", "music_path": "audio/a.wav", "video_duration_s": "10",
             "music_duration_s": "10", "meta_movie_start_s": "100", "meta_movie_end_s": "110"},
            {"pair_id": "b", "video_path": "video/b.mp4", "music_path": "audio/b.wav", "video_duration_s": "8",
             "music_duration_s": "8", "meta_movie_start_s": "", "meta_movie_end_s": ""},
            {"pair_id": "c", "video_path": "video/c.mp4", "music_path": "audio/c.wav", "video_duration_s": "5",
             "music_duration_s": "5", "meta_movie_start_s": "", "meta_movie_end_s": ""}]
    edits = {"a": {"action": "trim", "start_s": 2.0, "end_s": 6.0}, "b": {"action": "exclude"}}   # older form
    out = build_pairs(rows, edits, {"a": [{"video_duration_s": 4.0, "music_duration_s": 4.0}]})
    assert [r["pair_id"] for r in out] == ["a", "c"]
    a, c = out
    assert a["video_path"] == "trimmed/video/a.mp4" and a["music_path"] == "trimmed/audio/a.wav"
    assert float(a["video_duration_s"]) == 4.0 and a["trim_start_s"] == "2.000000" and a["trim_source_id"] == "a"
    assert float(a["meta_movie_start_s"]) == 102.0 and float(a["meta_movie_end_s"]) == 106.0
    assert c["video_path"] == "video/c.mp4" and c["trim_start_s"] == ""


def test_build_pairs_splits_multi_part_trims_into_pairs():
    rows = [{"pair_id": "a", "video_path": "video/a.mp4", "music_path": "audio/a.wav", "music_id": "a",
             "track_id": "trk_1", "video_duration_s": "60", "music_duration_s": "60",
             "meta_movie_start_s": "100", "meta_movie_end_s": "160"}]
    edits = {"a": {"action": "trim", "ranges": [[0.0, 30.0], [35.0, 45.0], [53.0, 59.0]]}}
    info = {"a": [{"video_duration_s": d, "music_duration_s": d} for d in (30.0, 10.0, 6.0)]}
    out = build_pairs(rows, edits, info)
    assert [r["pair_id"] for r in out] == ["a_p1", "a_p2", "a_p3"]
    assert [r["music_id"] for r in out] == ["a_p1", "a_p2", "a_p3"]       # own music files -> own feature cache
    assert {r["track_id"] for r in out} == {"trk_1"} and {r["trim_source_id"] for r in out} == {"a"}
    assert out[1]["video_path"] == "trimmed/video/a_p2.mp4" and out[1]["music_path"] == "trimmed/audio/a_p2.wav"
    assert (float(out[1]["trim_start_s"]), float(out[1]["trim_end_s"])) == (35.0, 45.0)
    assert (float(out[2]["meta_movie_start_s"]), float(out[2]["meta_movie_end_s"])) == (153.0, 159.0)


@pytest.fixture()
def root(synth_pairs, tmp_path):
    """A dataset folder in the dataset_1248 layout, built from the separate-music synthetic pairs."""
    (tmp_path / "video").mkdir()
    (tmp_path / "audio").mkdir()
    rows = []
    for p in [p for p in synth_pairs if not p.embedded]:
        shutil.copy(p.video_path, tmp_path / "video" / Path(p.video_path).name)
        shutil.copy(p.music_path, tmp_path / "audio" / Path(p.music_path).name)
        rows.append({"pair_id": p.pair_id, "video_path": f"video/{Path(p.video_path).name}",
                     "music_path": f"audio/{Path(p.music_path).name}", "music_source": "file", "music_id": p.pair_id,
                     "video_duration_s": p.duration_s, "music_duration_s": sf.info(p.music_path).duration,
                     "fps": p.fps, "meta_film": "synth", "meta_movie_start_s": "", "meta_movie_end_s": "",
                     "meta_source": "synth"})
    with open(tmp_path / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def _frames(path: Path, w: int = 32, h: int = 32) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def test_trim_keeps_video_and_music_in_sync(root):
    st = Store(root)
    pid = st.rows[0]["pair_id"]
    fps = float(st.rows[0]["fps"])
    ed = st.trim(pid, [[2.013, 7.49]])                    # off-grid request -> snapped to frames
    assert ed["ranges"] == [[pytest.approx(2.0), pytest.approx(7.48)]]

    y0, sr = sf.read(st.src(pid, "music"), always_2d=True)
    y1, sr1 = sf.read(st.dst(pid, "music"), always_2d=True)
    assert sr1 == sr and y1.shape[1] == y0.shape[1] and sf.info(st.dst(pid, "music")).subtype == sf.info(st.src(pid, "music")).subtype
    a, b = int(round(2.0 * sr)), int(round(7.48 * sr))
    np.testing.assert_array_equal(y1, y0[a:b])              # sample-exact cut at the same instants

    f0, f1 = _frames(st.src(pid, "video")), _frames(st.dst(pid, "video"))
    k = int(round(2.0 * fps))
    assert len(f1) == int(round((7.48 - 2.0) * fps))
    assert np.abs(f1[0] - f0[k]).mean() < 3.0               # first kept frame is source frame k
    assert np.abs(f1[-1] - f0[k + len(f1) - 1]).mean() < 3.0

    rows = list(csv.DictReader(open(root / "trimmed" / "pairs.csv")))
    r = next(r for r in rows if r["pair_id"] == pid)
    assert r["video_path"] == f"trimmed/video/{Path(st.rows[0]['video_path']).name}"
    assert float(r["video_duration_s"]) == pytest.approx(5.48, abs=1 / fps)
    assert float(r["music_duration_s"]) == pytest.approx(5.48, abs=1e-4)
    assert len(rows) == len(st.rows)

    again = Store(root)                                     # state survives a restart
    assert again.edits[pid]["action"] == "trim"


def test_multi_part_trim_writes_one_synced_pair_per_part(root):
    st = Store(root)
    pid = st.rows[0]["pair_id"]
    fps = float(st.rows[0]["fps"])
    st.trim(pid, [[1.0, 5.0]])
    assert st.dst(pid, "video").exists()
    ed = st.trim(pid, [[6.0, 8.0], [0.0, 3.0]])           # re-cut into two parts, given out of order
    assert ed["ranges"] == [[0.0, 3.0], [6.0, 8.0]]
    assert not st.dst(pid, "video").exists() and not st.dst(pid, "music").exists()   # the old single cut is gone
    y0, sr = sf.read(st.src(pid, "music"), always_2d=True)
    f0 = _frames(st.src(pid, "video"))
    for k, (s, e) in enumerate(ed["ranges"], 1):
        y1, _ = sf.read(st.dst(pid, "music", k), always_2d=True)
        np.testing.assert_array_equal(y1, y0[int(round(s * sr)):int(round(e * sr))])
        f1 = _frames(st.dst(pid, "video", k))
        assert len(f1) == int(round((e - s) * fps)) and np.abs(f1[0] - f0[int(round(s * fps))]).mean() < 3.0
    rows = {r["pair_id"]: r for r in csv.DictReader(open(root / "trimmed" / "pairs.csv"))}
    assert pid not in rows and {f"{pid}_p1", f"{pid}_p2"} <= set(rows) and len(rows) == len(st.rows) + 1
    assert float(rows[f"{pid}_p2"]["video_duration_s"]) == pytest.approx(2.0, abs=1 / fps)
    st.reset(pid)
    assert not st.dst(pid, "video", 1).exists() and not st.dst(pid, "music", 2).exists()


def test_old_single_cut_edits_still_load(root):
    st = Store(root)
    pid = st.rows[0]["pair_id"]
    st.trim(pid, [[1.0, 4.0]])
    old = {pid: {"action": "trim", "start_s": 1.0, "end_s": 4.0, "edited_at": "x"}}
    (root / "trimmed" / "edits.json").write_text(json.dumps(old))
    again = Store(root)
    assert again.edits[pid]["ranges"] == [[1.0, 4.0]] and again.listing()[0]["ranges"] == [[1.0, 4.0]]
    again.reset(pid)
    assert not again.dst(pid, "video").exists()


def test_exclude_then_reset(root):
    st = Store(root)
    pid = st.rows[0]["pair_id"]
    st.trim(pid, [[1.0, 5.0]])
    st.exclude(pid)
    assert not st.dst(pid, "video").exists() and not st.dst(pid, "music").exists()
    rows = list(csv.DictReader(open(root / "trimmed" / "pairs.csv")))
    assert pid not in {r["pair_id"] for r in rows} and len(rows) == len(st.rows) - 1
    st.reset(pid)
    rows = list(csv.DictReader(open(root / "trimmed" / "pairs.csv")))
    r = next(r for r in rows if r["pair_id"] == pid)
    assert r["video_path"] == st.rows[0]["video_path"] and r["trim_start_s"] == ""
    assert st.src(pid, "video").exists()                    # originals untouched throughout


def test_http_api_and_range(root):
    st = Store(root)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        pairs = json.loads(urllib.request.urlopen(base + "/api/pairs").read())
        pid = pairs[0]["pair_id"]
        assert pairs[0]["status"] == "untouched"
        req = urllib.request.Request(f"{base}/media/original/video/{pid}", headers={"Range": "bytes=10-109"})
        resp = urllib.request.urlopen(req)
        assert resp.status == 206 and resp.headers["Content-Range"].endswith(f"/{st.src(pid, 'video').stat().st_size}")
        assert resp.read() == st.src(pid, "video").read_bytes()[10:110]
        body = json.dumps({"pair_id": pid, "start_s": 1.0, "end_s": 4.0}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/api/trim", data=body, method="POST")).read())
        assert r["action"] == "trim"
        assert urllib.request.urlopen(f"{base}/media/trimmed/music/{pid}").status == 200
        body = json.dumps({"pair_id": pid, "ranges": [[0.0, 2.0], [3.0, 5.0]]}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/api/trim", data=body, method="POST")).read())
        assert r["ranges"] == [[0.0, 2.0], [3.0, 5.0]]
        assert urllib.request.urlopen(f"{base}/media/trimmed/video/{pid}?part=2").status == 200
        body = json.dumps({"pair_id": pid, "ranges": [[0.0, 2.0], [1.0, 5.0]]}).encode()
        with pytest.raises(urllib.error.HTTPError) as err:      # overlapping parts are refused
            urllib.request.urlopen(urllib.request.Request(base + "/api/trim", data=body, method="POST"))
        assert err.value.code == 400
        assert len(json.loads(urllib.request.urlopen(f"{base}/api/peaks/{pid}").read())) > 100
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/media/original/video/..%2Fpairs.csv")
    finally:
        srv.shutdown()


def test_apply_cuts_csv(root, tmp_path):
    from tools.pair_trimmer import apply_cuts
    st = Store(root)
    a, b = st.rows[0]["pair_id"], st.rows[1]["pair_id"]
    cuts = tmp_path / "cuts.csv"
    cuts.write_text(f"pair_id,start_s,end_s\n{a},1.6,6.0\n{b},exclude,\nnot_a_pair,1,2\n{a},3.0,3.1\n{a},0.0,1.0\n")
    failures = apply_cuts(st, cuts)
    assert [p for p, _ in failures] == ["not_a_pair", a]     # bad rows reported, good ones applied
    assert st.edits[a]["action"] == "trim" and st.edits[a]["ranges"] == [[0.0, 1.0], [pytest.approx(1.6), 6.0]]
    assert st.edits[b]["action"] == "exclude"
