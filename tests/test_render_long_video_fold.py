"""The long-video shortlist must not use other folds' heads on a dataset clip (test-fold leak, 2026-11-17)."""
from bgm.render.run import long_video_fold


def test_dataset_clip_uses_its_fold_only():
    ids = ["sunnxt_85_85_bgm_0002", "combined_full_bgm_dataset__Geet gaya Patharon Ne__Geet gaya Patharon Ne_bgm_0001"]
    assert long_video_fold("data/interim/video/sunnxt_85_85_bgm_0002.mp4", ids, 4) == 4
    assert long_video_fold("data/interim/video/combined_full_bgm_dataset__Geet gaya Patharon Ne__Geet gaya Patharon Ne_bgm_0001.mp4", ids, 0) == 0


def test_outside_video_uses_every_fold():
    assert long_video_fold("godfather.mp4", ["sunnxt_85_85_bgm_0002"], 0) is None
