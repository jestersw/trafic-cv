from __future__ import annotations

import numpy as np
import pytest

from traffic_cv.tracking import SORT, ByteTrack, build_tracker
from traffic_cv.tracking.base import iou_matrix


def det(x, y, w=60, h=40, score=0.9, cls=0):
    return [x, y, x + w, y + h, score, cls]


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


@pytest.mark.parametrize("tracker", [SORT(min_hits=1), ByteTrack()])
def test_two_objects_keep_ids(tracker):
    ids_a, ids_b = set(), set()
    for f in range(30):
        dets = np.array([det(50 + 8 * f, 100), det(600 - 8 * f, 300, cls=1)])
        out = tracker.update(dets)
        for r in out:
            (ids_a if r[6] == 0 else ids_b).add(int(r[4]))
    assert len(ids_a) == 1 and len(ids_b) == 1
    assert ids_a != ids_b


def test_bytetrack_survives_occlusion_sort_does_not():
    def run(tracker):
        ids = set()
        for f in range(40):
            missing = 15 <= f < 20  # 5-frame full occlusion
            dets = np.empty((0, 6)) if missing else np.array([det(50 + 6 * f, 200)])
            for r in tracker.update(dets):
                ids.add(int(r[4]))
        return ids

    assert len(run(ByteTrack(track_buffer=30))) == 1
    assert len(run(SORT(max_age=1, min_hits=1))) == 2


def test_bytetrack_uses_low_score_boxes():
    bt = ByteTrack()
    ids = set()
    for f in range(20):
        score = 0.2 if 8 <= f < 12 else 0.9  # detector confidence drops (blur/occlusion)
        out = bt.update(np.array([det(50 + 6 * f, 200, score=score)]))
        assert len(out) == 1 or f == 0
        ids |= {int(r[4]) for r in out}
    assert ids == {1}


def test_class_majority_vote():
    bt = ByteTrack()
    for f in range(10):
        cls = 2 if f == 5 else 0  # single misclassified frame
        out = bt.update(np.array([det(50 + 5 * f, 200, cls=cls)]))
    assert int(out[0, 6]) == 0


def test_empty_input_and_reset():
    for name in ("sort", "bytetrack"):
        t = build_tracker(name)
        assert t.update(np.empty((0, 6))).shape == (0, 7)
        t.update(np.array([det(0, 0)]))
        t.update(np.array([det(2, 0)]))
        t.reset()
        out = t.update(np.array([det(0, 0)]))
        assert all(int(r[4]) == 1 for r in out)


def test_unknown_tracker():
    with pytest.raises(ValueError):
        build_tracker("deepsort")
