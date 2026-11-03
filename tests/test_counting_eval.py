from __future__ import annotations

import numpy as np

from traffic_cv.events.evaluate import count_line, dominant_flow, errors, gt_frames, suggest_line
from traffic_cv.tracking.evaluate import MotSequence

NAMES = ["car", "bus", "van", "others"]


def vertical_flow_sequence(n_frames: int = 40) -> MotSequence:
    seq = MotSequence("toy", im_dir=None, length=n_frames)
    for f in range(1, n_frames + 1):
        rows = []
        for tid, x, cls, step in ((1, 200, 0, 12), (2, 600, 1, -12)):
            y = (50 + step * f) if step > 0 else (530 + step * f)
            rows.append([tid, x, y - 30, 40, 30, 1, cls, 1.0])
        seq.gt[f] = np.array(rows, dtype=float)
    return seq


def test_count_line_on_gt_tracks():
    seq = vertical_flow_sequence()
    c = count_line(gt_frames(seq), [0, 300, 960, 300], NAMES, seq.ignored)
    assert c["car"] == {"in": 1, "out": 0}
    assert c["bus"] == {"in": 0, "out": 1}


def test_ignored_region_drops_tracks():
    seq = vertical_flow_sequence()
    seq.ignored = np.array([[0, 0, 400, 540]], dtype=float)
    c = count_line(gt_frames(seq), [0, 300, 960, 300], NAMES, seq.ignored)
    assert c["car"] == {"in": 0, "out": 0}
    assert c["bus"] == {"in": 0, "out": 1}


def test_suggest_line_is_across_the_flow():
    seq = vertical_flow_sequence()
    assert dominant_flow(seq) == "horizontal"
    ln = suggest_line(seq, 960, 540, NAMES)
    assert ln[1] == ln[3] and 0.5 * 540 <= ln[1] <= 0.85 * 540


def test_errors():
    zero = {c: {"in": 0, "out": 0} for c in NAMES}
    per_seq = {
        "a": {
            "gt": {**zero, "car": {"in": 10, "out": 10}},
            "pred": {**zero, "car": {"in": 9, "out": 10}, "van": {"in": 1, "out": 0}},
        },
        "b": {
            "gt": {**zero, "bus": {"in": 0, "out": 20}},
            "pred": {**zero, "bus": {"in": 0, "out": 18}},
        },
    }
    e = errors(per_seq, NAMES)
    assert (e["gt"], e["pred"]) == (40, 38)
    assert e["total_error"] == 0.05
    assert e["wape_direction"] == 0.05
    assert e["wape_class_direction"] == 0.1
    assert e["seq_within_5pct"] == 0.5
