from __future__ import annotations

import csv
import json

import numpy as np

from traffic_cv.events import DwellZone, EventWriter, LineCounter
from traffic_cv.events.geometry import point_in_polygon, segments_intersect

NAMES = ["car", "bus", "van", "others"]


def trk(tid, cx, bottom, cls=0, w=40, h=30):
    return [cx - w / 2, bottom - h, cx + w / 2, bottom, tid, 0.9, cls]


def test_segments_intersect():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (5, 0), (5, 10))


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)


def test_line_counts_direction_and_class():
    lc = LineCounter((0, 300), (960, 300), NAMES)
    for f in range(10):
        tracks = np.array(
            [
                trk(1, 100, 250 + 12 * f),
                trk(2, 500, 360 - 12 * f, cls=1),
                trk(3, 800, 100 + f),
            ]
        )
        lc.update(tracks, f, 25.0)
    assert lc.counts["car"] == {"in": 1, "out": 0}
    assert lc.counts["bus"] == {"in": 0, "out": 1}
    assert (lc.total_in, lc.total_out) == (1, 1)


def test_line_ignores_crossing_beyond_segment():
    lc = LineCounter((0, 300), (400, 300), NAMES)
    for f in range(10):
        lc.update(np.array([trk(1, 700, 250 + 12 * f)]), f, 25.0)
    assert lc.total_in == lc.total_out == 0


def test_line_count_once_suppresses_jitter():
    ys = [290, 305, 295, 306, 294, 310, 320]
    once = LineCounter((0, 300), (960, 300), NAMES)
    every = LineCounter((0, 300), (960, 300), NAMES, count_once=False)
    for f, y in enumerate(ys):
        t = np.array([trk(7, 100, y)])
        once.update(t, f, 25.0)
        every.update(t, f, 25.0)
    assert once.total_in + once.total_out == 1
    assert every.total_in + every.total_out > 1


def test_dwell_zone_events():
    zone = DwellZone(
        [(0, 0), (200, 0), (200, 200), (0, 200)],
        max_seconds=1.0,
        class_names=NAMES,
        exit_grace_frames=2,
    )
    kinds = []
    fps = 10.0
    for f in range(30):
        tracks = np.array([trk(1, 100, 150)]) if f < 20 else np.empty((0, 7))
        kinds += [e.event for e in zone.update(tracks, f, fps)]
    assert kinds == ["zone_enter", "dwell_exceeded", "zone_exit"]


def test_event_writer_csv_and_jsonl(tmp_path):
    lc = LineCounter((0, 300), (960, 300), NAMES)
    events = []
    for f in range(10):
        events += lc.update(np.array([trk(1, 100, 250 + 12 * f)]), f, 25.0)
    assert len(events) == 1

    with EventWriter(tmp_path / "e.csv") as w:
        w.write(events)
    rows = list(csv.DictReader(open(tmp_path / "e.csv")))
    assert rows[0]["event"] == "line_cross" and rows[0]["detail"] == "in"
    assert rows[0]["class_name"] == "car"

    with EventWriter(tmp_path / "e.jsonl") as w:
        w.write(events)
    d = json.loads((tmp_path / "e.jsonl").read_text().splitlines()[0])
    assert d["track_id"] == 1 and d["frame"] == events[0].frame
