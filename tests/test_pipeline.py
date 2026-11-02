from __future__ import annotations

import csv

import cv2
import numpy as np

from traffic_cv.events import LineCounter
from traffic_cv.pipeline import run_video
from traffic_cv.tracking import build_tracker


def _make_video(path, n=40, w=320, h=240):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


class FakeDetector:
    """Two cars moving down the frame, one of them at a lower confidence."""

    def __init__(self):
        self.f = 0

    def __call__(self, frame):
        self.f += 1
        y = 20 + 5 * self.f
        return np.array([[40, y, 80, y + 30, 0.9, 0], [200, y, 250, y + 40, 0.6, 1]])


def test_run_video_end_to_end(tmp_path):
    video = tmp_path / "in.mp4"
    _make_video(video)
    names = {0: "car", 1: "bus"}
    line = LineCounter((0, 150), (320, 150), names)

    summary = run_video(
        video,
        FakeDetector(),
        build_tracker("bytetrack"),
        names,
        [line],
        out_video=tmp_path / "out.mp4",
        events_path=tmp_path / "events.csv",
        tracks_path=tmp_path / "tracks.txt",
    )

    assert summary["perf"]["frames"] == 40
    assert summary["totals"]["line"] == {"in": 2, "out": 0}
    assert summary["counts"]["line"] == {"car": {"in": 1, "out": 0}, "bus": {"in": 1, "out": 0}}

    events = list(csv.DictReader(open(tmp_path / "events.csv")))
    assert {e["class_name"] for e in events} == {"car", "bus"}

    ids = {line.split(",")[1] for line in (tmp_path / "tracks.txt").read_text().splitlines()}
    assert len(ids) == 2

    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
