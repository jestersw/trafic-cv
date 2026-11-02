from __future__ import annotations

from collections import Counter
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_cv.tracking.kalman import KalmanBoxFilter

EMPTY_TRACKS = np.empty((0, 7), dtype=np.float64)


class Tracker(Protocol):
    def update(self, dets: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = a[:, None, :4], b[None, :, :4]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def associate(
    track_boxes: np.ndarray, det_boxes: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    n_t, n_d = len(track_boxes), len(det_boxes)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))
    iou = iou_matrix(track_boxes, det_boxes)
    rows, cols = linear_sum_assignment(-iou)
    matches = [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= iou_threshold]
    mt = {r for r, _ in matches}
    md = {c for _, c in matches}
    return (
        matches,
        [i for i in range(n_t) if i not in mt],
        [j for j in range(n_d) if j not in md],
    )


class Track:
    __slots__ = (
        "id",
        "kf",
        "score",
        "class_votes",
        "hits",
        "hit_streak",
        "age",
        "time_since_update",
        "activated",
    )

    def __init__(self, track_id: int, det: np.ndarray) -> None:
        self.id = track_id
        self.kf = KalmanBoxFilter(det[:4])
        self.score = float(det[4])
        self.class_votes: Counter[int] = Counter({int(det[5]): 1})
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.activated = False

    def predict(self) -> np.ndarray:
        box = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return box

    def update(self, det: np.ndarray) -> None:
        self.kf.update(det[:4])
        self.score = float(det[4])
        self.class_votes[int(det[5])] += 1
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    @property
    def class_id(self) -> int:
        return self.class_votes.most_common(1)[0][0]

    def as_row(self) -> list[float]:
        return [*self.kf.box.tolist(), float(self.id), self.score, float(self.class_id)]


def stack(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64) if rows else EMPTY_TRACKS.copy()
