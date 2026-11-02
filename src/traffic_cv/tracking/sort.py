from __future__ import annotations

import numpy as np

from traffic_cv.tracking.base import Track, associate, stack


class SORT:
    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        min_score: float = 0.25,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, dets: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        dets = dets[dets[:, 4] >= self.min_score]

        predicted = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        ok = np.isfinite(predicted).all(axis=1)
        self.tracks = [t for t, k in zip(self.tracks, ok, strict=True) if k]
        predicted = predicted[ok]

        matches, _, unmatched_d = associate(predicted, dets, self.iou_threshold)
        for ti, di in matches:
            self.tracks[ti].update(dets[di])
        for di in unmatched_d:
            self.tracks.append(Track(self._next_id, dets[di]))
            self._next_id += 1

        out = []
        for t in self.tracks:
            if t.time_since_update < 1 and (
                t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                out.append(t.as_row())
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return stack(out)
