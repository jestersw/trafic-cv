"""ByteTrack (Zhang et al., 2022): associate *every* detection box, not only confident ones.

Two-stage association per frame:
  1. high-score detections  <-> all tracks (tracked + recently lost)
  2. low-score detections   <-> tracks left unmatched in stage 1 that were tracked last frame
     (recovers occluded / blurred vehicles whose score dropped, instead of killing the track)
  3. unconfirmed (1-frame-old) tracks <-> remaining high-score detections
New tracks start only from unmatched high-score detections; lost tracks are kept for
``track_buffer`` frames so an object can re-appear with the same ID.

Motion model is the SORT Kalman filter (xysr) rather than the original xyah one, to share
code with SORT and keep the comparison about the association strategy.
"""

from __future__ import annotations

import numpy as np

from traffic_cv.tracking.base import Track, associate, stack


class ByteTrack:
    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_iou: float = 0.2,
        low_match_iou: float = 0.5,
        unconfirmed_match_iou: float = 0.3,
        track_buffer: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.unconfirmed_match_iou = unconfirmed_match_iou
        self.track_buffer = track_buffer
        self.reset()

    def reset(self) -> None:
        self.tracked: list[Track] = []  # seen in the previous frame
        self.lost: list[Track] = []  # missed recently, kept for re-identification
        self.frame_count = 0
        self._next_id = 1

    def _new_track(self, det: np.ndarray) -> Track:
        t = Track(self._next_id, det)
        self._next_id += 1
        # first frame of a video: activate immediately (nothing to confirm against)
        t.activated = self.frame_count == 1
        return t

    def update(self, dets: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        high = dets[dets[:, 4] >= self.track_thresh]
        low = dets[(dets[:, 4] >= self.low_thresh) & (dets[:, 4] < self.track_thresh)]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.predict()

        # --- stage 1: high-score dets vs confirmed + lost tracks
        pool_boxes = np.array([t.kf.box for t in pool]).reshape(-1, 4)
        m1, um_t1, um_d1 = associate(pool_boxes, high, self.match_iou)
        next_tracked: list[Track] = []
        for ti, di in m1:
            pool[ti].update(high[di])
            next_tracked.append(pool[ti])

        # --- stage 2: low-score dets vs still-unmatched tracks that were tracked last frame
        remain = [pool[i] for i in um_t1 if pool[i] in confirmed]
        remain_boxes = np.array([t.kf.box for t in remain]).reshape(-1, 4)
        m2, um_t2, _ = associate(remain_boxes, low, self.low_match_iou)
        for ti, di in m2:
            remain[ti].update(low[di])
            next_tracked.append(remain[ti])

        still_lost = [remain[i] for i in um_t2]
        still_lost += [pool[i] for i in um_t1 if pool[i] not in confirmed]  # were lost already

        # --- stage 3: unconfirmed tracks vs leftover high dets
        left_high = high[um_d1]
        unc_boxes = np.array([t.kf.box for t in unconfirmed]).reshape(-1, 4)
        m3, _, um_d3 = associate(unc_boxes, left_high, self.unconfirmed_match_iou)
        for ti, di in m3:
            unconfirmed[ti].update(left_high[di])
            unconfirmed[ti].activated = True
            next_tracked.append(unconfirmed[ti])
        # unmatched unconfirmed tracks are dropped (likely false positives)

        # --- new tracks
        for di in um_d3:
            if left_high[di, 4] >= self.new_track_thresh:
                next_tracked.append(self._new_track(left_high[di]))

        for t in next_tracked:
            if t.hits >= 2:
                t.activated = True
        self.tracked = next_tracked
        self.lost = [t for t in still_lost if t.time_since_update <= self.track_buffer]

        return stack([t.as_row() for t in self.tracked if t.activated])
