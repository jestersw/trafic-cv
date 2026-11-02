"""Traffic events computed from tracker output.

* :class:`LineCounter` — counts crossings of a line segment per class and direction.
* :class:`DwellZone` — fires when an object stays in a polygon longer than N seconds
  ("no stopping" zone).

Both consume tracker rows ``(x1, y1, x2, y2, track_id, score, class_id)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from traffic_cv.events.geometry import Point, anchor, point_in_polygon, segments_intersect, side


@dataclass(frozen=True, slots=True)
class Event:
    frame: int
    time_s: float
    track_id: int
    class_id: int
    class_name: str
    event: str  # "line_cross" | "zone_enter" | "dwell_exceeded" | "zone_exit"
    detail: str  # direction for lines, zone name for zones
    source: str  # name of the line / zone

    def to_dict(self) -> dict:
        return asdict(self)


class LineCounter:
    """Counts objects whose anchor point moves across segment ``p1 -> p2``.

    Direction ``in``  = anchor ends on the positive side, cross(p1->p2, q) > 0. In image
    coordinates (y down) that is the side to the *right* of p1->p2 when walking from p1
    to p2: for a horizontal line drawn left->right, ``in`` means moving down the frame.
    Direction ``out`` = the opposite. Swap p1/p2 to flip the naming.

    A crossing is registered only when the anchor's motion segment (previous position ->
    current position) actually intersects the counting segment, so objects passing
    beyond the ends of the line are not counted. With ``count_once`` each track is
    counted at most once, which suppresses double counts from box jitter on the line.
    """

    def __init__(
        self,
        p1: Point,
        p2: Point,
        class_names: dict[int, str] | list[str],
        name: str = "line",
        anchor_mode: str = "bottom_center",
        count_once: bool = True,
    ) -> None:
        self.p1, self.p2 = (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))
        self.names = dict(enumerate(class_names)) if isinstance(class_names, list) else class_names
        self.name = name
        self.anchor_mode = anchor_mode
        self.count_once = count_once
        self.reset()

    def reset(self) -> None:
        self._last: dict[int, Point] = {}
        self._counted: set[int] = set()
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    @property
    def total_in(self) -> int:
        return sum(c["in"] for c in self.counts.values())

    @property
    def total_out(self) -> int:
        return sum(c["out"] for c in self.counts.values())

    def update(self, tracks: np.ndarray, frame: int, fps: float) -> list[Event]:
        events: list[Event] = []
        for row in np.asarray(tracks).reshape(-1, 7):
            tid, cls = int(row[4]), int(row[6])
            cur = anchor(row, self.anchor_mode)
            prev = self._last.get(tid)
            self._last[tid] = cur
            if prev is None or (self.count_once and tid in self._counted):
                continue
            if not segments_intersect(prev, cur, self.p1, self.p2):
                continue
            s_cur = side(self.p1, self.p2, cur)
            if s_cur == 0:  # landed exactly on the line: wait for the next frame
                self._last[tid] = prev
                continue
            direction = "in" if s_cur > 0 else "out"
            cname = self.names.get(cls, str(cls))
            self.counts[cname][direction] += 1
            self._counted.add(tid)
            events.append(
                Event(frame, frame / fps, tid, cls, cname, "line_cross", direction, self.name)
            )
        return events


class DwellZone:
    """Emits zone_enter / dwell_exceeded (once) / zone_exit for a polygon."""

    def __init__(
        self,
        polygon: list[Point],
        max_seconds: float,
        class_names: dict[int, str] | list[str],
        name: str = "zone",
        anchor_mode: str = "bottom_center",
        exit_grace_frames: int = 5,
    ) -> None:
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.max_seconds = max_seconds
        self.names = dict(enumerate(class_names)) if isinstance(class_names, list) else class_names
        self.name = name
        self.anchor_mode = anchor_mode
        self.exit_grace_frames = exit_grace_frames
        self.reset()

    def reset(self) -> None:
        self._entered: dict[int, int] = {}  # track_id -> entry frame
        self._last_seen: dict[int, int] = {}
        self._class: dict[int, int] = {}
        self._alerted: set[int] = set()

    def _ev(self, frame: int, fps: float, tid: int, kind: str) -> Event:
        cls = self._class[tid]
        return Event(
            frame, frame / fps, tid, cls, self.names.get(cls, str(cls)), kind, self.name, self.name
        )

    def update(self, tracks: np.ndarray, frame: int, fps: float) -> list[Event]:
        events: list[Event] = []
        for row in np.asarray(tracks).reshape(-1, 7):
            tid = int(row[4])
            if not point_in_polygon(anchor(row, self.anchor_mode), self.polygon):
                continue
            self._class[tid] = int(row[6])
            self._last_seen[tid] = frame
            if tid not in self._entered:
                self._entered[tid] = frame
                events.append(self._ev(frame, fps, tid, "zone_enter"))
            if tid not in self._alerted and (frame - self._entered[tid]) / fps > self.max_seconds:
                self._alerted.add(tid)
                events.append(self._ev(frame, fps, tid, "dwell_exceeded"))

        for tid in [t for t, f in self._last_seen.items() if frame - f > self.exit_grace_frames]:
            events.append(self._ev(frame, fps, tid, "zone_exit"))
            for d in (self._entered, self._last_seen, self._class):
                d.pop(tid, None)
            self._alerted.discard(tid)
        return events

    def dwell_seconds(self, tid: int, frame: int, fps: float) -> float:
        return (frame - self._entered[tid]) / fps if tid in self._entered else 0.0
