from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from traffic_cv.events import DwellZone, EventWriter, LineCounter
from traffic_cv.tracking import Tracker
from traffic_cv.viz import draw


@dataclass
class RunStats:
    frames: int = 0
    seconds: float = 0.0
    det_ms: list[float] = field(default_factory=list)
    trk_ms: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0

    def summary(self) -> dict:
        def pct(xs, q):
            return round(float(np.percentile(xs, q)), 2) if xs else None

        return {
            "frames": self.frames,
            "fps_end_to_end": round(self.fps, 2),
            "detector_ms_p50": pct(self.det_ms, 50),
            "detector_ms_p95": pct(self.det_ms, 95),
            "tracker_ms_p50": pct(self.trk_ms, 50),
        }


def run_video(
    video: str | Path,
    detector: Callable[[np.ndarray], np.ndarray],
    tracker: Tracker,
    names: dict[int, str],
    lines: list[LineCounter],
    zones: list[DwellZone] | None = None,
    out_video: str | Path | None = None,
    events_path: str | Path | None = None,
    tracks_path: str | Path | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> dict:
    zones = zones or []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if max_frames:
        total = min(total or max_frames, max_frames)

    writer = None
    if out_video:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), vfps, (w, h))
    ev_writer = EventWriter(events_path) if events_path else None
    trk_file = None
    if tracks_path:
        Path(tracks_path).parent.mkdir(parents=True, exist_ok=True)
        trk_file = open(tracks_path, "w")

    stats = RunStats()
    recent: deque[float] = deque(maxlen=30)
    t_start = time.perf_counter()
    frame_idx = 0
    try:
        with tqdm(total=total, unit="f") as bar:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames and frame_idx >= max_frames):
                    break
                frame_idx += 1
                t0 = time.perf_counter()
                dets = detector(frame)
                t1 = time.perf_counter()
                tracks = tracker.update(dets)
                t2 = time.perf_counter()
                stats.det_ms.append((t1 - t0) * 1e3)
                stats.trk_ms.append((t2 - t1) * 1e3)

                events = []
                for c in (*lines, *zones):
                    events += c.update(tracks, frame_idx, vfps)
                if ev_writer:
                    ev_writer.write(events)
                if trk_file:
                    for r in tracks:
                        trk_file.write(
                            f"{frame_idx},{int(r[4])},{r[0]:.2f},{r[1]:.2f},{r[2] - r[0]:.2f},"
                            f"{r[3] - r[1]:.2f},{r[5]:.3f},{int(r[6])},-1\n"
                        )

                recent.append(time.perf_counter() - t0)
                if writer or show:
                    draw(
                        frame,
                        tracks,
                        names,
                        lines,
                        zones,
                        fps=len(recent) / sum(recent),
                        frame_idx=frame_idx,
                        video_fps=vfps,
                    )
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("traffic-cv", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                bar.update()
    finally:
        stats.frames = frame_idx
        stats.seconds = time.perf_counter() - t_start
        cap.release()
        if writer:
            writer.release()
        if ev_writer:
            ev_writer.close()
        if trk_file:
            trk_file.close()
        if show:
            cv2.destroyAllWindows()

    return {
        "video": str(video),
        "counts": {ln.name: {k: dict(v) for k, v in ln.counts.items()} for ln in lines},
        "totals": {ln.name: {"in": ln.total_in, "out": ln.total_out} for ln in lines},
        "perf": stats.summary(),
    }


def save_summary(summary: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=2))
