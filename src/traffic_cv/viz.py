from __future__ import annotations

import colorsys

import cv2
import numpy as np

from traffic_cv.events.counters import DwellZone, LineCounter


def id_color(track_id: int) -> tuple[int, int, int]:
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img, text: str, org: tuple[int, int], color, scale: float = 0.5) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(y, h + base)
    cv2.rectangle(img, (x, y - h - base), (x + w + 2, y), color, -1)
    cv2.putText(
        img, text, (x + 1, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA
    )


def draw(
    frame: np.ndarray,
    tracks: np.ndarray,
    names: dict[int, str],
    lines: list[LineCounter] = (),
    zones: list[DwellZone] = (),
    fps: float | None = None,
    frame_idx: int = 0,
    video_fps: float = 25.0,
) -> np.ndarray:
    img = frame
    for z in zones:
        pts = np.array(z.polygon, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    for ln in lines:
        p1 = tuple(int(v) for v in ln.p1)
        p2 = tuple(int(v) for v in ln.p2)
        cv2.line(img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        _label(
            img,
            f"{ln.name}  in {ln.total_in}  out {ln.total_out}",
            (p1[0], p1[1] - 6),
            (0, 255, 255),
        )

    alerted = set().union(*(z._alerted for z in zones)) if zones else set()
    for row in tracks:
        x1, y1, x2, y2 = (int(v) for v in row[:4])
        tid, cls = int(row[4]), int(row[6])
        color = (0, 0, 255) if tid in alerted else id_color(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _label(img, f"#{tid} {names.get(cls, cls)}", (x1, y1 - 2), color)

    rows = []
    if fps is not None:
        rows.append(f"FPS {fps:5.1f}   t={frame_idx / video_fps:6.1f}s")
    for ln in lines:
        for cname in sorted(ln.counts):
            c = ln.counts[cname]
            rows.append(f"{ln.name}/{cname:<7} in {c['in']:>3}  out {c['out']:>3}")
    if rows:
        pad, lh = 8, 20
        w = max(cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for r in rows) + 2 * pad
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, pad + lh * len(rows)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        for i, r in enumerate(rows):
            cv2.putText(
                img,
                r,
                (pad, pad + lh * (i + 1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return img
