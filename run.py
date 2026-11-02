#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_line(s: str) -> tuple[tuple[float, float], tuple[float, float]]:
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("--line expects x1,y1,x2,y2")
    return (v[0], v[1]), (v[2], v[3])


def parse_polygon(s: str) -> list[tuple[float, float]]:
    pts = [tuple(float(c) for c in p.split(",")) for p in s.split(";") if p.strip()]
    if len(pts) < 3 or any(len(p) != 2 for p in pts):
        raise argparse.ArgumentTypeError('--zone expects "x,y;x,y;x,y[;...]" (>= 3 points)')
    return pts


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help=".pt / .onnx / .engine weights")
    ap.add_argument(
        "--line",
        type=parse_line,
        action="append",
        default=[],
        help="counting line x1,y1,x2,y2 (repeatable)",
    )
    ap.add_argument(
        "--zone",
        type=parse_polygon,
        action="append",
        default=[],
        help='no-stopping zone "x,y;x,y;x,y;..." (repeatable)',
    )
    ap.add_argument("--dwell", type=float, default=5.0, help="seconds allowed in a zone")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "sort"])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    ap.add_argument(
        "--classes", default=None, help="comma-separated class names to keep, e.g. car,bus,truck"
    )
    ap.add_argument("--half", action="store_true", help="FP16 inference (GPU)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    from traffic_cv.detection.detector import Detector
    from traffic_cv.events import DwellZone, LineCounter
    from traffic_cv.pipeline import run_video, save_summary
    from traffic_cv.tracking import build_tracker

    out_dir = Path(args.out_dir or Path("outputs") / Path(args.video).stem)
    det = Detector(
        args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        classes=args.classes.split(",") if args.classes else None,
    )
    names = det.names
    lines = [
        LineCounter(p1, p2, names, name=f"line{i + 1}") for i, (p1, p2) in enumerate(args.line)
    ]
    zones = [
        DwellZone(poly, args.dwell, names, name=f"zone{i + 1}") for i, poly in enumerate(args.zone)
    ]

    summary = run_video(
        args.video,
        det,
        build_tracker(args.tracker),
        names,
        lines,
        zones,
        out_video=None if args.no_video else out_dir / "video.mp4",
        events_path=out_dir / "events.csv",
        tracks_path=out_dir / "tracks.txt",
        show=args.show,
        max_frames=args.max_frames,
    )
    summary.update(model=args.model, tracker=args.tracker)
    save_summary(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
