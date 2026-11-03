from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from traffic_cv.events.evaluate import count_line, errors, gt_frames, hyp_frames, suggest_line
from traffic_cv.tracking import TRACKERS
from traffic_cv.tracking.evaluate import accumulate, detect_sequence, load_mot_sequence, run_tracker

NAMES = ["car", "bus", "van", "others"]

GRIDS = {
    "bytetrack": {
        "track_thresh": [0.4, 0.5, 0.6, 0.7],
        "new_track_thresh_delta": [0.0, 0.1, 0.2],
        "match_iou": [0.2, 0.3],
        "low_thresh": [0.1, 0.2],
    },
    "sort": {
        "min_score": [0.25, 0.4, 0.5, 0.6],
        "max_age": [1, 3, 5],
        "min_hits": [1, 3],
        "iou_threshold": [0.2, 0.3],
    },
}


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, values, strict=True))
        if "new_track_thresh_delta" in p:
            p["new_track_thresh"] = round(p["track_thresh"] + p.pop("new_track_thresh_delta"), 3)
        out.append(p)
    return out


def score(tracker: str, params: dict, seqs, dets, lines, gt_counts) -> dict:
    import motmetrics as mm

    accs, per = [], {}
    for s in seqs:
        hyp, _ = run_tracker(TRACKERS[tracker](**params), dets[s.name])
        accs.append(accumulate(s, hyp))
        pred = count_line(hyp_frames(s, hyp), lines[s.name], NAMES, s.ignored)
        per[s.name] = {"gt": gt_counts[s.name], "pred": pred}
    o = (
        mm.metrics.create()
        .compute_many(
            accs,
            metrics=["mota", "idf1", "num_switches"],
            names=[s.name for s in seqs],
            generate_overall=True,
        )
        .loc["OVERALL"]
    )
    e = errors(per, NAMES)
    return {
        "params": params,
        "mota": round(float(o["mota"]), 4),
        "idf1": round(float(o["idf1"]), 4),
        "id_switches": int(o["num_switches"]),
        "count_total_error": e["total_error"],
        "count_wape_direction": e["wape_direction"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", default="data/processed/detrac/mot/val")
    ap.add_argument("--dets-cache", default="outputs/dets/finetune/val")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "sort"])
    ap.add_argument("--out", default="reports/tracking/tuning_val.json")
    args = ap.parse_args(argv)

    seqs = [load_mot_sequence(d) for d in sorted(Path(args.mot_root).iterdir())]
    dets = {s.name: detect_sequence(None, s, Path(args.dets_cache)) for s in seqs}
    lines = {s.name: suggest_line(s, 960, 540, NAMES) for s in seqs}
    gt_counts = {s.name: count_line(gt_frames(s), lines[s.name], NAMES, s.ignored) for s in seqs}

    report = {}
    for tracker in args.trackers:
        rows = [score(tracker, p, seqs, dets, lines, gt_counts) for p in expand(GRIDS[tracker])]
        rows.sort(key=lambda r: r["mota"] + r["idf1"], reverse=True)
        report[tracker] = rows
        best = rows[0]
        print(tracker, json.dumps(best))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
