from __future__ import annotations

import argparse
import configparser
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml

from traffic_cv.events.counters import LineCounter
from traffic_cv.tracking import build_tracker
from traffic_cv.tracking.evaluate import (
    MotSequence,
    _covered_by_ignored,
    detect_sequence,
    load_mot_sequence,
    run_tracker,
)

DIRECTIONS = ("in", "out")


def frame_size(seq_dir: Path) -> tuple[int, int]:
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    s = ini["Sequence"]
    return int(s["imWidth"]), int(s["imHeight"])


def gt_frames(seq: MotSequence) -> Iterator[tuple[int, np.ndarray]]:
    for frame in range(1, seq.length + 1):
        g = seq.gt.get(frame)
        if g is None or not len(g):
            yield frame, np.empty((0, 7))
            continue
        xyxy = np.c_[g[:, 1:3], g[:, 1:3] + g[:, 3:5]]
        yield frame, np.c_[xyxy, g[:, 0], np.ones(len(g)), g[:, 6]]


def hyp_frames(seq: MotSequence, hyp: dict[int, np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
    for frame in range(1, seq.length + 1):
        yield frame, hyp.get(frame, np.empty((0, 7)))


def count_line(
    frames: Iterator[tuple[int, np.ndarray]],
    line: list[float],
    names: list[str],
    ignored: np.ndarray,
    fps: float = 25.0,
    margin: float = 0.0,
) -> dict[str, dict[str, int]]:
    lc = LineCounter((line[0], line[1]), (line[2], line[3]), names, margin=margin)
    for frame, rows in frames:
        if len(rows):
            rows = rows[~_covered_by_ignored(rows[:, :4], ignored)]
        lc.update(rows, frame, fps)
    return {c: dict(lc.counts.get(c, {"in": 0, "out": 0})) for c in names}


def candidate_lines(
    width: int,
    height: int,
    orientation: str = "both",
    y_band: tuple[float, float] = (0.35, 0.85),
    x_band: tuple[float, float] = (0.25, 0.75),
) -> list[list[float]]:
    lines = []
    if orientation in ("both", "horizontal"):
        ys = np.arange(y_band[0], y_band[1] + 1e-6, 0.05)
        lines += [[0.0, height * f, float(width), height * f] for f in ys]
    if orientation in ("both", "vertical"):
        xs = np.arange(x_band[0], x_band[1] + 1e-6, 0.05)
        lines += [[width * f, 0.0, width * f, float(height)] for f in xs]
    return [[round(float(v), 1) for v in ln] for ln in lines]


def dominant_flow(seq: MotSequence) -> str:
    first: dict[int, np.ndarray] = {}
    last: dict[int, np.ndarray] = {}
    for _, rows in gt_frames(seq):
        for r in rows:
            c = np.array([(r[0] + r[2]) / 2, r[3]])
            first.setdefault(int(r[4]), c)
            last[int(r[4])] = c
    d = np.array([np.abs(last[t] - first[t]) for t in first]).reshape(-1, 2).sum(axis=0)
    return "horizontal" if d[1] >= d[0] else "vertical"


def suggest_line(
    seq: MotSequence,
    width: int,
    height: int,
    names: list[str],
    y_band: tuple[float, float] = (0.5, 0.85),
    x_band: tuple[float, float] = (0.3, 0.7),
) -> list[float]:
    def score(ln: list[float]) -> tuple[int, float]:
        c = count_line(gt_frames(seq), ln, names, seq.ignored)
        total = sum(v for d in c.values() for v in d.values())
        cx, cy = (ln[0] + ln[2]) / 2, (ln[1] + ln[3]) / 2
        return total, -abs(cx / width - 0.5) - abs(cy / height - 0.65)

    lines = candidate_lines(width, height, dominant_flow(seq), y_band, x_band)
    return max(lines, key=score)


def totals(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    return {d: sum(c[d] for c in counts.values()) for d in DIRECTIONS}


def errors(per_seq: dict[str, dict], names: list[str]) -> dict:
    gt_sum = pred_sum = abs_dir = abs_cls = 0
    by_class = {c: {"gt": 0, "pred": 0, "abs_err": 0} for c in names}
    seq_err = []
    for r in per_seq.values():
        gt_t, pr_t = totals(r["gt"]), totals(r["pred"])
        n_gt = sum(gt_t.values())
        if n_gt:
            seq_err.append(sum(abs(pr_t[d] - gt_t[d]) for d in DIRECTIONS) / n_gt)
        for d in DIRECTIONS:
            gt_sum += gt_t[d]
            pred_sum += pr_t[d]
            abs_dir += abs(pr_t[d] - gt_t[d])
            for c in names:
                g, p = r["gt"][c][d], r["pred"][c][d]
                abs_cls += abs(p - g)
                by_class[c]["gt"] += g
                by_class[c]["pred"] += p
                by_class[c]["abs_err"] += abs(p - g)
    for v in by_class.values():
        v["wape"] = round(v["abs_err"] / v["gt"], 4) if v["gt"] else None
    return {
        "gt": gt_sum,
        "pred": pred_sum,
        "total_error": round(abs(pred_sum - gt_sum) / gt_sum, 4),
        "wape_direction": round(abs_dir / gt_sum, 4),
        "wape_class_direction": round(abs_cls / gt_sum, 4),
        "seq_error_median": round(float(np.median(seq_err)), 4),
        "seq_within_5pct": round(float(np.mean(np.array(seq_err) <= 0.05)), 4),
        "by_class": by_class,
    }


def evaluate(
    seqs: list[MotSequence],
    dets: dict[str, list[np.ndarray]],
    lines: dict[str, list[float]],
    trackers: list[str],
    names: list[str],
    margin: float = 0.0,
) -> dict:
    gt = {s.name: count_line(gt_frames(s), lines[s.name], names, s.ignored) for s in seqs}
    report: dict = {}
    for name in trackers:
        per_seq = {}
        for s in seqs:
            hyp, _ = run_tracker(build_tracker(name), dets[s.name])
            pred = count_line(hyp_frames(s, hyp), lines[s.name], names, s.ignored, margin=margin)
            per_seq[s.name] = {"line": lines[s.name], "gt": gt[s.name], "pred": pred}
        report[name] = {"overall": errors(per_seq, names), "per_sequence": per_seq}
    return report


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def compare_table(report: dict, names: list[str]) -> str:
    head = ["tracker", "GT", "pred", "total err", "WAPE dir", "WAPE class+dir"]
    head += ["median seq err", "seqs <=5%"]
    head += [f"WAPE {c}" for c in names]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for name, r in report.items():
        o = r["overall"]
        cells = [name, str(o["gt"]), str(o["pred"]), pct(o["total_error"])]
        cells += [pct(o["wape_direction"]), pct(o["wape_class_direction"])]
        cells += [pct(o["seq_error_median"]), pct(o["seq_within_5pct"])]
        cells += [pct(o["by_class"][c]["wape"]) for c in names]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def sequence_table(report: dict, tracker: str) -> str:
    head = ["sequence", "GT in", "pred in", "GT out", "pred out", "abs err"]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for seq, r in report[tracker]["per_sequence"].items():
        g, p = totals(r["gt"]), totals(r["pred"])
        err = sum(abs(p[d] - g[d]) for d in DIRECTIONS)
        out.append(f"| {seq} | {g['in']} | {p['in']} | {g['out']} | {p['out']} | {err} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/counting.yaml")
    ap.add_argument("--weights", default="models/finetune.pt")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "sort"])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--suggest-lines", action="store_true")
    ap.add_argument("--out", default="reports/counting")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    names = cfg["classes"]
    mot_root = Path(cfg["mot_root"])
    seq_dirs = sorted(p for p in mot_root.iterdir() if (p / "seqinfo.ini").exists())
    seqs = [load_mot_sequence(d) for d in seq_dirs]

    if args.suggest_lines:
        cfg["lines"] = {
            s.name: suggest_line(s, *frame_size(d), names)
            for s, d in zip(seqs, seq_dirs, strict=True)
        }
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
        print(f"wrote {len(cfg['lines'])} lines to {cfg_path}")
        return

    cache = Path("outputs/dets") / Path(args.weights).stem / mot_root.name
    detector = None
    if not all((cache / f"{s.name}.npz").exists() for s in seqs):
        from traffic_cv.detection.detector import Detector

        detector = Detector(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device)
    dets = {s.name: detect_sequence(detector, s, cache) for s in seqs}

    margin = float(cfg.get("margin", 0.0))
    report = evaluate(seqs, dets, cfg["lines"], args.trackers, names, margin)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_out = {**report, "_meta": {"weights": args.weights, "margin": margin}}
    (out / "metrics.json").write_text(json.dumps(report_out, indent=2))
    table = compare_table(report, names)
    best = min(report, key=lambda k: report[k]["overall"]["wape_direction"])
    text = table + f"\n### {best}: per sequence\n\n" + sequence_table(report, best)
    (out / "compare.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
