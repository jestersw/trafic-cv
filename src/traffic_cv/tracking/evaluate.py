from __future__ import annotations

import argparse
import configparser
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from traffic_cv.tracking import build_tracker

METRICS = [
    "mota",
    "idf1",
    "motp",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "mostly_lost",
    "num_unique_objects",
    "num_frames",
]


@dataclass
class MotSequence:
    name: str
    im_dir: Path
    length: int
    gt: dict[int, np.ndarray] = field(default_factory=dict)
    ignored: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))

    def frame_path(self, frame: int) -> Path:
        return self.im_dir / f"img{frame:05d}.jpg"


def load_mot_sequence(seq_dir: Path, repo_root: Path = Path(".")) -> MotSequence:
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    s = ini["Sequence"]
    im_dir = Path(s["imDir"])
    if not im_dir.is_absolute():
        im_dir = repo_root / im_dir
    seq = MotSequence(s["name"], im_dir, int(s["seqLength"]))

    gt_path = seq_dir / "gt" / "gt.txt"
    raw = (
        np.loadtxt(gt_path, delimiter=",", ndmin=2) if gt_path.stat().st_size else np.empty((0, 9))
    )
    for frame in np.unique(raw[:, 0]).astype(int):
        seq.gt[int(frame)] = raw[raw[:, 0] == frame][:, 1:6]

    ign = seq_dir / "ignored_regions.json"
    if ign.exists():
        seq.ignored = np.asarray(json.loads(ign.read_text()), dtype=float).reshape(-1, 4)
    return seq


def _covered_by_ignored(xyxy: np.ndarray, ignored: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if len(xyxy) == 0 or len(ignored) == 0:
        return np.zeros(len(xyxy), bool)
    a, r = xyxy[:, None, :], ignored[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], r[..., 2]) - np.maximum(a[..., 0], r[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], r[..., 3]) - np.maximum(a[..., 1], r[..., 1]), 0, None)
    area = np.maximum((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]), 1e-9)
    return ((iw * ih) / area >= thr).any(axis=1)


def detect_sequence(
    detector: Callable[[np.ndarray], np.ndarray], seq: MotSequence, cache: Path | None = None
) -> list[np.ndarray]:
    cache_file = cache / f"{seq.name}.npz" if cache else None
    if cache_file and cache_file.exists():
        z = np.load(cache_file)
        return [z[f"f{i}"] for i in range(1, seq.length + 1)]

    import cv2
    from tqdm import tqdm

    dets = []
    for frame in tqdm(range(1, seq.length + 1), desc=f"detect {seq.name}", leave=False):
        img = cv2.imread(str(seq.frame_path(frame)))
        if img is None:
            raise FileNotFoundError(seq.frame_path(frame))
        dets.append(np.asarray(detector(img), dtype=np.float64).reshape(-1, 6))
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, **{f"f{i + 1}": d for i, d in enumerate(dets)})
    return dets


def run_tracker(tracker, dets: list[np.ndarray]) -> tuple[dict[int, np.ndarray], float]:
    tracker.reset()
    out: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for i, d in enumerate(dets, start=1):
        out[i] = tracker.update(d)
    ms = (time.perf_counter() - t0) * 1e3 / max(len(dets), 1)
    return out, ms


def accumulate(seq: MotSequence, hyp: dict[int, np.ndarray]):
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in range(1, seq.length + 1):
        g = seq.gt.get(frame, np.empty((0, 5)))
        g_xyxy = np.c_[g[:, 1:3], g[:, 1:3] + g[:, 3:5]] if len(g) else np.empty((0, 4))
        g = g[~_covered_by_ignored(g_xyxy, seq.ignored)]

        h = hyp.get(frame, np.empty((0, 7)))
        h = h[~_covered_by_ignored(h[:, :4], seq.ignored)] if len(h) else h
        h_xywh = np.c_[h[:, :2], h[:, 2:4] - h[:, :2]] if len(h) else np.empty((0, 4))

        dist = mm.distances.iou_matrix(g[:, 1:5], h_xywh, max_iou=0.5)
        acc.update(g[:, 0].astype(int).tolist(), h[:, 4].astype(int).tolist(), dist, frameid=frame)
    return acc


def write_mot(path: Path, hyp: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, rows in sorted(hyp.items()):
            for r in rows:
                f.write(
                    f"{frame},{int(r[4])},{r[0]:.2f},{r[1]:.2f},{r[2] - r[0]:.2f},"
                    f"{r[3] - r[1]:.2f},{r[5]:.3f},-1,-1,-1\n"
                )


def evaluate(
    seqs: list[MotSequence],
    dets: dict[str, list[np.ndarray]],
    trackers: list[str],
    out: Path | None = None,
) -> dict:
    import motmetrics as mm

    mh = mm.metrics.create()
    report: dict = {}
    for name in trackers:
        accs, ms = [], []
        for seq in seqs:
            hyp, t_ms = run_tracker(build_tracker(name), dets[seq.name])
            accs.append(accumulate(seq, hyp))
            ms.append(t_ms)
            if out:
                write_mot(out / name / f"{seq.name}.txt", hyp)
        summary = mh.compute_many(
            accs, metrics=METRICS, names=[s.name for s in seqs], generate_overall=True
        )
        report[name] = {
            "overall": {k: _num(v) for k, v in summary.loc["OVERALL"].items()},
            "per_sequence": {
                s: {k: _num(v) for k, v in summary.loc[s].items()}
                for s in summary.index
                if s != "OVERALL"
            },
            "tracker_ms_per_frame": round(float(np.mean(ms)), 3),
        }
    return report


def _num(v) -> float | int:
    v = float(v)
    return int(v) if v.is_integer() else round(v, 4)


def compare_table(report: dict) -> str:
    cols = [
        ("MOTA", "mota", "{:.3f}"),
        ("IDF1", "idf1", "{:.3f}"),
        ("MOTP (1-IoU)", "motp", "{:.3f}"),
        ("ID sw.", "num_switches", "{}"),
        ("FP", "num_false_positives", "{}"),
        ("FN", "num_misses", "{}"),
        ("MT", "mostly_tracked", "{}"),
        ("ML", "mostly_lost", "{}"),
    ]
    head = ["tracker", *[c[0] for c in cols], "ms/frame"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for name, r in report.items():
        o = r["overall"]
        lines.append(
            "| "
            + " | ".join(
                [name, *[f.format(o[k]) for _, k, f in cols], f"{r['tracker_ms_per_frame']:.2f}"]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mot-root", default="data/processed/detrac/mot/test")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "sort"])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--dets-cache", default=None, help="default: outputs/dets/<weights-stem>/<mot-root-name>"
    )
    ap.add_argument("--max-seqs", type=int, default=None, help="quick runs on a subset")
    ap.add_argument("--out", default="reports/tracking")
    args = ap.parse_args(argv)

    mot_root = Path(args.mot_root)
    seq_dirs = sorted(p for p in mot_root.iterdir() if (p / "seqinfo.ini").exists())
    if args.max_seqs:
        seq_dirs = seq_dirs[: args.max_seqs]
    if not seq_dirs:
        raise SystemExit(f"no MOT sequences under {mot_root}")
    seqs = [load_mot_sequence(d) for d in seq_dirs]

    cache = Path(args.dets_cache or Path("outputs/dets") / Path(args.weights).stem / mot_root.name)
    detector = None
    if not all((cache / f"{s.name}.npz").exists() for s in seqs):
        from traffic_cv.detection.detector import Detector

        detector = Detector(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device)
    dets = {s.name: detect_sequence(detector, s, cache) for s in seqs}

    out = Path(args.out)
    report = evaluate(seqs, dets, args.trackers, out)
    report["_meta"] = {
        "weights": args.weights,
        "mot_root": str(mot_root),
        "sequences": len(seqs),
        "detector_conf": args.conf,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    table = compare_table({k: v for k, v in report.items() if not k.startswith("_")})
    (out / "compare.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
