from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traffic_cv.tracking.base import iou_matrix


def load_rust_dets(path: Path) -> dict[int, np.ndarray]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["frame"]] = np.asarray(r["dets"], dtype=float).reshape(-1, 6)
    return out


def match_frame(
    a: np.ndarray, b: np.ndarray, thr: float = 0.5
) -> tuple[int, list[float], list[float]]:
    if not len(a) or not len(b):
        return 0, [], []
    m = iou_matrix(a[:, :4], b[:, :4])
    matched, ious, coord = 0, [], []
    for i in range(len(a)):
        j = int(m[i].argmax())
        if m[i, j] >= thr and a[i, 5] == b[j, 5]:
            matched += 1
            ious.append(float(m[i, j]))
            coord.append(float(np.abs(a[i, :4] - b[j, :4]).max()))
    return matched, ious, coord


def parity(frames_dir: Path, rust: dict[int, np.ndarray], model: str, n: int) -> dict:
    import cv2

    from traffic_cv.detection.detector import Detector

    det = Detector(model, conf=0.1, device="cpu")
    frames = sorted(rust)[:n]
    total_py = total_rs = matched = 0
    ious: list[float] = []
    coord: list[float] = []
    for f in frames:
        img = cv2.imread(str(frames_dir / f"img{f:05d}.jpg"))
        py = det(img)
        rs = rust[f]
        keep_py, keep_rs = py[py[:, 4] >= 0.25], rs[rs[:, 4] >= 0.25]
        total_py += len(keep_py)
        total_rs += len(keep_rs)
        m, i, c = match_frame(keep_py, keep_rs)
        matched += m
        ious += i
        coord += c
    return {
        "frames": len(frames),
        "python_boxes": total_py,
        "rust_boxes": total_rs,
        "matched": round(matched / max(total_py, 1), 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "max_coord_diff_px": round(float(np.max(coord)), 2) if coord else None,
        "p99_coord_diff_px": round(float(np.percentile(coord, 99)), 2) if coord else None,
    }


def perf_row(name: str, s: dict, binary_mb: float | None) -> list[str]:
    p = s["perf"]
    det = p.get("detector_ms_p50")
    if det is None:
        det = p["preprocess_ms_p50"] + p["inference_ms_p50"] + p["postprocess_ms_p50"]
    t = s["totals"]["line1"]
    return [
        name,
        f"{p['fps_end_to_end']:.1f}",
        f"{p['frame_ms_p50']:.1f}",
        f"{p['frame_ms_p95']:.1f}",
        f"{det:.1f}",
        f"{p['tracker_ms_p50']:.2f}",
        f"{p['peak_rss_mb']:.0f}",
        "-" if binary_mb is None else f"{binary_mb:.1f}",
        f"{t['in']}/{t['out']}",
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="data/raw/UA-DETRAC/DETRAC-Images/MVI_39031")
    ap.add_argument("--rust-dir", default="outputs/rust/MVI_39031")
    ap.add_argument("--python-dir", default="outputs/python_onnx/MVI_39031")
    ap.add_argument("--model", default="models/finetune.onnx")
    ap.add_argument("--binary", default="rust-infer/target/release/traffic-infer")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="reports/rust")
    args = ap.parse_args(argv)

    rust = load_rust_dets(Path(args.rust_dir) / "dets.jsonl")
    par = parity(Path(args.frames_dir), rust, args.model, args.n)
    rs = json.loads((Path(args.rust_dir) / "summary.json").read_text())
    py = json.loads((Path(args.python_dir) / "summary.json").read_text())
    bin_mb = Path(args.binary).stat().st_size / 2**20 if Path(args.binary).exists() else None

    head = ["pipeline", "FPS", "frame ms p50", "frame ms p95", "detector ms p50"]
    head += ["tracker ms p50", "peak RSS MB", "binary MB", "line in/out"]
    rows = [
        perf_row("python (ultralytics + onnxruntime)", py, None),
        perf_row("rust (ort + opencv)", rs, bin_mb),
    ]
    table = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    table += ["| " + " | ".join(r) + " |" for r in rows]
    same_counts = py["counts"] == rs["counts"]
    text = "\n".join(table) + "\n\n"
    text += (
        f"detections on {par['frames']} frames: {par['matched'] * 100:.1f}% of python boxes "
        f"matched (IoU>=0.5, same class), mean IoU {par['mean_iou']}, "
        f"max coord diff {par['max_coord_diff_px']} px; "
        f"line counts identical: {same_counts}\n"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(text)
    (out / "parity.json").write_text(
        json.dumps(
            {"detections": par, "same_counts": same_counts, "python": py, "rust": rs}, indent=2
        )
    )
    print(text)


if __name__ == "__main__":
    main()
