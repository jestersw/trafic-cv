from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from traffic_cv.tracking.base import iou_matrix


def model_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 2**20, 1)


def sample_frames(images_dir: Path, n: int) -> list[np.ndarray]:
    import cv2

    files = sorted(images_dir.glob("*.jpg"))
    step = max(len(files) // n, 1)
    return [cv2.imread(str(f)) for f in files[::step][:n]]


def predict(model, frame: np.ndarray, device: str, imgsz: int, conf: float, half: bool):
    return model.predict(frame, imgsz=imgsz, conf=conf, device=device, half=half, verbose=False)[0]


def as_array(r) -> np.ndarray:
    b = r.boxes
    if b is None or len(b) == 0:
        return np.empty((0, 6))
    return np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()]


def latency(model, frames, device: str, imgsz: int, half: bool, warmup: int = 10) -> dict:
    for f in frames[:warmup]:
        predict(model, f, device, imgsz, 0.1, half)
    rows = [predict(model, f, device, imgsz, 0.1, half).speed for f in frames]
    pre = np.median([r["preprocess"] for r in rows])
    inf = np.median([r["inference"] for r in rows])
    post = np.median([r["postprocess"] for r in rows])
    total = pre + inf + post
    return {
        "preprocess_ms": round(float(pre), 2),
        "inference_ms": round(float(inf), 2),
        "postprocess_ms": round(float(post), 2),
        "total_ms": round(float(total), 2),
        "fps": round(1000.0 / total, 1),
        "inference_fps": round(1000.0 / inf, 1),
    }


def parity(ref, model, frames, device, ref_device, imgsz: int, half: bool) -> dict:
    ious, matched, total, conf_diff = [], 0, 0, []
    for f in frames:
        a = as_array(predict(ref, f, ref_device, imgsz, 0.25, False))
        b = as_array(predict(model, f, device, imgsz, 0.25, half))
        total += len(a)
        if not len(a) or not len(b):
            continue
        m = iou_matrix(a[:, :4], b[:, :4])
        best = m.argmax(axis=1)
        for i, j in enumerate(best):
            if m[i, j] >= 0.5 and a[i, 5] == b[j, 5]:
                matched += 1
                ious.append(m[i, j])
                conf_diff.append(abs(a[i, 4] - b[j, 4]))
    return {
        "ref_boxes": total,
        "matched": round(matched / max(total, 1), 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "max_conf_diff": round(float(np.max(conf_diff)), 4) if conf_diff else None,
    }


def benchmark(entry: str, args) -> dict:
    from ultralytics import YOLO

    from traffic_cv.detection.evaluate import evaluate_weights

    path_s, _, device = entry.partition("@")
    path = Path(path_s)
    device = device or "cpu"
    half = path.suffix == ".pt" and device not in ("cpu", "mps") and args.half_pt
    model = YOLO(str(path), task="detect")
    frames = sample_frames(Path(args.frames_dir), args.n_frames)
    res = {
        "model": path.name,
        "format": path.suffix.lstrip("."),
        "device": device,
        "size_mb": model_size_mb(path),
        "latency": latency(model, frames, device, args.imgsz, half),
    }
    if path.suffix != ".pt" or half:
        ref = YOLO(args.reference, task="detect")
        ref_dev = "cpu" if device == "mps" else device
        res["parity"] = parity(
            ref, model, frames[: args.n_parity], device, ref_dev, args.imgsz, half
        )
    if not args.skip_map:
        batch = 16 if path.suffix == ".pt" else 1
        m = evaluate_weights(path, args.data, "test", args.imgsz, batch, device)
        res["map50"], res["map50_95"] = m["map50"], m["map50_95"]
    return res


def fmt(v, spec="{:.4f}") -> str:
    return "-" if v is None else spec.format(v)


def compare_table(rows: list[dict], ref_map: float | None) -> str:
    head = ["host", "model", "device", "size MB", "infer ms", "total ms", "FPS", "mAP@0.5"]
    head += ["mAP@0.5:0.95", "ΔmAP@0.5", "match vs .pt"]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rows:
        lat = r["latency"]
        d = None
        if ref_map and r.get("map50") is not None:
            d = (r["map50"] - ref_map) / ref_map
        cells = [r["host"], r["model"], r["device"], f"{r['size_mb']:.1f}"]
        cells += [f"{lat['inference_ms']:.1f}", f"{lat['total_ms']:.1f}", f"{lat['fps']:.0f}"]
        cells += [fmt(r.get("map50")), fmt(r.get("map50_95"))]
        cells += ["-" if d is None else f"{d * 100:+.1f}%"]
        p = r.get("parity")
        cells += [f"{p['matched'] * 100:.1f}%" if p else "ref"]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def collect(out_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        data = json.loads(f.read_text())
        rows += [{"host": data["host"], **r} for r in data["results"]]
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="path[@device]")
    ap.add_argument("--host", default=platform.node().split(".")[0])
    ap.add_argument("--reference", default="models/finetune.pt")
    ap.add_argument("--data", default="data/processed/detrac/dataset.yaml")
    ap.add_argument("--frames-dir", default="data/processed/detrac/images/test")
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--n-parity", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half-pt", action="store_true")
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--ref-map", type=float, default=None)
    ap.add_argument("--out", default="reports/export")
    args = ap.parse_args(argv)

    results = []
    for entry in args.models:
        r = benchmark(entry, args)
        print(json.dumps(r))
        results.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.host}.json").write_text(
        json.dumps({"host": args.host, "results": results}, indent=2)
    )
    rows = collect(out)
    ref_map = args.ref_map
    if ref_map is None:
        refs = [r["map50"] for r in rows if r["model"].endswith(".pt") and r.get("map50")]
        ref_map = refs[0] if refs else None
    table = compare_table(rows, ref_map)
    (out / "compare.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
