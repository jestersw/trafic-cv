"""Evaluate detector weights on a dataset split and build a comparison table.

Usage::

    python -m traffic_cv.detection.evaluate \\
        --weights models/baseline.pt models/finetune.pt --split test --out reports/detection

    # or only build the table from metrics already written by train.py
    python -m traffic_cv.detection.evaluate \\
        --results reports/detection/baseline.json reports/detection/finetune.json

Writes ``<out>/<weights-stem>.json`` per model and ``<out>/compare.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_weights(
    weights: str | Path,
    data: str | Path,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    conf: float = 0.001,
) -> dict:
    """mAP via Ultralytics' validator (COCO-style 101-point AP)."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        plots=False,
        verbose=False,
    )
    names = m.names
    per_class = {
        names[int(c)]: {"ap50": round(float(ap50), 4), "ap50_95": round(float(ap), 4)}
        for c, ap50, ap in zip(m.box.ap_class_index, m.box.ap50, m.box.ap, strict=True)
    }
    return {
        "weights": str(weights),
        "split": split,
        "map50": round(float(m.box.map50), 4),
        "map50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class": per_class,
        "speed_ms": {k: round(float(v), 2) for k, v in m.speed.items()},
    }


def compare_table(results: list[dict]) -> str:
    classes = sorted({c for r in results for c in r["per_class"]})
    head = ["model", "mAP@0.5", "mAP@0.5:0.95", "P", "R", *[f"AP50 {c}" for c in classes]]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    best50 = max(r["map50"] for r in results)
    for r in results:
        name = Path(r["weights"]).stem
        m50 = f"**{r['map50']:.4f}**" if r["map50"] == best50 else f"{r['map50']:.4f}"
        cells = [name, m50, f"{r['map50_95']:.4f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}"]
        cells += [f"{r['per_class'].get(c, {}).get('ap50', float('nan')):.4f}" for c in classes]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", nargs="+", help="evaluate these weights")
    src.add_argument("--results", nargs="+", help="existing metrics JSON files")
    ap.add_argument("--data", default="data/processed/detrac/dataset.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/detection")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [json.loads(Path(p).read_text()) for p in args.results or []]
    for w in args.weights or []:
        r = evaluate_weights(w, args.data, args.split, args.imgsz, args.batch, args.device)
        (out / f"{Path(w).stem}.json").write_text(json.dumps(r, indent=2))
        results.append(r)
    table = compare_table(results)
    (out / "compare.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
