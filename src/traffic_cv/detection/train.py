"""Train a detector experiment from ``configs/train.yaml`` and log it to MLflow.

Usage::

    python -m traffic_cv.detection.train --exp baseline
    python -m traffic_cv.detection.train --exp finetune --device 0 --epochs 50

Outputs: ``runs/detect/<exp>/`` (Ultralytics run), ``models/<exp>.pt`` (best weights),
``reports/detection/<exp>.json`` (test-split metrics). One MLflow run per experiment holds
per-epoch metrics (Ultralytics' integration), the resolved config, the augmentation
pipeline and the final test metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from traffic_cv.detection.augment import build_augmentations
from traffic_cv.detection.evaluate import evaluate_weights

# overrides accepted on the command line (type-converted by argparse)
CLI_OVERRIDES = {
    "epochs": int,
    "batch": int,
    "imgsz": int,
    "device": str,
    "fraction": float,
    "workers": int,
    "model": str,
    "data": str,
}


def resolve_config(cfg: dict, exp: str, overrides: dict[str, Any]) -> tuple[dict, list | None]:
    if exp not in cfg["experiments"]:
        raise SystemExit(f"unknown experiment {exp!r}; have {sorted(cfg['experiments'])}")
    args = {**cfg["base"], **(cfg["experiments"][exp] or {})}
    args.update({k: v for k, v in overrides.items() if v is not None})
    aug_spec = args.pop("augmentations", None)
    args["name"] = exp
    args["exist_ok"] = True  # rerunning an experiment overwrites it (DVC-friendly)
    return args, aug_spec


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def train(
    cfg: dict, exp: str, overrides: dict[str, Any], models_dir: Path, reports_dir: Path
) -> dict:
    import mlflow
    from ultralytics import YOLO

    args, aug_spec = resolve_config(cfg, exp, overrides)
    augmentations = build_augmentations(aug_spec)

    ml = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(ml.get("tracking_uri", "sqlite:///mlflow.db"))
    experiment = ml.get("experiment", "traffic-cv-detection")
    mlflow.set_experiment(experiment)
    # Ultralytics' callback calls set_experiment(project) unless this is set
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment

    # Start the run ourselves: Ultralytics' MLflow callback then logs into it and leaves
    # it open, so test metrics land in the same run.
    with mlflow.start_run(run_name=exp) as run:
        mlflow.set_tags({"experiment": exp, "git_commit": _git_rev()})
        mlflow.log_dict({"args": args, "augmentations": aug_spec}, "train_config.yaml")

        model = YOLO(args.pop("model"))
        model.train(**args, augmentations=augmentations)

        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        models_dir.mkdir(parents=True, exist_ok=True)
        final = models_dir / f"{exp}.pt"
        shutil.copy2(best, final)

        result = evaluate_weights(
            final,
            args["data"],
            split="test",
            imgsz=args["imgsz"],
            batch=args["batch"],
            device=args.get("device"),
        )
        result["mlflow_run_id"] = run.info.run_id
        mlflow.log_metrics(
            {
                "test/mAP50": result["map50"],
                "test/mAP50-95": result["map50_95"],
                "test/precision": result["precision"],
                "test/recall": result["recall"],
            }
        )
        mlflow.log_metrics({f"test/AP50_{c}": v["ap50"] for c, v in result["per_class"].items()})

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{exp}.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports/detection")
    for k, t in CLI_OVERRIDES.items():
        ap.add_argument(f"--{k}", type=t, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    overrides = {k: getattr(args, k) for k in CLI_OVERRIDES}
    result = train(cfg, args.exp, overrides, Path(args.models_dir), Path(args.reports_dir))
    print(json.dumps({k: result[k] for k in ("map50", "map50_95", "per_class")}, indent=2))


if __name__ == "__main__":
    main()
