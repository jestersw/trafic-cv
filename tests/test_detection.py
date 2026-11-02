from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traffic_cv.detection.augment import build_augmentations, build_transform
from traffic_cv.detection.evaluate import compare_table
from traffic_cv.detection.train import resolve_config

CFG = yaml.safe_load(Path("configs/train.yaml").read_text())


def test_resolve_config_merges_base_experiment_and_cli():
    args, aug = resolve_config(CFG, "finetune", {"epochs": 3, "device": None})
    assert args["epochs"] == 3
    assert args["model"] == CFG["base"]["model"]
    assert args["fliplr"] == 0.5
    assert args["name"] == "finetune" and args["exist_ok"] is True
    assert "augmentations" not in args and aug


def test_baseline_uses_ultralytics_defaults():
    _, aug = resolve_config(CFG, "baseline", {})
    assert aug is None and build_augmentations(aug) is None


def test_unknown_experiment():
    with pytest.raises(SystemExit):
        resolve_config(CFG, "nope", {})


def test_finetune_augmentations_are_image_only():
    import albumentations as A
    import numpy as np

    _, spec = resolve_config(CFG, "finetune", {})
    transforms = build_augmentations(spec)
    flat = [t for x in transforms for t in (x.transforms if isinstance(x, A.BaseCompose) else [x])]
    assert not any(isinstance(t, A.DualTransform) for t in flat)
    img = np.random.default_rng(0).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    assert A.Compose(transforms)(image=img)["image"].shape == img.shape


def test_build_transform_rejects_bad_spec():
    with pytest.raises(ValueError):
        build_transform({"NotATransform": {}})
    with pytest.raises(ValueError):
        build_transform({"Blur": {}, "CLAHE": {}})


def test_compare_table_marks_best():
    res = [
        {
            "weights": "models/baseline.pt",
            "map50": 0.6,
            "map50_95": 0.4,
            "precision": 0.7,
            "recall": 0.6,
            "per_class": {"car": {"ap50": 0.7}},
        },
        {
            "weights": "models/finetune.pt",
            "map50": 0.7,
            "map50_95": 0.5,
            "precision": 0.8,
            "recall": 0.7,
            "per_class": {"car": {"ap50": 0.8}, "bus": {"ap50": 0.6}},
        },
    ]
    t = compare_table(res)
    assert "| finetune | **0.7000** |" in t
    assert "| baseline | 0.6000 |" in t
    assert "nan" in t
