"""Detector wrapper: one interface over PyTorch / ONNX / TensorRT weights.

Ultralytics picks the backend from the file extension (.pt, .onnx, .engine, ...), so
the rest of the pipeline only sees ``(N, 6)`` arrays: x1, y1, x2, y2, score, class_id.
"""

from __future__ import annotations

import numpy as np


class Detector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.1,  # low on purpose: ByteTrack uses low-score boxes in stage 2
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | None = None,
        half: bool = False,
        classes: list[str] | None = None,  # keep only these class names (e.g. for COCO)
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.conf, self.iou, self.imgsz, self.device, self.half = conf, iou, imgsz, device, half
        self.class_ids = None
        if classes:
            lookup = {n: i for i, n in self.model.names.items()}
            missing = [c for c in classes if c not in lookup]
            if missing:
                raise ValueError(f"classes {missing} not in model: {sorted(lookup)}")
            self.class_ids = [lookup[c] for c in classes]

    @property
    def names(self) -> dict[int, str]:
        return dict(self.model.names)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        r = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            classes=self.class_ids,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.empty((0, 6))
        return np.concatenate(
            [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]],
            axis=1,
        ).astype(np.float64)
