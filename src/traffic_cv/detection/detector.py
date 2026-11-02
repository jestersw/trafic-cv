from __future__ import annotations

import numpy as np


class Detector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.1,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | None = None,
        half: bool = False,
        classes: list[str] | None = None,
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
