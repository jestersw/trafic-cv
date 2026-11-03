from __future__ import annotations

import albumentations as A
import numpy as np


class Haze(A.ImageOnlyTransform):
    def __init__(
        self,
        strength: tuple[float, float] = (0.15, 0.45),
        falloff: tuple[float, float] = (0.3, 0.9),
        airlight: tuple[float, float] = (170.0, 235.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.strength = strength
        self.falloff = falloff
        self.airlight = airlight

    def get_params(self) -> dict[str, float]:
        r = self.py_random
        return {
            "strength": r.uniform(*self.strength),
            "falloff": r.uniform(*self.falloff),
            "airlight": r.uniform(*self.airlight),
        }

    def apply(
        self,
        img: np.ndarray,
        strength: float = 0.3,
        falloff: float = 0.6,
        airlight: float = 200.0,
        **params,
    ) -> np.ndarray:
        h = img.shape[0]
        a = strength * np.linspace(1.0, 1.0 - falloff, h, dtype=np.float32)
        a = a.reshape(h, *([1] * (img.ndim - 1)))
        out = img.astype(np.float32) * (1.0 - a) + airlight * a
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("strength", "falloff", "airlight")
