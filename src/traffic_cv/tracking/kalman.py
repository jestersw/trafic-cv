"""Constant-velocity Kalman filter over boxes, as in SORT (Bewley et al., 2016).

State:       [cx, cy, s, r, vcx, vcy, vs]   (s = area, r = aspect w/h, r is constant)
Measurement: [cx, cy, s, r]

Kept deliberately small (plain numpy, no filterpy) so it ports line-by-line to Rust.
"""

from __future__ import annotations

import numpy as np

_F = np.eye(7)
_F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
_H = np.eye(4, 7)

_R = np.eye(4)
_R[2:, 2:] *= 10.0

_Q = np.eye(7)
_Q[-1, -1] *= 0.01
_Q[4:, 4:] *= 0.01

_P0 = np.eye(7) * 10.0
_P0[4:, 4:] *= 1000.0  # high uncertainty for unobserved velocities


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def x_to_xyxy(x: np.ndarray) -> np.ndarray:
    s, r = max(float(x[2]), 0.0), float(x[3])
    w = np.sqrt(s * r) if r > 0 else 0.0
    h = s / w if w > 0 else 0.0
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxFilter:
    __slots__ = ("x", "P")

    def __init__(self, box_xyxy: np.ndarray) -> None:
        self.x = np.zeros(7)
        self.x[:4] = xyxy_to_z(box_xyxy)
        self.P = _P0.copy()

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:  # area must stay positive
            self.x[6] = 0.0
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        return x_to_xyxy(self.x)

    def update(self, box_xyxy: np.ndarray) -> None:
        z = xyxy_to_z(box_xyxy)
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ _H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return x_to_xyxy(self.x)
