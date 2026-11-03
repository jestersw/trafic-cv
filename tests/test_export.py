from __future__ import annotations

import numpy as np
import pytest

from traffic_cv.export.benchmark import compare_table
from traffic_cv.export.compare_rust import match_frame
from traffic_cv.export.export import letterbox, to_tensor
from traffic_cv.tracking import TRACKERS


def test_letterbox_keeps_aspect_and_pads():
    img = np.full((540, 960, 3), 255, dtype=np.uint8)
    out = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert (out[:140] == 114).all() and (out[-140:] == 114).all()
    assert (out[140:500] == 255).all()


def test_to_tensor_layout():
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    img[..., 2] = 255
    x = to_tensor(img, 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert x[0, 0, 320, 320] == 1.0 and x[0, 2, 320, 320] == 0.0


def test_match_frame():
    a = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.8, 1]], dtype=float)
    b = np.array([[0.5, 0, 10, 10.5, 0.9, 0], [20, 20, 40, 40, 0.8, 2]], dtype=float)
    m, ious, coord = match_frame(a, b)
    assert m == 1 and ious[0] > 0.9 and coord[0] == 0.5


def test_compare_table():
    lat = {"inference_ms": 2.0, "total_ms": 4.0, "fps": 250.0}
    rows = [
        {
            "host": "h",
            "model": "m.pt",
            "device": "0",
            "size_mb": 18.3,
            "latency": lat,
            "map50": 0.68,
        },
        {
            "host": "h",
            "model": "m.engine",
            "device": "0",
            "size_mb": 12.0,
            "latency": lat,
            "map50": 0.663,
            "parity": {"matched": 0.97},
        },
    ]
    t = compare_table(rows, 0.68)
    assert "-2.5%" in t and "97.0%" in t and "ref" in t


@pytest.mark.skipif("bytetrack-rs" not in TRACKERS, reason="traffic_rs not installed")
@pytest.mark.parametrize("name", ["bytetrack", "sort"])
def test_rust_tracker_matches_python(name):
    rng = np.random.default_rng(0)
    py, rs = TRACKERS[name](), TRACKERS[f"{name}-rs"]()
    base = rng.uniform(0, 800, size=(12, 2))
    vel = rng.uniform(-6, 6, size=(12, 2))
    for f in range(60):
        c = base + vel * f
        dets = np.c_[c, c + 40, rng.uniform(0.05, 1.0, 12), rng.integers(0, 4, 12)]
        dets = dets[rng.random(12) > 0.15]
        a, b = py.update(dets), rs.update(dets)
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-6)
