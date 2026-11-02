from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traffic_cv.tracking.evaluate import (
    MotSequence,
    _covered_by_ignored,
    compare_table,
    detect_sequence,
    evaluate,
    load_mot_sequence,
)


def _make_seq(tmp_path: Path, n_frames=60, occlude=range(25, 31)) -> tuple[MotSequence, list]:
    """Two cars driving in opposite directions; car 1 vanishes for a few frames."""
    d = tmp_path / "MVI_TEST"
    (d / "gt").mkdir(parents=True)
    rows, dets = [], []
    rng = np.random.default_rng(0)
    for f in range(1, n_frames + 1):
        boxes = [(1, 50 + 6 * f, 200, 60, 40), (2, 700 - 6 * f, 350, 80, 50)]
        fd = []
        for tid, x, y, w, h in boxes:
            rows.append(f"{f},{tid},{x},{y},{w},{h},1,0,1")
            if tid == 1 and f in occlude:
                continue  # detector misses the object (occlusion)
            jx, jy = rng.normal(0, 1.0, 2)
            fd.append([x + jx, y + jy, x + w + jx, y + h + jy, 0.9, 0])
        dets.append(np.array(fd).reshape(-1, 6))
    (d / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname=MVI_TEST\nimDir={tmp_path}/imgs\nframeRate=25\n"
        f"seqLength={n_frames}\nimWidth=960\nimHeight=540\nimExt=.jpg\n"
    )
    (d / "ignored_regions.json").write_text(json.dumps([[900, 0, 960, 50]]))
    return load_mot_sequence(d), dets


def test_load_mot_sequence(tmp_path):
    seq, _ = _make_seq(tmp_path)
    assert seq.length == 60
    assert seq.gt[1].shape == (2, 5)
    assert seq.ignored.shape == (1, 4)


def test_covered_by_ignored():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]], float)
    reg = np.array([[0, 0, 10, 10]], float)
    assert _covered_by_ignored(boxes, reg).tolist() == [True, True, False]


def test_bytetrack_beats_sort_through_occlusion(tmp_path):
    seq, dets = _make_seq(tmp_path)
    report = evaluate([seq], {seq.name: dets}, ["bytetrack", "sort"], out=tmp_path / "res")

    bt, so = report["bytetrack"]["overall"], report["sort"]["overall"]
    assert bt["mota"] > 0.85
    assert bt["idf1"] > 0.9
    assert bt["num_switches"] == 0
    assert so["idf1"] < bt["idf1"]  # SORT starts a new ID after the gap
    assert (tmp_path / "res" / "bytetrack" / "MVI_TEST.txt").exists()

    table = compare_table(report)
    assert "| bytetrack |" in table and "| sort |" in table


def test_detect_sequence_uses_cache(tmp_path):
    seq, dets = _make_seq(tmp_path, n_frames=3)
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez_compressed(cache / "MVI_TEST.npz", **{f"f{i + 1}": d for i, d in enumerate(dets)})

    def boom(_):
        raise AssertionError("detector must not run when cache exists")

    loaded = detect_sequence(boom, seq, cache)
    assert len(loaded) == 3
    np.testing.assert_allclose(loaded[0], dets[0])


@pytest.mark.parametrize("missing_frame", [True])
def test_detect_sequence_missing_image_raises(tmp_path, missing_frame):
    seq, _ = _make_seq(tmp_path, n_frames=2)
    with pytest.raises(FileNotFoundError):
        detect_sequence(lambda img: np.empty((0, 6)), seq, None)
