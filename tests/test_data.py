from __future__ import annotations

import json
from pathlib import Path

import yaml

from traffic_cv.data.convert import main as convert_main
from traffic_cv.data.convert import overlap_fraction, yolo_line
from traffic_cv.data.detrac import Box, parse_sequence
from traffic_cv.data.split import SeqInfo, split_sequences


def test_parse_sequence(fake_detrac: Path):
    seq = parse_sequence(fake_detrac / "DETRAC-Train-Annotations-XML" / "MVI_20000.xml")
    assert seq.name == "MVI_20000"
    assert seq.weather == "sunny"
    assert len(seq.ignored_regions) == 1
    assert seq.num_frames == 6
    t = seq.frames[1][0]
    assert (t.track_id, t.vehicle_type) == (1, "car")
    assert t.box == Box(105, 300, 80, 60)


def test_yolo_line_normalised():
    line = yolo_line(2, Box(0, 0, 960, 540), 960, 540)
    assert line == "2 0.500000 0.500000 1.000000 1.000000"


def test_overlap_fraction():
    assert overlap_fraction(Box(0, 0, 10, 10), Box(5, 0, 10, 10)) == 0.5
    assert overlap_fraction(Box(0, 0, 10, 10), Box(20, 20, 5, 5)) == 0.0


def _infos(n_train=20, n_test=5):
    weathers = ["sunny", "night", "rainy", "cloudy"]
    return [SeqInfo(f"tr{i}", "train", weathers[i % 4]) for i in range(n_train)] + [
        SeqInfo(f"te{i}", "test", weathers[i % 4]) for i in range(n_test)
    ]


def test_split_official_is_disjoint_and_stratified():
    sp = split_sequences(_infos(), strategy="official", val_fraction=0.2, seed=1)
    assert set(sp["test"]) == {f"te{i}" for i in range(5)}
    assert not set(sp["train"]) & set(sp["val"])
    assert len(sp["train"]) + len(sp["val"]) == 20
    weathers = {i.name: i.stratum for i in _infos()}
    assert sorted(weathers[n] for n in sp["val"]) == ["cloudy", "night", "rainy", "sunny"]


def test_split_is_deterministic():
    assert split_sequences(_infos(), seed=7) == split_sequences(_infos(), seed=7)


def test_split_random_strategy():
    sp = split_sequences(_infos(), strategy="random", val_fraction=0.15, test_fraction=0.2)
    all_names = sp["train"] + sp["val"] + sp["test"]
    assert len(all_names) == len(set(all_names)) == 25
    assert sp["test"]


def test_convert_end_to_end(fake_detrac: Path, tmp_path: Path):
    out = tmp_path / "processed"
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    cfg.update(raw_root=str(fake_detrac), out_root=str(out))
    cfg["frame_step"] = {"train": 2, "val": 3, "test": 3}
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    convert_main(["--config", str(cfg_path), "--workers", "2"])

    split = json.loads((out / "split.json").read_text())
    assert len(split["test"]) == 3
    assert len(split["train"]) + len(split["val"]) == 8
    assert split["val"]

    for part, names in split.items():
        for name in names:
            imgs = list((out / "images" / part).glob(f"{name}_*.jpg"))
            assert imgs, (part, name)
            for other in {"train", "val", "test"} - {part}:
                assert not list((out / "images" / other).glob(f"{name}_*.jpg"))

    train_seq = split["train"][0]
    lbls = sorted((out / "labels" / "train").glob(f"{train_seq}_*.txt"))
    assert len(lbls) == 3
    rows = lbls[0].read_text().strip().splitlines()
    assert len(rows) == 2
    assert rows[0].startswith("0 ") and rows[1].startswith("1 ")

    import cv2

    img = cv2.imread(str(next((out / "images" / "train").glob("*.jpg"))))
    assert abs(int(img[50, 880, 0]) - 114) <= 3
    assert abs(int(img[300, 300, 0]) - 200) <= 3

    gt = (out / "mot" / "test" / split["test"][0] / "gt" / "gt.txt").read_text().splitlines()
    assert len(gt) == 6 * 3

    ds = yaml.safe_load((out / "dataset.yaml").read_text())
    assert ds["names"] == {0: "car", 1: "bus", 2: "van", 3: "others"}
    assert "path" not in ds


def test_discover_official_drive_layout(tmp_path: Path):
    from conftest import make_sequence

    from traffic_cv.data.convert import discover

    root = tmp_path / "UA-DETRAC"
    make_sequence(
        root, "DETRAC-Images", "nested/DETRAC-Train-Annotations-XML", "MVI_20011", "sunny"
    )
    make_sequence(root, "DETRAC-Images", "nested/DETRAC-Test-Annotations-XML", "MVI_39031", "night")
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    cfg["raw_root"] = str(root)

    found = {info.name: (info.official_split, img_dir) for info, _, img_dir in discover(cfg)}
    assert found == {
        "MVI_20011": ("train", root / "DETRAC-Images" / "MVI_20011"),
        "MVI_39031": ("test", root / "DETRAC-Images" / "MVI_39031"),
    }


def test_download_extracts_existing_archives(tmp_path: Path):
    import zipfile

    from traffic_cv.data.download import ARCHIVES
    from traffic_cv.data.download import main as download_main

    archives = tmp_path / "zips"
    archives.mkdir()
    for name in ARCHIVES:
        with zipfile.ZipFile(archives / name, "w") as zf:
            zf.writestr(f"{name[:-4]}/MVI_1.txt", "x")
            zf.writestr("__MACOSX/junk", "x")
    out = tmp_path / "raw"
    download_main(["--out", str(out), "--archives", str(archives), "--no-download"])
    assert sorted(p.name for p in out.iterdir()) == sorted(n[:-4] for n in ARCHIVES)


def test_download_reports_missing_archives(tmp_path: Path, capsys):
    import pytest

    from traffic_cv.data.download import main as download_main

    with pytest.raises(SystemExit):
        download_main(
            ["--out", str(tmp_path / "raw"), "--archives", str(tmp_path), "--no-download"]
        )
    assert "drive.google.com" in capsys.readouterr().out
