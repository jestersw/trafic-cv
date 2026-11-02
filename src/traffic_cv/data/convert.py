"""Convert UA-DETRAC to YOLO format with a video-level split.

Usage::

    python -m traffic_cv.data.convert --config configs/data.yaml

Produces under ``out_root``::

    images/{train,val,test}/MVI_20011_img00001.jpg
    labels/{train,val,test}/MVI_20011_img00001.txt   # YOLO: cls cx cy w h (normalised)
    mot/{split}/MVI_xxxxx/gt/gt.txt                  # MOTChallenge GT, every frame (val/test)
    mot/{split}/MVI_xxxxx/seqinfo.ini
    boxes.csv                                         # one row per exported box, for EDA
    sequences.csv                                     # one row per sequence
    split.json
    dataset.yaml                                      # Ultralytics dataset config
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

from traffic_cv.data.detrac import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    Box,
    Sequence,
    frame_filename,
    parse_sequence,
)
from traffic_cv.data.split import SeqInfo, split_sequences

IGNORE_FILL = (114, 114, 114)  # same grey Ultralytics uses for letterbox padding


@dataclass(frozen=True, slots=True)
class SeqJob:
    xml_path: Path
    image_dir: Path
    split: str
    frame_step: int
    out_root: Path
    class_ids: dict[str, int]  # raw vehicle_type -> class id
    mask_ignored: bool
    min_box_side: float
    export_mot: bool


def overlap_fraction(box: Box, region: Box) -> float:
    """Fraction of ``box`` area covered by ``region``."""
    ax1, ay1, ax2, ay2 = box.xyxy
    bx1, by1, bx2, by2 = region.xyxy
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = box.width * box.height
    return (iw * ih) / area if area > 0 else 0.0


def is_ignored(box: Box, regions: list[Box], threshold: float = 0.5) -> bool:
    return any(overlap_fraction(box, r) >= threshold for r in regions)


def yolo_line(cls_id: int, box: Box, img_w: int, img_h: int) -> str:
    cx = (box.left + box.width / 2) / img_w
    cy = (box.top + box.height / 2) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {box.width / img_w:.6f} {box.height / img_h:.6f}"


def _write_image(src: Path, dst: Path, regions: list[Box], mask: bool) -> None:
    if dst.exists():
        dst.unlink()
    if not mask or not regions:
        # hard link when possible (same FS, no extra space, survives DVC), else copy
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    import cv2

    img = cv2.imread(str(src))
    if img is None:
        raise OSError(f"cannot read image {src}")
    for r in regions:
        x1, y1, x2, y2 = (int(round(v)) for v in r.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), IGNORE_FILL, thickness=-1)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _write_mot(seq: Sequence, job: SeqJob, n_frames: int) -> None:
    """MOTChallenge GT: frame,id,x,y,w,h,conf,class,visibility (1-based frames)."""
    seq_dir = job.out_root / "mot" / job.split / seq.name
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    with open(seq_dir / "gt" / "gt.txt", "w") as f:
        for num in sorted(seq.frames):
            for t in seq.frames[num]:
                cls_id = job.class_ids.get(t.vehicle_type)
                if cls_id is None:
                    continue
                b = t.box
                vis = 1.0 - t.truncation_ratio
                f.write(
                    f"{num},{t.track_id},{b.left:.2f},{b.top:.2f},{b.width:.2f},"
                    f"{b.height:.2f},1,{cls_id},{vis:.3f}\n"
                )
    (seq_dir / "ignored_regions.json").write_text(
        json.dumps([list(r.xyxy) for r in seq.ignored_regions])
    )
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={seq.name}\n"
        f"imDir={job.image_dir}\n"
        "frameRate=25\n"
        f"seqLength={n_frames}\n"
        f"imWidth={IMAGE_WIDTH}\nimHeight={IMAGE_HEIGHT}\nimExt=.jpg\n"
    )


def convert_sequence(job: SeqJob) -> tuple[dict, list[dict]]:
    seq = parse_sequence(job.xml_path)
    img_out = job.out_root / "images" / job.split
    lbl_out = job.out_root / "labels" / job.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    frame_nums = sorted(int(p.stem[3:]) for p in job.image_dir.glob("img*.jpg"))
    if not frame_nums:
        raise FileNotFoundError(f"no frames in {job.image_dir}")

    box_rows: list[dict] = []
    n_exported = 0
    for num in frame_nums[:: job.frame_step]:
        stem = f"{seq.name}_{frame_filename(num)[:-4]}"
        lines: list[str] = []
        for t in seq.frames.get(num, []):
            cls_id = job.class_ids.get(t.vehicle_type)
            if cls_id is None:
                continue
            box = t.box.clip(IMAGE_WIDTH, IMAGE_HEIGHT)
            if min(box.width, box.height) < job.min_box_side:
                continue
            if job.mask_ignored and is_ignored(box, seq.ignored_regions):
                continue
            lines.append(yolo_line(cls_id, box, IMAGE_WIDTH, IMAGE_HEIGHT))
            box_rows.append(
                {
                    "sequence": seq.name,
                    "split": job.split,
                    "frame": num,
                    "track_id": t.track_id,
                    "class_id": cls_id,
                    "width": round(box.width, 2),
                    "height": round(box.height, 2),
                    "cx": round(box.left + box.width / 2, 2),
                    "cy": round(box.top + box.height / 2, 2),
                    "truncation": t.truncation_ratio,
                    "weather": seq.weather,
                }
            )
        _write_image(
            job.image_dir / frame_filename(num),
            img_out / f"{stem}.jpg",
            seq.ignored_regions,
            job.mask_ignored,
        )
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_exported += 1

    if job.export_mot:
        _write_mot(seq, job, len(frame_nums))

    seq_row = {
        "sequence": seq.name,
        "split": job.split,
        "weather": seq.weather,
        "camera_state": seq.camera_state,
        "num_frames": len(frame_nums),
        "exported_frames": n_exported,
        "num_tracks": len({t.track_id for ts in seq.frames.values() for t in ts}),
        "num_ignored_regions": len(seq.ignored_regions),
    }
    return seq_row, box_rows


def discover(cfg: dict) -> list[tuple[SeqInfo, Path, Path]]:
    raw = Path(cfg["raw_root"])
    found = []
    for src in cfg["sources"]:
        ann_dir, img_dir = raw / src["annotations"], raw / src["images"]
        if not ann_dir.is_dir():
            print(f"[warn] annotations dir missing, skipping: {ann_dir}")
            continue
        for xml in sorted(ann_dir.glob("*.xml")):
            seq_img_dir = img_dir / xml.stem
            if not seq_img_dir.is_dir():
                print(f"[warn] no images for {xml.stem}, skipping")
                continue
            # cheap attribute read; full parse happens in the worker
            seq = parse_sequence(xml)
            info = SeqInfo(xml.stem, src["official_split"], seq.weather, seq.num_frames)
            found.append((info, xml, seq_img_dir))
    return found


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(cfg["out_root"])
    found = discover(cfg)
    if not found:
        raise SystemExit(f"no sequences found under {cfg['raw_root']} — see configs/data.yaml")

    sp = cfg["split"]
    split = split_sequences(
        (f[0] for f in found),
        strategy=sp["strategy"],
        val_fraction=sp["val_fraction"],
        test_fraction=sp.get("test_fraction", 0.2),
        seed=sp["seed"],
    )
    seq_split = {name: part for part, names in split.items() for name in names}

    # clean rebuild: stale files from a previous config must not linger
    for sub in ("images", "labels", "mot"):
        shutil.rmtree(out_root / sub, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    classes: list[str] = cfg["classes"]
    class_ids = {raw: classes.index(name) for raw, name in cfg["class_map"].items()}

    jobs = [
        SeqJob(
            xml_path=xml,
            image_dir=img_dir,
            split=seq_split[info.name],
            frame_step=cfg["frame_step"][seq_split[info.name]],
            out_root=out_root,
            class_ids=class_ids,
            mask_ignored=cfg["mask_ignored_regions"],
            min_box_side=cfg["min_box_side"],
            export_mot=seq_split[info.name] in ("val", "test"),
        )
        for info, xml, img_dir in found
    ]

    from tqdm import tqdm

    seq_rows, box_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(convert_sequence, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sequences"):
            s, b = fut.result()
            seq_rows.append(s)
            box_rows.extend(b)

    seq_rows.sort(key=lambda r: r["sequence"])
    box_rows.sort(key=lambda r: (r["sequence"], r["frame"], r["track_id"]))
    write_csv(out_root / "sequences.csv", seq_rows)
    write_csv(out_root / "boxes.csv", box_rows)
    (out_root / "split.json").write_text(json.dumps(split, indent=2))
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                # no "path" key: Ultralytics then resolves splits relative to this
                # file, so the dataset stays portable after `dvc pull` elsewhere.
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(classes)),
            },
            sort_keys=False,
        )
    )
    counts = {p: len(v) for p, v in split.items()}
    print(f"done: sequences per split {counts}, boxes exported {len(box_rows)}")


if __name__ == "__main__":
    main()
