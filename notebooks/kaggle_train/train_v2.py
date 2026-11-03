import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/jestersw/trafic-cv.git"
EXP = "finetune_v2"
WORK = Path("/kaggle/working")
OUT = WORK / "out"


def sh(*cmd: str, check: bool = True) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check)


os.chdir(WORK)
if not (WORK / "trafic-cv").exists():
    sh("git", "clone", "--depth", "1", REPO)
os.chdir(WORK / "trafic-cv")
sh(sys.executable, "-m", "pip", "install", "-q", "-e", ".[train,track,export]")
sh(sys.executable, "-m", "pip", "install", "-q", "tensorrt-cu12")
sh("nvidia-smi")

src = next(Path("/kaggle/input").rglob("detrac/images")).parent
clip = next(Path("/kaggle/input").rglob("clip/MVI_39031"))
data = Path("data/processed/detrac").resolve()
for split in ("train", "val", "test"):
    img = data / "images" / split
    img.parent.mkdir(parents=True, exist_ok=True)
    if not img.exists():
        img.symlink_to(src / "images" / split, target_is_directory=True)
    lab = data / "labels" / split
    if not lab.exists():
        shutil.copytree(src / "labels" / split, lab)
(data / "dataset.yaml").write_text(
    "train: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: car\n  1: bus\n  2: van\n  3: others\n"
)

OUT.mkdir(exist_ok=True)
stem = f"models/{EXP}"
try:
    sh(sys.executable, "-m", "traffic_cv.detection.train", "--exp", EXP, "--device", "0")
    sh(
        sys.executable,
        "-m",
        "traffic_cv.export.export",
        "--weights",
        f"{stem}.pt",
        "--formats",
        "onnx",
        "onnx-int8",
        "engine-fp16",
        "engine-int8",
        "--calib-dir",
        str(data / "images/val"),
        "--data",
        str(data / "dataset.yaml"),
        "--engine-fraction",
        "0.25",
    )
    sh(
        sys.executable,
        "-m",
        "traffic_cv.export.benchmark",
        "--host",
        "kaggle-t4-v2",
        "--reference",
        f"{stem}.pt",
        "--data",
        str(data / "dataset.yaml"),
        "--frames-dir",
        str(data / "images/test"),
        "--models",
        f"{stem}.pt@0",
        f"{stem}_fp16.engine@0",
        f"{stem}_int8.engine@0",
        f"{stem}.onnx@cpu",
        f"{stem}_int8.onnx@cpu",
        check=False,
    )
    for eng in ("int8", "fp16"):
        sh(
            sys.executable,
            "run.py",
            "--video",
            f"{clip}/img%05d.jpg",
            "--model",
            f"{stem}_{eng}.engine",
            "--line",
            "0,324,960,324",
            "--device",
            "0",
            "--no-video",
            "--out-dir",
            str(OUT / f"run_{eng}_engine"),
            check=False,
        )
finally:
    files = [
        Path(f"{stem}.pt"),
        Path(f"{stem}.onnx"),
        Path(f"{stem}_int8.onnx"),
        Path(f"{stem}_fp16.engine"),
        Path(f"{stem}_int8.engine"),
        Path(f"reports/detection/{EXP}.json"),
        Path("reports/export/kaggle-t4-v2.json"),
        Path(f"runs/detect/{EXP}/results.csv"),
        Path("mlflow.db"),
    ]
    for f in files:
        if f.exists():
            shutil.copy2(f, OUT / f.name)
    os.chdir(WORK)
    shutil.rmtree(WORK / "trafic-cv", ignore_errors=True)
    print("done", sorted(p.name for p in OUT.iterdir()))
