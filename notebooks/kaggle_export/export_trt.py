import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/jestersw/trafic-cv.git"
WORK = Path("/kaggle/working")
OUT = WORK / "out"


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


os.chdir(WORK)
if not (WORK / "trafic-cv").exists():
    sh("git", "clone", "--depth", "1", REPO)
os.chdir(WORK / "trafic-cv")
sh(sys.executable, "-m", "pip", "install", "-q", "-e", ".[train,export]")
sh(sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu", "tensorrt-cu12")
sh("nvidia-smi")

src = next(Path("/kaggle/input").rglob("finetune.pt")).parent
data = Path("data/processed/detrac").resolve()
data.mkdir(parents=True, exist_ok=True)
for kind in ("images", "labels"):
    for split in ("val", "test"):
        dst = data / kind / split
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(next(src.rglob(f"{kind}/{split}")), dst)
(data / "dataset.yaml").write_text(
    f"path: {data}\ntrain: images/val\nval: images/val\ntest: images/test\n"
    "names:\n  0: car\n  1: bus\n  2: van\n  3: others\n"
)
Path("models").mkdir(exist_ok=True)
shutil.copy2(src / "finetune.pt", "models/finetune.pt")

try:
    sh(
        sys.executable,
        "-m",
        "traffic_cv.export.export",
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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "traffic_cv.export.benchmark",
            "--host",
            "kaggle-t4",
            "--data",
            str(data / "dataset.yaml"),
            "--frames-dir",
            str(data / "images/test"),
            "--models",
            "models/finetune.pt@0",
            "models/finetune_fp16.engine@0",
            "models/finetune_int8.engine@0",
            "models/finetune.onnx@cpu",
            "models/finetune_int8.onnx@cpu",
            "models/finetune.onnx@0",
        ]
    )
finally:
    OUT.mkdir(exist_ok=True)
    for f in [Path("reports/export/kaggle-t4.json"), *Path("models").glob("finetune_*.engine")]:
        if f.exists():
            shutil.copy2(f, OUT / f.name)
    os.chdir(WORK)
    shutil.rmtree(WORK / "trafic-cv", ignore_errors=True)
    print("done", sorted(p.name for p in OUT.iterdir()))
