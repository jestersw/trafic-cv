from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np

FORMATS = ("onnx", "onnx-int8", "engine-fp16", "engine-int8")


def letterbox(img: np.ndarray, size: int = 640, pad: int = 114) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    top, left = round((size - nh) / 2 - 0.1), round((size - nw) / 2 - 0.1)
    out[top : top + nh, left : left + nw] = resized
    return out


def to_tensor(img_bgr: np.ndarray, size: int = 640) -> np.ndarray:
    x = letterbox(img_bgr, size)[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0


def calibration_images(images_dir: Path, n: int, seed: int = 0) -> list[Path]:
    files = sorted(images_dir.glob("*.jpg"))
    random.Random(seed).shuffle(files)
    return files[:n]


class ImageReader:
    def __init__(self, files: list[Path], input_name: str, size: int = 640) -> None:
        self.files = iter(files)
        self.input_name = input_name
        self.size = size

    def get_next(self) -> dict | None:
        import cv2

        f = next(self.files, None)
        if f is None:
            return None
        return {self.input_name: to_tensor(cv2.imread(str(f)), self.size)}

    def rewind(self) -> None:
        pass


def head_nodes(onnx_path: Path, head_prefix: str) -> list[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    return [n.name for n in model.graph.node if n.name.startswith(head_prefix)]


def export_onnx(weights: Path, imgsz: int, out: Path) -> Path:
    from ultralytics import YOLO

    produced = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True))
    if produced.resolve() != out.resolve():
        shutil.move(produced, out)
    return out


def quantize_int8(
    fp32: Path,
    out: Path,
    calib_dir: Path,
    n_calib: int,
    imgsz: int,
    exclude_head: bool,
) -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = out.with_name(out.stem + "_prep.onnx")
    quant_pre_process(str(fp32), str(prep))
    input_name = (
        ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    )
    reader = ImageReader(calibration_images(calib_dir, n_calib), input_name, imgsz)
    exclude = head_nodes(prep, "/model.23/") if exclude_head else []
    quantize_static(
        str(prep),
        str(out),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
    )
    prep.unlink(missing_ok=True)
    return out


def export_engine(
    weights: Path, imgsz: int, out: Path, int8: bool, data: str, fraction: float
) -> Path:
    from ultralytics import YOLO

    kw = {"int8": True, "data": data, "fraction": fraction} if int8 else {"half": True}
    produced = Path(YOLO(str(weights)).export(format="engine", imgsz=imgsz, workspace=4, **kw))
    shutil.move(produced, out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/finetune.pt")
    ap.add_argument("--formats", nargs="+", default=["onnx", "onnx-int8"], choices=FORMATS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default="data/processed/detrac/dataset.yaml")
    ap.add_argument("--calib-dir", default="data/processed/detrac/images/train")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--quantize-head", action="store_true")
    ap.add_argument("--engine-fraction", type=float, default=0.1)
    args = ap.parse_args(argv)

    weights = Path(args.weights)
    stem = weights.with_suffix("")
    fp32 = stem.with_suffix(".onnx")
    for fmt in args.formats:
        if fmt == "onnx":
            print("->", export_onnx(weights, args.imgsz, fp32))
        elif fmt == "onnx-int8":
            if not fp32.exists():
                export_onnx(weights, args.imgsz, fp32)
            out = Path(f"{stem}_int8.onnx")
            q = quantize_int8(
                fp32, out, Path(args.calib_dir), args.n_calib, args.imgsz, not args.quantize_head
            )
            print("->", q)
        else:
            int8 = fmt == "engine-int8"
            out = Path(f"{stem}_{'int8' if int8 else 'fp16'}.engine")
            print(
                "->", export_engine(weights, args.imgsz, out, int8, args.data, args.engine_fraction)
            )


if __name__ == "__main__":
    main()
