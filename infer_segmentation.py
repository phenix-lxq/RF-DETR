"""Run an exported custom RF-DETR segmentation model without PyTorch or RF-DETR."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_chw(array, height, width):
    """Half-pixel bilinear interpolation, align_corners=False, antialias=False."""
    source_h, source_w = array.shape[-2:]
    if (source_h, source_w) == (height, width):
        return np.asarray(array, dtype=np.float32)

    def axis_indices(source, target):
        positions = (np.arange(target, dtype=np.float32) + np.float32(0.5)) * np.float32(source / target)
        positions = np.clip(positions - np.float32(0.5), 0, source - 1)
        lower = positions.astype(np.intp)
        fraction = positions - lower.astype(np.float32)
        return lower, np.minimum(lower + 1, source - 1), fraction

    x0, x1, fx = axis_indices(source_w, width)
    y0, y1, fy = axis_indices(source_h, height)
    rows = array[..., x0] * (1 - fx) + array[..., x1] * fx
    return np.asarray(rows[..., y0, :] * (1 - fy[:, None]) + rows[..., y1, :] * fy[:, None], dtype=np.float32)


def preprocess(rgb, metadata):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("predict() expects an RGB uint8 array with shape [height, width, 3]")
    config = metadata["input"]
    _, _, height, width = config["shape"]
    chw = resize_chw(rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255), height, width)
    mean = np.asarray(config["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(config["std"], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((chw - mean) / std)[None], dtype=np.float32)


@dataclass
class Prediction:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    masks: np.ndarray
    timings_ms: dict

    def records(self, classes):
        return [
            {"class_id": int(label), "class_name": classes[int(label)],
             "confidence": float(score), "box_xyxy": box.tolist(), "mask_index": index,
             "mask_area_pixels": int(mask.sum())}
            for index, (box, score, label, mask) in enumerate(zip(self.boxes, self.scores, self.class_ids, self.masks))
        ]


def decode(raw, image_size, metadata, threshold):
    boxes, logits, mask_logits = (np.asarray(value, dtype=np.float32) for value in raw)
    if (boxes.ndim != 3 or logits.ndim != 3 or mask_logits.ndim != 4
            or boxes.shape[0] != 1 or logits.shape[0] != 1 or mask_logits.shape[0] != 1
            or boxes.shape[-1] != 4 or boxes.shape[1] != logits.shape[1]
            or boxes.shape[1] != mask_logits.shape[1]):
        raise ValueError("Unexpected segmentation outputs; expected boxes [1,Q,4], logits [1,Q,C], masks [1,Q,H,W]")
    if logits.shape[-1] != len(metadata["classes"]) + 1:
        raise ValueError("Model class slots do not match metadata (including the final background slot)")
    if not all(np.isfinite(value).all() for value in (boxes, logits, mask_logits)):
        raise ValueError("Model produced nonfinite outputs")
    scores = np.float32(1) / (np.float32(1) + np.exp(-np.clip(logits[0], -80, 80)))
    # Match RF-DETR's global query/class top-k and deterministic tie ordering.
    indices = np.argsort(-scores.ravel(), kind="stable")[:metadata["num_select"]]
    confidences = scores.ravel()[indices]
    labels = indices % scores.shape[-1]
    queries = indices // scores.shape[-1]
    keep = (confidences > threshold) & (labels < len(metadata["classes"]))
    confidences, labels, queries = confidences[keep], labels[keep], queries[keep]
    selected = boxes[0, queries]
    xyxy = np.concatenate((selected[:, :2] - selected[:, 2:] / 2, selected[:, :2] + selected[:, 2:] / 2), axis=1)
    width, height = image_size
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    xyxy = np.clip(xyxy * scale, 0, scale)
    masks = np.empty((len(queries), height, width), dtype=bool)
    # Resize float logits before thresholding; process one mask at a time to bound temporary memory.
    for index, query in enumerate(queries):
        masks[index] = resize_chw(mask_logits[0, query:query + 1], height, width)[0] > 0
    return Prediction(xyxy, confidences, labels.astype(np.int64), masks, {})


class Segmenter:
    """Reusable synchronous predictor. RGB inputs; pixel boxes and original-size boolean masks."""

    def __init__(self, model_dir, device="GPU", precision="auto"):
        import openvino as ov

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 1 or self.metadata.get("task") != "instance_segmentation":
            raise ValueError("Unsupported deployment metadata")
        if self.metadata["input"].get("resize") != "bilinear_half_pixel_no_antialias":
            raise ValueError("Unsupported preprocessing recipe")
        self.classes = self.metadata["classes"]
        if not self.classes or any(not isinstance(name, str) or not name for name in self.classes):
            raise ValueError("Metadata must contain a nonempty ordered class list")
        for name in ("model.xml", "model.bin"):
            if sha256(self.model_dir / name) != self.metadata["files"][name]:
                raise ValueError(f"Model checksum mismatch: {name}; copy the complete deployment directory")
        device = device.upper()
        if device != "CPU" and device != "GPU" and not device.startswith("GPU."):
            raise ValueError("Choose CPU, GPU or GPU.N explicitly")
        if precision not in ("auto", "f16", "f32"):
            raise ValueError("Precision must be auto, f16 or f32")
        core = ov.Core()
        available = core.available_devices
        if device.startswith("GPU") and not any(name.startswith("GPU") for name in available):
            raise RuntimeError(f"No OpenVINO GPU is available; available devices: {available}")
        if precision == "auto":
            precision = "f16" if device.startswith("GPU") else "f32"
        network = core.read_model(str(self.model_dir / "model.xml"))
        expected = self.metadata["input"]["shape"]
        if len(network.inputs) != 1 or len(network.outputs) != 3 or list(network.input().shape) != expected:
            raise ValueError("IR input/output signature does not match metadata")
        for index, port in enumerate(network.outputs):
            if list(port.shape) != self.metadata["outputs"][index]["shape"]:
                raise ValueError(f"Output {index} shape differs from metadata")
        self.compiled = core.compile_model(network, device, {
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": precision,
        })
        execution = list(self.compiled.get_property("EXECUTION_DEVICES"))
        if device.startswith("GPU") and not all(name.startswith("GPU") for name in execution):
            raise RuntimeError(f"Requested GPU but execution devices are {execution}")
        self.device_info = {
            "requested_device": device, "execution_devices": execution,
            "precision_hint": str(self.compiled.get_property("INFERENCE_PRECISION_HINT")),
            "openvino": ov.__version__,
        }
        self.request = self.compiled.create_infer_request()
        self.lock = threading.Lock()

    def predict(self, rgb, threshold=None):
        threshold = self.metadata["default_threshold"] if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        started = time.perf_counter()
        array = preprocess(rgb, self.metadata)
        prepared = time.perf_counter()
        with self.lock:
            self.request.infer({0: array})
            raw = tuple(self.request.get_output_tensor(index).data.copy() for index in range(3))
        inferred = time.perf_counter()
        prediction = decode(raw, (rgb.shape[1], rgb.shape[0]), self.metadata, threshold)
        finished = time.perf_counter()
        prediction.timings_ms = {
            "preprocess": (prepared - started) * 1000, "inference_and_output_copy": (inferred - prepared) * 1000,
            "postprocess": (finished - inferred) * 1000, "total": (finished - started) * 1000,
        }
        return prediction


def save_prediction(rgb, prediction, predictor, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    colors = np.asarray([[255, 100, 40], [40, 200, 100], [70, 130, 255], [220, 70, 200]], dtype=np.float32)
    for label, mask in zip(prediction.class_ids, prediction.masks):
        overlay[mask] = (overlay[mask] * 0.55 + colors[int(label) % len(colors)] * 0.45).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(prediction.boxes, prediction.scores, prediction.class_ids):
        draw.rectangle(tuple(float(value) for value in box), outline="yellow", width=2)
        draw.text((float(box[0]), max(0, float(box[1]) - 15)), f"{predictor.classes[int(label)]}: {score:.2f}", fill="yellow")
    image.save(output / "overlay.png")
    np.savez_compressed(output / "masks.npz", masks=prediction.masks, class_ids=prediction.class_ids,
                        scores=prediction.scores, boxes=prediction.boxes)
    report = {
        "model_id": predictor.metadata["model_id"], "image_size": [rgb.shape[1], rgb.shape[0]],
        "device": predictor.device_info, "timings_ms": prediction.timings_ms,
        "instances": prediction.records(predictor.classes), "masks_file": "masks.npz",
    }
    (output / "predictions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent, help="Deployment directory")
    parser.add_argument("--device", default="GPU", help="GPU, GPU.N or CPU")
    parser.add_argument("--precision", choices=("auto", "f16", "f32"), default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path, default=Path("segmentation-result"))
    parser.add_argument("--list-devices", action="store_true")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--source", help="Camera index, video path or RTSP URL; requires OpenCV")
    parser.add_argument("--max-frames", type=int, default=0, help="Video frame limit; 0 means until interrupted")
    args = parser.parse_args(argv)
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    try:
        if args.list_devices:
            import openvino as ov
            core = ov.Core()
            for device in core.available_devices:
                print(device, core.get_property(device, "FULL_DEVICE_NAME"))
            return 0
        if args.image is None and args.source is None:
            parser.error("Specify --image or --source")
        predictor = Segmenter(args.model, args.device, args.precision)
        if args.image is not None:
            with Image.open(args.image) as image:
                rgb = np.array(image.convert("RGB"))
            report = save_prediction(rgb, predictor.predict(rgb, args.threshold), predictor, args.output)
            print(json.dumps(report, ensure_ascii=True, indent=2))
        else:
            import cv2
            source = int(args.source) if args.source.isdecimal() else args.source
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Cannot open video source: {args.source}")
            frames, last = 0, None
            try:
                while args.max_frames == 0 or frames < args.max_frames:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    prediction = predictor.predict(rgb, args.threshold)
                    last = rgb, prediction
                    frames += 1
                    if frames == 1 or frames % 30 == 0:
                        print(json.dumps({"frame": frames, "instances": len(prediction.scores),
                                          "timings_ms": prediction.timings_ms}), flush=True)
            except KeyboardInterrupt:
                pass
            finally:
                capture.release()
            if last is None:
                raise RuntimeError("No video frames were received")
            save_prediction(*last, predictor, args.output)
            print(f"Processed {frames} frames; saved last frame to {args.output}")
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
