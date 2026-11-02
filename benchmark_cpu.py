"""比较 CPU 后端速度，并核对优化前后的类别、置信度和分割掩膜。"""

import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from PIL import Image
import torch

from camera_segmentation import SegmentationPredictor, parse_args as camera_args


def measure(predictor, frames, warmup, iterations):
    for index in range(warmup):
        predictor.predict(frames[index % len(frames)])
    timings = []
    for index in range(iterations):
        start = time.perf_counter()
        annotated, detections = predictor.predict(frames[index % len(frames)])
        timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    return {
        "median_ms": median * 1000, "fps": 1 / median,
        "mean_ms": statistics.mean(timings) * 1000,
        "p95_ms": float(np.percentile(timings, 95)) * 1000,
        "objects_last_frame": len(detections),
    }


def compare_predictions(reference, candidate):
    from scipy.optimize import linear_sum_assignment

    overlaps = np.zeros((len(reference), len(candidate)))
    if reference.mask is not None and candidate.mask is not None:
        for i in range(len(reference)):
            for j in range(len(candidate)):
                if reference.class_id[i] == candidate.class_id[j]:
                    union = np.count_nonzero(reference.mask[i] | candidate.mask[j])
                    overlaps[i, j] = np.count_nonzero(reference.mask[i] & candidate.mask[j]) / max(1, union)
    rows, columns = linear_sum_assignment(-overlaps)
    pairs = [(int(i), int(j)) for i, j in zip(rows, columns) if overlaps[i, j] >= 0.5]
    return {
        "reference_objects": len(reference), "candidate_objects": len(candidate),
        "matched_at_mask_iou_0_5": len(pairs),
        "missed_vs_reference": len(reference) - len(pairs),
        "extra_vs_reference": len(candidate) - len(pairs),
        "classes": candidate.class_id.tolist(),
        "confidence": candidate.confidence.tolist(),
        "matched_mask_iou": [float(overlaps[i, j]) for i, j in pairs],
        "max_confidence_change": max((abs(float(reference.confidence[i] - candidate.confidence[j])) for i, j in pairs), default=None),
        "max_box_change_pixels": max((float(np.max(np.abs(reference.xyxy[i] - candidate.xyxy[j]))) for i, j in pairs), default=None),
    }


def compare_backends(args):
    from cpu_inference import enable_openvino

    originals, camera_frames = {}, []
    paths = [args.image, *args.extra_images]
    for path in paths:
        with Image.open(path) as image:
            native = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            frame = cv2.cvtColor(np.array(image.convert("RGB").resize((640, 480))), cv2.COLOR_RGB2BGR)
        originals[f"{path.name}:native"] = native
        originals[f"{path.name}:640x480"] = frame
        originals[f"{path.name}:640x480:flip"] = cv2.flip(frame, 1)
        camera_frames.extend([frame, cv2.flip(frame, 1)])
    settings = camera_args(["--device", "cpu", "--backend", "torch", "--checkpoint", str(args.checkpoint)])
    predictor = SegmentationPredictor(settings)
    original_torch_threads, original_cv_threads = torch.get_num_threads(), cv2.getNumThreads()
    references = {name: predictor.predict(frame)[1] for name, frame in originals.items()}
    results = {
        "checkpoint": str(settings.checkpoint), "camera_size": [640, 480],
        "resolution": predictor.model.model_config.resolution,
        "scope": "preprocess + model + postprocess + draw; excludes capture/JPEG/browser",
        "warmup": args.warmup, "iterations": args.iterations, "backends": [],
    }
    for backend in args.backends:
        if backend == "torch":
            torch.set_num_threads(original_torch_threads)
            cv2.setNumThreads(original_cv_threads)
            predictor.model.inference(compile=False, dtype=torch.float32)
        else:
            enable_openvino(
                predictor.model, settings.checkpoint,
                precision="int8" if backend == "openvino-int8" else "fp32",
            )
            cv2.setNumThreads(1)
        row = {
            "backend": backend, "torch_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
            **measure(predictor, camera_frames, args.warmup, args.iterations),
        }
        row["validation"] = {
            name: compare_predictions(references[name], predictor.predict(frame)[1])
            for name, frame in originals.items()
        }
        results["backends"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # 最后再测一次 FP32，排查温度和后台负载对不同后端测试顺序的影响。
    torch.set_num_threads(original_torch_threads)
    cv2.setNumThreads(original_cv_threads)
    predictor.model.inference(compile=False, dtype=torch.float32)
    results["torch_recheck"] = measure(predictor, camera_frames, args.warmup, args.iterations)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PyTorch recheck:", json.dumps(results["torch_recheck"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--extra-images", type=Path, nargs="*", default=[])
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "runs/segmentation/checkpoint_best_ema.pth")
    parser.add_argument("--compare", action="store_true", help="对比多个后端，默认分别测试 PyTorch、OpenVINO FP32、INT8")
    parser.add_argument("--backends", nargs="+", choices=("torch", "openvino", "openvino-int8"), default=["torch", "openvino", "openvino-int8"])
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 6, 8, 10, 14])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runs/segmentation/cpu_benchmark/thread_sweep.json"))
    parser.add_argument("--torchscript", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or any(n < 1 for n in args.threads):
        parser.error("测试次数和线程数应大于 0，预热次数不能小于 0")
    if args.compare:
        if args.output == Path("runs/segmentation/cpu_benchmark/thread_sweep.json"):
            args.output = Path("runs/segmentation/cpu_benchmark/backend_comparison.json")
        compare_backends(args)
        return
    with Image.open(args.image) as image:
        frame = cv2.cvtColor(np.array(image.convert("RGB").resize((640, 480))), cv2.COLOR_RGB2BGR)
    frames = [frame, cv2.flip(frame, 1)]
    settings = camera_args(["--device", "cpu", "--backend", "torch", "--checkpoint", str(args.checkpoint)])
    predictor = SegmentationPredictor(settings)
    results = []

    def record(mode, threads):
        result = {"mode": mode, "torch_threads": threads, "opencv_threads": cv2.getNumThreads()}
        result.update(measure(predictor, frames, args.warmup, args.iterations))
        print(json.dumps(result), flush=True)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    record("baseline", torch.get_num_threads())
    cv2.setNumThreads(1)
    for threads in args.threads:
        torch.set_num_threads(threads)
        record("eager", threads)
    if args.torchscript:
        best = min(results[1:], key=lambda row: row["median_ms"])
        torch.set_num_threads(best["torch_threads"])
        started = time.perf_counter()
        predictor.model.inference(compile=True, dtype=torch.float32, compile_backend="torchscript")
        print("TorchScript setup seconds:", time.perf_counter() - started, flush=True)
        record("torchscript", best["torch_threads"])


if __name__ == "__main__":
    main()
