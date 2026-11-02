"""对比 CPU INT8 输入尺寸，输出速度、真实标注匹配及 FP32 结果对照。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from benchmark_cpu import compare_predictions, measure
from camera_segmentation import SegmentationPredictor, parse_args as camera_args
from cpu_inference import enable_openvino
from evaluate_segmentation import comparison, draw_instances, score_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "runs/segmentation/checkpoint_best_ema.pth")
    parser.add_argument("--sizes", type=int, nargs="+", default=[312, 288, 264, 240])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("runs/segmentation/cpu_benchmark/resolutions"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.threads < 1 or any(size < 1 for size in args.sizes):
        parser.error("迭代、线程和尺寸必须大于 0，预热不能小于 0")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = camera_args(["--device", "cpu", "--backend", "openvino", "--checkpoint", str(args.checkpoint)])
    predictor = SegmentationPredictor(settings)
    coco = COCO(str(args.data / "test/_annotations.coco.json"))
    classes = list(predictor.model.class_names)
    category_to_label = {category_id: classes.index(category["name"]) for category_id, category in coco.cats.items()}
    originals, camera_frames = {}, []
    for meta in coco.imgs.values():
        with Image.open(args.data / "test" / meta["file_name"]) as image:
            image = image.convert("RGB")
            native = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame = cv2.cvtColor(np.array(image.resize((640, 480))), cv2.COLOR_RGB2BGR)
        annotations = [ann for ann in coco.loadAnns(coco.getAnnIds(imgIds=[meta["id"]])) if not ann.get("iscrowd", 0)]
        gt_masks = np.array([coco.annToMask(ann).astype(bool) for ann in annotations])
        gt_labels = [category_to_label[ann["category_id"]] for ann in annotations]
        originals[meta["file_name"]] = {
            "image": image, "native": native, "camera": frame, "gt_masks": gt_masks, "gt_labels": gt_labels,
            "reference_native": predictor.predict(native)[1],
            "reference_camera": predictor.predict(frame)[1],
        }
        camera_frames.extend([frame, cv2.flip(frame, 1)])
    result = {
        "scope": "CPU INT8 preprocess + model + postprocess + draw, camera images 640x480 plus flips",
        "reference": "original-resolution OpenVINO FP32",
        "iterations": args.iterations, "warmup": args.warmup, "threads": args.threads,
        "test_images": len(originals), "variants": [],
    }
    for size in args.sizes:
        enable_openvino(predictor.model, settings.checkpoint, precision="int8", threads=args.threads, resolution=size)
        predictor.resolution = size
        row = {"resolution": size, **measure(predictor, camera_frames, args.warmup, args.iterations), "images": []}
        for name, item in originals.items():
            native_prediction = predictor.predict(item["native"])[1]
            camera_prediction = predictor.predict(item["camera"])[1]
            evaluation = score_image(item["gt_masks"], item["gt_labels"], native_prediction)
            evaluation.update(
                image=name,
                native_vs_fp32=compare_predictions(item["reference_native"], native_prediction),
                camera_vs_fp32=compare_predictions(item["reference_camera"], camera_prediction),
            )
            row["images"].append(evaluation)
            gt = draw_instances(item["image"], item["gt_masks"], item["gt_labels"], classes)
            predicted = draw_instances(
                item["image"], native_prediction.mask if native_prediction.mask is not None else [],
                native_prediction.class_id, classes, native_prediction.confidence,
            )
            comparison(item["image"], gt, predicted, f"INT8 input {size}, {name}").save(
                args.output / f"input_{size}_{Path(name).stem}.png"
            )
        row["ground_truth_matched"] = sum(item["matched_at_mask_iou_0_5"] for item in row["images"])
        row["false_positives"] = sum(item["false_positives_at_mask_iou_0_5"] for item in row["images"])
        mask_ious = [match["mask_iou"] for item in row["images"] for match in item["matches"]]
        row["mean_matched_mask_iou_vs_gt"] = float(np.mean(mask_ious)) if mask_ious else None
        result["variants"].append(row)
        (args.output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in row.items() if key != "images"}), flush=True)


if __name__ == "__main__":
    main()
