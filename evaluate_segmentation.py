"""评估已有实例分割权重，保存测试指标和真实标注/预测对照图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment


PROJECT_DIR = Path(__file__).resolve().parent
COLORS = [(230, 110, 25), (35, 155, 210), (155, 65, 205), (45, 175, 105)]


def draw_instances(image, masks, labels, classes, scores=None):
    canvas = image.convert("RGBA")
    for mask, label in zip(masks, labels):
        color = COLORS[int(label) % len(COLORS)]
        layer = Image.new("RGBA", image.size, color + (0,))
        layer.putalpha(Image.fromarray(np.asarray(mask, dtype=np.uint8) * 95))
        canvas = Image.alpha_composite(canvas, layer)
    result = canvas.convert("RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
    for index, (mask, label) in enumerate(zip(masks, labels)):
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        color = COLORS[int(label) % len(COLORS)]
        x, y = int(xs.min()), max(0, int(ys.min()) - 36)
        text = classes[int(label)]
        if scores is not None:
            text += f" {float(scores[index]):.2f}"
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle((box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3), fill=color)
        draw.text((x, y), text, font=font, fill="white")
    return result


def comparison(image, gt, predicted, title):
    scale = min(1.0, 850 / image.width, 850 / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    board = Image.new("RGB", (size[0] * 2, size[1] + 70), "white")
    board.paste(gt.resize(size), (0, 70))
    board.paste(predicted.resize(size), (size[0], 70))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    draw.text((12, 8), title, font=font, fill="black")
    draw.text((12, 38), "Ground truth", font=font, fill="black")
    draw.text((size[0] + 12, 38), "Prediction (confidence >= 0.50)", font=font, fill="black")
    return board


def score_image(gt_masks, gt_labels, prediction):
    pred_masks = prediction.mask
    pred_labels = prediction.class_id
    if pred_masks is None:
        pred_masks = np.zeros((0, *gt_masks.shape[1:]), dtype=bool)
    overlaps = np.zeros((len(gt_masks), len(pred_masks)))
    for i, (gt_mask, gt_label) in enumerate(zip(gt_masks, gt_labels)):
        for j, (pred_mask, pred_label) in enumerate(zip(pred_masks, pred_labels)):
            if gt_label != pred_label:
                continue
            union = np.count_nonzero(gt_mask | pred_mask)
            overlaps[i, j] = np.count_nonzero(gt_mask & pred_mask) / max(1, union)
    rows, columns = linear_sum_assignment(-overlaps)
    matches = [(int(i), int(j), float(overlaps[i, j])) for i, j in zip(rows, columns) if overlaps[i, j] >= 0.5]
    return {
        "ground_truth_count": len(gt_masks),
        "prediction_count": len(pred_masks),
        "matched_at_mask_iou_0_5": len(matches),
        "missed_at_mask_iou_0_5": len(gt_masks) - len(matches),
        "false_positives_at_mask_iou_0_5": len(pred_masks) - len(matches),
        "matches": [{"ground_truth_index": i, "prediction_index": j, "mask_iou": iou} for i, j, iou in matches],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "runs/segmentation/checkpoint_best_ema.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "runs/segmentation/evaluation")
    args = parser.parse_args()
    args.data, args.checkpoint, args.output = (path.resolve() for path in (args.data, args.checkpoint, args.output))
    args.output.mkdir(parents=True, exist_ok=True)

    from rfdetr import RFDETR

    model = RFDETR.from_checkpoint(args.checkpoint, device="cuda", gradient_checkpointing=False)
    print("Loaded:", type(model).__name__, "classes:", model.class_names, "resolution:", model.model_config.resolution, flush=True)
    metrics = model.evaluate(
        dataset_dir=str(args.data), split="test", device="cuda", batch_size=1,
        num_workers=0, tensorboard=False, wandb=False, use_ema=False,
        log_per_class_metrics=True, compute_test_loss=False,
        multi_scale=False, expanded_scales=False,
    )
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Test evaluation completed.", flush=True)

    annotation_path = args.data / "test/_annotations.coco.json"
    coco = COCO(str(annotation_path))
    classes = list(model.class_names)
    category_to_label = {category_id: classes.index(coco.cats[category_id]["name"]) for category_id in coco.cats}
    results = []
    for image_id in coco.getImgIds():
        meta = coco.imgs[image_id]
        with Image.open(args.data / "test" / meta["file_name"]) as source:
            image = source.convert("RGB")
        annotations = [ann for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])) if not ann.get("iscrowd", 0)]
        gt_masks = np.array([coco.annToMask(ann).astype(bool) for ann in annotations])
        gt_labels = [category_to_label[ann["category_id"]] for ann in annotations]
        prediction = model.predict(image, threshold=0.5)
        gt = draw_instances(image, gt_masks, gt_labels, classes)
        predicted = draw_instances(image, prediction.mask if prediction.mask is not None else [], prediction.class_id, classes, prediction.confidence)
        output_path = args.output / f"{Path(meta['file_name']).stem}_comparison.png"
        comparison(image, gt, predicted, meta["file_name"]).save(output_path)
        entry = score_image(gt_masks, gt_labels, prediction)
        entry.update(
            image=meta["file_name"], comparison_image=str(output_path),
            predictions=[{"class": classes[int(label)], "confidence": float(score)} for label, score in zip(prediction.class_id, prediction.confidence)],
        )
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    history_path = args.checkpoint.parent / "metrics.csv"
    history = {}
    if history_path.is_file():
        with history_path.open(encoding="utf-8") as handle:
            validation_rows = [row for row in csv.DictReader(handle) if row.get("val/segm_mAP_50_95")]
        if validation_rows:
            best = max(validation_rows, key=lambda row: float(row["val/segm_mAP_50_95"]))
            history = {
                "completed_validation_epochs": len(validation_rows),
                "best_epoch_one_based": int(best["epoch"]) + 1,
                "best_validation_segm_map_50_95": float(best["val/segm_mAP_50_95"]),
                "last_epoch_one_based": int(validation_rows[-1]["epoch"]) + 1,
                "last_validation_segm_map_50_95": float(validation_rows[-1]["val/segm_mAP_50_95"]),
            }
    summary = {
        "checkpoint": str(args.checkpoint), "data": str(args.data),
        "classes": classes, "resolution": model.model_config.resolution,
        "history": history, "test_metrics": metrics,
        "visualization_confidence_threshold": 0.5,
        "images": results,
    }
    (args.output / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results saved to:", args.output, flush=True)


if __name__ == "__main__":
    main()
