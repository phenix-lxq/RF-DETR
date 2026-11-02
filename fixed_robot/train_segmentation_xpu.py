"""RF-DETR Seg Nano training using the remote machine's verified XPU adapter."""

import argparse
import inspect
import json
import math
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return number


def check_dataset(root):
    """Check COCO metadata and image references; mask decoding is checked by training."""
    if (root / "annotations" / "instances_train2017.json").is_file():
        dataset_file = "coco"
        splits = (
            ("train", root / "train2017", root / "annotations" / "instances_train2017.json"),
            ("valid", root / "val2017", root / "annotations" / "instances_val2017.json"),
        )
    elif (root / "train" / "_annotations.coco.json").is_file():
        dataset_file = "roboflow"
        splits = tuple(
            (split, root / split, root / split / "_annotations.coco.json")
            for split in ("train", "valid")
        )
    else:
        raise ValueError(
            "Expected annotations/instances_train2017.json or train/_annotations.coco.json. "
            "Export instance masks as COCO polygons or RLE; detection boxes alone are insufficient."
        )

    train_categories = None
    for split, image_root, annotation_path in splits:
        data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        images = {image["id"]: image for image in data["images"]}
        categories = {category["id"]: category["name"] for category in data["categories"]}
        if not images or not categories:
            raise ValueError(f"Empty images or categories: {annotation_path}")
        if any(not isinstance(category_id, int) or category_id < 0 for category_id in categories):
            raise ValueError(f"Category IDs must be nonnegative integers: {annotation_path}")
        if train_categories is None:
            train_categories = categories
        elif categories != train_categories:
            raise ValueError("Train/validation category IDs and names must match")
        for image in images.values():
            image_path = image_root / image["file_name"]
            if not image_path.is_file():
                raise ValueError(f"Missing image: {image_path}")
        masks = 0
        for annotation in data["annotations"]:
            if annotation["image_id"] not in images or annotation["category_id"] not in categories:
                raise ValueError(f"Unknown image/category reference: {annotation_path}")
            if annotation.get("iscrowd", 0):
                continue
            box = annotation.get("bbox", [])
            if len(box) != 4 or not all(math.isfinite(value) for value in box) or min(box[2:]) <= 0:
                raise ValueError(f"Invalid bounding box in annotation {annotation.get('id')}")
            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, list):
                valid_mask = bool(segmentation) and all(
                    isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
                    and all(math.isfinite(value) for value in polygon)
                    for polygon in segmentation
                )
            elif isinstance(segmentation, dict):
                size = segmentation.get("size", [])
                valid_mask = (
                    len(size) == 2 and all(isinstance(value, int) and value > 0 for value in size)
                    and isinstance(segmentation.get("counts"), (str, list))
                    and bool(segmentation["counts"])
                )
            else:
                valid_mask = False
            if not valid_mask:
                raise ValueError(f"Missing/invalid instance mask in annotation {annotation.get('id')}")
            masks += 1
        if not masks:
            raise ValueError(f"No non-crowd instance masks: {annotation_path}")
        print(f"{split}: {len(images)} images, {masks} instance masks")

    # Standard COCO keeps source IDs; Roboflow remaps them to contiguous labels.
    num_classes = max(1, max(train_categories)) if dataset_file == "coco" else len(train_categories)
    return dataset_file, num_classes


def train(args, dataset_file, num_classes):
    import torch
    from pytorch_lightning import seed_everything
    from pytorch_lightning.strategies import SingleDeviceStrategy
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    from train_xpu import DeviceProof, XPUAccelerator

    if not torch.xpu.is_available():
        raise RuntimeError("XPU required; CPU fallback is disabled. Run in the remote cleaning environment.")
    seed_everything(42, workers=True)

    class SegmentationProof(DeviceProof):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            super().on_train_batch_start(trainer, module, batch, batch_idx)
            for target in batch[1]:
                masks = target.get("masks")
                if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
                    raise RuntimeError("Expected target masks with shape [instances, height, width]")
                self.check(masks)
                if len(masks) != len(target["labels"]):
                    raise RuntimeError("Instance mask count does not match label count")

        def on_after_backward(self, trainer, module):
            super().on_after_backward(trainer, module)
            gradients = [
                parameter.grad for name, parameter in module.named_parameters()
                if "segmentation_head" in name and parameter.grad is not None
            ]
            if any(gradient.count_nonzero().item() for gradient in gradients):
                self.report["mask_gradient_steps"] = self.report.get("mask_gradient_steps", 0) + 1
                self.report["mask_gradient_tensors"] = len(gradients)

    weights = str(args.weights.resolve()) if args.weights else (
        "rf-detr-seg-nano.pt" if args.pretrained else None
    )
    model_config = RFDETRSegNanoConfig(
        pretrain_weights=weights, device="xpu:0", amp=False, num_classes=num_classes,
    )
    train_kwargs = dict(
        dataset_dir=str(args.dataset), dataset_file=dataset_file,
        output_dir=str(args.output), epochs=args.epochs, batch_size=1, grad_accum_steps=1,
        lr=args.lr, lr_encoder=args.lr, num_workers=0,
        multi_scale=False, expanded_scales=False, use_ema=False,
        tensorboard=False, wandb=False, run_test=False, compute_val_loss=True,
    )
    # Older remote RF-DETR uses amp=False; current versions also expose amp_dtype.
    if "amp_dtype" in SegmentationTrainConfig.model_fields:
        train_kwargs["amp_dtype"] = None
    train_config = SegmentationTrainConfig(**train_kwargs)
    module_kwargs = {}
    if "load_encoder_weights" in inspect.signature(RFDETRModelModule).parameters:
        module_kwargs["load_encoder_weights"] = False
    module = RFDETRModelModule(model_config, train_config, **module_kwargs)
    if module.model.segmentation_head is None:
        raise RuntimeError("Segmentation head was not initialized")
    data = RFDETRDataModule(model_config, train_config)
    proof = SegmentationProof()
    trainer = build_trainer(
        train_config, model_config, accelerator=XPUAccelerator(),
        strategy=SingleDeviceStrategy(device="xpu:0"), devices=1, precision="32-true",
        limit_train_batches=1.0 if args.full_epoch else 2,
        limit_val_batches=1.0 if args.full_epoch else 1,
        num_sanity_val_steps=0, log_every_n_steps=1,
    )
    # Append the device checks while keeping RF-DETR's evaluation/checkpoint callbacks.
    trainer.callbacks.append(proof)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = trainer.validate(module, datamodule=data)
    (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    trainer.fit(module, datamodule=data)
    trainer.save_checkpoint(args.output / "last.ckpt")
    proof.report.update(
        optimizer_steps=trainer.global_step, device_name=torch.xpu.get_device_name(0),
        task="instance_segmentation", dataset_file=dataset_file, resolution=model_config.resolution,
    )
    (args.output / "device-proof.json").write_text(json.dumps(proof.report, indent=2), encoding="utf-8")
    if proof.report["backward_steps"] < 2 or trainer.global_step < 2:
        raise RuntimeError("Verification requires two optimizer steps; add images or increase --epochs")
    if not proof.report.get("mask_gradient_steps"):
        raise RuntimeError("No nonzero segmentation-head gradients; include positive images in the training batches")
    print(json.dumps(proof.report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="COCO instance segmentation dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "segmentation-xpu")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--lr", type=positive_float, default=1e-5)
    parser.add_argument("--pretrained", action="store_true", help="Load official Seg Nano weights")
    parser.add_argument("--weights", type=Path, help="Use a local segmentation checkpoint instead")
    parser.add_argument("--full-epoch", action="store_true", help="Use all training/validation batches")
    parser.add_argument("--check-data", action="store_true", help="Check metadata without importing PyTorch or training")
    args = parser.parse_args(argv)
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    try:
        dataset_file, num_classes = check_dataset(args.dataset)
        print(f"Dataset layout: {dataset_file}; configured classes: {num_classes}")
        if args.check_data:
            print("Metadata check passed. Image decoding, mask decoding and XPU training have not been tested.")
            return 0
        if args.weights and not args.weights.expanduser().is_file():
            raise ValueError(f"Checkpoint does not exist: {args.weights}")
        if args.weights:
            args.weights = args.weights.expanduser()
        train(args, dataset_file, num_classes)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
