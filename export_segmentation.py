"""Package a fine-tuned segmentation checkpoint for OpenVINO deployment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from infer_segmentation import sha256


PROJECT_DIR = Path(__file__).resolve().parent


def verify(model, output, paths, threshold):
    import numpy as np
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from infer_segmentation import Segmenter, preprocess
    import torch
    from torchvision.transforms import functional as F

    candidate = Segmenter(output, device="CPU", precision="f32")
    records = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        rgb = np.array(image)
        size = candidate.metadata["input"]["shape"][-2:]
        official = F.normalize(F.resize(F.to_tensor(image), size, antialias=False), model.means, model.stds)
        preprocess_error = float(np.max(np.abs(preprocess(rgb, candidate.metadata)[0] - official.numpy())))
        with torch.inference_mode():
            reference = model.predict(image, threshold=threshold)
        result = candidate.predict(rgb, threshold)
        foreground = reference.class_id < len(candidate.classes)
        labels = reference.class_id[foreground]
        masks = reference.mask[foreground]
        boxes = reference.xyxy[foreground]
        scores = reference.confidence[foreground]
        overlaps = np.zeros((len(labels), len(result.class_ids)))
        for i, (label, mask) in enumerate(zip(labels, masks)):
            for j, (other_label, other_mask) in enumerate(zip(result.class_ids, result.masks)):
                if label == other_label:
                    union = np.count_nonzero(mask | other_mask)
                    overlaps[i, j] = np.count_nonzero(mask & other_mask) / max(1, union)
        rows, columns = linear_sum_assignment(-overlaps)
        ious = overlaps[rows, columns]
        score_error = float(np.max(np.abs(scores[rows] - result.scores[columns]))) if len(rows) else 0.0
        box_error = float(np.max(np.abs(boxes[rows] - result.boxes[columns]))) if len(rows) else 0.0
        passed = (preprocess_error < 3e-5 and len(labels) == len(result.class_ids)
                  and bool(np.all(ious >= 0.99)) and score_error < 0.01 and box_error < 1.0)
        records.append({
            "image": Path(path).name, "reference_instances": len(labels), "openvino_instances": len(result.class_ids),
            "max_preprocess_error": preprocess_error, "matched_mask_iou": ious.tolist(),
            "max_confidence_error": score_error, "max_box_error_pixels": box_error, "passed": passed,
        })
    report = {
        "reference": "PyTorch CPU FP32 predict; foreground classes only",
        "candidate": "OpenVINO CPU FP32, standalone NumPy preprocessing/postprocessing",
        "scope": "Export consistency on supplied images; not a dataset accuracy or remote GPU benchmark",
        "passed": all(record["passed"] for record in records), "images": records,
    }
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    # Third-party exporter logs include Unicode symbols unsupported by some Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "runs/segmentation/checkpoint_best_ema.pth")
    parser.add_argument("--output", type=Path, help="New or empty deployment directory")
    parser.add_argument("--verify-image", type=Path, nargs="+", help="Compare exported predictions against PyTorch")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if args.verify_image:
        args.verify_image = [path.expanduser().resolve() for path in args.verify_image]
        for path in args.verify_image:
            if not path.is_file():
                parser.error(f"Verification image does not exist: {path}")
    if args.output and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be new or empty; use a new directory for each model version")

    import openvino as ov
    import torch
    from rfdetr import RFDETR

    torch.set_num_threads(1)
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    model = RFDETR.from_checkpoint(checkpoint, device="cpu", gradient_checkpointing=False)
    config = model.model_config
    if not config.segmentation_head or config.num_channels != 3:
        parser.error("Expected an RGB instance segmentation checkpoint")
    classes = list(model.class_names)
    if not classes or config.num_classes != len(classes) or any(not name for name in classes):
        parser.error("Expected a fine-tuned model with contiguous class IDs and saved class names")
    digest = sha256(checkpoint)
    size = int(config.resolution)
    output = (args.output or PROJECT_DIR / "deployment" / f"segmentation-{digest[:12]}-{size}").expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Deployment directory is not empty; choose a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {type(model).__name__}, {size}x{size}, classes={classes}", flush=True)
    with TemporaryDirectory(prefix="export_", dir=output) as work:
        # The automatically cleaned directory must remain within this deployment directory.
        if Path(work).resolve().parent != output:
            raise RuntimeError("Export work directory escaped the deployment directory")
        exported = model.export(format="openvino", output_dir=work, output_name="source",
                                openvino_precision="float32", shape=(size, size), batch_size=1)
        network = ov.Core().read_model(str(exported))
        network.reshape([1, 3, size, size])
        if len(network.outputs) != 3:
            raise RuntimeError("Expected segmentation boxes, logits and masks outputs")
        shapes = [list(port.shape) for port in network.outputs]
        if (shapes[0][-1] != 4 or shapes[1][-1] != len(classes) + 1 or len(shapes[2]) != 4):
            raise RuntimeError(f"Unexpected output signature: {shapes}")
        ov.save_model(network, str(output / "model.xml"), compress_to_fp16=False)
        del network
        gc.collect()
    metadata = {
        "schema_version": 1, "task": "instance_segmentation", "model_type": type(model).__name__,
        "model_id": f"{digest[:12]}-{size}", "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_name": checkpoint.name, "checkpoint_sha256": digest,
        "versions": {name: version(name) for name in ("rfdetr", "torch", "openvino", "numpy")},
        "storage_precision": "float32", "classes": classes, "num_select": int(config.num_select),
        "default_threshold": args.threshold,
        "input": {"shape": [1, 3, size, size], "dtype": "float32", "color": "RGB", "scale": "1/255",
                  "mean": list(model.means), "std": list(model.stds), "resize": "bilinear_half_pixel_no_antialias"},
        "outputs": [{"index": index, "meaning": name, "shape": shape}
                    for index, (name, shape) in enumerate(zip(("boxes_cxcywh", "class_logits", "mask_logits"), shapes))],
        "files": {name: sha256(output / name) for name in ("model.xml", "model.bin")},
        "verification": {"status": "not_run"},
    }
    for source, destination in (("infer_segmentation.py", "infer_segmentation.py"),
                                ("requirements-deploy.txt", "requirements-deploy.txt"),
                                ("DEPLOYMENT.md", "README.md")):
        shutil.copyfile(PROJECT_DIR / source, output / destination)
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.verify_image:
        print("Checking export consistency against a fresh PyTorch model...", flush=True)
        del model
        gc.collect()
        reference = RFDETR.from_checkpoint(checkpoint, device="cpu", gradient_checkpointing=False)
        report = verify(reference, output, args.verify_image, args.threshold)
        metadata["verification"] = {"status": "passed" if report["passed"] else "failed", "report": "verification.json"}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if not report["passed"]:
            parser.exit(1, f"Export consistency check failed; inspect {output / 'verification.json'}\n")
        print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    print(f"Deployment ready: {output}", flush=True)
    print("Copy this entire directory to the remote machine; run its infer_segmentation.py.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
