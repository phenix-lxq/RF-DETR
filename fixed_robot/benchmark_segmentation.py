import argparse
import gc
import json
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image, ImageDraw
from rfdetr import RFDETRSegNano
from rfdetr.export.inference import OpenVINOInference
from rfdetr.models.postprocess import PostProcess
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import functional as transforms

from benchmark_openvino import stats


def preprocess(image, resolution):
    tensor = transforms.resize(transforms.to_tensor(image), [resolution, resolution], antialias=True)
    tensor = transforms.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return np.ascontiguousarray(tensor.unsqueeze(0).numpy())


def load_samples(root):
    dataset = root / "datasets/coco128-seg"
    if not dataset.exists():
        with zipfile.ZipFile(root / "coco128-seg.zip") as archive:
            archive.extractall(root / "datasets")
    category_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                    44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                    88, 89, 90]
    paths = sorted((dataset / "images/train2017").glob("*.jpg"))[:32]
    assert len(paths) == 32
    samples = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        masks, boxes, labels = [], [], []
        for line in (dataset / "labels/train2017" / (path.stem + ".txt")).read_text().splitlines():
            values = [float(value) for value in line.split()]
            points = np.array(values[1:]).reshape(-1, 2) * [width, height]
            mask = Image.new("L", image.size)
            ImageDraw.Draw(mask).polygon([tuple(point) for point in points], fill=1)
            masks.append(np.array(mask, dtype=bool))
            boxes.append([*points.min(axis=0), *points.max(axis=0)])
            labels.append(category_ids[int(values[0])])
        target = dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                      labels=torch.tensor(labels, dtype=torch.long),
                      masks=torch.from_numpy(np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool)))
        samples.append((path.name, image, target))
    return samples


def visualize(samples, predictions, path):
    canvas = Image.new("RGB", (960, 4 * 270), "white")
    colors = np.array([[255, 90, 60], [40, 220, 100], [70, 130, 255], [230, 80, 220]])
    for index, ((name, image, target), prediction) in enumerate(zip(samples[:4], predictions[:4])):
        for column, masks in enumerate((target["masks"], prediction["masks"][prediction["scores"] > 0.5])):
            overlay = np.array(image).copy()
            for mask_index, mask in enumerate(masks):
                selected = mask.numpy().astype(bool)
                overlay[selected] = (overlay[selected] * 0.5 + colors[mask_index % len(colors)] * 0.5).astype(np.uint8)
            tile = Image.fromarray(overlay)
            tile.thumbnail((480, 240))
            canvas.paste(tile, (column * 480, index * 270 + 25))
            ImageDraw.Draw(canvas).text((column * 480 + 5, index * 270 + 5),
                                        f'{name}: {"labels" if column == 0 else "prediction > 0.5"}', fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=312)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1 or args.resolution < 12 or args.resolution % 12:
        parser.error("rounds and iterations must be positive; resolution must be a positive multiple of 12")
    root = Path(__file__).resolve().parent
    output = root / f"segmentation-benchmark-{args.resolution}"
    output.mkdir(exist_ok=True)
    samples = load_samples(root)
    inputs = [preprocess(image, args.resolution) for _, image, _ in samples]
    decoder = PostProcess(num_select=100)

    def decode(raw, image, threshold=None):
        assert len(raw) == 3
        prediction = decoder(dict(zip(("pred_boxes", "pred_logits", "pred_masks"),
                                      [torch.from_numpy(value) for value in raw])),
                             torch.tensor([[image.height, image.width]]), score_threshold=threshold)[0]
        prediction["masks"] = prediction["masks"].squeeze(1)
        return prediction

    model = RFDETRSegNano(device="cpu", resolution=args.resolution)
    xml = output / "rfdetr-seg-nano.xml"
    if not xml.exists():
        model.export(output_dir=str(output), format="openvino", openvino_precision="float32")
        exported = list(output.glob("*.xml"))
        assert len(exported) == 1, exported
        xml = exported[0]
    results = dict(torch=torch.__version__, openvino=ov.__version__, model="RFDETRSegNano",
                   resolution=args.resolution, batch=1, warmup=10, rounds=args.rounds,
                   iterations_per_round=args.iterations, threshold_for_timing=0.5,
                   samples=[name for name, _, _ in samples],
                   dataset="coco128-seg train2017 subset; overlaps pretraining, not held-out accuracy",
                   timing="serial; CPU input to FP32 CPU outputs; full includes preprocessing and original-size mask decoding",
                   backends={})
    del model
    gc.collect()
    for backend in ("torch_xpu_fp32", "torch_xpu_fp16", "ov_gpu_fp16"):
        model = network = session = infer = None
        proof = {}
        started = time.perf_counter()
        try:
            if backend.startswith("torch"):
                model = RFDETRSegNano(device="cpu", resolution=args.resolution)
                network = model.model.model.eval()
                network.export()
                dtype = torch.float16 if backend.endswith("fp16") else torch.float32
                network.to(device="xpu:0", dtype=dtype)
                assert all(parameter.device == torch.device("xpu:0") for parameter in network.parameters())

                def infer(array):
                    with torch.inference_mode():
                        tensor_input = torch.from_numpy(array).to(device="xpu:0", dtype=dtype)
                        raw = network(tensor_input)
                        if not proof:
                            assert all(value.device == torch.device("xpu:0") for value in raw)
                            proof.update(parameter_device=str(next(network.parameters()).device),
                                         parameter_dtype=str(next(network.parameters()).dtype),
                                         input_device=str(tensor_input.device), input_dtype=str(tensor_input.dtype),
                                         output_devices=[str(value.device) for value in raw],
                                         output_dtypes=[str(value.dtype) for value in raw],
                                         output_shapes=[list(value.shape) for value in raw])
                        return tuple(value.float().cpu().numpy() for value in raw)
            else:
                session = OpenVINOInference(xml, device="GPU", inference_precision="f16",
                                            config={"PERFORMANCE_HINT": "LATENCY"})
                proof.update(execution_devices=str(session.compiled_model.get_property("EXECUTION_DEVICES")),
                             precision_hint=str(session.compiled_model.get_property("INFERENCE_PRECISION_HINT")))
                infer = session.infer
            setup_seconds = time.perf_counter() - started
            for index in range(10):
                decode(infer(inputs[index]), samples[index][1], 0.5)
            raw_times, full_times = [], []
            for index in range(args.rounds * args.iterations):
                slot = index % len(samples)
                image = samples[slot][1]
                started = time.perf_counter()
                infer(inputs[slot])
                raw_times.append((time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                decode(infer(preprocess(image, args.resolution)), image, 0.5)
                full_times.append((time.perf_counter() - started) * 1000)
            print(backend, "timing complete; evaluating masks", flush=True)
            metric = MeanAveragePrecision(iou_type=("bbox", "segm"))
            previews = []
            counts = []
            for index, (_, image, target) in enumerate(samples):
                raw = infer(inputs[index])
                assert all(np.isfinite(value).all() for value in raw)
                prediction = decode(raw, image)
                assert prediction["masks"].shape[-2:] == (image.height, image.width)
                metric.update([prediction], [target])
                counts.append(int((prediction["scores"] > 0.5).sum()))
                if index < 4:
                    previews.append(prediction)
            metrics = metric.compute()
            record = dict(setup_seconds=setup_seconds, device_proof=proof,
                          infer_with_transfers=stats(raw_times), preprocess_infer_postprocess=stats(full_times),
                          bbox_map=float(metrics["bbox_map"]), bbox_map50=float(metrics["bbox_map_50"]),
                          segm_map=float(metrics["segm_map"]), segm_map50=float(metrics["segm_map_50"]),
                          detections_above_05_per_image=counts, finite_outputs=True,
                          rounds=[dict(inference=stats(raw_times[start:start + args.iterations]),
                                       full=stats(full_times[start:start + args.iterations]))
                                  for start in range(0, len(raw_times), args.iterations)])
            visualize(samples, previews, output / f"{backend}.jpg")
            results["backends"][backend] = record
            print(backend, json.dumps(record), flush=True)
        except Exception as error:
            traceback.print_exc()
            results["backends"][backend] = dict(error=repr(error))
        (output / "benchmark.json").write_text(json.dumps(results, indent=2))
        model = network = session = infer = None
        gc.collect()
        torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
