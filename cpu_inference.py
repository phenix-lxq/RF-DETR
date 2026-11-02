"""使用官方导出 API 加速 CPU；保留 RF-DETR 原有预处理和后处理。"""

from __future__ import annotations

import hashlib
import gc
from importlib.metadata import version
import json
from pathlib import Path
import threading
from tempfile import TemporaryDirectory

import numpy as np
import torch


def model_signature(model, checkpoint, resolution=None):
    with Path(checkpoint).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    signature = {
        "checkpoint_sha256": digest,
        "resolution": int(model.model_config.resolution),
        "classes": list(model.class_names),
        "rfdetr": version("rfdetr"),
        "openvino": version("openvino"),
        "recipe": 1,
    }
    if resolution and resolution != signature["resolution"]:
        signature["inference_resolution"] = int(resolution)
    return signature


def export_paths(checkpoint, precision, resolution=None):
    checkpoint = Path(checkpoint)
    root = checkpoint.parent / "openvino" / checkpoint.stem
    if resolution is not None:
        root = root / f"input_{resolution}"
    return root / f"model_{precision}.xml", root / f"model_{precision}.json"


def cached_export(checkpoint, precision, signature):
    xml, metadata = export_paths(checkpoint, precision, signature.get("inference_resolution"))
    if not xml.is_file() or not xml.with_suffix(".bin").is_file() or not metadata.is_file():
        return None
    try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return xml if saved.get("signature") == signature and saved.get("precision") == precision else None


def save_metadata(path, signature, precision, **details):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"signature": signature, "precision": precision, **details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_fp32_export(model, checkpoint, signature=None):
    import openvino as ov

    signature = signature or model_signature(model, checkpoint)
    cached = cached_export(checkpoint, "fp32", signature)
    if cached is not None:
        return cached
    xml, metadata = export_paths(checkpoint, "fp32", signature.get("inference_resolution"))
    size = signature.get("inference_resolution", signature["resolution"])
    print("首次为当前权重生成 OpenVINO FP32 模型，后续启动复用缓存。", flush=True)
    xml.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="export_", dir=xml.parent) as temporary_dir:
        exported = model.export(
            format="openvino", output_dir=temporary_dir, output_name=xml.stem,
            openvino_precision="float32", shape=(size, size),
        )
        network = ov.Core().read_model(str(exported))
        # 当前官方导出的 IR 仍带动态维度；固定 batch 和训练分辨率让 CPU 充分优化。
        network.reshape([1, 3, size, size])
        # Windows 会锁定 read_model 映射的 .bin，必须另存，再释放映射。
        ov.save_model(network, str(xml), compress_to_fp16=False)
        del network
        gc.collect()
    save_metadata(metadata, signature, "fp32", checkpoint=str(Path(checkpoint).resolve()))
    return xml


class OpenVINOForward(torch.nn.Module):
    """只替换模型前向计算，沿用官方 predict 的缩放、归一化、框和掩膜解码。

    与当前 rfdetr 1.11.2 的 inference_model 接口适配；升级后需重新验证。
    """

    def __init__(self, xml, threads=0):
        super().__init__()
        import openvino as ov

        core = ov.Core()
        network = core.read_model(str(xml))
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32",
            "CACHE_DIR": str(Path(xml).parent / "compiled_cache"),
        }
        if threads:
            config["INFERENCE_NUM_THREADS"] = threads
        self.compiled = core.compile_model(network, "CPU", config)
        self.request = self.compiled.create_infer_request()
        self.lock = threading.Lock()
        self.threads = int(self.compiled.get_property("INFERENCE_NUM_THREADS"))
        self.input_resolution = int(self.compiled.input(0).shape[-1])
        if len(self.compiled.outputs) != 3:
            raise ValueError("OpenVINO 分割模型应输出 boxes、logits、masks 三个张量")

    def forward(self, batch):
        if batch.device.type != "cpu" or batch.dtype != torch.float32:
            raise ValueError("OpenVINO 适配器只接受 CPU float32 输入")
        with self.lock:
            self.request.infer({0: np.ascontiguousarray(batch.detach().numpy())})
            # InferRequest 下次会重用输出内存，复制后才能交给官方后处理。
            return tuple(torch.from_numpy(self.request.get_output_tensor(i).data.copy()) for i in range(3))


def enable_openvino(model, checkpoint, *, precision="fp32", threads=0, resolution=None):
    signature = model_signature(model, checkpoint, resolution)
    if precision == "fp32":
        xml = ensure_fp32_export(model, checkpoint, signature)
    else:
        xml = cached_export(checkpoint, "int8", signature)
        if xml is None:
            raise ValueError(
                "没有与当前权重匹配的 INT8 模型。请先运行："
                'python optimize_cpu.py --calibration-data "你的数据集/train"'
                + (f" --resolution {resolution}" if resolution else "")
            )
    adapter = OpenVINOForward(xml, threads)
    model.inference(compile=False, dtype=torch.float32)
    model.model.inference_model = adapter
    # predict(shape=...) 必须与导出的输入尺寸一致；原训练配置和权重不变。
    model._optimized_resolution = adapter.input_resolution
    # PyTorch 仅处理少量缩放和掩膜运算，避免与 OpenVINO 争用大量线程。
    torch.set_num_threads(1)
    print(f"CPU 引擎：OpenVINO {precision.upper()}；计算线程：{adapter.threads}；模型：{xml}", flush=True)
    return xml


def quantize_int8(model, checkpoint, calibration_root, max_images=100, resolution=None):
    import nncf
    import openvino as ov
    from PIL import Image
    from torchvision.transforms import functional as F

    signature = model_signature(model, checkpoint, resolution)
    fp32_xml = ensure_fp32_export(model, checkpoint, signature)
    calibration_root = Path(calibration_root).expanduser().resolve()
    annotation = calibration_root / "_annotations.coco.json"
    if annotation.is_file():
        entries = json.loads(annotation.read_text(encoding="utf-8"))["images"]
        paths = [calibration_root / item["file_name"] for item in entries]
    else:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(path for path in calibration_root.rglob("*") if path.suffix.lower() in extensions)
    paths = paths[:max_images]
    if not paths:
        raise ValueError(f"校准目录中没有图片：{calibration_root}")
    size = signature.get("inference_resolution", signature["resolution"])
    calibration = []
    for path in paths:
        with Image.open(path) as image:
            tensor = F.to_tensor(image.convert("RGB"))
        tensor = F.resize(tensor, [size, size], antialias=False)
        tensor = F.normalize(tensor, model.means, model.stds)
        calibration.append(np.ascontiguousarray(tensor.unsqueeze(0).numpy()))
    print(f"仅使用提供的 {len(paths)} 张图片校准 INT8；不更新原始训练权重。", flush=True)
    network = ov.Core().read_model(str(fp32_xml))
    quantized = nncf.quantize(
        network, nncf.Dataset(calibration), subset_size=len(calibration),
        preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.CPU,
        model_type=nncf.ModelType.TRANSFORMER,
    )
    xml, metadata = export_paths(checkpoint, "int8", signature.get("inference_resolution"))
    # 计算密集层采用 INT8；不支持或敏感的算子仍保留浮点计算。
    ov.save_model(quantized, str(xml), compress_to_fp16=False)
    save_metadata(
        metadata, signature, "int8", checkpoint=str(Path(checkpoint).resolve()),
        calibration_images=[str(path) for path in paths], nncf=version("nncf"),
    )
    return xml
