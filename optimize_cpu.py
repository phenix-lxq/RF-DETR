"""导出当前分割权重的 CPU 模型；可选用训练图片校准 INT8。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "runs/segmentation/checkpoint_best_ema.pth")
    parser.add_argument("--calibration-data", type=Path, help="用于 INT8 校准的训练图片目录，例如 dataset/train")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=0, help="CPU 输入尺寸，0 沿用训练尺寸；降低尺寸可能影响精度")
    args = parser.parse_args()
    if args.max_images < 1:
        parser.error("--max-images 必须大于 0")
    if args.resolution < 0:
        parser.error("--resolution 不能小于 0")
    if not args.checkpoint.is_file():
        parser.error(f"权重文件不存在：{args.checkpoint}")
    if args.calibration_data is not None and not args.calibration_data.is_dir():
        parser.error(f"校准目录不存在：{args.calibration_data}")

    import torch
    from rfdetr import RFDETR
    from cpu_inference import ensure_fp32_export, model_signature, quantize_int8

    torch.set_num_threads(1)
    model = RFDETR.from_checkpoint(args.checkpoint, device="cpu", gradient_checkpointing=False)
    if not model.model_config.segmentation_head:
        parser.error("请传入实例分割模型权重")
    block_size = model.model_config.patch_size * model.model_config.num_windows
    if args.resolution and args.resolution % block_size:
        parser.error(f"--resolution 必须是 {block_size} 的倍数")
    xml = ensure_fp32_export(model, args.checkpoint, model_signature(model, args.checkpoint, args.resolution))
    print(f"FP32 模型：{xml}", flush=True)
    if args.calibration_data is not None:
        xml = quantize_int8(model, args.checkpoint, args.calibration_data, args.max_images, args.resolution)
        print(f"INT8 模型：{xml}", flush=True)
        print("INT8 会产生数值差异，请对照 FP32 检查分割效果。", flush=True)


if __name__ == "__main__":
    main()
