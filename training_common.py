"""三个训练脚本共用的命令行参数、数据检查和配置。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
TASKS = ("detection", "segmentation", "keypoints")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def batch_size(value: str) -> int | str:
    return "auto" if value == "auto" else positive_int(value)


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数值")
    return number


def parse_args(task: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"RF-DETR {task} 训练脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(task=task)
    parser.add_argument("--data", type=Path, help=f"数据集根目录，默认 dataset/{task}")
    parser.add_argument("--output", type=Path, help=f"结果目录，默认 runs/{task}")
    parser.add_argument("--epochs", type=positive_int, default=100, help="训练总轮数")
    parser.add_argument("--batch-size", type=batch_size, default=1, help="每批图片数，或 auto")
    parser.add_argument(
        "--grad-accum-steps", type=positive_int, default=16,
        help="梯度累积次数；batch-size=auto 时由官方 API 自动决定",
    )
    parser.add_argument(
        "--lr", type=positive_float, default=1e-5 if task == "keypoints" else 1e-4,
        help="学习率",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="训练设备")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载进程数，Windows 建议先用 0")
    parser.add_argument("--resolution", type=positive_int, help="输入分辨率，省略时使用模型默认值")
    parser.add_argument("--resume", type=Path, help="从完整训练检查点 last.ckpt 恢复")
    parser.add_argument("--tensorboard", action="store_true", help="开启 TensorBoard 日志")
    parser.add_argument("--run-test", action="store_true", help="训练结束后评估 test 数据")
    parser.add_argument("--check", action="store_true", help="检查数据和配置，不下载权重、不开始训练")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers 不能小于 0")
    if args.device == "cpu" and args.batch_size == "auto":
        parser.error("CPU 训练需要指定整数 --batch-size")
    args.data = (args.data or PROJECT_DIR / "dataset" / args.task).expanduser().resolve()
    args.resume = args.resume.expanduser().resolve() if args.resume else None
    # 恢复时默认沿用检查点目录，保留最佳模型跟踪状态。
    args.output = (
        args.output or (args.resume.parent if args.resume else PROJECT_DIR / "runs" / args.task)
    ).expanduser().resolve()
    return args


def check_coco_annotations(root: Path, task: str, run_test: bool) -> None:
    """在下载模型前检查 COCO 文件、图片引用和任务需要的标注。"""
    splits = ("train", "valid", "test") if run_test else ("train", "valid")
    for split in splits:
        annotation_file = root / split / "_annotations.coco.json"
        if not annotation_file.is_file():
            raise ValueError(f"缺少标注文件：{annotation_file}")
        with annotation_file.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"COCO 标注必须是 JSON 对象：{annotation_file}")
        images = data.get("images")
        categories = data.get("categories")
        annotations = data.get("annotations")
        if not images or not categories or not isinstance(annotations, list):
            raise ValueError(f"标注需要非空 images、categories 和 annotations 列表：{annotation_file}")
        image_ids = {image["id"] for image in images}
        categories_by_id = {category["id"]: category for category in categories}
        category_ids = set(categories_by_id)
        for image in images:
            image_path = root / split / image["file_name"]
            if not image_path.is_file():
                raise ValueError(f"标注引用的图片不存在：{image_path}")
        foreground = 0
        for annotation in annotations:
            if annotation["image_id"] not in image_ids or annotation["category_id"] not in category_ids:
                raise ValueError(f"标注引用了不存在的 image_id 或 category_id：{annotation_file}")
            if annotation.get("iscrowd", 0):
                continue
            foreground += 1
            box = annotation.get("bbox", [])
            if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"目标框应为 [x, y, 正宽度, 正高度]：{annotation_file}")
            if task == "segmentation" and not annotation.get("segmentation"):
                raise ValueError(f"实例分割需要 segmentation 多边形或掩膜标注：{annotation_file}")
            if task == "keypoints":
                points = annotation.get("keypoints", [])
                if not points or len(points) % 3:
                    raise ValueError(f"关键点标注需要 [x, y, v] 三元组：{annotation_file}")
                category = categories_by_id[annotation["category_id"]]
                names = category.get("keypoints")
                if names and len(points) != 3 * len(names):
                    raise ValueError(f"关键点坐标数量与 categories.keypoints 定义不一致：{annotation_file}")
        if not foreground:
            raise ValueError(f"该划分没有可用的目标标注，请先放入已标注的数据：{annotation_file}")
        print(f"{split}: {len(images)} 张图片，{foreground} 个目标")


def prepare_configs(
    args: argparse.Namespace,
    model_config_class,
    train_config_class,
    *,
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
):
    from rfdetr.datasets import detect_roboflow_format

    dataset_format = detect_roboflow_format(args.data)
    model_kwargs = {"device": args.device, "gradient_checkpointing": True}
    train_kwargs = dict(
        dataset_dir=str(args.data), output_dir=str(args.output),
        epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        lr=args.lr, device=args.device, num_workers=args.num_workers,
        multi_scale=False, expanded_scales=False, amp_dtype="auto",
        tensorboard=args.tensorboard, wandb=False, run_test=args.run_test,
    )
    if dataset_format == "coco":
        check_coco_annotations(args.data, args.task, args.run_test)
    model_kwargs.update(model_overrides or {})
    train_kwargs.update(train_overrides or {})
    model_config = model_config_class(**model_kwargs)
    if args.resolution is not None:
        divisor = model_config.patch_size * model_config.num_windows
        if args.resolution % divisor:
            raise ValueError(f"该模型的 --resolution 必须是 {divisor} 的倍数")
        model_config.resolution = args.resolution
        # 分辨率交给高级 API 更新，以保持位置编码和推理配置一致。
        train_kwargs["resolution"] = args.resolution
    if args.resume:
        train_kwargs["resume"] = str(args.resume)
    config_kwargs = {key: value for key, value in train_kwargs.items() if key not in ("device", "resolution")}
    config_kwargs["accelerator"] = "gpu" if args.device == "cuda" else "cpu"
    train_config = train_config_class(**config_kwargs)
    print(f"任务：{args.task}；分辨率：{model_config.resolution}")
    print(f"数据：{args.data}\n输出：{args.output}")
    return model_kwargs, train_kwargs, model_config, train_config


def check_dataset_loading(model_config, train_config, run_test: bool) -> None:
    from rfdetr.training import RFDETRDataModule

    # 预检查固定使用 CPU 加载，每个划分取一个 batch，不探测自动批量大小。
    check_config = train_config.model_copy(update={
        "batch_size": 1, "grad_accum_steps": 1, "num_workers": 0,
        "accelerator": "cpu", "pin_memory": False,
    })
    data = RFDETRDataModule(model_config, check_config)
    data.setup("fit")
    for name, loader in (("train", data.train_dataloader()), ("valid", data.val_dataloader())):
        if not len(loader.dataset):
            raise ValueError(f"{name} 数据集为空")
        next(iter(loader))
        print(f"{name} 样本加载成功")
    if run_test:
        data.setup("test")
        loader = data.test_dataloader()
        if not len(loader.dataset):
            raise ValueError("test 数据集为空")
        next(iter(loader))
        print("test 样本加载成功")
    print("检查通过。尚未下载模型权重或开始训练。")


def validate_paths(args: argparse.Namespace) -> None:
    if not args.data.is_dir():
        raise ValueError(f"数据集目录不存在：{args.data}")
    if not (args.data / "train" / "_annotations.coco.json").is_file() and not any(
        (args.data / name).is_file() for name in ("data.yaml", "data.yml")
    ):
        raise ValueError(
            f"数据集尚未就绪：{args.data}\n"
            "请放入 COCO 标注 train/_annotations.coco.json，或 YOLO 配置 data.yaml；详见 README.md。"
        )
    if args.resume and not args.resume.is_file():
        raise ValueError(f"恢复训练的检查点不存在：{args.resume}")


def main(
    task: str, runner: Callable[[argparse.Namespace], None], argv: list[str] | None = None,
) -> int:
    args = parse_args(task, argv)
    try:
        validate_paths(args)
        if not args.check and args.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("CUDA 不可用，请检查 GPU 版 PyTorch，或使用 --device cpu")
        runner(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0
