# 在远端工控机训练实例分割

把 `train_segmentation_xpu.py` 放到远端 `wall_cleaning_system/ai/`，与已有
`train_xpu.py` 放在同一目录。新脚本复用原文件的 XPU 适配和设备/梯度检查。
保留远端现有的 `cleaning` 环境和 `rf-detr/` 源码。

模型使用 `RFDETRSegNanoConfig` 和 `SegmentationTrainConfig`，默认输入 312×312，
batch size 1，FP32。训练设备固定为 `xpu:0`；XPU 不可用时直接报错。
没有指定 `--pretrained` 或 `--weights` 时使用随机初始化，建议微调时添加 `--pretrained`。

## 准备数据

使用实例分割 COCO 导出，每个前景实例需要 `bbox` 和 `segmentation` 多边形或 RLE。
只有检测框的数据不能直接训练实例分割；已有的 random-marker 检测数据也需要增加实例掩膜。
同类别的不同实例分别标注。支持以下两种目录，自动识别：

```text
datasets/my-segmentation/
├── train2017/                         # 训练图片
├── val2017/                           # 验证图片
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

或者使用本地项目相同的 COCO 目录：

```text
datasets/my-segmentation/
├── train/
│   ├── _annotations.coco.json
│   └── ...训练图片
└── valid/
    ├── _annotations.coco.json
    └── ...验证图片
```

图片路径须与标注的 `images[].file_name` 对应，训练和验证类别 ID/名称须一致。
YOLO 分割格式需先转换成上述 COCO 格式。

## 远端运行

以下命令在远端 `wall_cleaning_system/` 下执行，示例数据位置为
`ai/datasets/my-segmentation/`，请替换为实际目录。

```bash
conda activate cleaning

# 只检查图片引用和标注基本结构，不下载权重、不训练
python ai/train_segmentation_xpu.py --dataset ai/datasets/my-segmentation --check-data

# 每轮 2 个训练 batch、1 个验证 batch，先验证 XPU 上的分割反向传播
python ai/train_segmentation_xpu.py --dataset ai/datasets/my-segmentation --output ai/segmentation-xpu-check --pretrained

# 通过小批量验证后，先运行完整 1 轮
python ai/train_segmentation_xpu.py --dataset ai/datasets/my-segmentation --output ai/segmentation-xpu-1epoch --pretrained --full-epoch --epochs 1

# 再根据验证集效果调整轮数和学习率，下面是 50 轮的起点示例
python ai/train_segmentation_xpu.py --dataset ai/datasets/my-segmentation --output ai/segmentation-xpu --pretrained --full-epoch --epochs 50 --lr 0.00001
```

需要至少两个优化步骤才能通过训练验证；极小数据集可用 `--epochs 2`。
如果前两个 batch 都是无目标图片，掩膜梯度检查可能没有证据，可用完整 1 轮验证。
默认无 EMA，保留 RF-DETR 自带评估与最佳模型回调。用验证集 mask mAP 判断分割效果。

输出目录包含 `baseline.json`、`last.ckpt`、`device-proof.json`，以及当前 RF-DETR
版本默认保存的最佳模型文件。`last.ckpt` 是训练状态文件；最佳权重文件名以远端实际输出为准，
远端原 README 中为 `checkpoint_best_regular.pth`。

`device-proof.json` 检查模型、输入、目标、梯度在 XPU，全部已生成梯度有限，
并要求分割头至少产生一次非零梯度。完成短跑仅说明这条训练路径能执行，
完整数据训练时间、内存占用和实际分割精度需要另外测量。

## 验证范围

本地已验证脚本参数、COCO 数据检查、官方数据加载器实际解码示例掩膜，
以及配置/回调兼容性；未初始化模型或在远端 Intel XPU 上执行训练。
原远端 README 证明的是检测训练和分割推理，尚未证明分割训练兼容。
新脚本对两种 `RFDETRModelModule` 构造接口做了适配；真实 XPU 算子支持、
内存和远端依赖版本仍需通过上述小批量训练确认。OpenVINO 用于训练后的推理部署。
