# 关键点数据集格式示例

本目录的 `example/coco` 和 `example/yolo` 提供同一组 384×256 合成图片的两种标注格式。
此例使用 `part` 类四边形的四个角作为自定义关键点，展示如何训练人体 17 点以外的定义。
每个划分各有一张不同图片，供对照格式；正式训练应使用真实标注数据。

关键点的顺序固定为：

| 顺序（从 0 开始） | 名称 | 含义 |
| --- | --- | --- |
| 0 | top_left | 左上角 |
| 1 | top_right | 右上角 |
| 2 | bottom_right | 右下角 |
| 3 | bottom_left | 左下角 |

## COCO 格式

```text
keypoints/
├── train/
│   ├── _annotations.coco.json
│   └── your_train_image.png
├── valid/
│   ├── _annotations.coco.json
│   └── your_valid_image.png
└── test/                         # 仅 --run-test 时需要
    ├── _annotations.coco.json
    └── your_test_image.png
```

打开 [完整训练标注示例](example/coco/train/_annotations.coco.json) 和
[对应图片](example/coco/train/sample_train.png)。
逐字段、关键点坐标、可见性及连线的中文说明见
[中文注释版 JSONC](example/coco/train/_annotations.coco.jsonc)。
train、valid、test 的 JSON 旁均有同名 `.jsonc` 对照文件；训练读取 `.json`。
标准 JSON 不支持注释，因此中文说明保存在 JSONC 中。

类别定义中的关键点名称决定坐标排列顺序：

```json
{
  "id": 1,
  "name": "part",
  "supercategory": "object",
  "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
  "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]]
}
```

相应的一个目标标注为：

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [64, 48, 192, 144],
  "area": 27648,
  "iscrowd": 0,
  "keypoints": [64, 48, 2, 256, 66, 2, 232, 192, 2, 88, 174, 2],
  "num_keypoints": 4
}
```

- `keypoints` 按 `[x1, y1, v1, x2, y2, v2, ...]` 排列，坐标单位为像素。
- `v=0`：该点未标注，通常写成 `[0, 0, 0]`；`v=1`：有坐标但被遮挡；`v=2`：可见。
- 每个目标都保留完整的关键点顺序，未标注的点使用占位三元组。
- `num_keypoints` 统计 `v>0` 的点数，坐标列表长度始终为关键点总数乘以 3。
- `skeleton` 描述连线，使用从 1 开始的点序号；左右对称点名用于官方辅助函数推断翻转对应关系。
- `bbox` 使用 `[x, y, width, height]`。`image_id`、`category_id` 必须关联有效的图片和类别。
- 训练、验证和测试必须保持关键点数量、名称及顺序一致。

## YOLO-Pose 格式

打开 [data.yaml](example/yolo/data.yaml) 和
[完整标签示例](example/yolo/train/labels/sample_train.txt)。

```text
keypoints/
├── data.yaml
├── train/
│   ├── images/sample_train.png
│   └── labels/sample_train.txt
├── valid/
│   ├── images/sample_valid.png
│   └── labels/sample_valid.txt
└── test/
    ├── images/sample_test.png
    └── labels/sample_test.txt
```

示例 YAML 中的关键配置：

```yaml
names:
  0: part
nc: 1
train: train/images
val: valid/images
test: test/images
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
kpt_names:
  0: [top_left, top_right, bottom_right, bottom_left]
```

- `kpt_shape: [4, 3]` 表示四个点，每个点三个值 `[x, y, v]`。
- `flip_idx` 使用从 0 开始的索引：水平翻转后 0 与 1 交换、2 与 3 交换。
- 每个目标标签依次为 `class_id xc yc w h x1 y1 v1 ... x4 y4 v4`。
- `xc yc w h` 和关键点坐标归一化到 0～1，可见性仍使用 0、1、2。
- 使用其他关键点数量时，应同时更新 `kpt_shape`、名称、翻转映射及标签中的全部坐标。

## 检查示例

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\train_keypoints.py --data .\dataset\keypoints\example\coco --check --run-test
.\.venv\Scripts\python.exe .\train_keypoints.py --data .\dataset\keypoints\example\yolo --check --run-test
```

检查时会读取到一个类别、四个关键点。实际 COCO 数据放入本目录的 train、valid、test；
使用 YOLO 时再添加 data.yaml 及 images、labels 子目录。
