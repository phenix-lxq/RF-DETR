# 目标检测数据集格式示例

本目录的 `example/coco` 和 `example/yolo` 提供同一组 384×256 合成图片的两种标注格式。
每种格式的 train、valid、test 各有一张不同图片，供整理数据时参照；正式训练应使用真实标注数据。

## COCO 格式

```text
detection/
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
[对应图片](example/coco/train/sample_train.png) 查看实际内容。

逐字段的中文行内注释见 [中文注释版 JSONC](example/coco/train/_annotations.coco.jsonc)。
train、valid、test 的 JSON 旁均有同名 `.jsonc` 对照文件；训练读取 `.json`。
标准 JSON 不支持注释，因此中文说明保存在 JSONC 中。

该图片中有一个 `part` 类目标，标注为：

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [64, 48, 192, 144],
  "area": 27648,
  "iscrowd": 0
}
```

- `bbox` 为 `[左上角 x, 左上角 y, 宽度, 高度]`，单位是像素。
- 这里目标框从 `(64, 48)` 延伸至 `(256, 192)`。
- `image_id` 对应 `images[].id`，`category_id` 对应 `categories[].id`。
- 图片放置位置与 `images[].file_name` 一致；每份 JSON 内的图片和标注 ID 应各自唯一。
- 同一张图片有多个目标时，在 `annotations` 中添加多条记录。

## YOLO 格式

打开 [data.yaml](example/yolo/data.yaml) 和
[完整标签示例](example/yolo/train/labels/sample_train.txt)。

```text
detection/
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

每个目标占一行，格式如下：

```text
class_id x_center y_center width height
0 0.41666667 0.46875000 0.50000000 0.56250000
```

第一行是字段说明；实际 `.txt` 文件只写第二行这样的数字。
类别从 0 开始，横坐标和宽度除以图片宽度，纵坐标和高度除以图片高度。
图片与标签的文件名主体必须一致。

## 检查示例

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --data .\dataset\detection\example\coco --check --run-test
.\.venv\Scripts\python.exe .\train_detection.py --data .\dataset\detection\example\yolo --check --run-test
```

实际 COCO 数据放入本目录的 train、valid、test；使用 YOLO 时再添加 data.yaml 及 images、labels 子目录。
