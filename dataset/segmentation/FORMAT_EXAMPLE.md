# 实例分割数据集格式示例

本目录的 `example/coco` 和 `example/yolo` 提供同一组 384×256 合成图片的两种标注格式。
每个划分各有一张不同图片，每张图片有一个四边形实例，类别为 `part`。
示例供对照格式，正式训练应使用真实标注数据。

## COCO 格式

```text
segmentation/
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
逐字段及轮廓坐标的中文说明见 [中文注释版 JSONC](example/coco/train/_annotations.coco.jsonc)。
train、valid、test 的 JSON 旁均有同名 `.jsonc` 对照文件；训练读取 `.json`。
标准 JSON 不支持注释，因此中文说明保存在 JSONC 中。

其中一个实例的标注如下：

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [64, 48, 192, 144],
  "area": 21168,
  "iscrowd": 0,
  "segmentation": [
    [64, 48, 256, 66, 232, 192, 88, 174]
  ]
}
```

- `segmentation` 内每组列表表示一个多边形，写作 `[x1, y1, x2, y2, ...]`，单位为像素。
- 此例四个顶点依次为 `(64,48)`、`(256,66)`、`(232,192)`、`(88,174)`，由最后一点闭合回第一点。
- `bbox` 是包围整个实例的目标框，格式为 `[x, y, width, height]`。
- `area` 是实例多边形的面积；这里是 21168，而目标框面积为 27648。
- 一个实例可包含多个多边形；也可以采用 COCO RLE 掩膜表示。
- 同一类别的不同实例应分别写入 `annotations`，使用不同的标注 ID。
- `image_id`、`category_id` 分别关联 `images` 和 `categories` 中的记录。

## YOLO-Seg 格式

打开 [data.yaml](example/yolo/data.yaml) 和
[完整标签示例](example/yolo/train/labels/sample_train.txt)。

```text
segmentation/
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

每个实例占一行：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

实际标签中只填写数字；类别从 0 开始，所有顶点坐标归一化到 0～1：
`x_norm = x / 图片宽度`，`y_norm = y / 图片高度`。
至少需要三个顶点，图片与标签文件名主体一致。

## 检查示例

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\train_segmentation.py --data .\dataset\segmentation\example\coco --check --run-test
.\.venv\Scripts\python.exe .\train_segmentation.py --data .\dataset\segmentation\example\yolo --check --run-test
```

实际 COCO 数据放入本目录的 train、valid、test；使用 YOLO 时再添加 data.yaml 及 images、labels 子目录。
