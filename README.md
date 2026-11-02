# RF-DETR 三任务训练

使用已安装的 RF-DETR 1.11.2 官方高级 API，三个任务分别通过独立脚本训练。
所有命令在项目根目录的 PowerShell 中执行。

| 任务 | 训练脚本 | 默认数据目录 | 默认结果目录 |
| --- | --- | --- | --- |
| 目标检测 | `train_detection.py` | `dataset/detection` | `runs/detection` |
| 实例分割 | `train_segmentation.py` | `dataset/segmentation` | `runs/segmentation` |
| 关键点检测 | `train_keypoints.py` | `dataset/keypoints` | `runs/keypoints` |

三个脚本分别选择模型并调用 `model.train()`，参数解析和数据检查共用 `training_common.py`。

## 数据目录

已创建以下空目录，放入实际图片和对应标注后即可使用：

```text
dataset/
├── detection/
│   ├── train/
│   ├── valid/
│   └── test/
├── segmentation/
│   ├── train/
│   ├── valid/
│   └── test/
└── keypoints/
    ├── train/
    ├── valid/
    └── test/
```

三个任务目录中都提供了格式说明和可读取的合成示例：

| 任务 | 格式说明 | 示例位置 |
| --- | --- | --- |
| 目标检测 | [FORMAT_EXAMPLE.md](dataset/detection/FORMAT_EXAMPLE.md) | `dataset/detection/example/coco`、`example/yolo` |
| 实例分割 | [FORMAT_EXAMPLE.md](dataset/segmentation/FORMAT_EXAMPLE.md) | `dataset/segmentation/example/coco`、`example/yolo` |
| 关键点检测 | [FORMAT_EXAMPLE.md](dataset/keypoints/FORMAT_EXAMPLE.md) | `dataset/keypoints/example/coco`、`example/yolo` |

示例包含图片和完整标注，每种格式的 train、valid、test 各有一张不同的合成图片。
可按各说明中的命令使用 `--data ... --check --run-test` 检查示例。
实际训练数据仍放在任务目录下的 train、valid、test，示例数据仅供格式参照。

每份 COCO 示例标注旁都有 `_annotations.coco.jsonc`，在对应字段和坐标处提供中文行内注释。
标准 JSON 不支持注释，训练继续读取 `_annotations.coco.json`；JSONC 用于阅读对照。

### COCO 格式（推荐）

每个任务的 `train`、`valid` 目录中放置图片和 `_annotations.coco.json`。
图片位置必须与 JSON 的 `images[].file_name` 一致。
例如：

```text
dataset/detection/
├── train/
│   ├── _annotations.coco.json
│   └── image001.jpg
└── valid/
    ├── _annotations.coco.json
    └── image002.jpg
```

- 检测：目标框 `bbox: [x, y, width, height]` 和类别。
- 分割：在目标框基础上提供 `segmentation` 多边形或 COCO RLE 掩膜。
- 关键点：提供目标框、`keypoints: [x, y, v, ...]` 和 `num_keypoints`；
  建议在 `categories[].keypoints` 中明确每个关键点的名字和顺序。
  `v=0` 表示未标注，`v=1` 表示标注但遮挡，`v=2` 表示可见。
  训练、验证和测试应使用一致的类别及关键点定义。

关键点训练会使用官方辅助函数读取类别、关键点数量和左右对称点对。
自定义关键点的 OKS sigma 使用官方辅助函数默认的 `0.1`，不同评估协议的指标不可直接比较。

### YOLO 格式

支持含 `data.yaml` 或 `data.yml` 的数据集，例如：

```text
dataset/detection/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

```yaml
names: [your_class]
nc: 1
train: train/images
val: valid/images
```

将示例类别替换成实际类别。分割需要 YOLO 多边形标签；关键点需要 YOLO pose 标签，
并在 YAML 中声明 `kpt_shape: [关键点数量, 3]`，左右对称时还需提供正确的 `flip_idx`。
脚本根据官方规则自动识别格式；同一目录同时存在两种格式时，COCO 优先。

## 检查数据

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --check
.\.venv\Scripts\python.exe .\train_segmentation.py --check
.\.venv\Scripts\python.exe .\train_keypoints.py --check
```

`--check` 检查配置和数据，使用官方数据加载器读取每个划分的一个 batch，
不初始化模型、不下载预训练权重、不开始训练。COCO 还会检查所有图片引用和任务需要的标注字段。
它不等同于完整遍历并验证全部图片内容。新创建的空目录在放入标注前会明确提示数据尚未就绪。

## 先跑 1 轮

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --epochs 1 --output .\runs\detection_check
.\.venv\Scripts\python.exe .\train_segmentation.py --epochs 1 --output .\runs\segmentation_check
.\.venv\Scripts\python.exe .\train_keypoints.py --epochs 1 --output .\runs\keypoints_check
```

每条命令分别启动一种任务。先完成一个任务的检查，再启动下一个，避免争用 8GB 显存。
默认使用 `RFDETRNano`、`RFDETRSegNano`、`RFDETRKeypointPreview`，
输入分辨率分别为 384、312、576。关键点功能为官方预览版本。

## 正式训练

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --epochs 100
.\.venv\Scripts\python.exe .\train_segmentation.py --epochs 100
.\.venv\Scripts\python.exe .\train_keypoints.py --epochs 100
```

数据默认来自 `dataset/<task>`，结果默认保存到 `runs/<task>`。
首次初始化对应模型会自动下载预训练权重。
8GB 显卡默认使用 batch size 1、梯度累积 16、梯度检查点、混合精度，并关闭多尺度训练。
实际显存需求与图片目标数量有关，1 轮检查用于验证当前数据和硬件能否正常训练。

指定其他数据集、输出目录或参数：

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --data "I:\my_dataset" --output .\runs\experiment1 --epochs 100 --batch-size 2 --grad-accum-steps 8 --tensorboard
```

也可使用 `--batch-size auto`，由官方 API 在 CUDA 上探测 batch size 并决定梯度累积次数。
`--resolution` 可以调整输入分辨率；检测 Nano 要求 32 的倍数，分割 Nano 要求 12 的倍数，关键点要求 24 的倍数。
关键点默认学习率和 backbone 学习率均为 `1e-5`，其他任务学习率为 `1e-4`；可用 `--lr` 调整。

## 恢复训练和结果

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --epochs 100 --resume .\runs\detection\last.ckpt
```

`--epochs` 是训练总轮数。恢复时应保持模型、数据和分辨率等设置一致；
省略 `--output` 时会自动沿用检查点所在目录。
完整恢复使用 `last.ckpt`，预测使用 `checkpoint_best_total.pth`。

手动中断训练时，可能尚未生成 `checkpoint_best_total.pth`，可使用已经保存的 `checkpoint_best_ema.pth`。
评估分割模型并生成测试集标注/预测对照图：

```powershell
.\.venv\Scripts\python.exe .\evaluate_segmentation.py --data "I:\rf-detr数据集\dataset" --checkpoint .\runs\segmentation\checkpoint_best_ema.pth
```

指标和图片默认保存到 `runs/segmentation/evaluation`，可用 `--output` 指定其他目录。

## 摄像头实时测试实例分割

当前训练得到的最佳权重是 `runs/segmentation/checkpoint_best_ema.pth`。
已提供 `camera_segmentation.py`，默认加载该模型，使用 GPU（CUDA FP16）推理，
在画面上叠加掩膜、目标框、类别和置信度。使用 CPU 时加 `--device cpu`。
默认打开编号 1，适用于常见的内置摄像头加外接 USB 摄像头配置。

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\camera_segmentation.py
```

脚本启动后自动打开本机浏览器预览页面 `http://127.0.0.1:8765`。
画面由本机摄像头采集，在本机处理和显示。页面提供“停止测试”和“保存截图”按钮；
也可以在终端按 `Ctrl+C` 停止。仅关闭浏览器页面不会停止程序。
截图保存到 `runs/segmentation/camera`。
现有环境的 OpenCV 不包含 GUI，本脚本使用浏览器预览，无需额外安装窗口组件。

也可以显式指定外接 USB 摄像头编号和阈值：

```powershell
.\.venv\Scripts\python.exe .\camera_segmentation.py --source 1 --threshold 0.5
```

编号由 Windows 的设备枚举顺序决定，1 不保证始终对应 USB 摄像头。
如果未打开正确设备，可尝试 `--source 0` 或 `--source 2`。
摄像头打开失败时，检查 Windows 的摄像头权限，并关闭占用设备的会议或相机应用。
如果预览端口被占用，可用 `--port 8766`；启动时会显示实际地址。
`--no-browser` 可关闭自动打开浏览器，改为自行访问地址。

采集分辨率默认请求 640×480，设备可能返回其他支持的尺寸；启动日志会打印实际尺寸。
模型仍自动缩放到训练时的 312×312 输入，原始画面用于显示。
画面中的 FPS 只统计推理和绘图，不包含采集与浏览器显示开销。

也支持视频文件和 RTSP 地址：

```powershell
.\.venv\Scripts\python.exe .\camera_segmentation.py --source "I:\your_video.mp4"
.\.venv\Scripts\python.exe .\camera_segmentation.py --source "rtsp://camera-address/stream"
```

单张图片测试：

```powershell
.\.venv\Scripts\python.exe .\camera_segmentation.py --image "I:\rf-detr数据集\dataset\test\img_14.jpg" --output .\runs\segmentation\camera\image_test.png
```

使用其他分割权重时指定 `--checkpoint`；需要 CPU 推理时加 `--device cpu`。
完整参数可通过 `camera_segmentation.py --help` 查看。

### CPU 加速

已经在本机 i9-12900HK（14 核、20 线程）上安装并验证 OpenVINO。
当前设备默认为 GPU；显式指定 `--device cpu` 后，自动优先使用与当前训练权重匹配的 OpenVINO INT8 缓存。
没有匹配的 INT8 缓存时使用 OpenVINO FP32；未安装 OpenVINO 时使用 PyTorch。
GPU FP16 可通过 `--device cuda` 使用。CPU 计算使用训练好的分割 Nano 权重，输入仍为 312×312。

```powershell
.\.venv\Scripts\python.exe .\camera_segmentation.py --device cpu --source 1
```

2026-10-29 使用两张测试图及其水平翻转、640×480 画面、预热 8 次后计时 40 帧：

| CPU 推理方式 | 推理和绘图 FPS 中位数 | 本次效果对比 |
| --- | ---: | --- |
| 原 PyTorch FP32 | 5.8（结束后复测 5.6） | 参照 |
| OpenVINO FP32 | 6.7 | 两张原图、缩放图及翻转图的分割掩膜均与参照一致 |
| OpenVINO INT8 混合精度（CPU 自动模式优先） | 11.5 | 分割有轻微变化；一张原图增加一个检测结果 |

这些 FPS 包括预处理、模型、后处理和绘图，不包含采集、JPEG 编码和浏览器显示。
实际摄像头帧率受画面目标数量、系统负载和笔记本温度影响。INT8 的测试规模很小，
仅使用现有 10 张训练图片校准；本次匹配目标的掩膜 IoU 为 0.962～0.997（与 FP32 对比）。
需要原 FP32 效果时指定 `--device cpu --backend openvino`；显式选择当前使用的 INT8：

```powershell
.\.venv\Scripts\python.exe .\camera_segmentation.py --device cpu --backend openvino-int8 --source 1
```

两种 CPU 模型已生成到 `runs/segmentation/openvino/checkpoint_best_ema/`。
原始 `checkpoint_best_ema.pth` 保留；启动时校验该文件的 SHA-256、分辨率、类别和依赖版本，
避免使用其他检查点的转换缓存。首次编译稍慢，后续复用缓存。
需要回到原版 CPU 推理可使用 `--device cpu --backend torch`。
`--cpu-threads` 可手动设定计算线程；默认由 OpenVINO 选择，本机为 6 个计算线程。
实测增加线程和使用 TorchScript 均没有稳定提速，因此未作为默认方案。

补测 OpenVINO FP16：i9-12900HK 的 CPU 插件没有原生 FP16 计算支持，
即使请求 FP16 计算也会返回 FP32。FP16 压缩能将导出权重文件从约 122 MB 缩小到 61 MB，
同轮实测 FP32 约 7.63 FPS、FP16 存储加 FP32 计算约 7.44 FPS，没有速度收益，
因此没有增加 CPU FP16 启动选项。FP16 在支持它的 GPU 上仍可用于加速。

在其他环境安装可选依赖，或更换权重后重新导出：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-cpu.txt
.\.venv\Scripts\python.exe .\optimize_cpu.py
# 只有需要 INT8 时才运行下面的校准；请使用训练图片。
.\.venv\Scripts\python.exe .\optimize_cpu.py --calibration-data "I:\rf-detr数据集\dataset\train"
```

FP32 缓存不匹配时，摄像头脚本会自动重新导出；auto 模式没有匹配的 INT8 缓存时回退到 FP32，
显式指定 `--backend openvino-int8` 而缓存不匹配时会提示重新校准。
不需要重新训练模型。适配器沿用 RF-DETR 官方的预处理和后处理，当前验证版本为 `rfdetr==1.11.2`；
升级 RF-DETR 后需要重新验证适配和输出。

复现后端速度及分割结果对比：

```powershell
.\.venv\Scripts\python.exe .\benchmark_cpu.py --compare --image "I:\rf-detr数据集\dataset\test\img_14.jpg" --extra-images "I:\rf-detr数据集\dataset\test\img_13.jpg" --warmup 8 --iterations 40
```

数据保存在 `runs/segmentation/cpu_benchmark/backend_comparison.json`；该测试不需要打开摄像头。

### 只用 CPU 继续提速：可选降低输入尺寸

已额外导出和校准 288、264、240 三种输入的 INT8 模型，原始权重和默认 312×312 输入保留。
模型文件分别位于 `runs/segmentation/openvino/checkpoint_best_ema/input_<尺寸>/`，缓存中记录输入尺寸，
避免把不同尺寸的模型混用。采集和显示画面仍为 640×480。

本机同轮测试（6 计算线程、预热 8 次、计时 40 帧；推理和绘图，不含采集/JPEG/浏览器）：

| INT8 模型输入 | FPS（按中位耗时） | 两张原图中按类别和掩膜 IoU≥0.5 匹配的真实目标 |
| --- | ---: | ---: |
| 312×312（当前默认） | 13.2 | 6/6 |
| 288×288 | 15.6 | 5/6 |
| 264×264 | 19.0 | 5/6 |
| 240×240 | 21.7 | 5/6 |

312 的测试仍有一个额外检测；低分辨率在一张图上出现将鼠标错分为耳机的情况，
因此没有自动降低默认输入尺寸。以上只对照现有两张测试图片，不能证明其他场景下的精度。
笔记本负载和温度会影响结果，历史不同轮测试的绝对 FPS 会有变化。

如果接受精度和速度的取舍，可以运行：

```powershell
# 264 输入，本轮约 19 FPS。
.\.venv\Scripts\python.exe .\camera_segmentation.py --device cpu --resolution 264 --backend openvino-int8 --cpu-threads 6
# 240 输入，本轮约 22 FPS。
.\.venv\Scripts\python.exe .\camera_segmentation.py --device cpu --resolution 240 --backend openvino-int8 --cpu-threads 6
# 恢复当前的训练输入尺寸：省略 --resolution，或指定 312。
.\.venv\Scripts\python.exe .\camera_segmentation.py --device cpu --resolution 312
```

其他环境或更换检查点后，需要按指定尺寸重新导出和校准，例如：

```powershell
.\.venv\Scripts\python.exe .\optimize_cpu.py --calibration-data "I:\rf-detr数据集\dataset\train" --resolution 264
```

原图标注/预测对照图及逐图结果保存在 `runs/segmentation/cpu_benchmark/resolutions/`。
复现尺寸对比：

```powershell
.\.venv\Scripts\python.exe .\benchmark_cpu_resolutions.py --data "I:\rf-detr数据集\dataset"
```

默认只需要训练集和验证集。使用 `--run-test` 时请准备真实的 test 数据；
COCO 必须有 `test/_annotations.coco.json`。YOLO 缺少 test 时，官方 API 可能回退到验证集，需留意输出提示。

查看全部参数：

```powershell
.\.venv\Scripts\python.exe .\train_detection.py --help
.\.venv\Scripts\python.exe .\train_segmentation.py --help
.\.venv\Scripts\python.exe .\train_keypoints.py --help
```

官方参考：[安装](https://rfdetr.roboflow.com/latest/getting-started/install/) ·
[训练](https://rfdetr.roboflow.com/latest/learn/train/) ·
[数据格式](https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/)

## 在本地训练，部署到远端 Intel GPU

已提供 `export_segmentation.py` 和独立的 `infer_segmentation.py`。
前者将自定义分割权重打包成 OpenVINO XML/BIN、类别与预处理配置；
后者直接在远端 OpenVINO GPU 上推理，不需要 RF-DETR 或 PyTorch。
导出时可对比真实图片的 PyTorch 和 OpenVINO 结果。
完整命令、部署包结构和摄像头/业务接口接入方式见 [DEPLOYMENT.md](DEPLOYMENT.md)。
