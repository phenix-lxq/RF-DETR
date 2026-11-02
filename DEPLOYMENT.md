# 本地训练，远端部署实例分割

`export_segmentation.py` 在本地读取已训练权重，导出 OpenVINO IR 并保存模型实际类别、
输入尺寸、预处理参数、输出顺序和校验值。`infer_segmentation.py` 在远端使用 OpenVINO
直接推理，不依赖 RF-DETR、PyTorch、CUDA，也不再加载官方默认权重。

当前已用现有 `checkpoint_best_ema.pth` 生成 `deployment/segmentation-v1/`，
类别为 `can`、`Earbuds`、`Mouse`，输入 312×312，可以直接复制这份部署包。
两张真实测试图片上，CPU FP32 与 PyTorch 匹配掩膜的 IoU 均为 1.0；
本机 Intel UHD GPU 的 FP16 与 CPU FP32 匹配掩膜 IoU 均高于 0.997。
这些结果不代表远端工控机已经完成验证；本地 GPU 对照记录在
`deployment-validation/backend-comparison.json`，视频入口也已用本地文件检查。

## 1. 本地导出

在 `I:\RF-DETR` 的 PowerShell 中执行。下面示例再次导出现有三类别权重，使用新目录 `segmentation-v2`，
并以两张真实测试图片检查导出一致性：

```powershell
.\.venv\Scripts\python.exe .\export_segmentation.py --checkpoint .\runs\segmentation\checkpoint_best_ema.pth --output .\deployment\segmentation-v2 --verify-image "I:\rf-detr数据集\dataset\test\img_13.jpg" "I:\rf-detr数据集\dataset\test\img_14.jpg"
```

后续训练完成后换成实际最佳分割权重，并使用新的输出目录。
省略 `--output` 时按权重 SHA-256 和分辨率生成目录名。输出目录必须为空，避免旧模型与新配置混用。
脚本从检查点恢复类别与分辨率，支持连续类别 ID 的自定义 RGB 实例分割模型。

`--verify-image` 比较 PyTorch CPU FP32 与独立部署入口的 OpenVINO CPU FP32 结果：
实例数量与类别匹配、掩膜 IoU 至少 0.99、置信度误差小于 0.01、框误差小于 1 像素。
它检查模型转换一致性，不替代真实测试集准确率评估或远端 GPU FP16 验证。

导出包包含：

```text
segmentation-v1/
├── model.xml
├── model.bin
├── metadata.json
├── infer_segmentation.py
├── requirements-deploy.txt
├── README.md
└── verification.json          # 指定 --verify-image 时生成
```

FP32 存储的 IR 可由远端 GPU 以 FP16 计算，无需在本地改成 XPU 训练。
原始 `.pth` 保留在本地，整个部署包复制到远端即可。不要复制本地 OpenVINO 编译缓存。

## 2. 远端运行

假设整个包复制到 `wall_cleaning_system/ai/deployment/segmentation-v1/`。
在远端 `wall_cleaning_system` 下执行：

```bash
conda activate cleaning

# 已有 OpenVINO/NumPy/Pillow 时可直接使用；缺依赖时再安装
python -m pip install -r ai/deployment/segmentation-v1/requirements-deploy.txt

# 确认 Intel GPU 可见
python ai/deployment/segmentation-v1/infer_segmentation.py --list-devices

# 替换为实际图片路径。默认使用脚本所在目录的模型
python ai/deployment/segmentation-v1/infer_segmentation.py --image /path/to/test.jpg --device GPU --precision f16 --output ai/segmentation-result
```

第一次编译可能较慢。启动时核对 XML/BIN 校验值、输入输出形状，报告实际执行设备与精度提示。
指定 GPU 后不自动降级到 CPU。CPU 对照可以使用 `--device CPU --precision f32`，指定单张图片得到的
`timings_ms` 不是完整测速结果，正式测速应先预热并遍历代表性图片。

输出为 `overlay.png`、`predictions.json` 和 `masks.npz`。JSON 每个实例包含类别、置信度、
原图像素坐标 `box_xyxy` 和对应掩膜下标；NPZ 包含原图尺寸的布尔掩膜 `[N,H,W]`。
没有目标时返回空数组。

摄像头、视频或 RTSP 输入需要 OpenCV（远端已有时直接复用）：

```bash
python -m pip install "opencv-python-headless>=4.8"
python ai/deployment/segmentation-v1/infer_segmentation.py --source 0 --device GPU --max-frames 30 --output ai/camera-segmentation-check
```

摄像头示例处理 30 帧后退出，保存最后一帧。默认不弹出窗口；不设置帧数限制时用 Ctrl+C 停止。
接入现有机器人程序时建议直接调用 Python 接口，避免每帧加载模型或落盘：

```python
from infer_segmentation import Segmenter
import cv2

segmenter = Segmenter("ai/deployment/segmentation-v1", device="GPU", precision="f16")

# 每帧：摄像头 OpenCV BGR 转 RGB。模型实例在循环外初始化一次。
rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
prediction = segmenter.predict(rgb, threshold=0.5)
for label, score, box, mask in zip(
    prediction.class_ids, prediction.scores, prediction.boxes, prediction.masks
):
    name = segmenter.classes[int(label)]
    # mask 为原图尺寸 bool 数组；按业务需要处理实例区域。
```

导入时将部署目录加入现有应用的 Python 模块搜索路径，或把入口作为本地模块集成。
模型输入使用 RGB、0..1 浮点值、检查点的 mean/std；缩放为 half-pixel 双线性、无抗锯齿。
掩膜先将浮点 logits 双线性放大到原图，再以 0 分界生成布尔掩膜。
置信度使用独立 sigmoid 和全局 top-k，过滤背景槽；不套用 COCO 80 类映射或额外 YOLO NMS。

## 3. 部署验收

把同一批真实测试图片带到远端，先运行 CPU FP32，再运行 GPU FP16。
对照原图、类别、置信度和掩膜边界；检查漏检、误检以及完整处理耗时。
FP16 可能产生数值差异，测试集 mask mAP 和业务阈值需要在远端重新确认。
现有 benchmark 中官方 COCO128-seg 的指标不能代替自定义模型的结果。

纯推理部署无需对齐两边 CUDA/XPU PyTorch 版本，也无需远端具备分割训练能力。
这套部署包以 OpenVINO 2026.4.1 验证；Intel GPU 仍需要远端驱动和 OpenCL/Level Zero 运行环境。
