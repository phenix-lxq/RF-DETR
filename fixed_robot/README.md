# RF-DETR training

The RF-DETR source is in `rf-detr/`. The `cleaning` environment has the
XPU build of PyTorch and the RF-DETR training extras installed.

## Smoke training

`rfdetr-smoke/` is a one-image COCO fixture. Run the completed CPU smoke test
with:

```bash
C:\Users\techpro\miniforge3\envs\cleaning\bin\python -c "import rfdetr; m=rfdetr.RFDETRNano(pretrain_weights=None); m.train(dataset_dir='ai/rfdetr-smoke',dataset_file='coco',device='cpu',epochs=1,batch_size=1,num_workers=0,output_dir='ai/rfdetr-smoke-output',resolution=384,tensorboard=False,wandb=False,use_ema=False,run_test=False,checkpoint_interval=1)"
```

The resulting checkpoint is `rfdetr-smoke-output/checkpoint_best_regular.pth`.

## XPU status

XPU training is verified with PyTorch 2.14.1+xpu, Compute Runtime
25.35.35096.9 and Level Zero loader 1.24.3 on Intel Arc integrated graphics.
From the project root, activate `cleaning` and run:

```bash
conda activate cleaning
python ai/train_xpu.py
```

`train_xpu.py` supplies a Lightning accelerator and an explicit single-device
strategy targeting `xpu:0`. It uses FP32 and fails if XPU is unavailable.
The upstream `RFDETRNano.train(device='xpu')` entry point still falls back to
CPU; use this script for the verified XPU path.

The smoke run uses random initial weights and the synthetic COCO fixture,
two training batches and one validation batch. Device assertions check every
model parameter, input/mask, target tensor, and generated gradient. Gradients
must be finite and not all zero; at least two optimizer steps must complete.
This verifies execution, not detection accuracy or full-dataset performance.

Results are in `rfdetr-xpu-verified/device-proof.json`, with the training
checkpoint at `rfdetr-xpu-verified/verified.ckpt`. The completed run reports
two backward/optimizer steps, 464 gradient tensors and `xpu:0` for the model,
input and gradients. The earlier `rfdetr-xpu-output/` artifacts were produced
by CPU fallback and are not XPU evidence.

The adapter currently supports only one XPU and FP32 training; distributed
training and mixed precision have not been validated.

## Small real-image training run

COCO128 was downloaded from https://ultralytics.com/assets/coco128.zip
(approximately 6.7 MB). Its class names come from the Ultralytics
`ultralytics/cfg/datasets/coco128.yaml` file, saved here as `coco128.yaml`.
`prepare_coco128.py` converts YOLO boxes to COCO annotations and selects a
deterministic, disjoint split with seed 42: 32 training images (242 boxes)
and 8 validation images (53 boxes). Dataset files are under
`datasets/coco128-tiny/`.

```bash
conda activate cleaning
python ai/prepare_coco128.py
python ai/train_xpu.py --dataset ai/datasets/coco128-tiny --output ai/rfdetr-coco128-xpu --full-epoch
```

The full one-epoch run completed 32 backward passes and optimizer updates on
`xpu:0`, followed by validation. Device/gradient evidence, metrics and checkpoint
are saved under `rfdetr-coco128-xpu/`; the captured log is
`logs/coco128-xpu.log`. This run starts from random weights to check training
compatibility; it is not a detection-quality benchmark.

## Pretrained effectiveness experiment

`prepare_coco128.py` now writes `datasets/coco128-effect` with the original
COCO category IDs (including gaps), matching the pretrained detection head.
The old `coco128-tiny` fixture remains available for the earlier smoke run.
The trainer uses `extra_callbacks` to retain RF-DETR's COCO evaluation and
best-checkpoint callbacks alongside the XPU assertions.

```bash
python ai/prepare_coco128.py
python ai/train_xpu.py --dataset ai/datasets/coco128-effect --output ai/rfdetr-effect-xpu --full-epoch --epochs 10 --pretrained
python ai/report_effect.py
```

This experiment uses RF-DETR Nano pretrained weights, seed 42, FP32,
384-pixel resolution, batch size 1, and learning rates of 1e-5 for both the
encoder and other parameters. The same disjoint 32/8 split is used throughout.
All 320 backward passes and optimizer steps passed the XPU device/gradient
assertions. Results on the eight validation images:

| Checkpoint | mAP 50:95 | mAP 50 | Validation loss |
| --- | --- | --- | --- |
| Pretrained baseline | 0.53994 | 0.74692 | 3.71898 |
| Best (epoch 1) | 0.55654 | 0.76892 | 3.72331 |
| Last (epoch 10) | 0.53001 | 0.77069 | 3.73843 |

Training loss fell from 3.58008 to 2.99550, while validation loss stayed near
3.72. The best mAP improved by 1.66 percentage points, but later epochs lost
that gain: this is consistent with overfitting or small-validation-set noise,
not evidence of sustained generalization improvement. COCO128 comes from
COCO and may overlap model pretraining data; these are diagnostic numbers,
not an independent benchmark or wall-cleaning-domain accuracy estimate.

Artifacts in `rfdetr-effect-xpu/` include `baseline.json`, `metrics.csv`,
`comparison.json`, `device-proof.json`, `learning-curves.png`, and
`prediction-comparison.jpg` (pretrained top row, best fine-tuned bottom row,
confidence threshold 0.3). Use `checkpoint_best_regular.pth` for the best
model; `verified.ckpt` contains the final training state.

## Random-marker controlled experiment

`generate_marker.py` procedurally draws a seeded 4-by-4 colored marker and
generates separate train/validation/test scenes using seeds 101/202/303.
The scenes vary target position, scale, rotation, perspective, brightness,
partial occlusion, background noise and colored geometric distractors.
There are 48/12/32 images; one quarter of every split contains no marker.
SHA256 checks confirmed no duplicate images within or across splits.

```bash
python ai/generate_marker.py
python ai/train_xpu.py --dataset ai/datasets/random-marker --output ai/marker-scratch --full-epoch --epochs 6 --lr 0.0001
python ai/train_xpu.py --dataset ai/datasets/random-marker --output ai/marker-pretrained --full-epoch --epochs 6 --lr 0.0001 --pretrained
python ai/evaluate_marker.py
```

Both models use seed 42, batch size 1, FP32, resolution 384, identical data
and learning rates, and 288 optimizer steps. The scratch run disables both
detector and encoder pretrained weights. The other run loads Nano COCO
weights with a task-specific one-class head. Best checkpoints are selected
using validation mAP; the 32 test images are evaluated only afterward.

| Test metric | Scratch | Pretrained |
| --- | --- | --- |
| mAP 50:95 | 33.29% | 95.75% |
| mAP 50 | 64.94% | 100% |
| Recall at confidence 0.5, IoU 0.5 | 0/24 | 24/24 |
| False positives at confidence 0.5 | 0 | 0 |
| Negative images triggering an alarm | 0/8 | 0/8 |

mAP uses predictions down to confidence 0.001; fixed-threshold detection uses
0.5. Thus scratch mAP can be nonzero while recall at 0.5 is zero. Six epochs
are insufficient for confident scratch predictions; zero false alarms alone
does not imply a useful detector. All 576 combined backward/optimizer steps
passed the XPU/gradient assertions.

The generated pattern was created for this experiment, while pretrained
visual features can still transfer. These simple backgrounds and ordinary
shape distractors do not prove recognition of the exact internal bit pattern
against similar markers, nor real camera performance. Test data comes from
the same generator distribution and is small.

Artifacts under `datasets/random-marker/`: `target.png`, `manifest.json`,
`learning-curves.png`, `test-predictions.jpg`, and `test-results.json`.
Prediction panels show scratch above pretrained, green ground-truth boxes
and red predictions at confidence 0.5. Weights are in
`marker-{scratch,pretrained}/checkpoint_best_regular.pth`; logs are under
`logs/marker-*.log`.

## OpenVINO benchmark

The pretrained random-marker best model was exported directly to OpenVINO
with `format="openvino", openvino_precision="float32"`. The IR is at
`marker-openvino/rfdetr-nano.xml` with a roughly 103 MiB `.bin` file.
No INT8 calibration or quantization was applied. GPU FP16 below refers to
the runtime computation hint, using the same FP32-storage IR.

```bash
python ai/benchmark_openvino.py
python ai/plot_openvino_benchmark.py
```

Hardware: Core Ultra 9 185H CPU / Intel Arc integrated GPU. Software:
PyTorch 2.14.1+xpu and OpenVINO 2026.4.1. Sequential batch=1, 384x384,
10 warmups, 100 measured iterations per timing path, LATENCY performance
hint for OpenVINO. The PyTorch model uses its eager export forward path.
All backends use the same normalization and CPU postprocessor. Inputs cycle
through the 32 test images. OpenVINO's exported dynamic shapes are retained.
Raw inference includes host/device transfers and CPU output materialization;
end-to-end adds preprocessing and postprocessing but excludes disk reads,
JPEG decoding, model loading/compilation and warmup. FPS is reciprocal serial
latency, not async or batched throughput. One run per configuration was measured.

| Backend | Inference mean ms | End-to-end mean ms | End-to-end p95 ms | Serial FPS | Test mAP 50:95 |
| --- | --- | --- | --- | --- | --- |
| PyTorch XPU FP32 | 56.92 | 60.91 | 66.03 | 16.42 | 95.75% |
| OpenVINO CPU FP32 | 127.17 | 131.96 | 155.67 | 7.58 | 95.75% |
| OpenVINO GPU FP32 | 129.01 | 135.50 | 153.57 | 7.38 | 95.75% |
| OpenVINO GPU FP16 | 33.31 | 36.71 | 40.75 | 27.24 | 95.90% |

All four scored AP50=100%, detected 24/24 targets at confidence 0.5 and
IoU 0.5, with no false-positive detections or alarms on the eight negative
images. GPU execution was confirmed as GPU.0. FP16 is approximately 1.66x
faster end-to-end than this PyTorch FP32 baseline; this comparison changes
both backend and precision, so it is not a pure backend speedup. FP16 is
approximately 3.69x faster than OpenVINO GPU FP32 here.

Outputs are not bitwise identical: raw query-level boxes/logits differ,
especially with FP16. Test-set detection metrics remain consistent; the
0.15-point AP difference is not evidence of improved accuracy on this small
synthetic set. Broader real-image validation remains necessary for deployment.
Results and setup/compile times are in `marker-openvino/benchmark.json`;
chart: `marker-openvino/benchmark-comparison.png`; logs:
`logs/openvino-export.log` and `logs/openvino-benchmark.log`.

## PyTorch XPU FP16 comparison

Run `python ai/benchmark_openvino.py --fp16-comparison --rounds 3`.
Same checkpoint, 32 test images, batch 1, 384x384; 10 warmup calls and
three consecutive blocks of 100 measurements per timing scope/backend.
PyTorch 2.14.1+xpu eager mode; OpenVINO 2026.4.1 LATENCY mode.
Inference includes transfers and FP32 CPU outputs. Full timing adds image
preprocessing and common CPU decoding, excluding disk reads and model setup.
Backends ran sequentially in table order; torch.compile was not tested.

| Backend | Inference ms | Full ms | Full p95 ms | Serial FPS | Test mAP |
| --- | --- | --- | --- | --- | --- |
| XPU FP32 | 58.65 | 62.05 | 67.77 | 16.12 | 95.75% |
| XPU autocast FP16 | 60.35 | 63.72 | 69.99 | 15.69 | 95.91% |
| XPU half weights and inputs | 42.77 | 46.29 | 49.76 | 21.60 | 95.66% |
| OpenVINO GPU FP16 | 29.13 | 32.12 | 35.00 | 31.13 | 95.90% |

Explicit half improves XPU serial FPS by 34% over FP32. OpenVINO still
delivers 1.44x the FPS of XPU half, reducing full latency by 30.6%.
Block mean full latency ranges: XPU half 44.97–47.45 ms; OpenVINO
31.85–32.63 ms. Autocast did not improve speed; its cause was not profiled.
These serial numbers are not directly comparable to asynchronous throughput.

Device assertions confirm parameters, inputs and outputs on xpu:0.
Half mode uses FP16 parameters, inputs and outputs. Autocast keeps FP32
parameters/inputs and produces mixed FP32/FP16 outputs. OpenVINO reports
GPU.0 and FP16 precision hint. All outputs were finite. All four achieved
AP50=100%, 24/24 detections at confidence 0.5 and IoU 0.5, zero false
positives and zero alarms on eight negatives. Small synthetic-set mAP
differences do not establish a general accuracy advantage.

Results and device evidence: `marker-openvino/fp16-benchmark.json`.
Log: `logs/fp16-benchmark.log`. OpenVINO GPU FP16 remains the preferred
measured inference option on this machine.

## RF-DETR Segmentation Nano inference

Run `python ai/benchmark_segmentation.py` in the cleaning environment.
The script uses official pretrained RFDETRSegNano weights and exports an
FP32-storage OpenVINO IR. Input is the model's default 312x312, batch 1;
PyTorch uses eager mode, OpenVINO uses GPU FP16 LATENCY mode.
Dataset: the first 32 sorted images from Ultralytics coco128-seg, downloaded
from https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip
to `ai/coco128-seg.zip`. YOLO polygons are rasterized at original image size,
with COCO category IDs restored for evaluation.

Each backend runs 10 warmups and three consecutive blocks of 100 measurements
per timing scope. Inference includes input transfers and FP32 CPU output copies.
Full timing adds preprocessing and CPU mask decoding to original image size,
retaining scores above 0.5. Disk reads, model setup and export are excluded.
Accuracy uses all top-100 predictions without the 0.5 score filter.

| Backend | Inference ms | Full ms | Full p95 ms | Serial FPS | Mask mAP | Mask AP50 |
| --- | --- | --- | --- | --- | --- | --- |
| XPU FP32 | 84.33 | 91.76 | 97.91 | 10.90 | 58.90% | 85.59% |
| XPU FP16 weights and inputs | 63.86 | 70.67 | 77.57 | 14.15 | 58.93% | 85.01% |
| OpenVINO GPU FP16 | 51.70 | 58.26 | 68.04 | 17.17 | 58.76% | 85.09% |

OpenVINO provides 21% higher serial FPS than XPU FP16 in this test. XPU FP16
provides 30% higher FPS than FP32. Parameter/input/output device evidence
confirms xpu:0 for PyTorch and GPU.0 for OpenVINO. All evaluated outputs are
finite. Raw mask logits have shape [1,100,78,78], then are resized to the
original image dimensions. Box mAP was 73.18%, 72.50%, and 73.10%, respectively.

This is a pretrained-model inference check, not a segmentation training run.
COCO128-seg uses training images that overlap pretraining; these metrics are
not held-out accuracy or official COCO scores. Polygon conversion also differs
from original COCO mask annotations. Compared with the previous marker detector
benchmark, architecture, weights, classes, images, input resolution and decoding
work differ, so latency differences cannot isolate the cost of a mask head.

Outputs: `segmentation-benchmark-312/benchmark.json`, exported XML/BIN, and
one JPG per backend showing ground-truth masks on the left and predictions
above confidence 0.5 on the right. Log: `logs/segmentation-benchmark.log`.

## OpenVINO GPU FP16 model size comparison

Run `python ai/benchmark_variants.py` and then
`python ai/plot_variant_benchmarks.py`. The benchmark covers all current
non-deprecated open-source detection sizes (Nano, Small, Medium, Large) and
segmentation sizes (Nano, Small, Medium, Large, XLarge, 2XLarge).
Each uses official pretrained weights and its own default input resolution.
Deprecated Base, old Large and SegPreview variants are outside this comparison.

Each model runs in an isolated subprocess with GPU FP16 LATENCY configuration,
batch 1, 10 warmups and three rounds of 32 measurements per timing scope.
All models use the same first 32 sorted COCO128-seg images, original image
sizes for decoding, and the same polygon-derived annotations. Full timing
includes preprocessing, transfers, CPU output copies and CPU postprocessing;
segmentation masks are resized only for scores above 0.5 during timing.
Accuracy evaluates the model's full top-k output, with COCO maxDets=100.
Setup, export, file reads and metric computation are excluded from timings.

Both box and mask mAP mean AP@[0.50:0.95]. These training-set subset metrics
can overlap pretraining and use converted polygon annotations, so they are
not official or held-out COCO validation scores. A larger model need not
score better on this small subset. The previous marker-trained detector
results are not comparable to these official pretrained models. Runtime
variation also means Nano timing may differ from earlier benchmark runs.

The generated `variant-benchmarks/report.md` contains the full comparison,
`variant-benchmarks/results.json` contains all measurements, and
`variant-benchmarks/speed-accuracy.png` plots the tradeoffs. Each model folder
contains an exported IR and benchmark JSON; segmentation folders also have
ground-truth/prediction comparison images. Per-model logs are in the parent
folder. Successful results are retained on rerun; `--model RFDETRSmall`
reruns a specific model (run the coordinator afterward to refresh the report).

## CPU + GPU asynchronous inference

```bash
python ai/benchmark_multi.py --jobs 4
python ai/benchmark_multi.py --jobs 8
python ai/plot_multi_benchmark.py
```

OpenVINO `MULTI:GPU,CPU` schedules independent image requests across devices;
it does not split a single image's model between CPU and GPU. Tested the same
IR and 32 synthetic test images using GPU f16 and CPU f32 compute hints,
THROUGHPUT mode, batch 1, 384x384, profiling enabled, 32 warmup images,
and three rounds of 96 images per configuration. Reported throughput is
the median of the three rounds. Images are preprocessed before timing;
timed work includes input submission, output copies in completion callbacks,
and draining the request queue, but excludes decoding and postprocessing.
Request latency starts after waiting for a free queue slot, and includes
internal scheduling wait. These numbers are not directly comparable to the
earlier serial end-to-end benchmark.

| Configuration | 4 requests FPS | 8 requests FPS | 4 requests test mAP | 8 requests test mAP |
| --- | --- | --- | --- | --- |
| GPU only | 40.04 | 39.90 | 95.90% | 95.90% |
| CPU only | 6.99 | Not tested | 95.75% | Not tested |
| MULTI GPU+CPU | 37.15 | 24.26 | 95.90% | 92.04% |
| MULTI GPU+CPU, CPU limited to 4 threads | 36.30 | 28.45 | 88.60% | 90.53% |

Mean request latency at 4 requests: GPU 98.4 ms, MULTI 105.9 ms.
At 8 requests: GPU 190.6 ms, MULTI 314.1 ms. More concurrent requests
increase queueing latency without improving throughput on this machine.
GPU-only four-request asynchronous inference is the recommended tested
throughput configuration; use serial latency mode for lower per-frame delay.

Compiled MULTI properties confirm GPU.0 and CPU are enabled. For eight
requests, CPU used f32, 18 threads and 6 streams; GPU used f16 and 2 streams.
Per-request profiling is partially unavailable in MULTI, so exact scheduling
counts are not claimed. Shared CPU/iGPU power and memory resources may explain
the slowdown, but were not measured directly. MULTI also showed a test-mAP
regression under some configurations; its cause is not established. Do not
deploy those configurations based on throughput alone. This is a small
synthetic test, not a general statement about OpenVINO MULTI accuracy.

Detailed results: `marker-openvino/multi-benchmark-4.json` and
`marker-openvino/multi-benchmark-8.json`. Chart:
`marker-openvino/multi-comparison.png`. Logs are `logs/multi-benchmark*.log`.
