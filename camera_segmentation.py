"""使用已训练的实例分割模型测试 USB 摄像头，也可处理单张图片。"""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.util import find_spec
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_DIR / "runs/segmentation/checkpoint_best_ema.pth",
        help="分割模型权重路径",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--source", default="1",
        help="摄像头编号、视频路径或 RTSP 地址；默认 1（常见的外接 USB 摄像头编号）",
    )
    inputs.add_argument("--image", type=Path, help="使用单张图片检查模型和绘图流程")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值，默认 0.5")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="默认 GPU（CUDA FP16）；使用 CPU 时指定 --device cpu")
    parser.add_argument(
        "--backend", choices=("auto", "torch", "openvino", "openvino-int8"), default="auto",
        help="auto：CPU 优先使用匹配当前权重的 OpenVINO INT8 缓存，否则使用 FP32；GPU 使用 PyTorch",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU 计算线程，0 为引擎自动选择")
    parser.add_argument("--resolution", type=int, default=0, help="CPU OpenVINO 输入尺寸，0 沿用训练尺寸；降低尺寸可能影响精度")
    parser.add_argument("--width", type=int, default=640, help="摄像头采集宽度，默认 640")
    parser.add_argument("--height", type=int, default=480, help="摄像头采集高度，默认 480")
    parser.add_argument("--fp32", action="store_true", help="禁用默认的 GPU FP16 推理")
    parser.add_argument("--output", type=Path, help="保存单张图片测试的结果")
    parser.add_argument("--snapshot-dir", type=Path, default=PROJECT_DIR / "runs/segmentation/camera")
    parser.add_argument("--no-display", action="store_true", help="处理视频但不显示预览页面，供流程检查")
    parser.add_argument("--port", type=int, default=8765, help="本机浏览器预览端口，默认 8765")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器，只输出预览地址")
    parser.add_argument("--max-frames", type=int, default=0, help="处理指定帧数后退出，0 表示持续运行")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须介于 0 和 1 之间")
    if args.width <= 0 or args.height <= 0 or args.max_frames < 0:
        parser.error("宽高必须大于 0，--max-frames 不能小于 0")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须介于 1 和 65535 之间")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads 不能小于 0")
    if args.resolution < 0:
        parser.error("--resolution 不能小于 0")
    if args.resolution and (args.device != "cpu" or args.backend == "torch"):
        parser.error("--resolution 目前仅支持 CPU OpenVINO 模式")
    if args.backend.startswith("openvino") and args.device != "cpu":
        parser.error("OpenVINO 后端请同时指定 --device cpu")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"权重文件不存在：{args.checkpoint}")
    return args


class SegmentationPredictor:
    def __init__(self, args):
        import supervision as sv
        import torch
        from rfdetr import RFDETR

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查 GPU 版 PyTorch，或使用 --device cpu")
        self.model = RFDETR.from_checkpoint(
            args.checkpoint, device=args.device, gradient_checkpointing=False,
        )
        if not self.model.model_config.segmentation_head:
            raise ValueError("该权重不是实例分割模型，请传入分割训练产生的权重")
        self.resolution = args.resolution or self.model.model_config.resolution
        block_size = self.model.model_config.patch_size * self.model.model_config.num_windows
        if self.resolution % block_size:
            raise ValueError(f"--resolution 必须是 {block_size} 的倍数")
        self.backend = args.backend
        if self.backend == "auto":
            self.backend = "torch"
            if args.device == "cpu" and find_spec("openvino"):
                from cpu_inference import cached_export, model_signature

                signature = model_signature(self.model, args.checkpoint, self.resolution)
                self.backend = "openvino-int8" if cached_export(args.checkpoint, "int8", signature) else "openvino"
        if args.resolution and self.backend == "torch":
            raise RuntimeError("指定 CPU 输入尺寸需要先安装 OpenVINO：python -m pip install -r requirements-cpu.txt")
        dtype = torch.float16 if args.device == "cuda" and not args.fp32 else torch.float32
        if self.backend.startswith("openvino"):
            if find_spec("openvino") is None:
                raise RuntimeError("请先安装 CPU 加速依赖：python -m pip install -r requirements-cpu.txt")
            from cpu_inference import enable_openvino

            enable_openvino(
                self.model, args.checkpoint,
                precision="int8" if self.backend == "openvino-int8" else "fp32",
                threads=args.cpu_threads, resolution=self.resolution,
            )
            cv2.setNumThreads(1)
        else:
            if args.device == "cpu" and args.cpu_threads:
                torch.set_num_threads(args.cpu_threads)
            # 使用官方推理优化 API，不编译以缩短首次启动时间。
            self.model.inference(compile=False, dtype=dtype)
            print(f"推理引擎：PyTorch；设备：{args.device}", flush=True)
        self.threshold = args.threshold
        self.mask_annotator = sv.MaskAnnotator(opacity=0.45)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
        print(f"已加载：{args.checkpoint}", flush=True)
        precision = "INT8 混合精度" if self.backend == "openvino-int8" else str(dtype)
        print(f"类别：{self.model.class_names}；模型输入：{self.resolution}；精度：{precision}", flush=True)

    def predict(self, frame):
        # OpenCV 采集的是 BGR，RF-DETR 接收 RGB。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(
            rgb, threshold=self.threshold, include_source_image=False,
            shape=(self.resolution, self.resolution),
        )
        annotated = frame.copy()
        if len(detections):
            names = detections.data.get("class_name")
            if names is None:
                names = [self.model.class_names[int(label)] for label in detections.class_id]
            labels = [f"{name} {float(score):.2f}" for name, score in zip(names, detections.confidence)]
            annotated = self.mask_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels=labels)
        return annotated, detections


def save_frame(frame, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
    print(f"图片已保存：{path}", flush=True)


class BrowserPreview:
    """通过本机 HTTP 预览兼容 OpenCV headless，无需额外 GUI 依赖。"""

    def __init__(self, args):
        self.running = True
        self.condition = threading.Condition()
        self.jpeg = None
        self.frame = None
        self.sequence = 0
        self.snapshot_dir = args.snapshot_dir
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *unused):
                pass

            def respond(self, content, content_type="text/plain; charset=utf-8", status=200):
                payload = content.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/":
                    self.respond(PREVIEW_HTML, "text/html; charset=utf-8")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sequence = -1
                    try:
                        while preview.running:
                            with preview.condition:
                                preview.condition.wait_for(
                                    lambda: preview.sequence != sequence or not preview.running, timeout=1,
                                )
                                sequence, jpeg = preview.sequence, preview.jpeg
                            if jpeg is None or not preview.running:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self.respond("Not found", status=404)

            def do_POST(self):
                if self.path == "/stop":
                    self.respond("已停止摄像头测试，可以关闭页面。")
                    preview.close()
                elif self.path == "/snapshot":
                    with preview.condition:
                        frame = preview.frame.copy() if preview.frame is not None else None
                    if frame is None:
                        self.respond("正在等待摄像头画面。", status=409)
                    else:
                        name = datetime.now().strftime("camera_%Y%m%d_%H%M%S_%f.png")
                        path = preview.snapshot_dir / name
                        save_frame(frame, path)
                        self.respond(f"截图已保存：{path.resolve()}")
                else:
                    self.respond("Not found", status=404)

        self.server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        url = f"http://127.0.0.1:{self.server.server_address[1]}"
        print(f"预览地址：{url}；页面中可停止测试或保存截图。", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    def close(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()

    def update(self, frame):
        success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("预览画面编码失败")
        with self.condition:
            self.frame = frame.copy()
            self.jpeg = jpeg.tobytes()
            self.sequence += 1
            self.condition.notify_all()

    def shutdown(self):
        self.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


PREVIEW_HTML = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RF-DETR 摄像头实例分割</title>
<style>
body{margin:24px;background:#14202b;color:#eef4fa;font:16px system-ui,sans-serif}
main{max-width:1100px;margin:auto}h1{font-size:24px}p{line-height:1.6}
img{display:block;max-width:100%;max-height:75vh;border-radius:8px;background:#0a131c}
button{margin:16px 12px 0 0;padding:10px 18px;border:0;border-radius:6px;font-size:16px;cursor:pointer}
#message{white-space:pre-wrap;overflow-wrap:anywhere;color:#b8cee0}
</style><main><h1>摄像头实例分割</h1>
<p>显示分割掩膜、类别和置信度。关闭页面后，可在终端按 Ctrl+C 停止测试。</p>
<img id="video" src="/stream" alt="正在等待摄像头画面…">
<button onclick="action('/snapshot')">保存截图</button><button onclick="action('/stop')">停止测试</button>
<p id="message">当前模型类别：can、Earbuds、Mouse。画面仅在本机显示。</p>
<script>
async function action(path){
  const el=document.getElementById('message');
  try{const response=await fetch(path,{method:'POST'});el.textContent=await response.text();
    if(path==='/stop'){document.getElementById('video').removeAttribute('src');}}
  catch(error){el.textContent='连接已断开，请查看终端。';}
}
</script></main></html>"""


def open_capture(args):
    source = int(args.source) if args.source.isdecimal() else args.source
    # Windows USB/内置摄像头先尝试 DirectShow，再尝试默认后端。
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if isinstance(source, int) and sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(source, backend)
        if capture.isOpened():
            if isinstance(source, int):
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if ok:
                return capture, frame
        capture.release()
    raise RuntimeError(
        f"无法读取视频源 {args.source!r}。USB 摄像头编号由系统决定，可尝试 --source 0 或 --source 2；"
        "同时检查 Windows 摄像头权限和其他程序是否占用摄像头。"
    )


def run(args):
    predictor = SegmentationPredictor(args)
    if args.image:
        with Image.open(args.image) as image:
            frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        annotated, detections = predictor.predict(frame)
        save_frame(annotated, args.output or args.snapshot_dir / f"{args.image.stem}_prediction.png")
        print(f"检出 {len(detections)} 个目标：{detections.data.get('class_name', [])}")
        return

    capture, frame = open_capture(args)
    window = None
    frames = 0
    smoothed_fps = None
    try:
        if not args.no_display:
            window = BrowserPreview(args)
        print(f"视频源已打开，实际画面：{frame.shape[1]}×{frame.shape[0]}。终端按 Ctrl+C 或在预览页面点击停止。", flush=True)
        while window is None or window.running:
            started = time.perf_counter()
            annotated, detections = predictor.predict(frame)
            frames += 1
            elapsed = time.perf_counter() - started
            fps = 1 / max(elapsed, 1e-6)
            smoothed_fps = fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * fps
            status = f"Inference+draw: {smoothed_fps:.1f} FPS | Objects: {len(detections)}"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (25, 25, 25), -1)
            cv2.putText(annotated, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if window:
                window.update(annotated)
            if args.max_frames and frames >= args.max_frames:
                break
            if window is not None and not window.running:
                break
            ok, frame = capture.read()
            if not ok:
                print("视频已结束或摄像头连接中断。", flush=True)
                break
        print(f"已处理 {frames} 帧。", flush=True)
    finally:
        capture.release()
        if window:
            window.shutdown()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("摄像头测试已停止。")
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
