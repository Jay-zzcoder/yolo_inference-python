[中文](./README.zh.md) | [English](./README.md)
# YOLO Multi-Backend Inference Framework

This repository provides a complete YOLO object detection Python solution supporting three inference backends: **PyTorch (.pt)**, **ONNX Runtime**, and **TensorRT**. It also includes model export, class label extraction, performance benchmarking, and other utilities. The project is well-structured and suitable for rapid integration into real-world applications.

## Features

- **Multi-backend support**: Seamlessly switch between PyTorch, ONNX, or TensorRT inference via a unified interface.
- **Flexible deployment**: Support single image, batch processing of image directories, result visualization, and saving.
- **Performance analysis**: Built-in benchmarking tools to measure stage-wise latencies (preprocessing, inference, postprocessing, NMS, etc.).
- **Model export**: Complete pipeline from `.pt → ONNX → TensorRT`, with FP16 precision support.
- **Class management**: Automatically extract class names from the model and generate `.txt` label files for easy lookup and mapping.
---

## Directory Structure
```bash
├── export/                  # Model export scripts
│   ├── export_onnx.py       # PyTorch model → ONNX
│   └── export_trt.py        # ONNX → TensorRT engine
├── inference/               # Inference implementations
│   ├── yolo_detector.py     # PyTorch backend (based on ultralytics)
│   ├── yolo_detector_onnx.py # ONNX Runtime backend
│   └── yolo_detector_trt.py  # TensorRT backend
├── models/                  # Models and test data
│   └── fire/                # Example: fire/smoke detection model
│       ├── best.pt / .onnx / .engine
│       ├── fire_smoke_yolov8.*
│       ├── *.txt            # Class label files
│       └── test.jpg         # Test image
└── utils/                   # Helper tools
    └── clsname_from_pt.py   # Extract class names from .pt and generate .txt
├── requirements_win.txt     (Windows system dependencies, GPU: RTX3070ti, CUDA version: 12.0)
```

## 1. Model Export

### 1.1 Export to ONNX

Use export/export_onnx.py to convert a .pt model to ONNX format.

**Basic Usage**：

```bash
python export/export_onnx.py \
    --model models/fire/fire_smoke_yolov8.pt \
    --output models/fire/fire_smoke_yolov8.onnx \
    --imgsz 640 640 \
    --dynamic \
    --simplify \
    --opset 18
```
**参数说明**：
```bash
  --model      : Path to input .pt model (required)
  --output     : Path to output ONNX file (optional, default same as input with .onnx extension)
  --imgsz      : Input image size, can be an integer (square) or two integers (height width), default 640
  --batch      : Batch size, default 1
  --dynamic    : Enable dynamic axes (batch and spatial dimensions), default False
  --simplify   : Simplify the ONNX model using onnxsim, default False
  --opset      : ONNX opset version, default 12
  --device     : Export device, 'cpu' or 'cuda', default 'cpu'
  --half       : Export FP16 half‑precision model, default False
```
### 1.2 Export to TensorRT Engine
An ONNX file is required first. Then use export/export_trt.py to generate a TensorRT engine (.engine).

**Basic Usage**：
```bash
python export/export_trt.py \
    --onnx models/fire/fire_smoke_yolov8.onnx \
    --engine models/fire/fire_smoke_yolov8.engine \
# The above enables FP16 by default. To use FP32 instead, add --no-fp16:
python export/export_trt.py \
    --onnx models/fire/fire_smoke_yolov8.onnx \
    --engine models/fire/fire_smoke_yolov8.engine \
    --no-fp16
```


## 2. Model Inference
Three inference backend are provided with a unified interface for easy backend switching. All inference code are located under inference/.

### 2.1 PyTorch Backend（yolo_detector.py）
Depends on the ultralytics library and loads .pt models directly.
**Basic Usage**：
```python
from inference.yolo_detector import YOLODetecter

# Initializ
detector = YOLODetecter(
    model_path="models/fire/fire_smoke_yolov8.pt",
    device="cuda"          # 或 "cpu"
)

# Single image inference with display
detector.inference("test.jpg", conf_thres=0.5, show=True)

# Batch process a directory
detector.inference("images/", conf_thres=0.5, output_dir="results/")

# Benchmark (default 1000 runs, average latency, warmup enabled by default)
detector.benchmark("test.jpg")
```

### 2.2 ONNX Runtime Backend （yolo_detector_onnx.py）
Depends on onnxruntime-gpu and supports CPU and CUDA acceleration.
**Basic Usage**：
```python
from inference.yolo_detector_onnx import YOLODetecterONNX

detector = YOLODetecterONNX(
    model_path="models/fire/fire_smoke_yolov8.onnx",
    provider="cuda"        # or "cpu"
)

# Single image inference with display
detector.inference("test.jpg", conf_thres=0.5, show=True)

# Batch process a directory
detector.inference("images/", conf_thres=0.5, output_dir="results/")

# Benchmark (default 1000 runs, average latency, warmup enabled by default)
detector.benchmark("test.jpg")
```


### 2.3 TensorRT Backend（yolo_detector_trt.py）
Depends on tensorrt and pycuda, providing the highest inference performance.
**Basic Usage**：
```python
from inference.yolo_detector_trt import YOLODetecterTRT

detector = YOLODetecterTRT(
    engine_path="models/fire/fire_smoke_yolov8.engine",
    input_shape=(640, 640)   # 必须与导出时的尺寸一致
)
# Single image inference with display
detector.inference("test.jpg", conf_thres=0.5, show=True)

# Batch process a directory
detector.inference("images/", conf_thres=0.5, output_dir="results/")

# Returns detection results and per‑stage timings (preprocess, inference, parsing, NMS, filtering) for performance analysis.
detector.inference_profiling(image, conf_thres=0.6, visualization=False, return_timing=True)

# Automatically computes average stage latencies and percentages, outputting a detailed report.
detector.benchmark(image, num_runs=1000, conf_thres=0.6, warmup=5)
```

## 3. Automatic Class Label Loading
All detectors attempt to read class names from a file with the same name as the model but with .txt extension (e.g., best.pt → best.txt). The file content should be in the format:
```bash
0 fire
1 smoke
```
If such a file exists, class names will be displayed in visualizations instead of numeric IDs; otherwise, default numeric labels are used.

You can use utils/clsname_from_pt.py to extract class names from a .pt model and generate this file:
```python
python utils/clsname_from_pt.py models/yolo.pt
```


## 5. FAQ
If you encounter issues with CUDA acceleration for ONNX or TensorRT, possible reasons include:
1. CuDNN is not installed.
2. The onnxruntime-gpu version does not match your CUDA version.

## 🚀 Roadmap
- [ ] Explore more optimization techniques
- [ ] Implement YOLO C++ inference with different backends
- [ ] Integrate various detection models, ready to use without training

For any questions, feel free to discuss: 821482076@qq.com