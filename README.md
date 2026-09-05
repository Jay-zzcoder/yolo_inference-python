# YOLO 多后端推理框架

本仓库提供了一套完整的 YOLO 目标检测python解决方案，支持 **PyTorch (.pt)**、**ONNX Runtime** 和 **TensorRT** 三种推理后端，并包含模型导出、类别标签提取、性能基准测试等工具。项目结构清晰，适合快速集成到实际应用中。

## 特性

- **多后端支持**：同一套接口，可无缝切换 PyTorch、ONNX 或 TensorRT 推理。
- **灵活部署**：支持单张图片、图片目录批量处理，结果可视化与保存。
- **性能分析**：内置 Benchmark 工具，可统计各阶段耗时（预处理、推理、后处理、NMS 等）。
- **模型导出**：提供 `.pt → ONNX → TensorRT` 的完整导出流程，支持 FP16 精度。
- **类别管理**：自动从模型提取类别名称并生成 `.txt` 标签文件，便于查看和映射。

---

## 目录结构
```bash
├── export/ # 模型导出脚本
│ ├── export_onnx.py # PyTorch 模型 → ONNX
│ └── export_trt.py # ONNX → TensorRT 引擎
├── inference/ # 推理实现
│ ├── yolo_detector.py # PyTorch 后端（基于 ultralytics）
│ ├── yolo_detector_onnx.py # ONNX Runtime 后端
│ └── yolo_detector_trt.py # TensorRT 后端
├── models/ # 模型与测试数据
│ ├── fire/ # 示例：烟火检测模型
│ │ ├── best.pt / .onnx / .engine
│ │ ├── fire_smoke_yolov8.*
│ │ ├── *.txt # 类别标签文件
│ │ └── test.jpg # 测试图片
└── utils/ # 辅助工具
│ ├── clsname_from_pt.py # 从 .pt 提取类别名称生成 .txt
├── requirements_win.txt (windows系统下的依赖, GPU: RTX3070ti, cuda版本: 12.0)
```

## 1. 模型导出

### 1.1 导出为 ONNX

使用 `export/export_onnx.py` 脚本将 `.pt` 模型转换为 ONNX 格式。

**基本用法**：

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
  --model      : 输入的 .pt 模型路径（必需）
  --output     : 输出的 ONNX 文件路径（可选，默认与输入同名，扩展名 .onnx）
  --imgsz      : 输入图像尺寸，支持整数（正方形）或两个整数（高 宽），默认 640
  --batch      : 批处理大小，默认 1
  --dynamic    : 是否启用动态轴（batch 和尺寸），默认 False
  --simplify   : 是否使用 onnxsim 简化模型，默认 False
  --opset      : ONNX opset 版本，默认 12
  --device     : 导出设备，'cpu' 或 'cuda'，默认 'cpu'
  --half       : 是否导出 FP16 半精度模型，默认 False
```
### 1.2 导出为 TensorRT 引擎
需要先有 ONNX 文件，再使用 export/export_trt.py 生成 TensorRT 引擎（.engine）。

**基本用法**：
```bash
python export/export_trt.py \
    --onnx models/fire/fire_smoke_yolov8.onnx \
    --engine models/fire/fire_smoke_yolov8.engine \
# 上面是默认开启转FP16格式的，如果不想转fp16, 想直接使用fp32, 使用下面的命令:
python export/export_trt.py \
    --onnx models/fire/fire_smoke_yolov8.onnx \
    --engine models/fire/fire_smoke_yolov8.engine \
    --no-fp16
```


## 2. 模型推理
提供三种后端推理类，接口统一，便于切换。所有类均位于 inference/ 目录。

### 2.1 PyTorch 后端（yolo_detector.py）
依赖 ultralytics 库，直接加载 .pt 模型。
**基本用法**：
```python
from inference.yolo_detector import YOLODetecter

# 初始化
detector = YOLODetecter(
    model_path="models/fire/fire_smoke_yolov8.pt",
    device="cuda"          # 或 "cpu"
)

# 单张图片检测并显示
detector.inference("test.jpg", conf_thres=0.5, show=True)

# 批量处理目录
detector.inference("images/", conf_thres=0.5, output_dir="results/")

#耗时测试（默认推理1000次，计算平均耗时，默认开启warmup）
detector.benchmark("test.jpg")
```

### 2.2 ONNX Runtime 后端（yolo_detector_onnx.py）
依赖 onnxruntime，支持 CPU 和 CUDA 加速。
**基本用法**：
```python
from inference.yolo_detector_onnx import YOLODetecterONNX

detector = YOLODetecterONNX(
    model_path="models/fire/fire_smoke_yolov8.onnx",
    provider="cuda"        # 或 "cpu"
)

# 单张图片检测并显示
detector.inference("test.jpg", conf_thres=0.5, show=True)

# 批量处理目录
detector.inference("images/", conf_thres=0.5, output_dir="results/")

#耗时测试（默认推理1000次，计算平均耗时，默认开启warmup）
detector.benchmark("test.jpg")
```


### 2.3 TensorRT 后端（yolo_detector_trt.py）
依赖 tensorrt 和 pycuda，提供最高推理性能。
**基本用法**：
```python
from inference.yolo_detector_trt import YOLODetecterTRT

detector = YOLODetecterTRT(
    engine_path="models/fire/fire_smoke_yolov8.engine",
    input_shape=(640, 640)   # 必须与导出时的尺寸一致
)
# 单张图片检测并显示
detector.inference("test.jpg", conf_thres=0.5, show=True)

# 批量处理目录
detector.inference("images/", conf_thres=0.5, output_dir="results/")

#返回检测结果和各阶段耗时（预处理、推理、后处理解析、NMS、过滤），便于性能分析。
detector.inference_profiling(image, conf_thres=0.6, visualization=False, return_timing=True)

#自动统计各阶段平均耗时及占比，输出详细报告。
detector.benchmark(image, num_runs=1000, conf_thres=0.6, warmup=5)
```

## 3. 类别标签自动加载
所有检测器在初始化时会尝试从模型路径同名（扩展名 .txt）的文件中读取类别名称。例如模型 best.pt 对应 best.txt，内容格式：
```bash
0 fire
1 smoke
```
若存在，则可视化时显示类别名称而非 ID；否则使用默认数字标签。
可使用 utils/clsname_from_pt.py 从 .pt 模型提取并生成该文件：
```python
python utils/clsname_from_pt.py models/yolo.pt
```


## 5. 常见问题
如果遇到onnx或者tensorrt使用CUDA加速失败，可能存在以下原因：
1. 没有安装cudnn
2. onnxuntime-gpu版本与cuda版本不匹配



有问题欢迎大家一起讨论：821482076@qq.com