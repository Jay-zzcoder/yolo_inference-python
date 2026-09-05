# -*- coding: utf-8 -*-
import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt model to ONNX")
    parser.add_argument('--model', type=str, default="yolo11n.pt", help='Path to input .pt model')
    parser.add_argument('--output', type=str, default="yolo11n.onnx", help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640, 640], 
                        help='Input image size (int for square, or two ints for height width)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic axes for batch and size')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model using onnxsim')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Export device')
    parser.add_argument('--half', action='store_true', help='Export FP16 model')
    
    args = parser.parse_args()

    # 处理输入尺寸
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    elif len(args.imgsz) == 2:
        imgsz = tuple(args.imgsz)  # (height, width)
    else:
        raise ValueError("--imgsz must be one integer (square) or two integers (height width)")

    # 如果未指定输出路径，自动生成
    if args.output is None:
        base = os.path.splitext(args.model)[0]
        args.output = base + '.onnx'

    # 加载模型
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)

    # 导出参数
    export_kwargs = {
        'format': 'onnx',
        'imgsz': imgsz,
        'batch': args.batch,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'device': args.device,
        'half': args.half,
    }
    # 如果用户指定了输出路径，需要设置 file 参数（Ultralytics 8.x 以上）
    # 新版本支持 file 参数，但不同版本可能有差异，使用 'file' 或直接通过文件名。
    # 更通用：使用 model.export 的 file 参数（如果支持）
    # 检查 export 方法是否接受 'file' 参数
    import inspect
    sig = inspect.signature(model.export)
    if 'file' in sig.parameters:
        export_kwargs['file'] = args.output
        print(f"Exporting to {args.output}...")
    else:
        # 旧版本可能不支持 file，则忽略，之后重命名
        print("Exporting with default name, will rename later...")

    # 执行导出
    model.export(**export_kwargs)

    # 如果 export 不支持 file 参数，需要移动生成的文件
    if 'file' not in sig.parameters:
        # 默认导出文件名基于模型名，例如 "model.onnx"
        default_name = os.path.splitext(os.path.basename(args.model))[0] + '.onnx'
        if os.path.exists(default_name) and default_name != args.output:
            os.rename(default_name, args.output)
            print(f"Renamed to {args.output}")

    print(f"Export completed. ONNX model saved to {args.output}")


if __name__ == "__main__":
    main()