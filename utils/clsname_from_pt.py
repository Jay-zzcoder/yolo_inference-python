"""
功能描述：
    从训练好的 .pt 模型文件中提取类别标签与类别名的映射字典，
    并将每一对（标签, 类别名）逐行写入与 .pt 文件同目录、同名的 .txt 文件中。

输入：
    命令行参数：模型文件路径（必填），例如：
        python clsname_from_pt.py E:/Robotics/Code/yolo_inference-python/models/fire/fire_smoke_yolov8.pt

输出：
    生成一个 .txt 文件，每行格式为 "标签 类别名"（例如 "0 fire"）。

注意事项：
    1. 无需进行推理，直接读取模型元数据，高效快速。
    2. 如果模型文件不存在或未包含 names 属性，程序会报错，请确保模型有效。
    3. 生成的 .txt 文件会覆盖已有同名文件。
"""

import os
import argparse
from ultralytics import YOLO

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="从 YOLO .pt 模型提取类别标签并写入 .txt")
    parser.add_argument("model_path", type=str, help="YOLO 模型文件路径（.pt）")
    args = parser.parse_args()

    model_path = args.model_path

    # 检查模型文件是否存在
    if not os.path.isfile(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        return

    # 加载 YOLO 模型（自动解析元数据）
    model = YOLO(model_path)

    # 获取类别名称字典，格式：{0: 'fire', 1: 'smoke', ...}
    class_names = model.names

    # 构造输出 txt 文件路径（与 .pt 同目录、同主文件名）
    txt_path = os.path.splitext(model_path)[0] + '.txt'

    # 将标签和类别名逐行写入 txt 文件（覆盖写入）
    with open(txt_path, 'w', encoding='utf-8') as f:
        for label, name in class_names.items():
            f.write(f"{label} {name}\n")

    print(f"类别标签已写入：{txt_path}")

if __name__ == "__main__":
    main()