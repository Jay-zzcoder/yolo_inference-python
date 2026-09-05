# -*- coding: utf-8 -*-
"""
yolo_detector.py

功能：
  - 提供 YOLOv11Detecter 类，加载 yolov11n.pt 等模型，并对输入图像进行 person 检测。
  - 返回 (cls_id, conf, (x1, y1, x2, y2)) 格式列表。

算法细节：
  - 使用 ultralytics 库加载 YOLO 模型；
  - 检测后只关注 (cls_id, conf, bbox) 三元组，并可根据 conf_thres 过滤结果。

使用方法：
  - from yolo_detector import YOLOv11Detecter
  - detector = YOLOv11Detecter("yolo11n.pt")
  - detections = detector.detect_person(img, conf_thres=0.6)
"""
import os
from ultralytics import YOLO
import torch
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']      # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题

class YOLODetecter:
    def __init__(self, model_path="yolo11n.pt", device="cuda", class_names=None):
        """
        初始化 YOLOv11Detecter 类，加载指定路径的 YOLO 模型。

        :param model_path: 模型文件路径，默认值为 "yolo11n.pt"
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU.")
            device = "cpu"
        self.device = device
        self.model = YOLO(model_path)  # 加载 YOLO 模型

        if class_names is None:
            self.class_names = self._load_class_names(model_path)
        else:
            self.class_names = class_names

    def _load_class_names(self, model_path):
        """尝试从模型路径同名的txt加载类别名称，若失败则返回默认数字名称"""
        txt_path = os.path.splitext(model_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Loaded class names from {txt_path}")
            return names
        else:
            print(f"[WARN] Class names file not found: {txt_path}, using default numeric names.")
            print(f"[WARN] You can generate Class names file using utils/clsname_from_pt.py")
            # 默认：尝试从模型获取，或生成数字
            if hasattr(self.model, 'names') and self.model.names:
                # 如果模型自带names（如COCO），使用它
                return list(self.model.names.values())
            else:
                # 否则生成 "class0", "class1", ...
                return [f"class{i}" for i in range(80)]  # 80为COCO默认，可根据情况调整

    def _inference_single(self, image, conf_thres=0.6, save_path=None, show=True, class_names=None):
        """
        处理单张图片：检测、可视化、保存/显示。
        返回检测结果列表。
        """
        import cv2
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        import os

        # 加载图像
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"无法读取图像: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            detect_img = img_bgr
        else:
            detect_img = image.copy()
            img_rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

        # 检测
        detections = self.process(detect_img, conf_thres=conf_thres)

        if class_names is None:
            class_names = self.class_names

        # 可视化
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_rgb)
        ax.axis('off')

        for cls_id, conf, (x1, y1, x2, y2) in detections:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            if class_names and isinstance(class_names, list) and cls_id < len(class_names):
                label = f"{class_names[cls_id]} {conf:.2f}"
            else:
                label = f"cls{cls_id} {conf:.2f}"
            ax.text(
                x1, y1 - 5,
                label,
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.6)
            )

        # 保存或显示
        if save_path:
            # 如果 save_path 是目录，则自动生成文件名
            if os.path.isdir(save_path):
                # 若 image 是文件路径，取文件名；否则用默认名
                if isinstance(image, str):
                    base = os.path.basename(image)
                    name, ext = os.path.splitext(base)
                    save_path = os.path.join(save_path, f"{name}_detected{ext}")
                else:
                    save_path = os.path.join(save_path, "detected.jpg")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            print(f"可视化结果已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return detections

    def inference(self, image, conf_thres=0.6, save_path=None, show=True, class_names=None, output_dir=None):
        """
        对输入进行目标检测并可视化所有目标。

        参数：
            image       : 可以是：
                          - 单张图片路径 (str)
                          - numpy 数组 (BGR 格式)
                          - 目录路径 (str) —— 此时将批量处理该目录下所有图片
            conf_thres  : 置信度阈值（默认 0.6）
            save_path   : 单张图片时，保存路径（文件或目录）。若为目录，自动生成文件名。
                          批量模式时，此参数无效，请使用 output_dir。
            show        : 是否显示图像（批量时强制关闭显示）
            class_names : 类别名称列表（可选）
            output_dir  : 批量模式下的输出目录（默认在输入目录下创建 'detected' 文件夹）

        返回：
            - 单张模式：检测结果列表，每个元素为 (cls_id, conf, (x1,y1,x2,y2))
            - 批量模式：列表，每个元素为 {'image': 文件名, 'detections': 检测结果列表}
        """
        import os
        from pathlib import Path

        # ---------- 批量处理（输入为目录） ----------
        if isinstance(image, str) and os.path.isdir(image):
            img_dir = image
            if output_dir is None:
                output_dir = os.path.join(img_dir, 'detected')
            os.makedirs(output_dir, exist_ok=True)

            # 支持的图片扩展名
            img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]
            if not img_files:
                print(f"目录 {img_dir} 中没有图片文件")
                return []

            results = []
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                # 构建保存路径
                name, ext = os.path.splitext(img_file)
                save_path_img = os.path.join(output_dir, f"{name}_detected{ext}")
                # 调用单张处理（强制不显示）
                dets = self._inference_single(
                    img_path,
                    conf_thres=conf_thres,
                    save_path=save_path_img,
                    show=False,
                    class_names=class_names if class_names is not None else self.class_names
                )
                results.append({'image': img_file, 'detections': dets})

            print(f"批量检测完成，结果保存在 {output_dir}")
            return results

        # ---------- 单张处理 ----------
        else:
            return self._inference_single(image, conf_thres, save_path, show, 
                                          class_names=class_names if class_names is not None else self.class_names)

    def process(self, image, conf_thres=0.6):
        """
        对输入图像进行 person 检测。

        :param image: 输入图像 (numpy array)
        :param conf_thres: 置信度阈值，默认值为 0.6
        :return: 检测结果列表，每个元素为 (cls_id, conf, (x1, y1, x2, y2))
        """
        results = self.model(image, conf=conf_thres, device=self.device, verbose=False)  # 使用模型进行检测
        boxes = results[0].boxes  # 获取检测到的边框
        out_list = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()  # 获取边框坐标
            conf_ = float(b.conf.item())  # 获取置信度
            cls_id = int(b.cls.item())  # 获取类别 ID
            x1, y1, x2, y2 = map(int, xyxy)  # 将坐标转换为整数
            out_list.append((cls_id, conf_, (x1, y1, x2, y2)))  # 添加到输出列表
        
        return out_list  # 返回检测结果列表


    def benchmark(self, image, num_runs=1000, conf_thres=0.6, warmup=5):
        """
        对指定图片重复推理 num_runs 次，计算平均推理时间（端到端，含预处理/后处理）。

        参数:
            image       : 图像路径 (str) 或 numpy 数组 (BGR格式)
            num_runs    : 重复推理次数（计入统计）
            conf_thres  : 置信度阈值
            warmup      : 预热运行次数（不计入统计，默认5次）

        返回:
            avg_time_ms : 平均耗时（毫秒）
        """
        # 加载图像（若传入路径）
        if isinstance(image, str):
            import cv2
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image.copy()   # 避免意外修改原图

        # 预热运行，确保模型/设备状态稳定（不计时）
        for _ in range(warmup):
            self.process(img, conf_thres=conf_thres)

        # 正式计时
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.process(img, conf_thres=conf_thres)
            end = time.perf_counter()
            times.append((end - start) * 1000)   # 转换为毫秒

        avg_time = sum(times) / len(times)
        fps = 1000.0 / avg_time
        print(f"[Benchmark] 运行次数: {num_runs}, 平均推理时间: {avg_time:.2f} ms, FPS: {fps:.2f}")
        return avg_time



if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Test YOLO detector")
    parser.add_argument("--image", type=str, default=r"a.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default=r"yolov11.pt", help="Path to .pt model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "gpu", "cuda"],
                        help="Inference provider")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    args = parser.parse_args()

    detector = YOLODetecter(args.model, device="cuda")

    #单张推理
    #results = detector.inference("test.jpg", conf_thres=args.conf, show=True)

    #批量推理
    #results = detector.inference("images", conf_thres=args.conf, show=True)

    #耗时测试
    #results = detector.benchmark("test.jpg")