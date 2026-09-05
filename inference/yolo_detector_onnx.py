# -*- coding: utf-8 -*-
import numpy as np
import cv2
import onnxruntime as ort
import time
import os


class YOLODetecterONNX:
    def __init__(self, model_path="yolo11n.onnx", provider='cpu', class_names=None):
        """
        初始化 ONNX 推理会话。

        :param model_path: ONNX 模型文件路径
        :param provider: 推理后端，可选 'cpu' 或 'gpu'/'cuda'。默认 'cpu'。
        """
        if provider.lower() in ('gpu', 'cuda'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # (1, 3, H, W)
        self.img_size = (self.input_shape[2], self.input_shape[3])  # (H, W)
        self.num_classes = 3   # COCO
        self.output_names = [out.name for out in self.session.get_outputs()]
        if class_names is None:
            self.class_names = self._load_class_names(model_path)
        else:
            self.class_names = class_names
        self.num_classes = len(self.class_names) if self.class_names else 80

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

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """缩放并填充图像，保持宽高比。返回填充后的图像、缩放比例、填充偏移。"""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    # ========== 修改点1：_nms 接收 (x, y, w, h) 格式 ==========
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """
        非极大值抑制。
        :param boxes: list of (x, y, w, h)  (整数)
        :param scores: list of float
        :return: 保留的索引列表
        """
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = indices[0]
        return indices.flatten().tolist()

    def _inference_single(self, image, conf_thres=0.6, save_path=None, show=True, class_names=None):
        """
        处理单张图片：检测、可视化、保存/显示。
        返回检测结果列表。
        """
        import os
        import cv2
        import numpy as np

        # 加载图像
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"无法读取图像: {image}")
            img_display = img_bgr.copy()
        else:
            img_bgr = image
            img_display = img_bgr.copy()

        detections = self.process(img_bgr, conf_thres=conf_thres)
        if class_names is None:
            class_names = self.class_names

        # 绘制所有检测框
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            color = (0, 255, 0)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            if class_names and isinstance(class_names, list) and cls_id < len(class_names):
                label = f"{class_names[cls_id]} {conf:.2f}"
            else:
                label = f"cls{cls_id} {conf:.2f}"
            cv2.putText(img_display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存
        if save_path:
            if os.path.isdir(save_path):
                # 若 image 是文件路径，取文件名；否则用默认名
                if isinstance(image, str):
                    base = os.path.basename(image)
                    name, ext = os.path.splitext(base)
                    save_path = os.path.join(save_path, f"{name}_detected{ext}")
                else:
                    save_path = os.path.join(save_path, "detected.jpg")
            cv2.imwrite(save_path, img_display)
            print(f"可视化结果已保存至: {save_path}")

        if show:
            cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection", 800, 600)
            cv2.imshow("Detection", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

        # ---------- 批量处理（输入为目录） ----------
        if isinstance(image, str) and os.path.isdir(image):
            img_dir = image
            if output_dir is None:
                output_dir = os.path.join(img_dir, 'detected')
            os.makedirs(output_dir, exist_ok=True)

            img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]
            if not img_files:
                print(f"目录 {img_dir} 中没有图片文件")
                return []

            results = []
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                dets = self._inference_single(
                    img_path,
                    conf_thres=conf_thres,
                    save_path=output_dir,   # 传入目录，函数内部自动生成文件名
                    show=False,             # 批量时禁止弹窗
                    class_names=class_names
                )
                results.append({'image': img_file, 'detections': dets})

            print(f"批量检测完成，结果保存在 {output_dir}")
            return results

        # ---------- 单张处理 ----------
        else:
            return self._inference_single(image, conf_thres, save_path, show, class_names)

    def process(self, image, conf_thres=0.6):
        """
        对输入图像进行目标检测。
        :param image: BGR numpy array
        :param conf_thres: 置信度阈值
        :return: [(cls_id, conf, (x1, y1, x2, y2)), ...]
        """
        self.conf_thres = conf_thres

        # 1. 预处理（与原代码完全相同）
        img, ratio, (dw, dh) = self._letterbox(image, self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 2. 推理
        outputs = self.session.run(self.output_names, {self.input_name: img})
        pred = outputs[0]  # shape: (1, 84, num_grids) 或 (1, num_grids, 84)
        #if pred.shape[1] == 84 and pred.shape[2] != 84:
        pred = pred.transpose(0, 2, 1)
        pred = pred[0]  # (num_grids, 84)

        # 3. 解析检测框（假定输出为 cx, cy, w, h 像素坐标，即已解码）
        boxes_cxcywh = pred[:, :4]   # (num_grids, 4)
        scores = pred[:, 4:]         # (num_grids, 80)
        max_scores = np.max(scores, axis=1)
        cls_ids = np.argmax(scores, axis=1)
        valid = max_scores >= conf_thres
        if not np.any(valid):
            return []

        boxes_cxcywh = boxes_cxcywh[valid]
        max_scores = max_scores[valid]
        cls_ids = cls_ids[valid]

        # 4. 将 cx, cy, w, h 映射回原图坐标 (x1, y1, x2, y2)
        h_img, w_img = image.shape[:2]
        scale = min(self.img_size[0] / h_img, self.img_size[1] / w_img)  # 与 letterbox 的 r 一致
        cx = (boxes_cxcywh[:, 0] - dw) / scale
        cy = (boxes_cxcywh[:, 1] - dh) / scale
        w = boxes_cxcywh[:, 2] / scale
        h = boxes_cxcywh[:, 3] / scale
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        x1 = np.clip(x1, 0, w_img - 1)
        y1 = np.clip(y1, 0, h_img - 1)
        x2 = np.clip(x2, 0, w_img - 1)
        y2 = np.clip(y2, 0, h_img - 1)

        # ========== 修改点2：构建 NMS 输入为 (x, y, w, h) ==========
        boxes_xywh = []
        for i in range(len(x1)):
            box_w = int(x2[i] - x1[i])
            box_h = int(y2[i] - y1[i])
            boxes_xywh.append((int(x1[i]), int(y1[i]), box_w, box_h))
        scores_list = max_scores.tolist()

        keep = self._nms(boxes_xywh, scores_list, iou_threshold=0.45)

        # 5. 组装返回结果
        results = []
        for idx in keep:
            cls_id = int(cls_ids[idx])
            conf = float(max_scores[idx])
            box = (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]))
            results.append((cls_id, conf, box))
        
        return results


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

        # 预热运行，确保推理状态稳定（不计时）
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

    

def show_results(img, pred=None):
    if pred is not None:
        for cls_id, conf, (x1, y1, x2, y2) in pred:
                if cls_id != 0:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"person {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", 800, 600)  
    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Test YOLOv ONNX detector")
    parser.add_argument("--image", type=str, default="a.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="yolov11.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "gpu", "cuda"],
                        help="Inference provider")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    args = parser.parse_args()

    detector = YOLODetecterONNX(args.model, provider="cuda")

    #单张推理
    #results = detector.inference("test.jpg", conf_thres=args.conf, show=True)

    #批量推理
    #results = detector.inference("images", conf_thres=args.conf, show=True)

    #耗时测试
    #results = detector.benchmark("test.jpg")
