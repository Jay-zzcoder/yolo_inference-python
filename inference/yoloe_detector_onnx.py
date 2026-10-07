# -*- coding: utf-8 -*-
"""
YOLOE-Seg Prompt-Free ONNX 推理封装
"""
import os
import time
import numpy as np
import cv2
import onnxruntime as ort


class YOLOESegONNX:
    def __init__(self, model_path="yoloe-v8l-seg-pf.onnx", provider='cpu',
                 class_names=None, num_classes=None, box_dim=4, mask_dim=32,
                 input_size=(640, 640)):
        """
        初始化 YOLOE 分割 ONNX 推理会话。

        :param model_path: ONNX 模型文件路径
        :param provider: 推理后端，'cpu' 或 'gpu'/'cuda'
        :param class_names: 类别名称列表；为 None 时按 model_path 同名 txt
                            或 tools/ram_tag_list.txt 加载
        :param num_classes: 类别数；为 None 时从模型输出维度自动推断
        :param box_dim: box 分支维度（默认 4，cxcywh）
        :param mask_dim: mask 系数维度（默认 32）
        :param input_size: 输入尺寸 (H, W)；为 None 时从模型输入形状推断
        """
        if provider.lower() in ('gpu', 'cuda'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # (1, 3, H, W)
        self.img_size = tuple(input_size) if input_size else \
            (self.input_shape[2], self.input_shape[3])
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.box_dim = box_dim
        self.mask_dim = mask_dim

        # 从 output0 的形状推断类别数
        out0_shape = self.session.get_outputs()[0].shape  # (1, D, N)
        D = out0_shape[1]
        self.D = D
        self.num_classes = num_classes if num_classes is not None \
            else D - box_dim - mask_dim

        # 加载类别名称
        if class_names is None:
            self.class_names = self._load_class_names(model_path)
        else:
            self.class_names = class_names

        print(f"[INFO] YOLOE-Seg loaded: D={D}, box_dim={box_dim}, "
              f"mask_dim={mask_dim}, num_classes={self.num_classes}, "
              f"num_names={len(self.class_names) if self.class_names else 0}")


    def _load_class_names(self, model_path):
        """优先从 model_path 同名 txt 加载；否则尝试 tools/ram_tag_list.txt"""
        txt_path = os.path.splitext(model_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Loaded class names from {txt_path}")
            return names

        for alt in ('tools/ram_tag_list.txt', 'ram_tag_list.txt'):
            if os.path.exists(alt):
                with open(alt, 'r', encoding='utf-8') as f:
                    names = [line.strip() for line in f if line.strip()]
                print(f"[INFO] Loaded class names from {alt}")
                return names

        n = getattr(self, 'num_classes', 4585)
        print(f"[WARN] Class names file not found, using numeric names "
              f"(0..{n - 1})")
        return [f"class{i}" for i in range(n)]


    def _preprocess(self, img):
        orig_h, orig_w = img.shape[:2]
        m_h, m_w = self.img_size
        scale = min(m_h / orig_h, m_w / orig_w)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))

        resized = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, 0, m_h - new_h,
                                    0, m_w - new_w,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        return tensor, scale


    def _nms(self, boxes_xywh, scores, iou_threshold=0.45):
        """
        :param boxes_xywh: list of (x, y, w, h) 整数
        :param scores: list of float
        :return: 保留的索引列表
        """
        if len(boxes_xywh) == 0:
            return []
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores,
                                   self.conf_thres, iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = indices[0]
        return indices.flatten().tolist()

    def _multiclass_nms(self, boxes_xyxy, scores, class_ids,
                        iou_threshold=0.45):
        """按类别分别做 NMS"""
        keep_all = []
        for c in np.unique(class_ids):
            idx = np.where(class_ids == c)[0]
            sub_boxes = boxes_xyxy[idx]
            boxes_xywh = [(int(b[0]), int(b[1]),
                           int(b[2] - b[0]), int(b[3] - b[1]))
                          for b in sub_boxes]
            k = self._nms(boxes_xywh, scores[idx].tolist(), iou_threshold)
            if len(k) > 0:
                keep_all.append(idx[k])
        if not keep_all:
            return np.array([], dtype=np.int32)
        return np.concatenate(keep_all)


    def _generate_masks(self, mask_coeffs, protos, boxes, orig_shape, scale):
        protos = protos[0]                      # (C, Hp, Wp)
        C, Hp, Wp = protos.shape
        masks = mask_coeffs @ protos.reshape(C, -1)
        masks = np.clip(masks, -50.0, 50.0)
        masks = 1.0 / (1.0 + np.exp(-masks))
        masks = masks.reshape(-1, Hp, Wp)

        orig_h, orig_w = orig_shape
        final_masks = np.zeros((len(boxes), orig_h, orig_w), dtype=np.uint8)
        valid_h = int(orig_h * scale)
        valid_w = int(orig_w * scale)

        for i in range(len(boxes)):
            mask = cv2.resize(masks[i], (Wp * 4, Hp * 4),
                              interpolation=cv2.INTER_LINEAR)
            mask = mask[:valid_h, :valid_w]
            mask = cv2.resize(mask, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)

            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            crop_mask = np.zeros_like(mask)
            crop_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            final_masks[i] = (crop_mask > 0.5).astype(np.uint8)

        return final_masks


    def process(self, image, conf_thres=0.25, iou_thres=0.45):
        """
        对输入图像做分割推理。

        :param image: BGR numpy array
        :return: [(cls_id, conf, (x1, y1, x2, y2), mask), ...]
                 mask 为 (H, W) 的 uint8（0/1）
        """
        self.conf_thres = conf_thres

        # 1. 预处理
        tensor, scale = self._preprocess(image)
        orig_h, orig_w = image.shape[:2]

        # 2. 推理
        outputs = self.session.run(self.output_names,
                                   {self.input_name: tensor})
        raw0 = outputs[0]           # (1, D, N)
        protos = outputs[1]         # (1, 32, Hp, Wp)
        det = raw0[0].T             # (N, D)
        N, D = det.shape
        print("outputs[0] shape: ", outputs[0].shape)
        print("outputs[1] shape: ", outputs[1].shape)
        # 3. 拆分
        box_feat = det[:, :self.box_dim]
        cls_feat = det[:, self.box_dim:self.box_dim + self.num_classes]
        mask_feat = det[:, self.box_dim + self.num_classes:]

        # 4. 分类
        class_ids = np.argmax(cls_feat, axis=1).astype(np.int32)
        scores = cls_feat[np.arange(N), class_ids].astype(np.float32)

        # 5. 置信度过滤
        keep = scores > conf_thres
        if not np.any(keep):
            return []

        box_raw = box_feat[keep].astype(np.float32)
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_feat = mask_feat[keep]

        # 6. cxcywh -> xyxy
        cx, cy, w, h = box_raw[:, 0], box_raw[:, 1], box_raw[:, 2], box_raw[:, 3]
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2,
                               cx + w / 2, cy + h / 2], axis=1)

        # 7. 按类别 NMS
        keep_idx = self._multiclass_nms(boxes_xyxy, scores, class_ids,
                                        iou_thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        mask_feat = mask_feat[keep_idx]

        # 8. 坐标还原到原图
        boxes_xyxy /= scale
        boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, orig_w)
        boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, orig_h)

        # 9. 生成掩码
        masks = self._generate_masks(mask_feat, protos, boxes_xyxy,
                                     (orig_h, orig_w), scale)

        # 10. 组装结果
        results = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            results.append((int(class_ids[i]),
                            float(scores[i]),
                            (int(x1), int(y1), int(x2), int(y2)),
                            masks[i]))
        return results


    def _draw(self, img_display, detections, class_names, draw_mask=True):
        for cls_id, conf, (x1, y1, x2, y2), mask in detections:
            # 掩码叠加
            if draw_mask and mask is not None:
                color = np.random.randint(60, 255, 3, dtype=np.uint8)
                region = mask == 1
                img_display[region] = (img_display[region] * 0.5
                                       + color * 0.5).astype(np.uint8)

            # 边界框
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if class_names and isinstance(class_names, list) \
                    and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                name = f"cls{cls_id}"
            label = f"{name} {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
            cv2.rectangle(img_display, (x1, y1 - th - 4),
                          (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img_display, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img_display

    def _inference_single(self, image, conf_thres=0.25, iou_thres=0.45,
                          save_path=None, show=True, class_names=None,
                          draw_mask=True):
        # 加载图像
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"无法读取图像: {image}")
            img_display = img_bgr.copy()
        else:
            img_bgr = image
            img_display = img_bgr.copy()

        detections = self.process(img_bgr, conf_thres=conf_thres,
                                  iou_thres=iou_thres)
        if class_names is None:
            class_names = self.class_names

        img_display = self._draw(img_display, detections,
                                 class_names, draw_mask=draw_mask)

        # 保存
        if save_path:
            if os.path.isdir(save_path):
                if isinstance(image, str):
                    base = os.path.basename(image)
                    name, ext = os.path.splitext(base)
                    save_path = os.path.join(save_path,
                                             f"{name}_detected{ext}")
                else:
                    save_path = os.path.join(save_path, "detected.jpg")
            cv2.imwrite(save_path, img_display)
            print(f"可视化结果已保存至: {save_path}")

        # 显示
        if show:
            cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection", 1200, 900)
            cv2.imshow("Detection", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections

 
    def inference(self, image, conf_thres=0.25, iou_thres=0.45,
                  save_path=None, show=True, class_names=None,
                  output_dir=None, draw_mask=True):
        """
        :param image: 图片路径 / BGR ndarray / 目录路径
        :return:
            单张  -> [(cls_id, conf, box, mask), ...]
            目录  -> [{'image': filename, 'detections': [...]}, ...]
        """
        # ---------- 目录批量 ----------
        if isinstance(image, str) and os.path.isdir(image):
            img_dir = image
            if output_dir is None:
                output_dir = os.path.join(img_dir, 'detected')
            os.makedirs(output_dir, exist_ok=True)

            img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            img_files = [f for f in os.listdir(img_dir)
                         if f.lower().endswith(img_exts)]
            if not img_files:
                print(f"目录 {img_dir} 中没有图片文件")
                return []

            results = []
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                dets = self._inference_single(
                    img_path, conf_thres=conf_thres, iou_thres=iou_thres,
                    save_path=output_dir, show=False,
                    class_names=class_names, draw_mask=draw_mask)
                results.append({'image': img_file, 'detections': dets})

            print(f"批量检测完成，结果保存在 {output_dir}")
            return results

        # ---------- 单张 ----------
        return self._inference_single(image, conf_thres=conf_thres,
                                      iou_thres=iou_thres,
                                      save_path=save_path, show=show,
                                      class_names=class_names,
                                      draw_mask=draw_mask)



    def benchmark(self, image, num_runs=200, conf_thres=0.25,
                  iou_thres=0.45, warmup=5):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image.copy()

        for _ in range(warmup):
            self.process(img, conf_thres=conf_thres, iou_thres=iou_thres)

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.process(img, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_time = sum(times) / len(times)
        fps = 1000.0 / avg_time
        print(f"[Benchmark] runs={num_runs}, "
              f"avg={avg_time:.2f} ms, FPS={fps:.2f}")
        return avg_time



def visualize(image, detections, class_names=None, draw_mask=True):

    result = image.copy()
    for cls_id, conf, (x1, y1, x2, y2), mask in detections:
        if draw_mask and mask is not None:
            color = np.random.randint(60, 255, 3, dtype=np.uint8)
            region = mask == 1
            result[region] = (result[region] * 0.5
                              + color * 0.5).astype(np.uint8)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if class_names and 0 <= cls_id < len(class_names):
            name = class_names[cls_id]
        else:
            name = str(cls_id)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw, y1),
                      (0, 255, 0), -1)
        cv2.putText(result, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOE-Seg ONNX 推理")
    parser.add_argument("--image", type=str,
                        default="test.jpeg",
                        help="输入图片路径或目录")
    parser.add_argument("--model", type=str,
                        default="yoloe-v8s-seg-pf.onnx",
                        help="ONNX 模型路径")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--provider", type=str, default="cuda",
                        choices=["cpu", "gpu", "cuda"])
    parser.add_argument("--output", type=str, default=None,
                        help="保存路径（文件或目录）")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹窗显示")
    parser.add_argument("--no-mask", action="store_true",
                        help="不绘制掩码，只画框")
    args = parser.parse_args()

    model = YOLOESegONNX(args.model, provider=args.provider)
    """
    # ---------- 单张推理 ----------
    dets = model.inference(
        args.image,
        conf_thres=args.conf,
        iou_thres=args.iou,
        save_path=args.output,
        show=not args.no_show,
        draw_mask=not args.no_mask,
    )
    print(f"检测到 {len(dets)} 个实例")
    """
    # ---------- 目录批量推理 ----------
    model.inference("data/", conf_thres=args.conf, show=False)

    # ---------- 性能测试 ----------
    # model.benchmark(args.image, num_runs=200)