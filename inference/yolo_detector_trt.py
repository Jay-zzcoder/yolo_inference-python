# -*- coding: utf-8 -*-
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 
import time
import os

class YOLODetecterTRT:
    def __init__(self, engine_path, input_shape=(640, 640),  class_names=None):

        if class_names is None:
            self.class_names = self._load_class_names(engine_path)
        else:
            self.class_names = class_names
        if self.class_names:
            self.num_classes = len(self.class_names)
        self.conf_thres = 0.6

        # 加载引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # 获取张量信息
        self.tensor_names = []
        self.input_names = []
        self.output_names = []
        self.tensor_shapes_orig = {}
        self.tensor_dtypes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_names.append(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.tensor_shapes_orig[name] = shape
            self.tensor_dtypes[name] = dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # 确定输入尺寸
        input_name = self.input_names[0]
        orig_in_shape = self.tensor_shapes_orig[input_name]
        if -1 in orig_in_shape:
            self.input_shape = (input_shape[0], input_shape[1])
            self._is_dynamic = True
            print(f"[INFO] Dynamic input, using shape: {self.input_shape}")
        else:
            if len(orig_in_shape) == 4:
                self.input_shape = (orig_in_shape[2], orig_in_shape[3])
            else:
                raise ValueError(f"Unsupported input shape: {orig_in_shape}")
            self._is_dynamic = False
            print(f"[INFO] Fixed input shape: {self.input_shape}")

        # 缓冲区（稍后分配）
        self.buffers = {}
        self.buffer_shapes = {}
        self.stream = cuda.Stream()



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

    def _ensure_buffers(self):
        """
        确保当前上下文中所有张量的缓冲区已分配且形状匹配。
        若形状变化则重新分配，否则复用。
        """
        for name in self.tensor_names:
            shape = self.context.get_tensor_shape(name)
            if -1 in shape:
                raise RuntimeError(f"Tensor '{name}' still has dynamic dims: {shape}")

            # 检查是否已存在且形状一致
            if name in self.buffers and np.array_equal(self.buffer_shapes[name], shape):
                continue  # 直接复用

            # 需要分配（或重新分配）
            size = trt.volume(shape)
            dtype = self.tensor_dtypes[name]

            # 释放旧内存（如有）
            if name in self.buffers:
                # pycuda 对象在 del 时自动释放，显式删除帮助 GC
                del self.buffers[name]
                del self.buffer_shapes[name]

            # 分配新的主机锁页内存和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.buffers[name] = (host_mem, device_mem)
            self.buffer_shapes[name] = shape

            # 设置张量地址（只需在分配时设置一次）
            self.context.set_tensor_address(name, int(device_mem))

    def _allocate_buffers(self):
        """根据当前上下文的实际形状分配主机/设备内存，并设置张量地址"""
        self.buffers = {}
        for name in self.tensor_names:
            shape = self.context.get_tensor_shape(name)
            if -1 in shape:
                # 若仍有动态维度，可能无法处理，但通常输入已设置，输出已知
                raise RuntimeError(f"Tensor '{name}' still has dynamic dims: {shape}")
            size = trt.volume(shape)
            dtype = self.tensor_dtypes[name]
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.buffers[name] = (host_mem, device_mem)
            # 关键：设置张量地址
            self.context.set_tensor_address(name, int(device_mem))

    def _letterbox(self, img, new_shape, color=(114, 114, 114)):
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

    def _nms(self, boxes, scores, iou_threshold=0.45):
        #print("boxes length: ", len(boxes))
        #print("scores length: ", len(scores))
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = indices[0]
        return indices.flatten().tolist()

    def process(self, image, conf_thres=0.6):

        self.conf_thres = conf_thres

        # 1. 预处理
        img, ratio, (dw, dh) = self._letterbox(image, self.input_shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 2. 设置输入形状（若动态）
        input_name = self.input_names[0]
        if self._is_dynamic:
            actual_shape = list(self.tensor_shapes_orig[input_name])
            for i, d in enumerate(actual_shape):
                if d == -1:
                    if i == 0:      actual_shape[i] = img.shape[0]
                    elif i == 1:    actual_shape[i] = img.shape[1]
                    elif i == 2:    actual_shape[i] = img.shape[2]
                    elif i == 3:    actual_shape[i] = img.shape[3]
            self.context.set_input_shape(input_name, actual_shape)
        else:
            expected_h, expected_w = self.input_shape
            if img.shape[2] != expected_h or img.shape[3] != expected_w:
                raise ValueError(f"Image size {img.shape[2:]} != engine input {expected_h}x{expected_w}")

        # 3. 分配缓冲区
        self._ensure_buffers()

        # 4. 复制输入数据到GPU
        host_mem, device_mem = self.buffers[input_name]
        expected_size = host_mem.size
        assert img.size == expected_size, f"Input size mismatch: {img.size} vs {expected_size}"
        np.copyto(host_mem, img.ravel())
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)

        # 5. 执行推理
        self.context.execute_async_v3(self.stream.handle)

        # 6. 复制输出回CPU
        for name in self.output_names:
            host_mem, device_mem = self.buffers[name]
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
        self.stream.synchronize()

        # 7. 解析输出
        output_name = self.output_names[0]
        host_mem, _ = self.buffers[output_name]
        output_shape = self.context.get_tensor_shape(output_name)
        if -1 in output_shape:
            total_size = host_mem.size
            if total_size % (self.num_classes + 4) == 0:
                num_detections = total_size // (self.num_classes + 4)
                pred = host_mem.reshape(1, num_detections, self.num_classes + 4)
                if pred.shape[2] != self.num_classes + 4:
                    pred = pred.transpose(0, 2, 1)
            else:
                raise RuntimeError("Cannot infer output shape")
        else:
            pred = host_mem.reshape(output_shape)
            if pred.shape[1] == self.num_classes + 4 and pred.shape[2] != self.num_classes + 4:
                pred = pred.transpose(0, 2, 1)
            elif pred.shape[2] == self.num_classes + 4 and pred.shape[1] != self.num_classes + 4:
                pass
            else:
                pred = pred.transpose(0, 2, 1)
        pred = pred[0]  # (N, 84)

        # 8. 提取有效检测
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:]
        max_scores = np.max(scores, axis=1)
        cls_ids = np.argmax(scores, axis=1)
        valid = max_scores >= conf_thres
        if not np.any(valid):
            return []

        boxes_cxcywh = boxes_cxcywh[valid]
        max_scores = max_scores[valid]
        cls_ids = cls_ids[valid]

        # 9. 映射回原图
        h_img, w_img = image.shape[:2]
        scale = min(self.input_shape[0] / h_img, self.input_shape[1] / w_img)
        cx = (boxes_cxcywh[:, 0] - dw) / scale
        cy = (boxes_cxcywh[:, 1] - dh) / scale
        w = boxes_cxcywh[:, 2] / scale
        h = boxes_cxcywh[:, 3] / scale
        x1 = np.clip(cx - w/2, 0, w_img - 1)
        y1 = np.clip(cy - h/2, 0, h_img - 1)
        x2 = np.clip(cx + w/2, 0, w_img - 1)
        y2 = np.clip(cy + h/2, 0, h_img - 1)

        # 10. NMS
        boxes_xywh = [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in range(len(x1))]
        keep = self._nms(boxes_xywh, max_scores.tolist(), iou_threshold=0.45)

        # 11. 返回所有类别（不过滤）
        results = []
        for idx in keep:
            cls_id = int(cls_ids[idx])
            conf = float(max_scores[idx])
            box = (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]))
            results.append((cls_id, conf, box))
        return results

    def _inference_single(self, image, conf_thres=0.6, save_path=None, show=True, class_names=None):
        """
        处理单张图片：检测、可视化、保存/显示（所有类别）。
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

        # 检测所有类别
        detections = self.process(img_bgr, conf_thres=conf_thres)

        if class_names is None:
            class_names = self.class_names

        # 绘制所有检测框（颜色随机或按类分配，此处统一为绿色）
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            color = (0, 255, 0)  # 绿色，可改为根据类别变化
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
                    save_path=output_dir,
                    show=False,
                    class_names=class_names
                )
                results.append({'image': img_file, 'detections': dets})

            print(f"批量检测完成，结果保存在 {output_dir}")
            return results

        # ---------- 单张处理 ----------
        else:
            return self._inference_single(image, conf_thres, save_path, show, class_names)

    def detect_person(self, image, conf_thres=0.6, visualization=False):
        self.conf_thres = conf_thres

        # 1. 预处理
        img, ratio, (dw, dh) = self._letterbox(image, self.input_shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 2. 设置输入形状（若动态）
        input_name = self.input_names[0]
        if self._is_dynamic:
            actual_shape = list(self.tensor_shapes_orig[input_name])
            for i, d in enumerate(actual_shape):
                if d == -1:
                    if i == 0:      actual_shape[i] = img.shape[0]
                    elif i == 1:    actual_shape[i] = img.shape[1]
                    elif i == 2:    actual_shape[i] = img.shape[2]
                    elif i == 3:    actual_shape[i] = img.shape[3]
            self.context.set_input_shape(input_name, actual_shape)
        else:
            expected_h, expected_w = self.input_shape
            if img.shape[2] != expected_h or img.shape[3] != expected_w:
                raise ValueError(f"Image size {img.shape[2:]} != engine input {expected_h}x{expected_w}")

        # 3. 分配缓冲区（内部调用 set_tensor_address）
        #self._allocate_buffers()
        self._ensure_buffers()
        # 4. 复制输入数据到GPU
        host_mem, device_mem = self.buffers[input_name]
        expected_size = host_mem.size
        assert img.size == expected_size, f"Input size mismatch: {img.size} vs {expected_size}"
        np.copyto(host_mem, img.ravel())
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)

        # 5. 执行推理（新API：只需传入流句柄）
        self.context.execute_async_v3(self.stream.handle)

        # 6. 复制输出回CPU
        for name in self.output_names:
            host_mem, device_mem = self.buffers[name]
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
        self.stream.synchronize()

        # 7. 解析输出（自动适应维度）
        output_name = self.output_names[0]
        host_mem, _ = self.buffers[output_name]
        output_shape = self.context.get_tensor_shape(output_name)
        if -1 in output_shape:
            total_size = host_mem.size
            if total_size % (self.num_classes + 4) == 0:
                num_detections = total_size // (self.num_classes + 4)
                pred = host_mem.reshape(1, num_detections, self.num_classes + 4)
                if pred.shape[2] != self.num_classes + 4:
                    pred = pred.transpose(0, 2, 1)
            else:
                raise RuntimeError("Cannot infer output shape")
        else:
            pred = host_mem.reshape(output_shape)
            # 调整为 (1, N, 84)
            if pred.shape[1] == self.num_classes + 4 and pred.shape[2] != self.num_classes + 4:
                pred = pred.transpose(0, 2, 1)
            elif pred.shape[2] == self.num_classes + 4 and pred.shape[1] != self.num_classes + 4:
                pass
            else:
                pred = pred.transpose(0, 2, 1)  # 尝试转置
        pred = pred[0]  # (N, 84)

        # 8. 提取有效检测
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:]
        max_scores = np.max(scores, axis=1)
        cls_ids = np.argmax(scores, axis=1)
        valid = max_scores >= conf_thres
        if not np.any(valid):
            return []

        boxes_cxcywh = boxes_cxcywh[valid]
        max_scores = max_scores[valid]
        cls_ids = cls_ids[valid]

        # 9. 映射回原图
        h_img, w_img = image.shape[:2]
        scale = min(self.input_shape[0] / h_img, self.input_shape[1] / w_img)
        cx = (boxes_cxcywh[:, 0] - dw) / scale
        cy = (boxes_cxcywh[:, 1] - dh) / scale
        w = boxes_cxcywh[:, 2] / scale
        h = boxes_cxcywh[:, 3] / scale
        x1 = np.clip(cx - w/2, 0, w_img - 1)
        y1 = np.clip(cy - h/2, 0, h_img - 1)
        x2 = np.clip(cx + w/2, 0, w_img - 1)
        y2 = np.clip(cy + h/2, 0, h_img - 1)

        # 10. NMS
        boxes_xywh = [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in range(len(x1))]
        keep = self._nms(boxes_xywh, max_scores.tolist(), iou_threshold=0.45)

        # 11. 仅保留person
        results = []
        for idx in keep:
            cls_id = int(cls_ids[idx])
            conf = float(max_scores[idx])
            if cls_id == 0:
                box = (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]))
                results.append((cls_id, conf, box))

        # 12. 可视化
        if visualization and results:
            for cls_id, conf, (x1, y1, x2, y2) in results:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"person {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection", 800, 600)
            cv2.imshow("Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    
    def inference_profiling(self, image, conf_thres=0.6,  return_timing=False):
        self.conf_thres = conf_thres
        timing = {}

        # 1. 预处理
        t0 = time.perf_counter()
        img, ratio, (dw, dh) = self._letterbox(image, self.input_shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        timing['preprocess'] = (time.perf_counter() - t0) * 1000

        # 2. 设置输入形状（若动态）
        input_name = self.input_names[0]
        if self._is_dynamic:
            actual_shape = list(self.tensor_shapes_orig[input_name])
            for i, d in enumerate(actual_shape):
                if d == -1:
                    if i == 0:      actual_shape[i] = img.shape[0]
                    elif i == 1:    actual_shape[i] = img.shape[1]
                    elif i == 2:    actual_shape[i] = img.shape[2]
                    elif i == 3:    actual_shape[i] = img.shape[3]
            self.context.set_input_shape(input_name, actual_shape)
        else:
            expected_h, expected_w = self.input_shape
            if img.shape[2] != expected_h or img.shape[3] != expected_w:
                raise ValueError(f"Image size {img.shape[2:]} != engine input {expected_h}x{expected_w}")

        # 3. 分配缓冲区（内部调用 set_tensor_address）
        self._ensure_buffers()

        # ---------- 推理阶段（拷贝输入 → 执行 → 拷贝输出） ----------
        t_infer_start = time.perf_counter()

        # 4. 复制输入数据到GPU
        host_mem, device_mem = self.buffers[input_name]
        expected_size = host_mem.size
        assert img.size == expected_size, f"Input size mismatch: {img.size} vs {expected_size}"
        np.copyto(host_mem, img.ravel())
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)

        # 5. 执行推理（新API：只需传入流句柄）
        self.context.execute_async_v3(self.stream.handle)

        # 6. 复制输出回CPU
        for name in self.output_names:
            host_mem, device_mem = self.buffers[name]
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
        self.stream.synchronize()

        timing['inference'] = (time.perf_counter() - t_infer_start) * 1000
        # ------------------------------------------------

        # 7. 解析输出
        t_parse_start = time.perf_counter()
        output_name = self.output_names[0]
        host_mem, _ = self.buffers[output_name]
        output_shape = self.context.get_tensor_shape(output_name)
        if -1 in output_shape:
            total_size = host_mem.size
            if total_size % (self.num_classes + 4) == 0:
                num_detections = total_size // (self.num_classes + 4)
                pred = host_mem.reshape(1, num_detections, self.num_classes + 4)
                if pred.shape[2] != self.num_classes + 4:
                    pred = pred.transpose(0, 2, 1)
            else:
                raise RuntimeError("Cannot infer output shape")
        else:
            pred = host_mem.reshape(output_shape)
            # 调整为 (1, N, 84)
            if pred.shape[1] == self.num_classes + 4 and pred.shape[2] != self.num_classes + 4:
                pred = pred.transpose(0, 2, 1)
            elif pred.shape[2] == self.num_classes + 4 and pred.shape[1] != self.num_classes + 4:
                pass
            else:
                pred = pred.transpose(0, 2, 1)  # 尝试转置
        pred = pred[0]  # (N, 84)

        # 8. 提取有效检测
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:]
        max_scores = np.max(scores, axis=1)
        cls_ids = np.argmax(scores, axis=1)
        valid = max_scores >= conf_thres
        if not np.any(valid):
            timing['postprocess_parse'] = (time.perf_counter() - t_parse_start) * 1000
            timing['nms'] = 0.0
            timing['postprocess_filter'] = 0.0
            if return_timing:
                return [], timing
            return []

        boxes_cxcywh = boxes_cxcywh[valid]
        max_scores = max_scores[valid]
        cls_ids = cls_ids[valid]

        # 9. 映射回原图
        h_img, w_img = image.shape[:2]
        scale = min(self.input_shape[0] / h_img, self.input_shape[1] / w_img)
        cx = (boxes_cxcywh[:, 0] - dw) / scale
        cy = (boxes_cxcywh[:, 1] - dh) / scale
        w = boxes_cxcywh[:, 2] / scale
        h = boxes_cxcywh[:, 3] / scale
        x1 = np.clip(cx - w/2, 0, w_img - 1)
        y1 = np.clip(cy - h/2, 0, h_img - 1)
        x2 = np.clip(cx + w/2, 0, w_img - 1)
        y2 = np.clip(cy + h/2, 0, h_img - 1)
        boxes_xywh = [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in range(len(x1))]
        timing['postprocess_parse'] = (time.perf_counter() - t_parse_start) * 1000

        # 10. NMS
        t_nms_start = time.perf_counter()
        keep = self._nms(boxes_xywh, max_scores.tolist(), iou_threshold=0.45)
        timing['nms'] = (time.perf_counter() - t_nms_start) * 1000

        # 11. 仅保留person
        t_filter_start = time.perf_counter()
        results = []
        for idx in keep:
            cls_id = int(cls_ids[idx])
            conf = float(max_scores[idx])
            if cls_id == 0:
                box = (int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]))
                results.append((cls_id, conf, box))
        timing['postprocess_filter'] = (time.perf_counter() - t_filter_start) * 1000

        if return_timing:
            return results, timing
        return results

    def benchmark(self, image, num_runs=1000, conf_thres=0.6, warmup=5):
        """
        对指定图片重复推理 num_runs 次，统计各阶段平均耗时及占比。

        参数:
            image       : 图像路径 (str) 或 numpy 数组 (BGR格式)
            num_runs    : 重复推理次数（计入统计）
            conf_thres  : 置信度阈值
            warmup      : 预热运行次数（不计入统计，默认5次）

        返回:
            avg_detect_ms : 平均总耗时（毫秒）
        """
        # 加载图像（若传入路径）
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
        else:
            img = image.copy()   # 避免意外修改原图

        # 预热运行，确保 GPU 状态稳定（不计时）
        for _ in range(warmup):
            self.inference_profiling(img, conf_thres=conf_thres, visualization=False, return_timing=False)

        # 正式计时
        total_detect = 0.0
        total_preprocess = 0.0
        total_inference = 0.0
        total_parse = 0.0
        total_nms = 0.0
        total_filter = 0.0

        for _ in range(num_runs):
            start = time.perf_counter()
            _, timing = self.inference_profiling(img, conf_thres=conf_thres, visualization=False, return_timing=True)
            end = time.perf_counter()
            total_detect += (end - start) * 1000
            total_preprocess += timing.get('preprocess', 0.0)
            total_inference += timing.get('inference', 0.0)
            total_parse += timing.get('postprocess_parse', 0.0)
            total_nms += timing.get('nms', 0.0)
            total_filter += timing.get('postprocess_filter', 0.0)

        avg_detect = total_detect / num_runs
        avg_preprocess = total_preprocess / num_runs
        avg_inference = total_inference / num_runs
        avg_parse = total_parse / num_runs
        avg_nms = total_nms / num_runs
        avg_filter = total_filter / num_runs

        # 计算占比（以总检测时间为分母）
        ratio_preprocess = (avg_preprocess / avg_detect) * 100
        ratio_inference = (avg_inference / avg_detect) * 100
        ratio_parse = (avg_parse / avg_detect) * 100
        ratio_nms = (avg_nms / avg_detect) * 100
        ratio_filter = (avg_filter / avg_detect) * 100
        # 其他未被单独统计的时间（如函数调用开销等）
        other = avg_detect - avg_preprocess - avg_inference - avg_parse - avg_nms - avg_filter
        ratio_other = 100 - ratio_preprocess - ratio_inference - ratio_parse - ratio_nms - ratio_filter

        print(f"[Benchmark] 运行次数: {num_runs}")
        print(f"  平均检测总耗时: {avg_detect:.4f} ms")
        print(f"  其中:")
        print(f"    - 预处理 (letterbox, 归一化等): {avg_preprocess:.4f} ms  ({ratio_preprocess:.1f}%)")
        print(f"    - 推理 (含数据拷贝与执行): {avg_inference:.4f} ms  ({ratio_inference:.1f}%)")
        print(f"    - 后处理解析 (reshape, 阈值筛选, 坐标映射): {avg_parse:.4f} ms  ({ratio_parse:.1f}%)")
        print(f"    - NMS (非极大值抑制): {avg_nms:.4f} ms  ({ratio_nms:.1f}%)")
        print(f"    - 后处理过滤 (仅保留person): {avg_filter:.4f} ms  ({ratio_filter:.1f}%)")
        print(f"    - 其他 (未计时的微小开销): {other:.4f} ms  ({ratio_other:.1f}%)")
        print(f"  整体 FPS: {1000.0 / avg_detect:.2f}")

        return avg_detect


def show_results(img, pred=None):
    """独立可视化函数（保留原接口）"""
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

    parser = argparse.ArgumentParser(description="Test YOLOv11 TensorRT detector")
    parser.add_argument("--image", type=str, default="a.jpg", help="Path to input image")
    parser.add_argument("--engine", type=str, default=r"E:\Robotics\Code\yolo_inference-python\models\fire\fire_smoke_yolov8.engine", help="Path to TensorRT engine file (.engine)")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image (not implemented)")
    args = parser.parse_args()

    detector = YOLODetecterTRT(args.engine, input_shape=(480, 640))

    #单张推理
    #results = detector.inference("test.jpg", conf_thres=args.conf, show=True)

    #批量推理
    #results = detector.inference("images", conf_thres=args.conf, show=True)

    #耗时测试
    #results = detector.benchmark("test.jpg")
