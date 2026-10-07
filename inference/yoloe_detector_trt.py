# -*- coding: utf-8 -*-
"""
YOLOE-Seg Prompt-Free TensorRT 推理封装
"""
import os
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class DDSOutputAllocator(trt.IOutputAllocator):

    _ALIGN = 1 << 20  # 1MB

    def __init__(self, debug=False):
        super().__init__()
        self.device_ptrs = {}   # name -> (DeviceAllocation, allocated_size)
        self.shapes = {}        # name -> tuple
        self.debug = debug

    def reset(self):
        self.shapes.clear()

    def reallocate_output(self, tensor_name, memory, size, alignment):
        if self.debug:
            print(f"[DDS] reallocate_output('{tensor_name}', size={size})")

        if tensor_name in self.device_ptrs:
            old_ptr, old_size = self.device_ptrs[tensor_name]
            if old_size >= size:
                # 复用旧 buffer
                return int(old_ptr)
            # 需要更大的 buffer：先释放旧的 DeviceAllocation
            try:
                old_ptr.free()
            except Exception:
                pass
            del self.device_ptrs[tensor_name]

        alloc_size = ((size + self._ALIGN - 1) // self._ALIGN) * self._ALIGN
        ptr = cuda.mem_alloc(alloc_size)
        self.device_ptrs[tensor_name] = (ptr, alloc_size)
        return int(ptr)

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(int(d) for d in shape)
        if self.debug:
            print(f"[DDS] notify_shape('{tensor_name}', "
                  f"shape={self.shapes[tensor_name]})")
        return None


class YOLOESegTRT:
    def __init__(self, engine_path, input_shape=(640, 640),
                 class_names=None, num_classes=None,
                 box_dim=4, mask_dim=32,
                 debug=False):
        self.box_dim = box_dim
        self.mask_dim = mask_dim
        self.debug = debug
        self.conf_thres = 0.25

        # ---------- 加载引擎 ----------
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # ---------- 张量信息 ----------
        self.tensor_names, self.input_names, self.output_names = [], [], []
        self.tensor_shapes_orig, self.tensor_dtypes = {}, {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_names.append(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            self.tensor_shapes_orig[name] = shape
            self.tensor_dtypes[name] = trt.nptype(trt_dtype)
            is_input = (self.engine.get_tensor_mode(name)
                        == trt.TensorIOMode.INPUT)
            (self.input_names if is_input
             else self.output_names).append(name)
            print(f"[INFO] Engine tensor '{name}': "
                  f"{'INPUT ' if is_input else 'OUTPUT'}, "
                  f"dtype={trt_dtype}, shape={shape}")

        # ---------- 输入尺寸 ----------
        input_name = self.input_names[0]
        orig_in_shape = self.tensor_shapes_orig[input_name]
        if -1 in orig_in_shape:
            self.input_shape = tuple(input_shape)   # (H, W)
            self._is_dynamic = True
            print(f"[INFO] Dynamic input, using shape: {self.input_shape}")
        else:
            self.input_shape = (orig_in_shape[2], orig_in_shape[3])
            self._is_dynamic = False
            print(f"[INFO] Fixed input shape: {self.input_shape}")

        # ---------- 区分 det / proto ----------
        self.det_output_name, self.proto_output_name = None, None
        for name in self.output_names:
            s = self.tensor_shapes_orig[name]
            if len(s) == 4 and s[1] == self.mask_dim:
                self.proto_output_name = name
            elif len(s) == 3:
                self.det_output_name = name
        if self.det_output_name is None:
            self.det_output_name = self.output_names[0]
        if self.proto_output_name is None:
            self.proto_output_name = self.output_names[1]

        # ---------- num_classes ----------
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            inferred = None
            s = self.tensor_shapes_orig.get(self.det_output_name)
            if s and len(s) == 3:
                for d in s:
                    if d != -1 and d > self.box_dim + self.mask_dim:
                        cand = d - self.box_dim - self.mask_dim
                        if 0 < cand < 100000:
                            inferred = cand
                            break
            if inferred:
                self.num_classes = inferred
            else:
                self.num_classes = 4585

        # ---------- 类别名 ----------
        if class_names is None:
            self.class_names = self._load_class_names(engine_path)
        else:
            self.class_names = class_names

        print(f"[INFO] YOLOE-Seg TRT loaded: box_dim={self.box_dim}, "
              f"mask_dim={self.mask_dim}, num_classes={self.num_classes}, "
              f"num_names={len(self.class_names) if self.class_names else 0}")

        # ---------- 缓冲区 ----------
        self.buffers = {}          # name -> (host_np, DeviceAllocation)
        self.buffer_shapes = {}    # name -> tuple
        self.stream = cuda.Stream()

        # ---------- DDS ----------
        self.dds_outputs = set()
        self.allocators = {}

        # 先 prime 输入形状，再识别 DDS
        self._prime_shapes()
        self._setup_dds_outputs()

        # 提前分配一次缓冲区
        self._ensure_buffers()


    def _load_class_names(self, model_path):
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


    def _prime_shapes(self):
        input_name = self.input_names[0]
        h, w = self.input_shape
        try:
            self.context.set_input_shape(input_name, (1, 3, h, w))
        except Exception as e:
            if self._is_dynamic:
                raise
            if self.debug:
                print(f"[WARN] set_input_shape on fixed engine: {e}")
        self._infer_shapes_safe()
        if self.debug:
            for name in self.tensor_names:
                try:
                    s = tuple(self.context.get_tensor_shape(name))
                except Exception:
                    s = self.tensor_shapes_orig[name]
                print(f"[DEBUG] Tensor '{name}' shape after priming: {s}")

    def _infer_shapes_safe(self):
        if hasattr(self.context, 'infer_shapes'):
            try:
                self.context.infer_shapes()
            except Exception:
                pass

    def _setup_dds_outputs(self):

        for name in self.output_names:
            try:
                shape = tuple(self.context.get_tensor_shape(name))
            except Exception:
                shape = self.tensor_shapes_orig[name]
            if -1 in shape:
                self.dds_outputs.add(name)
                alloc = DDSOutputAllocator(debug=self.debug)
                self.allocators[name] = alloc
                self.context.set_output_allocator(name, alloc)
                print(f"[INFO] Output '{name}' is DDS (shape={shape}), "
                      f"using IOutputAllocator")

        if not self.dds_outputs:
            print(f"[INFO] No DDS outputs detected; using static buffers")


    def _get_resolved_shape(self, name):
        self._infer_shapes_safe()
        shape = tuple(self.context.get_tensor_shape(name))
        if -1 not in shape:
            return shape

        shape = list(shape)
        h, w = self.input_shape
        if name == self.proto_output_name and len(shape) == 4:
            if shape[2] == -1:
                shape[2] = h // 4
            if shape[3] == -1:
                shape[3] = w // 4
        else:
            raise RuntimeError(
                f"Cannot resolve dynamic shape for '{name}': {tuple(shape)}")
        resolved = tuple(shape)
        if self.debug:
            print(f"[DEBUG] Manually resolved shape for '{name}': {resolved}")
        return resolved


    def _ensure_buffers(self):
        """
        为非 DDS 张量分配 host/device buffer 并 set_tensor_address。
        DDS 输出由 allocator 处理，跳过。
        """
        for name in self.tensor_names:
            if name in self.dds_outputs:
                continue

            shape = self._get_resolved_shape(name)

            # 已分配且 shape 一致 → 复用
            if name in self.buffers and self.buffer_shapes.get(name) == shape:
                continue

            # shape 变化或首次 → 重新分配
            if name in self.buffers:
                try:
                    _, old_dev = self.buffers[name]
                    old_dev.free()      # ← DeviceAllocation.free()
                except Exception:
                    pass
                del self.buffers[name]
                self.buffer_shapes.pop(name, None)

            size = int(trt.volume(shape))
            dtype = self.tensor_dtypes[name]
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.buffers[name] = (host_mem, device_mem)
            self.buffer_shapes[name] = shape
            self.context.set_tensor_address(name, int(device_mem))
            if self.debug:
                print(f"[DEBUG] Allocated '{name}': shape={shape}, "
                      f"dtype={dtype}, size={size}")


    def _preprocess(self, img):
        orig_h, orig_w = img.shape[:2]
        m_h, m_w = self.input_shape
        scale = min(m_h / orig_h, m_w / orig_w)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        resized = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, 0, m_h - new_h,
                                    0, m_w - new_w,
                                    cv2.BORDER_CONSTANT,
                                    value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        return tensor, scale


    def _nms(self, boxes_xywh, scores, iou_threshold=0.45):
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


    def _decode_det_output(self, det_host, det_shape):
        D = self.num_classes + self.box_dim + self.mask_dim
        if len(det_shape) != 3:
            raise RuntimeError(f"Unsupported det output shape: {det_shape}")

        valid_size = int(trt.volume(det_shape))
        raw = np.asarray(det_host[:valid_size],
                         dtype=np.float32).reshape(det_shape)
        if det_shape[1] == D and det_shape[2] != D:
            det = raw[0].T
        elif det_shape[2] == D and det_shape[1] != D:
            det = raw[0]
        else:
            raise RuntimeError(
                f"Cannot determine det layout from shape {det_shape} with D={D}")

        box_feat = det[:, :self.box_dim]
        cls_feat = det[:, self.box_dim:self.box_dim + self.num_classes]
        mask_feat = det[:, self.box_dim + self.num_classes:]
        return det, box_feat, cls_feat, mask_feat


    def process(self, image, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres

        # 0. 每帧推理前重置 DDS allocator 的 shape 缓存
        for alloc in self.allocators.values():
            alloc.reset()

        # 1. 预处理
        tensor, scale = self._preprocess(image)
        orig_h, orig_w = image.shape[:2]

        # 2. 设置输入形状
        input_name = self.input_names[0]
        try:
            self.context.set_input_shape(input_name, tuple(tensor.shape))
        except Exception:
            if self._is_dynamic:
                raise
        self._infer_shapes_safe()

        # 3. 分配非 DDS buffer
        self._ensure_buffers()

        # 4. 拷贝输入（显式全量覆盖）
        host_mem, device_mem = self.buffers[input_name]
        assert tensor.size == host_mem.size, \
            f"Input size mismatch: {tensor.size} vs {host_mem.size}"
        host_mem[:] = tensor.ravel()
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)

        # 5. 执行
        self.context.execute_async_v3(self.stream.handle)

        # 6. DDS 输出：每次读 allocator 当前 shape/ptr
        for name in self.dds_outputs:
            alloc = self.allocators[name]
            if name not in alloc.shapes:
                raise RuntimeError(
                    f"DDS output '{name}' did not report shape. "
                    f"Check that IOutputAllocator is registered correctly.")
            shape = alloc.shapes[name]
            ptr, _ = alloc.device_ptrs[name]
            host = np.empty(int(np.prod(shape)),
                            dtype=self.tensor_dtypes[name])
            cuda.memcpy_dtoh_async(host, ptr, self.stream)
            self.buffers[name] = (host, ptr)
            self.buffer_shapes[name] = shape
            if self.debug:
                print(f"[DEBUG] DDS '{name}': shape={shape}, "
                      f"bytes={host.nbytes}")

        # 7. 非 DDS 输出拷贝
        for name in self.output_names:
            if name in self.dds_outputs:
                continue
            hm, dm = self.buffers[name]
            cuda.memcpy_dtoh_async(hm, dm, self.stream)

        # 8. 同步
        self.stream.synchronize()

        # 9. 解析 det
        det_shape = self.buffer_shapes[self.det_output_name]
        proto_shape = self.buffer_shapes[self.proto_output_name]
        if self.debug:
            print(f"[DEBUG] Runtime '{self.det_output_name}' shape: {det_shape}")
            print(f"[DEBUG] Runtime '{self.proto_output_name}' shape: {proto_shape}")

        det_host, _ = self.buffers[self.det_output_name]
        det, box_feat, cls_feat, mask_feat = self._decode_det_output(
            det_host, det_shape)
        N = det.shape[0]

        # 10. proto
        proto_host, _ = self.buffers[self.proto_output_name]
        proto_size = int(trt.volume(proto_shape))
        protos = np.asarray(proto_host[:proto_size],
                            dtype=np.float32).reshape(proto_shape)

        # 11. 分类
        class_ids = np.argmax(cls_feat, axis=1).astype(np.int32)
        scores = cls_feat[np.arange(N), class_ids].astype(np.float32)

        # 12. 阈值过滤
        keep = scores > conf_thres
        if not np.any(keep):
            return []
        box_raw = box_feat[keep].astype(np.float32)
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_feat = mask_feat[keep]

        # 13. cxcywh -> xyxy
        cx, cy, w, h = (box_raw[:, 0], box_raw[:, 1],
                        box_raw[:, 2], box_raw[:, 3])
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2,
                               cx + w / 2, cy + h / 2], axis=1)

        # 14. 按类别 NMS
        keep_idx = self._multiclass_nms(boxes_xyxy, scores, class_ids,
                                        iou_thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        mask_feat = mask_feat[keep_idx]

        # 15. 映射回原图
        boxes_xyxy = boxes_xyxy.astype(np.float32)
        boxes_xyxy /= scale
        boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, orig_w)
        boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, orig_h)

        # 16. 生成掩码
        masks = self._generate_masks(mask_feat, protos, boxes_xyxy,
                                     (orig_h, orig_w), scale)

        # 17. 组装结果
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
            if draw_mask and mask is not None:
                color = np.random.randint(60, 255, 3, dtype=np.uint8)
                region = mask == 1
                img_display[region] = (img_display[region] * 0.5
                                       + color * 0.5).astype(np.uint8)
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
        name = (class_names[cls_id] if class_names and
                0 <= cls_id < len(class_names) else str(cls_id))
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw, y1),
                      (0, 255, 0), -1)
        cv2.putText(result, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOE-Seg TensorRT 推理")
    parser.add_argument("--image", type=str,
                        default="test.jpg",
                        help="输入图片路径或目录")
    parser.add_argument("--engine", type=str,
                        default="yoloe-v8s-seg-pf.engine",
                        help="TensorRT engine 路径")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=str, default=None,
                        help="保存路径（文件或目录）")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹窗显示")
    parser.add_argument("--no-mask", action="store_true",
                        help="不绘制掩码，只画框")
    parser.add_argument("--debug", action="store_true",
                        help="打印调试信息")
    args = parser.parse_args()

    model = YOLOESegTRT(args.engine, debug=args.debug)

    # ---------- 单张 / 目录推理 ----------
    dets = model.inference(
        "data",
        conf_thres=args.conf,
        iou_thres=args.iou,
        save_path=args.output,
        show=args.no_show,
        draw_mask=not args.no_mask,
    )
    #if isinstance(dets, list) and dets and isinstance(dets[0], dict):
        #print(f"批量检测完成：{len(dets)} 张图像")
    #else:
    #    print(f"检测到 {len(dets)} 个实例")

    # ---------- 性能测试 ----------
    # model.benchmark(args.image, num_runs=200)