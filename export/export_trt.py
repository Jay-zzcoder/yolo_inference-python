import tensorrt as trt
import os
import argparse


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
    if fp16:
        try:
            import modelopt.onnx.autocast as autocast
            import onnx
            
            fp16_onnx_path = onnx_path.replace(".onnx", "_fp16.onnx")
            if not os.path.exists(fp16_onnx_path):
                print(f"Converting {onnx_path} to FP16 ONNX using ModelOpt...")
                converted_model = autocast.convert_to_mixed_precision(
                    onnx_path=onnx_path,
                    low_precision_type="fp16",
                    keep_io_types=True, 
                )
                onnx.save(converted_model, fp16_onnx_path)
                print(f"Saved FP16 ONNX to {fp16_onnx_path}")
            onnx_path = fp16_onnx_path
        except ImportError:
            raise RuntimeError(
                "TensorRT 11.x requires NVIDIA ModelOpt for FP16 conversion.\n"
                "Install it with: pip install 'nvidia-modelopt[onnx]>=0.44'"
            )

    # ========== 步骤 2：TensorRT 构建引擎 ==========
    builder = trt.Builder(TRT_LOGGER)
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flag = 1
    
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # TRT 11.x 已经移除了 FP16/INT8 等 builder flag，这里不再设置
    if fp16 and trt.__version__.startswith("10."):
        config.set_flag(trt.BuilderFlag.FP16)

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # 若模型有动态形状，添加 optimization profile
    if any(network.get_input(i).shape[0] == -1 for i in range(network.num_inputs)):
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = list(inp.shape)
            # 假设 batch 维度是动态的，固定为 1
            shape[0] = 1
            profile.set_shape(inp.name, tuple(shape), tuple(shape), tuple(shape))
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Build failed")

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine.")
    parser.add_argument("--onnx", type=str, default="yolo11n.onnx",
                        help="Path to input ONNX model (default: yolo11n.onnx)")
    parser.add_argument("--engine", type=str, default="yolo11n_fp16.engine",
                        help="Path to output TensorRT engine (default: yolo11n_fp16.engine)")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                        help="Enable FP16 precision (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable FP16 precision")
    args = parser.parse_args()
    build_engine(args.onnx, args.engine, fp16=args.fp16)