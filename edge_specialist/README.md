# Edge Specialist Role

## Task Overview

Optimize the trained YOLO model and deploy it to NVIDIA Jetson Orin Nano. Convert to ONNX → TensorRT, measure FPS, and ensure real-time performance.

## Key Activities

1. **Model Optimization**
   - Convert PyTorch `.pt` → ONNX.
   - Use ONNX to TensorRT engine builder.
   - Quantize (FP16 / INT8) to reduce size and increase FPS.

2. **Performance Benchmarking**
   - Measure FPS, latency, model size.
   - Compare PyTorch vs ONNX vs TensorRT on target hardware.

3. **Jetson Integration**
   - Install JetPack SDK dependencies (when on Jetson hardware).
   - Verify TensorRT + CUDA are available.
   - Run the model on Jetson and report throughput.

## Directory Layout
```
edge_specialist/
├── convert_to_trt.py       # ONNX → TensorRT conversion
├── benchmark.py           # FPS/latency measurement
├── jetpack_setup.md       # Jetson setup notes
└── optimized_models/      # Final .engine files (gitignored)
```

## Conversion Command (example)
```bash
# ONNX → TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

## Benchmark
```bash
python benchmark.py --model optimized_models/model.engine --frames 100
```

## Notes
- The `.engine` file is platform-specific (must be built on Jetson or compatible GPU).
- Document JetPack version and TensorRT version used.