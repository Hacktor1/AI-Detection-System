# Poznámky nastavení JetPacku pro Edge Specialista

## Přehled

Tento dokument obsahuje poznámky pro nastavení Jetson Orin Nano a
konverzi modelu do TensorRT enginy.

## 1. Instalace JetPack SDK

Viz [Průvodce instalací JetPacku](jetpack_install.md) pro detailní kroky.
Stručně:

1. **NVIDIA SDK Manager** (hostitelský PC) → flash JetPack image na SD kartu
2. První boot Jetson → dokončení nastavení OS
3. `sudo apt update && sudo apt upgrade -y`

## 2. Instalace závislostí pro edge_specialist

```bash
# ONNX Runtime (pro developerní testování)
pip install onnxruntime onnx onnx-simplifier

# TensorRT (již obsazen v JetPacku, pro Python):
/usr/src/tensorrt/python/tensorrt-10.*.whl  # nebo pip install tensorrt

# pycuda (pro TensorRT Python API)
pip install pycuda
```

## 3. TensorRT engine konverze

Na Jetsonu:

```bash
# ONNX → TensorRT (FP16, dynamické tvary)
/usr/src/tensorrt/bin/trtexec \
    --onnx=best_fp16_dynamic.onnx \
    --saveEngine=best_fp16_dynamic.engine \
    --fp16 \
    --workspace=2048 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:16x3x640x640

# Nebo pomocí Python API:
python -m edge_specialist.convert_to_trt \
    --onnx optimized_models/best_fp16_dynamic.onnx \
    --engine optimized_models/best_fp16_dynamic.engine \
    --fp16 --min-shape 1x3x640x640 --opt-shape 8x3x640x640 --max-shape 16x3x640x640
```

### INT8 kvantizace

Vyžaduje kalibrační dataset:

```bash
python -m edge_specialist.convert_to_trt \
    --onnx optimized_models/best.onnx \
    --engine optimized_models/best_int8.engine \
    --int8 \
    --calib-cache data/calibration_int8.cache \
    --min-shape 1x3x640x640 --opt-shape 8x3x640x640 --max-shape 16x3x640x640
```

## 4. Benchmark

```bash
python -m edge_specialist.benchmark \
    --engine-model optimized_models/best_fp16_dynamic.engine \
    --video pipeline_engineer/sample_data/test_thermal.mp4 \
    --frames 50
```

Očekávaný výkon na Orin Nano (8 GB):

| Model | Formát | FP | FPS @ 640×640 |
|-------|--------|-----|----------------|
| YOLOv8-nano | TensorRT | FP16 | 30–60 |
| YOLOv8-nano | ONNX Runtime | FP32 | 10–20 |
| YOLOv8-nano | TensorRT | INT8 | 40–80 |

## 5. Troubleshooting

- **`trtexec: command not found`** — JetPack není nainstalovaný, nebo cesta není v PATH
- **ONNX opset error** — použijte `--opset 13` pro export nebo nižší
- **Memory allocation failed** — snižte `--workspace` nebo použijte menší dávku
- **CUDA driver mismatch** — zkontrolujte verzi JetPacku oproti CUDA Toolkit
