# Nasazení modelu na Jetson Orin Nano

## Postup

### 1. Export z PyTorch do ONNX

Na vývojovém stroji:

```bash
cd edge_specialist
python3 export_to_onnx.py \
    --weights ../ai_ml_architect/experiments/uav_thermal_v1/weights/best.pt \
    --output optimized_models/ \
    --half \
    --dynamic \
    --simplify \
    --opset 14
```

Výsledek: `optimized_models/best_fp16_dynamic.onnx` (~6MB)

### 2. Konverze ONNX → TensorRT Engine

**Na Jetsonu Orin Nano** (není možné konvertovat na PC kvůli architektuře):

```bash
# FP16 konverze (doporučeno pro Orin Nano)
/usr/src/tensorrt/bin/trtexec \
    --onnx=optimized_models/best_fp16_dynamic.onnx \
    --saveEngine=optimized_models/best_fp16.engine \
    --fp16 \
    --workspace=2048 \
    --batch=1 \
    --verbose

# INT8 kalibrace (pokročilá)
/usr/src/tensorrt/bin/trtexec \
    --onnx=optimized_models/best_fp16_dynamic.onnx \
    --saveEngine=optimized_models/best_int8.engine \
    --fp16 \
    --int8 \
    --calib=calib_cache.bin \
    --workspace=4096
```

### 3. Inference na Jetsonu

```bash
# Použijte TensorRT engine v pipeline
python pipeline_engineer/run_pipeline.py \
    --video 0 \
    --model edge_specialist/optimized_models/best_fp16.engine \
    --conf-thres 0.4 \
    --no-display

# Nebo použijte ONNX Runtime (pomalejší, ale univerzální)
python pipeline_engineer/run_pipeline.py \
    --video 0 \
    --model edge_specialist/optimized_models/best_fp16_dynamic.onnx \
    --conf-thres 0.4
```

## Výkonnostní očekávání (Orin Nano 8GB)

| Backend | FPS (640x640) | Latence | Model size | Poznámka |
|---------|--------------|---------|------------|----------|
| PyTorch (.pt) | ~10-20 FPS | ~50-100ms | 6.0 MB | Pouze pro vývoj |
| ONNX Runtime | ~20-40 FPS | ~25-50ms | 6.3 MB | Univerzální |
| TensorRT (FP16) | **~30-60 FPS** | ~15-30ms | 6.0 MB | Doporučeno |
| TensorRT (INT8) | **~60-100 FPS** | ~10-15ms | 3.5 MB | Po kalibraci |

## Detekce formátů modelu

`edge_specialist/convert_to_trt.py` automaticky detekuje formát:

```python
# .pt  → PyTorch (pouze pro vývoj, pomalé na edge)
# .onnx → ONNX Runtime (potřebuje onnxruntime nebo CUDA)
# .engine → TensorRT (optimální pro Jetson)
```

## Poznámky pro TensorRT

- TensorRT engine je **platform-specific** — vytvořte ho přímo na Orin Nano
- FP16 je bezpečné pro termální data (nízký kontrast není citlivý na precision loss)
- INT8 vyžaduje calibrace dataset (použijte 100-500 reprezentativních obrázků)
- Workspace velikost: 2048 MB pro nano, 4096 MB pro Orin
