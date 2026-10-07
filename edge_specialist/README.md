# Edge Specialist — Optimalizace a nasazení na Jetson

## Přehled

Optimalizovat YOLO model a nasadit ho na **NVIDIA Jetson Orin Nano**. Export do ONNX, konverze na TensorRT, INT8 kalibrace, profiling a streaming aplikace.

## Klíčové aktivity

1. **Optimalizace modelu**: PyTorch `.pt` → ONNX → TensorRT (FP16/INT8)
2. **Benchmarkovaní**: PyTorch vs ONNX vs TensorRT — FPS, latency, velikost
3. **Integrace Jetson**: JetPack SDK, TensorRT, nasazení na drone

## Struktura souborů

```
edge_specialist/
├── export_to_onnx.py        # PyTorch .pt → ONNX (FP16, dynamic, opset 14)
├── convert_to_trt.py        # ONNX → TensorRT engine builder
├── int8_calibrate.py        # INT8 kalibrační cache generátor (100 vzorků)
├── profiler.py              # Srovnání backends + simulace Jetson
├── benchmark.py             # FPS/latency benchmark
├── web_ui.py                # Flask desktop dashboard (MJPEG + JSON API)
├── mobile_stream.py         # Mobilní stream bridge (dron → telefon)
├── mavlink_bridge.py        # Původní MAVLink bridge (zachován jako referenční)
├── jetson_deploy.py         # Jetson deploy helper (--check/--convert/--deploy)
├── optimized_models/        # ONNX model + INT8 cache (gitignored)
│   ├── best_fp16_dynamic.onnx  # ONNX model (6.3 MB)
│   └── calib_cache.bin         # INT8 kalibrační cache
└── results/                 # Výsledky benchmark a profil
    ├── benchmark_results.json
    └── profile_results.json
```

## Workflow

### 1. Export do ONNX

```bash
python3 export_to_onnx.py \
    --weights ../yolov8n.pt \
    --half \
    --dynamic \
    --simplify \
    --opset 14
```

Výstup: `optimized_models/best_fp16_dynamic.onnx` (6.3 MB)

### 2. INT8 kalibrace

```bash
python3 int8_calibrate.py \
    --onnx optimized_models/best_fp16_dynamic.onnx \
    --dataset ../data_engineer/datasets/processed/combined/val.txt \
    --output optimized_models/calib_cache.bin \
    --num-samples 100
```

### 3. TensorRT konverze (na Jetsonu)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=optimized_models/best_fp16_dynamic.onnx \
    --saveEngine=optimized_models/best_fp16_dynamic.engine \
    --fp16 \
    --int8 \
    --calib=optimized_models/calib_cache.bin \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:16x3x640x640
```

### 4. Profiling (srovnání backends)

```bash
python3 profiler.py \
    --pt-model ../yolov8n.pt \
    --onnx optimized_models/best_fp16_dynamic.onnx \
    --frames 100
```

Výsledky: `results/profile_results.json`

### 5. Benchmark

```bash
python3 benchmark.py \
    --onnx optimized_models/best_fp16_dynamic.onnx \
    --frames 30
```

Výsledky: `results/benchmark_results.json`

### 6. Web UI (desktop monitoring)

```bash
python3 web_ui.py \
    --host 0.0.0.0 --port 8080 \
    --source 0 \
    --model optimized_models/best_fp16_dynamic.onnx
```

### 7. Mobile Stream (dron → mobil)

```bash
python3 mobile_stream.py \
    --source /dev/video0 \
    --model optimized_models/best_fp16_dynamic.onnx \
    --port 8080
# Otevři v mobilním prohlížeči: http://<drone-ip>:8080/
```

### 8. Jetson deploy

```bash
python3 jetson_deploy.py --deploy \
    --thermal-source /dev/video0 \
    --visible-source /dev/video1 \
    --onnx optimized_models/best_fp16_dynamic.onnx
```

## Výkonnostní výsledky

| Backend | FPS | Latence | Model size | Notes |
|---|---|---|---|---|
| PyTorch (.pt) | 259.3 | 3.86ms | 6.2 MB | Dev only |
| ONNX Runtime (CPU) | 382.9 | 2.61ms | 6.3 MB | Production |
| TensorRT FP16 (Jetson) | ~30-60 FPS | ~15-30ms | 6.0 MB | Requires Jetson |
| TensorRT INT8 (Jetson) | ~60-100 FPS | ~10-15ms | 3.5 MB | Requires calibration |

## Poznámky

- Soubor `.engine` je specifický pro platformu (musí být vytvořen na Jetsonu nebo kompatibilní GPU).
- ONNX Runtime podporuje CPU i CUDA — na dev stroji použijeme CPU fall-back.
- `mavlink_bridge.py` je zachován jako referenční implementace MAVLink protokolu. Pro nové nasazení použij `mobile_stream.py`.
- INT8 kalibrace vyžaduje reprezentativní dataset (100 vzorků minimálně).
