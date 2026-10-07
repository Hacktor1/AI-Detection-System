# Průvodce simulací Jetson (Fáze 2)

## Přehled

Simulační prostředí umožňuje testovat hlavní video pipeline na vývojovém počítači
**bez fyzického hardware**. Simulace napodobuje omezení Jetson Orin Nano:

- **CPU thread limiting** — ONNX Runtime a PyTorch jsou omezeny na 4 thready (simulace 6-jádrového ARM Cortex-A78AE)
- **Thermal throttling** — po 30 sekundách nepřetržitého zatížení se simuluje progresivní útlum výkonu (až 1.5×)
- **Memory limits** — simulovaný limit 6 GB RAM (8 GB hardware – 2 GB pro OS)
- **Inference resolution** — 640×640 (optimalizované pro Jetson)

## Struktura

```
jetson_sim/
├── __init__.py              # Package init
├── config.yaml             # Hardware profile (CPU, GPU, thermal, power)
├── hardware_profile.py     # JetsonProfile a SimulationConfig třídy
└── make_sample_video.py    # Generování syntetické viditelné kamery z termálního videa
```

## Použití

### 1. Simulace dual kamery

Vygenerujte syntetické viditelné video z existujícího termálního:

```bash
python -m jetson_sim.make_sample_video \
    --thermal-input pipeline_engineer/sample_data/test_thermal.mp4 \
    --visible-output pipeline_engineer/sample_data/test_visible.mp4 \
    --frames 30
```

### 2. Spuštění dual-camera pipeline ve simulaci

```bash
cd pipeline_engineer
python dual_camera_pipeline.py \
    --thermal-source sample_data/test_thermal.mp4 \
    --visible-source sample_data/test_visible.mp4 \
    --model ../optimized_models/best_fp16_dynamic.onnx \
    --sim-jetson \
    --no-display \
    --output ../results/dual_cam_sim.mp4 \
    --max-frames 30
```

### 3. Benchmark modelů

```bash
cd edge_specialist
python benchmark.py \
    --pt-model ../yolov8n.pt \
    --onnx-model ../optimized_models/best_fp16_dynamic.onnx \
    --video ../pipeline_engineer/sample_data/test_thermal.mp4 \
    --frames 30 \
    --sim-jetson \
    --output ../results/benchmark.json
```

### 4. TensorRT konverze (dry-run na hostovi)

```bash
cd edge_specialist
python convert_to_trt.py \
    --onnx ../optimized_models/best_fp16_dynamic.onnx \
    --engine ../optimized_models/best_fp16_dynamic.engine \
    --fp16 --dry-run
```

Výstup: přesný příkaz `trtexec` pro spuštění na Jetsonu.

## Hardwarový profil

Viz `jetson_sim/config.yaml` pro kompletní specifikace. Klíčové hodnoty:

| Parametr | Hodnota |
|---------|---------|
| CPU | 6 cores ARM Cortex-A78AE @ 2.0 GHz |
| GPU | 1024 CUDA cores, 32 Tensor cores |
| RAM | 8 GB LPDDR5 |
| TDP | 15 W |
| Throttle temp | 80 °C (po 30 s) |
| Cílový FPS | 25 @ 640×640 |
| Cílová latence | 35 ms |

## Limitace simulace

- Simulace běží na x86 CPU/GPU, ne na ARM architecture JetPacku
- Thermal throttling je simulován softwarově (přídavný delay), ne skutečným termálním throttlingem
- ONNX Runtime není ekvivalentní TensorRT — skutečný výkon na Jetsonu bude lepší (TensorRT optimalizuje kernely pro ARM + GPU)
