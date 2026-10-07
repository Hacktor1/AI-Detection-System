# Testování detekčního systému

## Přehled
Průvodce testováním dvojité kamerové detekční pipeline na vývojových počítačích
i na Jetson Orin Nano.

## Předtestovací kontrola
- [ ] Kamery připojeny a detegovány (`/dev/video*`)
- [ ] JetPack SDK nainstalován na Jetsonu
- [ ] Model exportován jako TensorRT `.engine`
- [ ] `requirements.txt` nainstalován v virtuálním prostředí

## Test 1: Detekce kamer
Ověř, že jsou obě kamery detegovány:
```bash
# Seznam všech video zařízení
ls -la /dev/video*

# Ověř každý stream kamery
python pipeline_engineer/camera_io.py --source 0  # Kamera 1
python pipeline_engineer/camera_io.py --source 1  # Kamera 2
```

## Test 2: Inferenční model
Spusť inferenci na ukázkovém obrázku pro potvrzení načtení modelu:
```bash
python pipeline_engineer/detector.py --image data/test/sample.jpg --model models/best.engine
```

Očekávaný výstup:
```
[detector] TensorRT engine loaded: models/best.engine
[detector] Inference done: 2 people detected (conf: 0.85, 0.76)
```

## Test 3: Kompletní dvojitá kamera pipeline
Spusť kompletní pipeline se skvrnami:
```bash
python pipeline_engineer/dual_camera_pipeline.py \
    --thermal-source sample_data/test_thermal.mp4 \
    --visible-source sample_data/test_visible.mp4 \
    --model ../optimized_models/best_fp16_dynamic.onnx \
    --output-dir results/
```

Tímto se provede:
1. Získání snímků z obou kamer najednou (paralelní vlákna)
2. Spuštění detekce na každém snímku (ONNX Runtime nebo TensorRT engine)
3. Overlay bounding boxů na oba streamy
4. Vedle sebe (side-by-side) oraz svýzku a FPS/latence do `results/`

### Fáze 2: Simulace Jetson (bez hardware)
```bash
# 1. Vygenerujte syntetické viditelné video (pokud ještě neexistuje):
python -m jetson_sim.make_sample_video \
    --thermal-input pipeline_engineer/sample_data/test_thermal.mp4 \
    --visible-output pipeline_engineer/sample_data/test_visible.mp4 --frames 30

# 2. Spusťte simulaci s omezením CPU threadů:
python pipeline_engineer/dual_camera_pipeline.py \
    --thermal-source sample_data/test_thermal.mp4 \
    --visible-source sample_data/test_visible.mp4 \
    --model ../yolov8n.pt \
    --sim-jetson --no-display --max-frames 30

# 3. Benchmark porovnání backends:
python edge_specialist/benchmark.py \
    --pt-model ../yolov8n.pt \
    --onnx-model ../optimized_models/best_fp16_dynamic.onnx \
    --video ../pipeline_engineer/sample_data/test_thermal.mp4 \
    --frames 30 --sim-jetson --output results/benchmark.json
```

Viz [Průvodce simulací Jetson](jetson_simulation.md) pro detailní dokumentaci.

## Test 4: Hluchý režim (Jetson)
Na Jetsonu bez displeje:
```bash
# Bez GUI displeje
python pipeline_engineer/dual_camera_pipeline.py \
    --thermal-source /dev/video0 \
    --visible-source /dev/video1 \
    --model models/best.engine \
    --no-display \
    --output-dir /tmp/results/
```

## Výkonnostní metriky
Monitoruj FPS a využití zdrojů:
```bash
# GPU využití
tegrastats

# Pipeline log ukáže:
# [pipeline] Frame 100 | inference: 0.032s | boxes: 3
```

## Tip na ladění
- Pokud jedna kamera selže, zkontroluj `dmesg | grep -i camera` pro hardware chyby
- Pokud je inferenční model pomalý, zkus snížit rozlišení vstupu (např. 480x360)
- Zkontroluj termický throttling: `cat /sys/class/thermal/thermal_zone*/temp`