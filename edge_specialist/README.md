# Rolle: Edge Specialist

## Přehled úlohy

Optimalizovat trénovaný YOLO model a nasadit ho na NVIDIA Jetson Orin Nano. Převést do ONNX → TensorRT, měřit FPS a zajistit reálný výkon.

## Klíčové aktivity

1. **Optimalizace modelu**
   - Převést PyTorch `.pt` → ONNX.
   - Použít ONNX k TensorRT engine builder.
   - Kvantizovat (FP16 / INT8) ke snížení velikosti a zvýšení FPS.

2. **Benchmarkování výkonu**
   - Měřit FPS, latenci, velikost modelu.
   - Porovnávat PyTorch vs ONNX vs TensorRT na cílovém hardwaru.

3. **Integrace Jetson**
   - Nainstalovat závislosti JetPack SDK (když je k dispozici hardware Jetson).
   - Ověřit TensorRT + CUDA.
   - Spustit model na Jetsonu a hlásit propustnost.

## Struktura adresářů
```
edge_specialist/
├── convert_to_trt.py       # Konverze ONNX → TensorRT
├── benchmark.py           # Měření FPS/latency
├── jetpack_setup.md       # Poznámky nastavení Jetson
└── optimized_models/      # Finální .engine soubory (gitignored)
```

## Příkaz pro konverzi (příklad)
```bash
# ONNX → TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

## Benchmark
```bash
python benchmark.py --model optimized_models/model.engine --frames 100
```

## Poznámky
- Soubor `.engine` je specifický pro platformu (musí být vytvořen na Jetsonu nebo kompatibilní GPU).
- Dokumentuj verzi JetPacku a TensorRT použitou.