# Průvodce nasazením modelu (Jetson Orin Nano)

## Přehled
Tento průvodce vede krok za krokem export trénovaného YOLO modelu z PyTorch do TensorRT enginy optimalizovaného pro inferenci na Jetson Orin Nano.

## Požadavky
- Trénovaný YOLOv8/v9/v11 model (`.pt` soubor)
- Jetson Orin Nano s nainstalovaným JetPackem
- ONNX opset nainstalován (`onnx`, `onnx-simplifier`)

## Krok 1: Export do ONNX

Na tvém vývojovém počítači (nebo Jetsonu):
```bash
# Pomocí Ultralytics CLI
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=13 simplify=true

# Výstup: best.onnx
```

## Krok 2: Konverze ONNX do TensorRT Enginy

Na Jetson Orin Nano:
```bash
# Vytvoř TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=best.engine \
    --fp16 \
    --workspace=2048 \
    --minShapes=input0:1x3x640x640 \
    --optShapes=input0:16x3x640x640 \
    --maxShapes=input0:32x32x640x640

# Výstup: best.engine
```

## Krok 3: Načtení a spuštění v Pythonu

Použij `pipeline_engineer/detector.py` s TensorRT backendem:
```python
# Příklad použití v pipeline
from detector import PersonDetector

detector = PersonDetector(model_path="best.engine", conf_thres=0.4)
boxes = detector.infer(frame)
```

## Poznámky k výkonu
- **YOLOv8-nano**: ~30-60 FPS @ 640x640 na Orin Nano (FP16)
- **YOLOv11**: Lepší přesnost, mírně pomalejší
- **INT8 kvantizace**: Může zvýšit FPS, ale vyžaduje kalibrační dataset

## Řešení problémů
- Pokud `trtexec` selže, zkontroluj verzi ONNX opsetu (použij 13 nebo nižší pro kompatibilitu)
- Pokud vytvoření enginy selže, sniž `--workspace` velikost
- Pro dynamické tvary, ujisti se, že názvy vstupních tensorů odpovídají ONNX

## Další kroky
Po nasazení modelu, testuj pomocí [Dvojité kamery pipeline](testing.md).