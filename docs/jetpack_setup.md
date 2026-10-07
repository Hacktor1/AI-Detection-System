# Instalace JetPack SDK na Jetson Orin Nano

## Požadavky

- Jetson Orin Nano Developer Kit
- MicroSD karta (64GB+) nebo eMMC
- Internetové připojení (Ethernet doporučeno pro stabilitu)
- 12V power adapter (5A+)

## Kroky instalace

### 1. Stáhněte Jetson Flash Utility

#### Linux (doporučeno):
```bash
# Nainstalujte Jetson Flash
sudo apt-get install python3-pip
pip3 install Jetson-Flasher
# NEBO použijte balíčkovací skript:
wget https://developer.download.nvidia.com/compute/servers/jetpack/api/jetpack_6x.sh
chmod +x jetpack_6x.sh
./jetpack_6x.sh --install-type download-only
```

#### Windows/macOS:
```bash
# Použijte NVIDIA SDK Manager
# Staňte si z https://developer.nvidia.com/sdk-manager
```

### 2. Flash JetPack 6 (L4T R36.x)

```bash
# Flash image to SD card (beware: this erases the SD card)
sudo ./jetpack_6x.sh --flash sdcard --sd /dev/sdX
# Nebo pro eMMC: --flash emmc
```

### 3. První spuštění

1. Vložte SD kartu do Jetson
2. Připojte napájení
3. První boot trvá 5-10 minut (expanduje partition)
4. Default credentials: `nvidia` / `nvidia`

### 4. Instalace závislostí

```bash
# Aktualizujte systém
sudo apt update && sudo apt upgrade -y

# Nainstalujte CMake, build tools, a GPU knihovny
sudo apt install cmake build-essential libgtk2.0-dev pkg-config

# PyTorch pro Jetson (ARM64)
wget https://nvidia.box.com/v/jetson-torch-python310 -O torch-jetson.whl
pip3 install torch-jetson.whl

# Ultralytics YOLO
pip3 install ultralytics opencv-python-headless

# ONNX Runtime pro ARM64 (s CUDA podporou)
pip3 install onnxruntime-gpu

# TensorRT je již v JetPack
/usr/src/tensorrt/bin/trtexec --version
```

### 5. Ověření instalace

```bash
# GPU status
tegrastats &

# PyTorch
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# TensorRT
/usr/src/tensorrt/bin/trtexec --help | head -5

# ONNX
python3 -c "import onnx; print('ONNX:', onnx.__version__)"
```

## Poznámky

- JetPack 6 obsahuje CUDA 12.x, cuDNN 8.x, TensorRT 8.6+
- Pro detekci GPU použijte `nvidia-smi` (není dostupné na Jetson – použijte `tegrastats`)
- Všechny modely musí být převedeny na TensorRT engine **na Jetsonu** (ne na PC)
- SD karta musí být Class 10 nebo vyšší pro spolehlivost
