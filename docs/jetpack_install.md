# Průvodce instalací JetPack SDK

## Přehled
JetPack SDK je základní software pro platformy NVIDIA Jetson,
včetně Linux for Tegra (L4T), CUDA, cuDNN, TensorRT a VisionWorks.

## Požadavky
- Jetson Orin Nano Developer Kit
- microSD karta (>64GB doporučeno)
- Internetové připojení (Ethernet preferováno)
- Hostitelský PC (pro SDK Manager, volitelné)

## Metoda 1: Pomocí SDK Manager (Doporučeno)

### Na hostitelském PC:
1. Stáhni [NVIDIA SDK Manager](https://developer.nvidia.com/embedded/sdk-manager)
2. Nainstaluj a spusť SDK Manager
3. Vyber produkt: **Jetson Orin Nano**
4. Zvol úplnou instalaci (včetně obrazu OS)
5. Zapsi na SD kartu nebo flash na eMMC

### Na Jetsonu:
1. Vlož SD kartu / zapni napájení
2. Dokonči první boot průvodce (jazyk, Wi-Fi, uživatelský účet)
3. SDK Manager nainstaluje zbývající komponenty (CUDA, TensorRT, atd.)

## Metoda 2: Manuální SD karta image (Alternativa)

1. Stáhni **L4T SD karta image** pro Orin Nano z:
   https://developer.nvidia.com/embedded/downloads
2. Zapsi pomocí [Balena Etcher](https://www.balena.io/etcher/) nebo `dd`
3. Spusť z SD karty
4. Dokonči počáteční nastavení

## Po instalaci

### Ověř komponenty JetPacku:
```bash
# Kontrola CUDA
nvcc --version

# Kontrola TensorRT
/usr/src/tensorrt/bin/trtexec --version

# Kontrola Vision
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### Aktualizuj systém:
```bash
sudo apt update && sudo apt upgrade -y
```

## Další kroky
Po instalaci JetPacku a ověření:
1. Viz [Průvodce nasazením modelu](model_deployment.md) pro nasazení trénovaného modelu.
2. Viz [Průvodce připojením kamer](camera_wiring.md) pro připojení kamer.