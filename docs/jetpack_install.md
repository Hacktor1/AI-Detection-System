# JetPack SDK Installation Guide

## Overview
JetPack SDK is the foundational software for NVIDIA Jetson platforms,
including Linux for Tegra (L4T), CUDA, cuDNN, TensorRT, and VisionWorks.

## Prerequisites
- Jetson Orin Nano Developer Kit
- microSD card (>64GB recommended)
- Internet connection (Ethernet preferred)
- Host PC (for SDK Manager, optional)

## Method 1: Using SDK Manager (Recommended)

### On Host PC:
1. Download [NVIDIA SDK Manager](https://developer.nvidia.com/embedded/sdk-manager)
2. Install and launch SDK Manager
3. Select product: **Jetson Orin Nano**
4. Choose full installation (includes OS image)
5. Write to SD card or flash to eMMC

### On Jetson:
1. Insert SD card / power on
2. Complete first boot wizard (language, Wi-Fi, user account)
3. SDK Manager will install remaining components (CUDA, TensorRT, etc.)

## Method 2: Manual SD Card Image (Alternative)

1. Download **L4T SD Card Image** for Orin Nano from:
   https://developer.nvidia.com/embedded/downloads
2. Flash using [Balena Etcher](https://www.balena.io/etcher/) or `dd`
3. Boot from SD card
4. Complete initial setup

## After Installation

### Verify JetPack Components:
```bash
# Check CUDA
nvcc --version

# Check TensorRT
/usr/src/tensorrt/bin/trtexec --version

# Check Vision
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### Update System:
```bash
sudo apt update && sudo apt upgrade -y
```

## Next Steps
After JetPack is installed and verified:
1. See [Model Deployment Guide](model_deployment.md) to deploy your trained model.
2. See [Camera Wiring Guide](camera_wiring.md) to connect your cameras.
