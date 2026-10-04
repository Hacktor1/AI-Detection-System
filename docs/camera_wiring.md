# Camera Wiring Guide

## Overview
This guide explains how to connect two cameras (thermal + visible) to the Jetson Orin Nano.

## Jetson Orin Nano Camera Ports

| Port Type | Quantity | Description |
|-----------|----------|-------------|
| CSI-2 (MIPI) | 1x | Dedicated camera interface (high bandwidth) |
| USB 3.0 | 2x | Standard USB port (supports webcams) |
| USB 2.0 | 1x | Lower speed USB |

## Wiring Setup

### Option A: CSI + USB (Recommended)
- **Visible Camera** → CSI port (for highest performance)
- **Thermal Camera** → USB 3.0 port

```
         Jetson Orin Nano
    +-------------------------+
    |   [CSI-2]     [USB-C]   |
    |     |              |    |
    |   [CAM0]       [USB-A] |
    |     |              |    |
    |  Visible      Thermal   |
    +-------------------------+
```

### Option B: Dual USB
If using two USB cameras:
- Both cameras → USB 3.0 ports
- May need powered USB hub if power draw exceeds USB limits

## Supported Cameras

### CSI Cameras
- Raspberry Pi Camera Module v2 (Sony IMX219)
- ArduCam IMX219/MIPI
- Leopard Imaging LI-OV5640

### USB Cameras
- FLIR Blackfly (USB3)
- OAK-D Series
- Logitech C920/C922
- Generic UVC webcams

### Thermal Cameras
- FLIR Lepton (via breakout board + SPI-to-USB adapter)
- OAK Thermal (USB3)
- Seek Thermal (USB OTG, limited support)

## Initial Test

After connecting cameras, verify detection:
```bash
# List CSI devices
ls /dev/video*

# List USB devices
lsusb

# Check camera info
v4l2-ctl --list-devices
```

If cameras appear in `/dev/video*`, they are detected by the system.

## Next Steps
See [Jetpack Installation Guide](jetpack_install.md) to set up the OS and drivers.
