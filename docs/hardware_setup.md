# Hardware Setup Guide

## Required Components

### 1. Main Compute Module
| Component | Model | Notes |
|-----------|-------|-------|
| SBC | NVIDIA Jetson Orin Nano (8GB or 16GB) | Dev Kit recommended for prototyping |

### 2. Cameras (Dual Camera Setup)
| Purpose | Camera Model | Interface | Notes |
|---------|-------------|-----------|-------|
| Thermal Imaging | FLIR Lepton 3.5 / OAK Thermal | SPI / USB | For human/body heat detection |
| Visible Light | OAK-D / Raspberry Pi Camera v2 | CSI / USB | For detail detection (license plates, packages) |

### 3. Power Supply
| Component | Specification |
|-----------|--------------|
| Power Adapter | 12V DC, 5A minimum (for Jetson Nano) |

### 4. Storage
| Component | Specification |
|-----------|--------------|
| microSD Card | >= 64GB, Class 10 (UHS-I or higher) |

### 5. Optional (Optional but Recommended)
| Component | Purpose |
|-----------|---------|
| USB Hub (with power) | For connecting peripherals |
| HDMI Monitor | For initial setup/debugging |
| Keyboard & Mouse | For setup |
| USB-C to Ethernet Adapter | For stable network during development |

## Connection Diagram

```
+------------------+       +------------------+
| Thermal Camera   |-------| Jetson Orin Nano |<------ Power (12V)
| (SPI/USB)        |       | (CSI + USB 3.0)  |
+------------------+       +--------+---------+
                                     |
                                     |
+------------------+                |
| Visible Camera   |----------------+
| (CSI/MIPI)       |
+------------------+
```

## Purchase Links (Examples)
- Jetson Orin Nano Developer Kit: ~$500-600
- FLIR Lepton Breakout Board: ~$200-300
- OAK Thermal (w/ lepton): ~$400-500
- Raspberry Pi Camera v2: ~$25

## Notes
1. Ensure you buy the version with **M.2 Key M connector** for WiFi/BT module (optional).
2. For thermal camera, consider using **OAK Thermal** for easier integration (has built-in processing).
