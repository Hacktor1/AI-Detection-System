# Připojení kamer k Jetson Orin Nano

## Přehled

Tento dokument popisuje, jak připojit termální a viditelné kamery k Jetson Orin Nano.

## Pinout a konektory

### CSI (MIPI) konektor
- **Použití**: Viditelná kamera (Raspberry Pi Camera v2, OAK-D)
- **Připojení**: 22-pin CSI konektor (nahoře na dev kitu)
- **Podpora**: 1× kamera (alespoň pro začátek)

### USB 3.0 porty
- **Použití**: Termální kamera (FLIR Lepton přes USB, OAK Thermal)
- **Poznámka**: USB 3.0 je důležitý pro vysoký datový tok termálních videí

### SPI rozhraní
- **Použití**: FLIR Lepton přes SPI breakout board
- **Požadavky**: Manuální wiring, SPI musí být povoleno v Jetson IO

## Zapojení

### Viditelná kamera (CSI)
```
Jetson Orin Nano          Raspberry Pi Camera v2
CSI port (22-pin)   ↔    CSI konektor (22-pin)
```

### Termální kamera (USB)
```
Jetson Orin Nano          FLIR Lepton (přes USB adapter)
USB 3.0 port        ↔    USB-C/Micro USB
```

### Termální kamera (SPI)
```
Jetson Orin Nano          FLIR Lepton Breakout
Pin 19 (SPI0_MOSI)  ↔    MOSI
Pin 21 (SPI0_MISO)  ↔    MISO
Pin 23 (SPI0_SCLK)  ↔    SCK
Pin 24 (SPI0_CS0)   ↔    CS
Pin 6  (GND)        ↔    GND
Pin 1  (3V3)        ↔    VCC
```

## Ověření kamer

Po zapojení:

```bash
# Ověření USB kamer
lsusb

# Ověření CSI kamer
sudo tegrastats --interval 1000

# Test video vstupu
python3 -c "
import cv2
cap = cv2.VideoCapture(0)  # /dev/video0
ret, frame = cap.read()
print(f'Frame shape: {frame.shape}' if ret else 'Failed')
cap.release()
"

# Ověření všech video zařízení
ls -la /dev/video*
```

## Poznámky

- Termální kamera by měla být primární pro detekci lidí
- Viditelná kamera doplňuje pro detekci detailů (SPZ, text)
- Všechny kamery by měly mít synchronizovaný čas (pro dual-stream analýzu)
