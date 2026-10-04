# Průvodce připojením kamer

## Přehled
Tento průvodce vysvětluje, jak připojit dvě kamery (termální + viditelné světlo) k Jetson Orin Nano.

## Porty kamer na Jetson Orin Nano

| Typ portu | Množství | Popis |
|-----------|----------|-------------|
| CSI-2 (MIPI) | 1× | Vyhrazené rozhraní kamery (vysoká propustnost) |
| USB 3.0 | 2× | Standardní USB port (podporuje webkamery) |
| USB 2.0 | 1× | Nižší rychlost USB |

## Nastavení připojení

### Možnost A: CSI + USB (Doporučeno)
- **Viditelná kamera** → CSI port (pro nejvyšší výkon)
- **Termální kamera** → USB 3.0 port

```
         Jetson Orin Nano
    +-------------------------+
    |   [CSI-2]     [USB-C]   |
    |     |              |    |
    |   [CAM0]       [USB-A] |
    |     |              |    |
    |  Viditelná    Termální  |
    +-------------------------+
```

### Možnost B: Dvojité USB
Pokud používáš dvě USB kamery:
- Obě kamery → USB 3.0 porty
- Může být potřeba napájený USB hub, pokud příkon přesahuje limity USB

## Podporované kamery

### CSI kamery
- Raspberry Pi Camera Module v2 (Sony IMX219)
- ArduCam IMX219/MIPI
- Leopard Imaging LI-OV5640

### USB kamery
- FLIR Blackfly (USB3)
- OAK-D série
- Logitech C920/C922
- Generické UVC webkamery

### Termální kamery
- FLIR Lepton (přes breakout board + SPI-to-USB adapter)
- OAK Thermal (USB3)
- Seek Thermal (USB OTG, omezená podpora)

## Prvotní test

Po připojení kamer ověř detekci:
```bash
# Seznam všech video zařízení
ls /dev/video*

# Seznam USB zařízení
lsusb

# Kontrola informací o kameře
v4l2-ctl --list-devices
```

Pokud se kamery objeví v `/dev/video*`, systém je deteguje.

## Další kroky
Viz [Průvodce instalací JetPacku](jetpack_install.md) pro nastavení OS a ovladačů.