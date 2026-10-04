# Průvodce nastavením hardwaru

## Požadované komponenty

### 1. Hlavní výpočetní modul
| Komponenta | Model | Poznámka |
|-----------|-------|---------|
| SBC | NVIDIA Jetson Orin Nano (8GB nebo 16GB) | Dev Kit doporučeno pro prototypování |

### 2. Kamery (Dvojitý nastavení)
| Účel | Model kamery | Rozhraní | Poznámka |
|---------|-------------|-----------|-------|
| Termální snímání | FLIR Lepton 3.5 / OAK Thermal | SPI / USB | Pro detekci lidí a tepla těla |
| Viditelné světlo | OAK-D / Raspberry Pi Camera v2 | CSI / USB | Pro detailní detekci (SPZ, balíčky) |

### 3. Napájení
| Komponenta | Specifikace |
|-----------|--------------|
| Power Adapter | 12V DC, min. 5A (pro Jetson Nano) |

### 4. Úložiště
| Komponenta | Specifikace |
|-----------|--------------|
| microSD karta | >= 64GB, Class 10 (UHS-I nebo vyšší) |

### 5. Volitelné (Doporučeno)
| Komponenta | Účel |
|-----------|---------|
| USB Hub (s napájením) | Pro připojení periferií |
| HDMI Monitor | Pro počáteční nastavení/ladění |
| Klávesnice & Myš | Pro nastavení |
| USB-C k Ethernet Adapter | Pro stabilní síť během vývoje |

## Diagram připojení

```
+------------------+       +------------------+\n| Termální Kamera |-------| Jetson Orin Nano |<------ Power (12V)\n| (SPI/USB)       |       | (CSI + USB 3.0)   |\n+------------------+       +--------+---------+\n                                     |
                                     |
+------------------+                |
| Viditelná Kamera |----------------+
| (CSI/MIPI)       |
+------------------+
```

## Odkazy pro nákup (Příklady)
- Jetson Orin Nano Developer Kit: ~500-600 USD
- FLIR Lepton Breakout Board: ~200-300 USD
- OAK Thermal (s leptonem): ~400-500 USD
- Raspberry Pi Camera v2: ~25 USD

## Poznámky
1. Ujisti se, že koupíš verzi s **M.2 Key M konektorem** pro WiFi/BT modul (volitelné).
2. Pro termální kameru zvaž použití **OAK Thermal** pro snadnější integraci (má vestavěné zpracování).