# Rolle: Data Engineer

## Přehled úlohy

Zhromažďovat, ověřovat a předzpracovávat veřejné termální datasety s lidskýma anotacema pro trénink lidské detekce.

## Klíčové aktivity

1. **Objevování datasetů**
   - Hledat veřejné zdroje (Kaggle, akademické repozitáře, GitHub) pro infračervené/termální datasety s lidskýma bounding boxy.
   - Dokumentovat zdroj, licenci, velikost a formát anotací.

2. **Skript pro stahování**
   - Napsat skript pro stahování datasetů do `data_engineer/datasets/`.
   - Zpracovat ověřování tam, kde je požadováno (Kaggle API token).

3. **Předzpracování**
   - Převést termální obrázky do konzistentního formátu.
   - Normalizovat anotace do formátu YOLO (xywh, normalizované 0-1).
   - Rozdělit na tréninkové/validační/testovací sady.

4. **Ověření**
   - Ověřit integritu datasetu (počty souborů, chybějící soubory).
   - Vizuálně ověřit anotace, pokud je možné.

## Kde najít datasety

Umisťuj stažené datasety do:
```
data_engineer/datasets/
```

## Šablona skriptu

Vytvoř skript v tomto adresáři s názvem `_download_<název_datasetu>.py`:
```python
#!/usr/bin/env python3
"""
Stahovací skript pro <Název Datasetu>.
Zdroj: <URL>
 Licence: <licence>
"""
import urllib.request
import os

OUTPUT = os.path.join("datasets", "raw")
os.makedirs(OUTPUT, exist_ok=True)
# Implementuj stahování zde
```

## Poznámky
- Všechny skripty začínající na `_` jsou ignorovány gitem (viz `.gitignore`).
- Udržuj metadata datasetu v `dataset_registry.md` v tomto adresáři.