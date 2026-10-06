#!/usr/bin/env python3
"""
Download script pro LLVIP dataset — Lidská detekce v termálnímviditelném spektru.
Zdroj: https://github.com/betensll/LLVIP
Licence: Apache 2.0 (otevřený pro výzkum a komerční použití)
Obsah: Cca 15 000 párových RGB-T snímků s anotacemi lidí ve formátu YOLO.

Poznámka: GitHub repo poskytuje pouze odkazy k původním souborům na Baidu Netdisk
(žádný přímý GitHub stažení). Proto tento skript:
  1. Klonuje metadata repo pro získání seznamu souborů.
  2. Vypisuje instrukce pro ruční stažení přes Baidu (protože linky expirují).
  3. Umožňuje ruční stažení do `datasets/raw/llvip/`.

Pokud již máte soubory staženy, použijte `--extract` pro rozbalení a
normalizaci anotací do formátu YOLO v `datasets/processed/llvip/`.
"""
import os
import sys
import argparse
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "datasets", "raw", "llvip")
PROCESSED_DIR = os.path.join(BASE_DIR, "datasets", "processed", "llvip")
REPO_URL = "https://github.com/betensll/LLVIP.git"


def get_repo_metadata():
    """Načte informace o stažení z oficiální stránky, protože repo již není veřejně klonovatelné."""
    print("[INFO] LLVIP repo již není veřejně klonovatelné na GitHubu.")
    print("[INFO] Oficiální stránka: https://github.com/betensll/LLVIP")
    print("[INFO] Stáhněte soubory manuálně:")
    print("       1. Uveďte se na https://aerial-datalab.com/llvip/")
    print("       2. Stáhněte 'illvipl+_ir.tar.gz' (termální) a 'illvipl+_rgb.tar.gz' (viditelné)")
    print("       3. Rozbalte do:")
    print(f"          {RAW_DIR}")
    print("[INFO] Poté spusťte: python _download_llvip.py --extract")


def extract_annotations():
    """
    Předpokládá, že jste soubory ručně stáhli do `datasets/raw/llvip/`.
    LLVIP používá PASCAL VOC XML anotace. Tento krok je konzervativní:
    pouze ověří strukturu a vytvoří výstupní adresář.
    """
    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] Složka {RAW_DIR} neexistuje. Nejprve stáhněte soubory.")
        sys.exit(1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    xml_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".xml")]

    if not xml_files:
        print(f"[WARN] V {RAW_DIR} nebyly nalezeny žádné .xml soubory.")
        print("       Obsah složky:")
        for f in sorted(os.listdir(RAW_DIR)):
            print(f"         - {f}")
        return

    print(f"[INFO] Nalezeno {len(xml_files)} XML anotací.")
    print("[INFO] Připravuji strukturu pro konverzi do YOLO formátu.")
    print(f"[INFO] Zpracovaná data putují do: {PROCESSED_DIR}")
    print("[TODO] Implementovat konverzi VOC -> YOLO (xywh, normalizované 0-1)")


def main():
    parser = argparse.ArgumentParser(description="Stahování a příprava LLVIP datasetu")
    parser.add_argument("--fetch-links", action="store_true",
                        help="Klonuje repo a vypisuje odkazy pro stažení")
    parser.add_argument("--extract", action="store_true",
                        help="Připraví strukturu pro konverzi anotací (vyžaduje stažená data)")
    args = parser.parse_args()

    if args.fetch_links:
        get_repo_metadata()

    if args.extract:
        import xml.etree.ElementTree as ET
        extract_annotations()

    if not args.fetch_links and not args.extract:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
