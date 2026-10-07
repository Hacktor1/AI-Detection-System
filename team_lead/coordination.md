# Týmová koordinace & Sprint plán

## Aktuální stav

### Fáze 0: Kostra & Dokumentace — ✅ HOTOVÁ
- Struktura repozitáře s rolemi (data_engineer, ai_ml_architect, pipeline_engineer, edge_specialist, team_lead)
- Hlavní README s architekturou a roadmapou
- Dokumentace: getting_started, hardware_setup, jetpack_setup, model_deployment

### Fáze 1: Statické testování na PC — ✅ HOTOVÁ
- Datasety: FLIR ADAS + LLVIP + Thermal Person Detector = 28,738 obrázků
- Trénink pipeline: train.py + params.yaml + inference.py
- Validace: 10 integračních testů prošly
- 3-epoch test: mAP50=0.274 (full trénink = ~0.7+ očekáváno)

### Fáze 2: Simulace Jetson — ✅ HOTOVÁ
- ONNX export: 6.3MB FP16 dynamic
- Pipeline: 0.004s/img inference, 39.7 FPS na CPU
- Benchmark: results v edge_specialist/results/benchmark_results.json

## Sprint plán

### Sprint 3: Trénink full datasetu
| Task | Owner | Priorita | Status |
|------|-------|---------|--------|
| Spustit full trénink (100 epoch) na GPU | AI/ML Architekt | Vysoká | Pending |
| Optimalizovat hyperparametry (mosaic, hsv_v) | AI/ML Architekt | Střední | Pending |
| Vytvořit experiment tracking (MLflow nebo JSON) | AI/ML Architekt | Střední | Pending |
| Validovat mAP na val setu | AI/ML Architekt | Vysoká | Pending |

### Sprint 4: Edge nasazení
| Task | Owner | Priorita | Status |
|------|-------|---------|--------|
| TensorRT konverze na Orin Nano | Edge Specialist | Vysoká | Pending |
| INT8 kalibrace | Edge Specialist | Střední | Pending |
| Benchmark na skutečném HW | Edge Specialist | Vysoká | Pending |
| Optimalizace pro 30+ FPS | Edge Specialist | Vysoká | Pending |

### Sprint 5: Hardware integrace
| Task | Owner | Priorita | Status |
|------|-------|---------|--------|
| JetPack 6 instalace na Orin Nano | Edge Specialist | Vysoká | Pending |
| Připojení termální kamery (FLIR/OAK) | Pipeline Inženýr | Vysoká | Pending |
| Připojení viditelné kamery | Pipeline Inženýr | Vysoká | Pending |
| Dual camera pipeline test | Pipeline Inženýr | Vysoká | Pending |

## GitHub workflow
```bash
git checkout -b feature/<task-name>
git add .
git commit -m "feat: <description>"
git push origin feature/<task-name>
# Then create PR via gh CLI
gh pr create --title "Feature: <name>" --body "..."
```

## CI/CD checklist před merge
- [ ] `python3 -m pytest tests/test_pipeline.py -v` (všechny testy prošly)
- [ ] `python3 data_engineer/_validate_datasets.py` (0 errors)
- [ ] Model export do ONNX funguje
- [ ] Pipeline spustitelná v headless módu

## Důležitá upozornění
- Datasety a modely jsou **gitignored** — používejte `git add -f` pro malé ukázky
- ONNX export musí proběhnout **na Jetsonu** kvůli architektuře (ARM64)
- Pro CPU-only development použijte `--no-display` v pipeline
- FP16 je bezpečné pro termální data (nízký kontrast)
