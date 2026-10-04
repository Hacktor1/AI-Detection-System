# Data Engineer Role

## Task Overview

Gather, validate, and preprocess open-source thermal datasets with human annotations for training the people detector.

## Key Activities

1. **Dataset Discovery**
   - Search public sources (Kaggle, academic repos, GitHub) for infrared/thermal datasets with human bounding boxes.
   - Document source, license, size, and annotation format.

2. **Download Script**
   - Write a script to download datasets into `data_engineer/datasets/`.
   - Handle authentication where required (Kaggle API token).

3. **Preprocessing**
   - Convert thermal images to consistent format.
   - Normalize annotations to YOLO format (xywh, normalized 0-1).
   - Split into train/val/test sets.

4. **Validation**
   - Verify dataset integrity (file counts, missing files).
   - Spot-check annotations visually if possible.

## Where to Find Datasets

Place downloaded datasets in:
```
data_engineer/datasets/
```

## Example Script Template

Create a script in this directory, named `_download_<dataset_name>.py`:
```python
#!/usr/bin/env python3
"""
Download script for <Dataset Name>.
Source: <URL>
License: <license>
"""
import urllib.request
import os

OUTPUT = os.path.join("datasets", "raw")
os.makedirs(OUTPUT, exist_ok=True)
# Implement download here
```

## Notes
- All scripts starting with `_` are ignored by git (see `.gitignore`).
- Keep dataset metadata in `dataset_registry.md` within this directory.