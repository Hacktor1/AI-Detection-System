#!/usr/bin/env bash
# ===================================================================
# Phase 3: Jetson Orin Nano deployment script
# ===================================================================
# This script automates the deployment of the AI-Detection-System
# to a Jetson Orin Nano device with JetPack SDK pre-installed.
#
# It performs:
#   1. Environment verification (JetPack, CUDA, TensorRT, cameras)
#   2. Python virtualenv creation + dependency installation
#   3. ONNX → TensorRT engine conversion (FP16)
#   4. Dual-camera pipeline launch (thermal + visible CSI cameras)
#
# Usage:
#   ./deploy.sh                          # Full deploy + run
#   ./deploy.sh --check-only             # Verify environment only
#   ./deploy.sh --no-convert             # Skip TRT conversion
#   ./deploy.sh --no-run                 # Convert only, don't launch pipeline
#   ./deploy.sh --visible-source /dev/video2  # Override camera
#
set -euo pipefail

# --- Config ---
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
ONNX_MODEL="${REPO_DIR}/optimized_models/best_fp16_dynamic.onnx"
ENGINE_MODEL="${REPO_DIR}/optimized_models/best_fp16_dynamic.engine"
THERMAL_SOURCE="${THERMAL_SOURCE:-/dev/video0}"
VISIBLE_SOURCE="${VISIBLE_SOURCE:-/dev/video1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RESULTS_DIR="${REPO_DIR}/results"

# --- Parse args ---
CHECK_ONLY=false
DO_CONVERT=true
DO_RUN=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=true; DO_CONVERT=false; DO_RUN=false; shift ;;
        --no-convert) DO_CONVERT=false; shift ;;
        --no-run) DO_RUN=false; shift ;;
        --thermal-source) THERMAL_SOURCE="$2"; shift 2 ;;
        --visible-source) VISIBLE_SOURCE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  AI-Detection-System — Jetson Deploy (Phase 3)"
echo "=========================================="
echo "  Repo:    ${REPO_DIR}"
echo "  Thermal: ${THERMAL_SOURCE}"
echo "  Visible: ${VISIBLE_SOURCE}"
echo "  ONNX:    ${ONNX_MODEL}"
echo "  Engine:  ${ENGINE_MODEL}"
echo "=========================================="

# --- Step 1: Environment check ---
echo ""
echo "[1/4] Checking Jetson environment..."
python3 -m edge_specialist.jetson_deploy --check || {
    echo "❌ Environment check failed. Install JetPack SDK and connect cameras."
    exit 1
}

if [ "$CHECK_ONLY" = true ]; then
    echo "✅ Check-only mode — exiting."
    exit 0
fi

# --- Step 2: Virtualenv + dependencies ---
echo ""
echo "[2/4] Setting up Python environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
# Jetson-specific packages
pip install --quiet onnxruntime numpy pyyaml || true
echo "✅ Python environment ready."

# --- Step 3: Convert ONNX → TensorRT engine ---
# NOTE: ONNX Runtime's strict checker may flag dynamic-shape graphs.
# The TensorRT parser is more lenient and handles these correctly.
if [ "$DO_CONVERT" = true ]; then
    echo ""
    echo "[3/4] Converting ONNX → TensorRT engine (FP16)..."
    if [ -f "${ENGINE_MODEL}" ]; then
        echo "  Engine already exists — skipping conversion."
    elif [ -f "${ONNX_MODEL}" ]; then
        /usr/src/tensorrt/bin/trtexec \
            --onnx="${ONNX_MODEL}" \
            --saveEngine="${ENGINE_MODEL}" \
            --workspace=2048 \
            --minShapes=images:1x3x640x640 \
            --optShapes=images:8x3x640x640 \
            --maxShapes=images:16x3x640x640 \
            --fp16 \
            --skipInference \
            --duration=3 \
            || {
                echo "❌ TensorRT conversion failed."
                echo "   Trying with Python API as fallback..."
                python3 -m edge_specialist.convert_to_trt \
                    --onnx "${ONNX_MODEL}" \
                    --engine "${ENGINE_MODEL}" \
                    --fp16 --min-shape 1x3x640x640 --opt-shape 8x3x640x640 --max-shape 16x3x640x640
            }
        echo "✅ TensorRT engine created: ${ENGINE_MODEL}"
    else
        echo "⚠️  ONNX model not found: ${ONNX_MODEL}"
        echo "   Falling back to ONNX Runtime inference."
    fi
fi

# --- Step 4: Launch dual-camera pipeline ---
if [ "$DO_RUN" = true ]; then
    echo ""
    echo "[4/4] Launching dual-camera pipeline..."
    mkdir -p "${RESULTS_DIR}"

    # Use TensorRT engine if available, otherwise ONNX
    MODEL_PATH="${ENGINE_MODEL}"
    if [ ! -f "${ENGINE_PATH:-}" ] && [ ! -f "${ENGINE_MODEL}" ]; then
        MODEL_PATH="${ONNX_MODEL}"
    fi
    if [ ! -f "${MODEL_PATH}" ]; then
        MODEL_PATH="${REPO_DIR}/yolov8n.pt"
    fi

    echo "  Model: ${MODEL_PATH}"
    echo "  Thermal: ${THERMAL_SOURCE}"
    echo "  Visible: ${VISIBLE_SOURCE}"
    echo ""

    python3 "${REPO_DIR}/pipeline_engineer/dual_camera_pipeline.py" \
        --thermal-source "${THERMAL_SOURCE}" \
        --visible-source "${VISIBLE_SOURCE}" \
        --model "${MODEL_PATH}" \
        --no-display \
        --output-dir "${RESULTS_DIR}" \
        ${MAX_FRAMES:+--max-frames "${MAX_FRAMES}"}
fi

echo ""
echo "=========================================="
echo "  ✅ Phase 3 deployment complete!"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="
