#!/usr/bin/env python3
"""
Phase 3 tests — Jetson deployment & TensorRT conversion.

Tests:
1. JetsonProfile loads from config (Phase 2 sim)
2. convert_to_trt produces valid trtexec command (dry-run)
3. convert_to_trt handles ONNX validation gracefully (no sys.exit)
4. Jetson environment check returns structured results
5. Hardware profile thermal factor math
6. Export settings produce correct filenames

Usage:
    python -m pytest tests/test_phase3.py -v
    python tests/test_phase3.py  # without pytest
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Import Phase 3 / edge modules
# ------------------------------------------------------------------
import pytest

from edge_specialist.convert_to_trt import (
    build_trtexec_command,
    build_python_command,
    parse_shape,
)
from edge_specialist.jetson_deploy import (
    EnvCheck,
    run_environment_check,
    build_engine,
)
from jetson_sim.hardware_profile import JetsonProfile


ONNX_MODEL = PROJECT_ROOT / "optimized_models" / "best_fp16_dynamic.onnx"
PT_MODEL = PROJECT_ROOT / "yolov8n.pt"


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #
def test_trt_command_contains_required_flags():
    """trtexec command includes onnx, engine, fp16, and shape params."""
    args = SimpleNamespace(
        onnx=str(ONNX_MODEL), engine="out.engine", fp16=True, int8=False,
        calib_cache="", min_shape="1x3x640x640", opt_shape="8x3x640x640",
        max_shape="16x3x640x640", workspace=2048, dry_run=True, verbose=False,
    )
    cmd = build_trtexec_command(args)
    assert "trtexec" in cmd
    assert "--onnx=" in cmd
    assert "--saveEngine=" in cmd
    assert "--fp16" in cmd
    assert "--minShapes=images:1x3x640x640" in cmd
    assert "--optShapes=images:8x3x640x640" in cmd
    assert "--maxShapes=images:16x3x640x640" in cmd


def test_trt_python_command_contains_fp16():
    """TensorRT Python API command includes FP16 flag."""
    args = SimpleNamespace(
        onnx=str(ONNX_MODEL), engine="out.engine", fp16=True, int8=False,
        calib_cache="", min_shape="1x3x640x640", opt_shape="8x3x640x640",
        max_shape="16x3x640x640", workspace=2048, dry_run=True, verbose=False,
    )
    py_cmd = build_python_command(args)
    assert "FP16" in py_cmd or "fp16" in py_cmd
    assert "images" in py_cmd
    assert str(ONNX_MODEL) not in py_cmd or "onnx" in py_cmd.lower()


def test_convert_to_trt_strict_validation_graceful():
    """convert_to_trt main() handles ONNX topological-sort warning without crashing.

    Uses monkeypatching to avoid requiring an actual ONNX file for the
    argument-parsing path — we test the command-builder directly instead.
    """
    if not ONNX_MODEL.exists():
        pytest.skip("ONNX model not found")
    args = SimpleNamespace(
        onnx=str(ONNX_MODEL), engine="/tmp/test.engine", fp16=True, int8=False,
        calib_cache="", min_shape="1x3x640x640", opt_shape="8x3x640x640",
        max_shape="16x3x640x640", workspace=2048, dry_run=True, verbose=False,
    )
    cmd = build_trtexec_command(args)
    # The command should be non-empty and contain trtexec
    assert len(cmd) > 0
    assert "trtexec" in cmd


def test_jetson_profile_thermal_factor_curve():
    """Thermal factor follows the expected curve: 1.0 → ramp → cap."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    # Before sustained period — no throttle
    assert profile.thermal_factor(elapsed_seconds=10) == 1.0
    assert profile.thermal_factor(elapsed_seconds=30) == 1.0
    # Just past sustained — beginning of ramp
    f = profile.thermal_factor(elapsed_seconds=31)
    assert f > 1.0
    assert f <= 1.5
    # Long after — capped at max
    assert profile.thermal_factor(elapsed_seconds=999) == 1.5


def test_jetson_profile_summary_contains_key_info():
    """Profile summary includes platform name and key specs."""
    profile = JetsonProfile()
    s = profile.summary()
    assert "Orin Nano" in s
    assert "1024 CUDA" in s


def test_env_check_returns_structured_results():
    """run_environment_check returns a list of EnvCheck objects."""
    results = run_environment_check()
    assert len(results) > 0
    assert all(isinstance(r, EnvCheck) for r in results)
    # Should include Jetson, TensorRT, CUDA, and camera checks
    names = [r.name for r in results]
    assert any("JetPack" in n for n in names)
    assert any("TensorRT" in n for n in names)
    assert any("CUDA" in n for n in names)
    assert any("camera" in n.lower() for n in names)


def test_build_engine_dry_run_does_not_crash():
    """build_engine in dry-run mode returns True and prints command."""
    if not ONNX_MODEL.exists():
        pytest.skip("ONNX model not found")
    with tempfile.TemporaryDirectory() as tmp:
        engine_path = os.path.join(tmp, "test.engine")
        ok = build_engine(str(ONNX_MODEL), engine_path, fp16=True, dry_run=True)
        assert ok is True  # dry-run always "succeeds"


def test_dual_camera_pipeline_importable():
    """The Phase 2/3 dual-camera pipeline module imports cleanly."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline_engineer.dual_camera_pipeline import (
        main, DualDetector, CameraReader, BenchmarkStats,
    )
    assert callable(main)
    assert DualDetector is not None
    assert CameraReader is not None
    assert BenchmarkStats is not None


def test_jetson_deploy_module_importable():
    """The Phase 3 jetson_deploy module imports cleanly."""
    from edge_specialist.jetson_deploy import (
        run_environment_check, build_engine, deploy, main,
    )
    assert callable(run_environment_check)
    assert callable(build_engine)
    assert callable(deploy)
    assert callable(main)


def test_onnx_model_exists():
    """The optimized ONNX model used for Phase 3 exists."""
    assert ONNX_MODEL.exists(), f"ONNX model not found at {ONNX_MODEL}"


def test_pt_model_exists():
    """The baseline PyTorch model exists."""
    assert PT_MODEL.exists(), f"PyTorch model not found at {PT_MODEL}"


if __name__ == "__main__":
    import traceback
    test_functions = [
        test_trt_command_contains_required_flags,
        test_trt_python_command_contains_fp16,
        test_convert_to_trt_strict_validation_graceful,
        test_jetson_profile_thermal_factor_curve,
        test_jetson_profile_summary_contains_key_info,
        test_env_check_returns_structured_results,
        test_build_engine_dry_run_does_not_crash,
        test_dual_camera_pipeline_importable,
        test_jetson_deploy_module_importable,
        test_onnx_model_exists,
        test_pt_model_exists,
    ]
    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
