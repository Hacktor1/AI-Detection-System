#!/usr/bin/env python3
"""
Tests for dual-camera pipeline and Jetson simulation.

Tests:
1. JetsonProfile loads config correctly
2. Thermal throttling simulation
3. Visible-light video generation
4. Dual camera pipeline runs end-to-end
5. ONNX inference produces valid output
6. convert_to_trt dry-run produces valid command

Usage:
    python -m pytest tests/test_dual_camera.py -v
    python tests/test_dual_camera.py  # without pytest
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------
# Jetson hardware profile tests
# ------------------------------------------------------------------
def test_jetson_profile_loads():
    """Verify JetsonProfile loads from config.yaml."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    assert profile.name == "Jetson Orin Nano (simulated)"
    assert profile.cpu.cores == 6
    assert profile.gpu.cuda_cores == 1024
    assert profile.gpu.tdp_w == 15.0


def test_jetson_profile_cpu_threads():
    """Verify CPU thread limiting returns reduced count."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    threads = profile.cpu_threads_for_inference()
    assert threads == 4  # sim_cpu_threads in config


def test_thermal_throttle_no_throttle_initially():
    """Verify no throttling at start."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    profile.start_inference_timer()
    factor = profile.thermal_factor(elapsed_seconds=5)
    assert factor == 1.0


def test_thermal_throttle_after_sustained():
    """Verify throttling kicks in after sustained period."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    factor = profile.thermal_factor(elapsed_seconds=60)  # 30s sustained + 30s into ramp
    assert factor > 1.0
    assert factor <= 1.5  # max throttle factor


def test_thermal_throttle_caps():
    """Verify throttling caps at max factor."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    factor = profile.thermal_factor(elapsed_seconds=999)
    assert factor == 1.5  # max_throttle_factor


def test_thermal_throttle_disabled():
    """Verify throttle is disabled when sim_throttle is False."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    profile.sim.enable_throttle = False
    factor = profile.thermal_factor(elapsed_seconds=999)
    assert factor == 1.0


def test_memory_limit():
    """Verify memory limit checking."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    assert profile.check_memory(4096)  # Below 6144 MB limit
    assert not profile.check_memory(7000)  # Above limit


def test_profile_summary():
    """Verify summary string is generated."""
    from jetson_sim.hardware_profile import JetsonProfile
    profile = JetsonProfile()
    summary = profile.summary()
    assert "Orin Nano" in summary
    assert "CPU" in summary
    assert "GPU" in summary


# ------------------------------------------------------------------
# Visible video generation tests
# ------------------------------------------------------------------
def test_visible_video_generation():
    """Verify synthetic visible-light video is generated from thermal."""
    from jetson_sim.make_sample_video import generate_visible_video
    thermal_path = PROJECT_ROOT / "pipeline_engineer" / "sample_data" / "test_thermal.mp4"
    if not thermal_path.exists():
        pytest.skip("test_thermal.mp4 not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_visible.mp4")
        generate_visible_video(str(thermal_path), output_path, max_frames=5)

        assert os.path.exists(output_path)
        cap = cv2.VideoCapture(output_path)
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret
        assert frame.shape[2] == 3  # BGR
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == 5
        cap.release()


# ------------------------------------------------------------------
# Dual camera pipeline tests
# ------------------------------------------------------------------
def test_benchmark_stats():
    """Verify BenchmarkStats computation."""
    from pipeline_engineer.dual_camera_pipeline import BenchmarkStats
    stats = BenchmarkStats()
    stats.total_frames = 10
    stats.inference_times = [0.05, 0.04, 0.03]  # 30ms, 40ms, 50ms
    stats.capture_times = [0.01, 0.02]
    stats.render_times = [0.005]
    stats.total_time = 0.3

    assert stats.avg_fps == pytest.approx(33.3, abs=0.1)
    assert stats.avg_inference_ms == pytest.approx(40.0, abs=0.1)
    assert stats.p95_latency_ms == pytest.approx(49.0, abs=0.1)

    d = stats.to_dict()
    assert "avg_fps" in d
    assert "avg_inference_ms" in d


def test_dual_detector_creation():
    """Verify DualDetector can be created from model path."""
    from pipeline_engineer.dual_camera_pipeline import DualDetector
    model_path = str(PROJECT_ROOT / "yolov8n.pt")
    if not os.path.exists(model_path):
        pytest.skip("yolov8n.pt not found")
    detector = DualDetector(model_path, conf_thres=0.3)
    assert detector.thermal_detector is not None
    assert detector.visible_detector is not None


def test_camera_reader():
    """Verify CameraReader reads frames from a video file."""
    from pipeline_engineer.dual_camera_pipeline import CameraReader
    thermal_path = str(PROJECT_ROOT / "pipeline_engineer" / "sample_data" / "test_thermal.mp4")
    if not os.path.exists(thermal_path):
        pytest.skip("test_thermal.mp4 not found")

    reader = CameraReader(thermal_path, name="thermal")
    reader.start()
    time.sleep(0.5)  # Wait for frames to buffer

    frame_result = reader.get_frame(timeout=2.0)
    assert frame_result is not None
    assert frame_result.frame is not None
    assert frame_result.frame_idx >= 0

    reader.release()


# ------------------------------------------------------------------
# ONNX conversion dry-run test
# ------------------------------------------------------------------
def test_convert_to_trt_dry_run():
    """Verify convert_to_trt dry-run produces valid command."""
    onnx_path = str(PROJECT_ROOT / "optimized_models" / "best_fp16_dynamic.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("ONNX model not found")

    # Import the function directly
    from edge_specialist.convert_to_trt import build_trtexec_command, parse_shape

    # Create a mock args namespace
    class MockArgs:
        onnx = onnx_path
        engine = "/tmp/test.engine"
        fp16 = True
        int8 = False
        calib_cache = ""
        min_shape = "1x3x640x640"
        opt_shape = "8x3x640x640"
        max_shape = "16x3x640x640"
        workspace = 2048
        dry_run = True
        verbose = False

    cmd = build_trtexec_command(MockArgs())
    assert "trtexec" in cmd
    assert "--onnx=" in cmd
    assert "--saveEngine=" in cmd
    assert "--fp16" in cmd
    assert "--minShapes=images:1x3x640x640" in cmd

    # Test shape parsing
    shape = parse_shape("1x3x640x640")
    assert shape == (1, 3, 640, 640)


# ------------------------------------------------------------------
# ONNX inference test
# ------------------------------------------------------------------
def test_onnx_inference():
    """Verify ONNX model produces output on a test image."""
    onnx_path = PROJECT_ROOT / "optimized_models" / "best_fp16_dynamic.onnx"
    if not onnx_path.exists():
        pytest.skip("ONNX model not found")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    img = cv2.imread(str(PROJECT_ROOT / "ai_ml_architect" / "runs" / "inference_test" / "190001.jpg"))
    if img is None:
        pytest.skip("Test image not found")

    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

    output = sess.run(None, {input_name: img_chw})[0]
    assert output.shape[0] == 1  # batch
    assert output.shape[1] == 9   # 4 bbox + 1 obj + 4 classes
    assert output.shape[2] > 0    # anchors


if __name__ == "__main__":
    import traceback
    test_functions = [
        test_jetson_profile_loads,
        test_jetson_profile_cpu_threads,
        test_thermal_throttle_no_throttle_initially,
        test_thermal_throttle_after_sustained,
        test_thermal_throttle_caps,
        test_thermal_throttle_disabled,
        test_memory_limit,
        test_profile_summary,
        test_visible_video_generation,
        test_benchmark_stats,
        test_dual_detector_creation,
        test_camera_reader,
        test_convert_to_trt_dry_run,
        test_onnx_inference,
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
