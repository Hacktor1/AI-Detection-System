#!/usr/bin/env python3
"""Jetson Orin Nano deployment helper (Phase 3).

Provides two capabilities that run on the Jetson device itself:

1. **Environment check** — verifies JetPack, CUDA, TensorRT, and camera
   devices are present before deploying.
2. **Deploy** — converts the ONNX model to a TensorRT engine (if not
   already present), then launches the dual-camera pipeline in headless
   mode.

Usage:

    # Phase 3 prerequisite: verify the Jetson environment
    python -m edge_specialist.jetson_deploy --check

    # Phase 3: deploy model + launch pipeline (CSI cameras)
    python -m edge_specialist.jetson_deploy --deploy \
        --onnx ../optimized_models/best_fp16_dynamic.onnx \
        --thermal-source /dev/video0 \
        --visible-source /dev/video1

    # Dry-run: print the trtexec command without executing
    python -m edge_specialist.jetson_deploy --convert \
        --onnx optimized_models/best_fp16_dynamic.onnx \
        --fp16 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------ #
#  Environment checks
# ------------------------------------------------------------------ #
class EnvCheck:
    """Result of a single environment check."""

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        line = f"  [{status}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


def _check_command(cmd: str) -> tuple[bool, str]:
    """Return (found, version_string)."""
    path = shutil.which(cmd)
    if path is None:
        return False, "not found in PATH"
    return True, path


def _detect_jetpack() -> tuple[bool, str]:
    """Detect JetPack via /etc/nv_tegra_release or lsb_release."""
    # Method 1: NVIDIA L4T release file
    l4t = Path("/etc/nv_tegra_release")
    if l4t.exists():
        content = l4t.read_text()
        for line in content.splitlines():
            if "R" in line and "rev" in line:
                return True, line.strip()
        return True, content.strip()[:80]

    # Method 2: JetPack version package (dpkg-based)
    try:
        result = subprocess.run(
            ["dpkg", "-l", "nvidia-jetpack"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and "ii" in result.stdout:
            for line in result.stdout.splitlines():
                if "nvidia-jetpack" in line:
                    return True, line.strip()
    except (FileNotFoundError, OSError):
        pass  # dpkg not available (e.g. non-Debian host)

    # Method 3: lsb_release
    try:
        result = subprocess.run(
            ["lsb_release", "-d"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            desc = result.stdout.strip()
            if "L4T" in desc or "JetPack" in desc:
                return True, desc
    except (FileNotFoundError, OSError):
        pass

    return False, "JetPack not detected (not on Jetson?)"


def _detect_trt() -> tuple[bool, str]:
    """Detect TensorRT via trtexec and Python package."""
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    checks = []

    if os.path.exists(trtexec):
        checks.append("trtexec @ " + trtexec)
    else:
        found, info = _check_command("trtexec")
        if found:
            checks.append(f"trtexec in PATH ({info})")

    try:
        import tensorrt as trt  # noqa: F401
        checks.append(f"tensorrt py: {trt.__version__}")
    except ImportError:
        checks.append("tensorrt py: not installed")

    if checks:
        return True, " | ".join(checks)
    return False, "TensorRT not found"


def _detect_cuda() -> tuple[bool, str]:
    nvcc = shutil.which("nvcc")
    if nvcc:
        r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
        version_line = ""
        for line in r.stdout.splitlines():
            if "release" in line:
                version_line = line.strip()
                break
        return True, f"{nvcc} — {version_line}"
    # On Jetson CUDA is in /usr/local/cuda
    cuda_dir = Path("/usr/local/cuda")
    if cuda_dir.exists():
        return True, f"CUDA toolkit @ {cuda_dir}"
    return False, "CUDA not detected"


def _detect_cameras() -> tuple[bool, str]:
    """Detect /dev/video* devices."""
    video_devices = sorted(Path("/dev").glob("video*"))
    if not video_devices:
        return False, "No /dev/video* devices found"
    names = ", ".join(d.name for d in video_devices)
    return True, f"{len(video_devices)} device(s): {names}"


def _detect_tegrastats() -> tuple[bool, str]:
    tegrastats = shutil.which("tegrastats")
    if tegrastats:
        return True, tegrastats
    return False, "tegrastats not in PATH (JetPack not installed?)"


def run_environment_check() -> list[EnvCheck]:
    """Run all environment checks and return results."""
    checks = [
        EnvCheck("Jetson / JetPack", *_detect_jetpack()),
        EnvCheck("TensorRT", *_detect_trt()),
        EnvCheck("CUDA", *_detect_cuda()),
        EnvCheck("GPU profiling (tegrastats)", *_detect_tegrastats()),
        EnvCheck("Camera devices (/dev/video*)", *_detect_cameras()),
        EnvCheck("ONNX model", os.path.exists(PROJECT_ROOT / "optimized_models" / "best_fp16_dynamic.onnx"),
                 str(PROJECT_ROOT / "optimized_models" / "best_fp16_dynamic.onnx")),
        EnvCheck("Pipeline code",
                 (PROJECT_ROOT / "pipeline_engineer" / "dual_camera_pipeline.py").exists(),
                 ""),
    ]
    return checks


# ------------------------------------------------------------------ #
#  TensorRT conversion (wraps convert_to_trt)
# ------------------------------------------------------------------ #
def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 int8: bool = False, workspace: int = 2048,
                 dry_run: bool = False) -> bool:
    """Convert ONNX → TensorRT engine via trtexec (or Python API)."""
    from edge_specialist.convert_to_trt import build_trtexec_command, parse_shape
    # Build a lightweight args-like object
    class _A:
        pass
    args = _A()
    args.onnx = onnx_path
    args.engine = engine_path
    args.fp16 = fp16
    args.int8 = int8
    args.calib_cache = ""
    args.min_shape = "1x3x640x640"
    args.opt_shape = "8x3x640x640"
    args.max_shape = "16x3x640x640"
    args.workspace = workspace
    args.dry_run = dry_run
    args.verbose = False

    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if os.path.exists(trtexec_path) and not dry_run:
        cmd = build_trtexec_command(args).split()
        cmd = [c for c in cmd if c != "--dry-run"]
        print(f"[jetson_deploy] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    else:
        # Dry-run or no trtexec — print the command
        if dry_run:
            print(f"[jetson_deploy] Dry-run — trtexec command:")
            print(f"  {build_trtexec_command(args)}")
        else:
            cmd = build_trtexec_command(args).split()
            cmd = [c for c in cmd if c != "--dry-run"]
            print(f"[jetson_deploy] trtexec not found — command to run:")
            print(f"  {' '.join(cmd)}")
        return dry_run  # In dry-run we "succeed"; otherwise we fail gracefully


# ------------------------------------------------------------------ #
#  Full deploy
# ------------------------------------------------------------------ #
def deploy(onnx_path: str, thermal_source: str, visible_source: str,
           engine_path: str = "", fp16: bool = True,
           max_frames: int = 0) -> bool:
    """Convert model (if needed) and launch the dual-camera pipeline."""
    onnx = Path(onnx_path)
    if not onnx.exists():
        print(f"[jetson_deploy] ERROR: ONNX not found: {onnx}")
        return False

    engine = Path(engine_path) if engine_path else onnx.parent / "best_fp16_dynamic.engine"
    engine = engine if engine.suffix == ".engine" else onnx.parent / (onnx.stem + ".engine")

    # Convert to engine if not present
    if not engine.exists():
        print(f"[jetson_deploy] Converting {onnx} → {engine} ...")
        if not build_engine(str(onnx), str(engine), fp16=fp16, dry_run=False):
            print("[jetson_deploy] TensorRT conversion failed — falling back to ONNX")
            engine = onnx
    else:
        print(f"[jetson_deploy] Engine already exists: {engine}")

    # Launch the pipeline headless
    pipeline = str(PROJECT_ROOT / "pipeline_engineer" / "dual_camera_pipeline.py")
    cmd = [
        sys.executable, pipeline,
        "--thermal-source", thermal_source,
        "--visible-source", visible_source,
        "--model", str(engine),
        "--no-display",
        "--output-dir", str(PROJECT_ROOT / "results"),
    ]
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]

    print(f"[jetson_deploy] Launching pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano deployment helper (Phase 3)")
    parser.add_argument("--check", action="store_true",
                        help="Run environment checks and exit")
    parser.add_argument("--convert", action="store_true",
                        help="Convert ONNX → TensorRT engine only")
    parser.add_argument("--deploy", action="store_true",
                        help="Convert (if needed) and launch dual-camera pipeline")
    parser.add_argument("--onnx", type=str, default="optimized_models/best_fp16_dynamic.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--engine", type=str, default="",
                        help="Path for TensorRT engine (default: same dir as ONNX)")
    parser.add_argument("--thermal-source", type=str, default="/dev/video0",
                        help="Thermal camera source (default: /dev/video0)")
    parser.add_argument("--visible-source", type=str, default="/dev/video1",
                        help="Visible camera source (default: /dev/video1)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 precision (default: yes)")
    parser.add_argument("--int8", action="store_true", default=False,
                        help="Use INT8 precision (requires calibration cache)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit pipeline frames (0 = unlimited)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.check:
        print("=" * 60)
        print("[jetson_deploy] Jetson Orin Nano Environment Check")
        print("=" * 60)
        checks = run_environment_check()
        all_pass = True
        for c in checks:
            print(c)
            if not c.passed:
                all_pass = False
        print("=" * 60)
        if all_pass:
            print("✅ All checks passed — ready for Phase 3 deployment!")
        else:
            print("❌ Some checks failed — see details above.")
            print("   Ensure JetPack SDK is installed and cameras are connected.")
        return 0 if all_pass else 1

    if args.convert:
        onnx_path = args.onnx
        engine_path = args.engine or (Path(onnx_path).parent / "best_fp16_dynamic.engine").as_posix()
        ok = build_engine(onnx_path, engine_path, fp16=args.fp16, int8=args.int8, dry_run=args.dry_run)
        return 0 if ok else 1

    if args.deploy:
        ok = deploy(args.onnx, args.thermal_source, args.visible_source,
                    args.engine, fp16=args.fp16, max_frames=args.max_frames)
        return 0 if ok else 1

    # Default: run checks
    print("[jetson_deploy] No action specified. Use --check, --convert, or --deploy.")
    print("Run with --help for options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
