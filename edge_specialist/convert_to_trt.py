#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine for deployment on Jetson.

On the Jetson, this uses the TensorRT Python API or trtexec CLI to build
an optimized engine (.engine file) from the exported ONNX model.

On the host (development machine), the script can run in "dry-run" mode
to validate the ONNX model and print the exact trtexec command that
should be executed on the Jetson device.

Usage (on Jetson):
    python -m edge_specialist.convert_to_trt \
        --onnx ../optimized_models/best_fp16_dynamic.onnx \
        --engine optimized_models/best_fp16_dynamic.engine \
        --fp16

    # INT8 with calibration
    python -m edge_specialist.convert_to_trt \
        --onnx ../optimized_models/best.onnx \
        --engine optimized_models/best_int8.engine \
        --int8 --calib-cache data/calibration.cache

    # With dynamic shapes
    python -m edge_specialist.convert_to_trt \
        --onnx ../optimized_models/best_fp16_dynamic.onnx \
        --engine optimized_models/best_fp16_dynamic.engine \
        --fp16 \
        --min-shape 1x3x640x640 \
        --opt-shape 8x3x640x640 \
        --max-shape 16x3x640x640

Usage (dry-run / host validation):
    python -m edge_specialist.convert_to_trt \
        --onnx ../optimized_models/best_fp16_dynamic.onnx \
        --engine optimized_models/best_fp16_dynamic.engine \
        --fp16 --dry-run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX → TensorRT engine")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to input ONNX model")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path for output TensorRT engine file")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision (recommended for Jetson)")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 precision (requires calibration cache)")
    parser.add_argument("--calib-cache", type=str, default="",
                        help="Path to INT8 calibration cache file")
    parser.add_argument("--min-shape", type=str, default="1x3x640x640",
                        help="Minimum input shape (batchxCxHxW)")
    parser.add_argument("--opt-shape", type=str, default="8x3x640x640",
                        help="Optimum input shape (batchxCxHxW)")
    parser.add_argument("--max-shape", type=str, default="16x3x640x640",
                        help="Maximum input shape (batchxCxHxW)")
    parser.add_argument("--workspace", type=int, default=2048,
                        help="Max workspace size in MB (default: 2048)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print trtexec command without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose layer information")
    return parser.parse_args()


def parse_shape(shape_str: str):
    """Parse 'Bx3xHxW' into (B, C, H, W) tuple."""
    parts = shape_str.split("x")
    if len(parts) != 4:
        raise ValueError(f"Invalid shape format '{shape_str}'. Use BxCxHxW (e.g. 1x3x640x640).")
    return tuple(int(p) for p in parts)


def build_trtexec_command(args) -> str:
    """Build the trtexec command string for ONNX→TensorRT conversion."""
    cmd_parts = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--workspace={args.workspace}",
        f"--minShapes=images:{args.min_shape}",
        f"--optShapes=images:{args.opt_shape}",
        f"--maxShapes=images:{args.max_shape}",
    ]

    if args.fp16:
        cmd_parts.append("--fp16")
    if args.int8:
        cmd_parts.append("--int8")
        if args.calib_cache:
            cmd_parts.append(f"--calib={args.calib_cache}")

    if args.verbose:
        cmd_parts.append("--verbose")

    # Skip-NA and other useful flags
    cmd_parts.append("--skipInference")  # Don't run inference on host GPU (not available on Jetson x86)
    cmd_parts.append("--duration=3")

    return " ".join(cmd_parts)


def build_python_command(args) -> str:
    """
    Alternative: use the TensorRT Python API directly (instead of trtexec).
    This is used when trtexec is not available but the 'tensorrt' Python
    package is installed (standard on Jetson).
    """
    min_shape = parse_shape(args.min_shape)
    opt_shape = parse_shape(args.opt_shape)
    max_shape = parse_shape(args.max_shape)
    fp16_line = 'config.set_flag(trt.BuilderFlag.FP16)' if args.fp16 else ''
    int8_line = 'config.set_flag(trt.BuilderFlag.INT8)' if args.int8 else ''
    calib_line = f"config.int8_calibrator = trt.Calibrator('{args.calib_cache}')" if args.int8 and args.calib_cache else ''
    return f"""python -c "
import tensorrt as trt
import sys
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)
with open('{args.onnx}', 'rb') as f:
    if not parser.parse(f.read()):
        for e in range(parser.num_errors):
            print(parser.get_error(e))
        sys.exit(1)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, {args.workspace} * 1024 * 1024)
{fp16_line}
{int8_line}
{calib_line}
profile = builder.create_optimization_profile()
profile.set_shape('images', {min_shape}, {opt_shape}, {max_shape})
config.add_optimization_profile(profile)
engine = builder.build_engine(network, config)
with open('{args.engine}', 'wb') as f:
    f.write(engine.serialize())
print('Engine built and saved to {args.engine}')
print(f'Engine size: {{len(engine.serialize()) / 1024 / 1024:.1f}} MB')
"
"""


def main():
    args = parse_args()

    # Validate ONNX file
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Validate ONNX model — try strict check first, then fall back to
    # a lenient check that infers shapes (Ultralytics-exported dynamic
    # models sometimes trigger topological-sort warnings that TensorRT's
    # own parser tolerates).
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    onnx_input_names = []
    onnx_output_names = []

    try:
        import onnx
        model = onnx.load(str(onnx_path))
        try:
            onnx.checker.check_model(model)
            print(f"[convert_to_trt] ONNX model validated: {onnx_path}")
        except Exception as check_err:
            # The strict checker can flag topological ordering that
            # TensorRT's parser handles fine.  Infer shapes and retry
            # the lighter validation before giving up.
            print(f"[convert_to_trt] Strict check flagged: {check_err}")
            print("[convert_to_trt] Attempting shape inference / topological fix…")
            try:
                model = onnx.shape_inference.infer_shapes(model)
                onnx.checker.check_model(model)
                print(f"[convert_to_trt] ONNX model validated (after shape inference): {onnx_path}")
            except Exception:
                # Still not strictly valid — warn but do NOT abort.
                # trtexec / the TensorRT parser is more lenient and will
                # report the real error if the model is truly broken.
                print(f"[convert_to_trt] WARNING: ONNX strict validation failed.")
                print(f"[convert_to_trt]   {check_err}")
                print("[convert_to_trt] Proceeding — TensorRT parser will validate during build.")

        onnx_input_names = [(i.name, [d.dim_value if d.HasField("dim_value") else d.dim_param
                                       for d in i.type.tensor_type.shape.dim])
                            for i in model.graph.input]
        onnx_output_names = [(o.name, [d.dim_value if d.HasField("dim_value") else d.dim_param
                                         for d in o.type.tensor_type.shape.dim])
                             for o in model.graph.output]
        print(f"  Inputs:  {onnx_input_names}")
        print(f"  Outputs: {onnx_output_names}")
        print(f"  Size: {size_mb:.1f} MB")
    except ImportError:
        print("[convert_to_trt] onnx package not available — skipping validation")
    except Exception as e:
        print(f"[convert_to_trt] WARNING: Could not inspect ONNX model: {e}")

    # Build conversion command
    precision = "FP16" if args.fp16 else ("INT8" if args.int8 else "FP32")

    print(f"\n{'='*60}")
    print(f"[convert_to_trt] Conversion Parameters")
    print(f"{'='*60}")
    print(f"  Input ONNX:    {onnx_path}")
    print(f"  Output engine: {args.engine}")
    print(f"  Precision:     {precision}")
    print(f"  Min shape:     {args.min_shape}")
    print(f"  Opt shape:     {args.opt_shape}")
    print(f"  Max shape:     {args.max_shape}")
    print(f"  Workspace:     {args.workspace} MB")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[convert_to_trt] Dry-run mode — command to execute on Jetson:")
        print(f"\n  # Method 1: trtexec CLI\n  {build_trtexec_command(args)}\n")
        print(f"  # Method 2: TensorRT Python API\n  {build_python_command(args)}")
        return

    # Check if we're on Jetson (has trtexec or tensorrt package)
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    has_trtexec = os.path.exists(trtexec_path)
    has_tensorrt_py = False
    try:
        import tensorrt  # noqa: F401
        has_tensorrt_py = True
    except ImportError:
        pass

    if not has_trtexec and not has_tensorrt_py:
        print("\n[convert_to_trt] WARNING: TensorRT not found on this system.")
        print("  This script should be run on a Jetson device with JetPack SDK installed.")
        print(f"\n  Command to run on Jetson:")
        print(f"  {build_trtexec_command(args)}")
        print(f"\n  Or with Python API:")
        print(f"  {build_python_command(args)}")
        sys.exit(1)

    # Execute conversion
    if has_trtexec:
        cmd = build_trtexec_command(args).split()
        # Remove --dry-run flags
        cmd = [c for c in cmd if c != "--dry-run"]
        print(f"\n[convert_to_trt] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"[convert_to_trt] ERROR: trtexec failed with code {result.returncode}")
            sys.exit(result.returncode)
        print(f"[convert_to_trt] Engine saved to: {args.engine}")

    elif has_tensorrt_py:
        print(f"\n[convert_to_trt] Running TensorRT Python API conversion...")
        # Execute the Python API command
        py_cmd = build_python_command(args)
        result = subprocess.run([sys.executable, "-c", py_cmd[7:-2]],  # Extract the python code
                                capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"[convert_to_trt] ERROR: Python API conversion failed with code {result.returncode}")
            sys.exit(result.returncode)

    # Report engine size
    engine_path = Path(args.engine)
    if engine_path.exists():
        engine_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"\n[convert_to_trt] Engine file size: {engine_mb:.1f} MB")
        compression = (1 - engine_mb / size_mb) * 100 if size_mb > 0 else 0
        print(f"  Compression: {compression:.1f}% (ONNX: {size_mb:.1f} MB → TRT: {engine_mb:.1f} MB)")


if __name__ == "__main__":
    main()
