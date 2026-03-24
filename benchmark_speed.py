"""
ONNX model inference speed benchmark with profiling.

Measures FLOPs, parameters, and per-iteration latency/FPS for an ONNX model
using ONNX Runtime, then saves detailed results to CSV.

Usage:
    python benchmark_speed.py \
        --model_path model.onnx \
        --input_shape 640 \
        --num_iters 100 \
        --output_csv speed_results.csv
"""

import argparse
import csv
import gc
import time

import numpy as np
import onnx
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed and profile an ONNX object detection model"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the ONNX model file")
    parser.add_argument("--input_shape", type=int, default=None,
                        help="Input image size (e.g., 320, 440, 640). "
                             "Used as both height and width. "
                             "If not specified, uses the model's native resolution.")
    parser.add_argument("--num_iters", type=int, default=100,
                        help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--output_csv", default="speed_results.csv",
                        help="Path to save per-iteration results (default: speed_results.csv)")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Model profiling
# ──────────────────────────────────────────────────────────────
def profile_model(model_path, input_shape):
    """
    Calculate FLOPs (G) and Parameters (M) for an ONNX model.

    Uses onnx-tool for accurate ONNX-level profiling.
    Falls back to parameter counting from raw weights if onnx-tool
    is not installed.
    """
    total_flops = None
    total_params = None

    # Try onnx-tool first
    try:
        import onnx_tool

        model = onnx.load(model_path)
        # onnx_tool.model_profile expects shape as dict or list
        # Build dummy shape from model input
        input_name = model.graph.input[0].name
        shape_map = {input_name: [1, 3, input_shape, input_shape]}

        # model_profile returns (flops, params) or prints them
        stats = onnx_tool.model_profile(model_path, shape_map, savenode="")
        if stats is not None:
            total_flops, total_params = stats[0], stats[1]
    except ImportError:
        pass
    except Exception as e:
        print(f"  onnx-tool profiling failed: {e}")

    # Fallback: count parameters from ONNX initializers
    if total_params is None:
        model = onnx.load(model_path)
        total_params = 0
        for initializer in model.graph.initializer:
            param_count = 1
            for dim in initializer.dims:
                param_count *= dim
            total_params += param_count

    return total_flops, total_params


def print_profile(total_flops, total_params):
    """Print model profile summary."""
    print("\n── Model Profile ──")
    if total_flops is not None:
        gflops = total_flops / 1e9
        print(f"  FLOPs:      {gflops:.2f} G")
    else:
        print("  FLOPs:      N/A (install onnx-tool for FLOPs calculation)")
    params_m = total_params / 1e6
    print(f"  Parameters: {params_m:.2f} M")


# ──────────────────────────────────────────────────────────────
# Speed benchmarking
# ──────────────────────────────────────────────────────────────
def create_session(model_path):
    """Create an ONNX Runtime inference session with available providers."""
    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)
    active = session.get_providers()
    print(f"\n── ONNX Runtime ──")
    print(f"  Providers: {active}")
    return session


def _resolve_hw(model_shape_dims, input_shape, default=640):
    """Resolve H/W dimensions from model shape and user override.

    Returns (h, w, effective_input_shape) and prints a warning if the user
    override differs from the model's fixed resolution.
    """
    # Extract model's native H/W (last two spatial dims)
    # These may be int (fixed) or str/None (dynamic)
    h_dim, w_dim = model_shape_dims

    def _native(dim):
        return dim if isinstance(dim, int) else None

    native_h, native_w = _native(h_dim), _native(w_dim)

    if input_shape is not None:
        # User explicitly specified --input_shape
        if native_h is not None and native_h != input_shape:
            print(f"  ⚠ Warning: model native resolution is {native_h}x{native_w}, "
                  f"but --input_shape={input_shape} was specified. "
                  f"This may cause incorrect results for models with fixed positional encodings.")
        return input_shape, input_shape, input_shape

    # No user override: use model's native or fallback to default
    h = native_h if native_h is not None else default
    w = native_w if native_w is not None else default
    effective = h  # for display purposes
    return h, w, effective


def build_dummy_input(session, input_shape):
    """Build a dummy numpy input matching the model's expected format."""
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = list(input_meta.shape)

    if len(shape) == 3:
        # 3D input (no batch dim): [C, H, W]  e.g. Faster R-CNN
        h, w, effective = _resolve_hw(shape[1:3], input_shape)
        shape[1] = h
        shape[2] = w
        # Keep shape[0] (channels) as-is; resolve if dynamic
        if isinstance(shape[0], str) or shape[0] is None:
            shape[0] = 3
    elif len(shape) == 4:
        # 4D input (with batch dim): [N, C, H, W]
        h, w, effective = _resolve_hw(shape[2:4], input_shape)
        shape[2] = h
        shape[3] = w
        # Resolve batch and channel dims
        if isinstance(shape[0], str) or shape[0] is None:
            shape[0] = 1
        shape[0] = 1  # force batch=1
        if isinstance(shape[1], str) or shape[1] is None:
            shape[1] = 3
    else:
        raise ValueError(f"Unexpected input rank {len(shape)}: {shape}")

    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint8)": np.uint8,
    }
    np_dtype = dtype_map.get(input_meta.type, np.float32)
    dummy = np.random.rand(*shape).astype(np_dtype)

    print(f"  Input name: {input_name}")
    print(f"  Input shape: {list(shape)} ({np_dtype.__name__})")

    return input_name, dummy


def run_benchmark(session, input_name, dummy_input, num_iters, warmup=20):
    """
    Run warm-up + timed benchmark iterations.

    Returns list of per-iteration latency in milliseconds.
    """
    feed = {input_name: dummy_input}

    # Warm-up
    print(f"\n── Benchmark ──")
    print(f"  Warm-up: {warmup} iterations...", end=" ", flush=True)
    for _ in range(warmup):
        session.run(None, feed)
    print("done")

    # Timed runs
    print(f"  Benchmarking: {num_iters} iterations...", end=" ", flush=True)
    latencies = []
    gc.disable()
    for _ in range(num_iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
    gc.enable()
    print("done")

    return latencies


# ──────────────────────────────────────────────────────────────
# CSV output & aggregation
# ──────────────────────────────────────────────────────────────
def write_csv(latencies, output_path):
    """Write per-iteration latency and FPS to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_ms", "fps"])
        for i, lat in enumerate(latencies, start=1):
            fps = 1000.0 / lat if lat > 0 else 0.0
            writer.writerow([i, round(lat, 4), round(fps, 2)])
    print(f"\n  Per-iteration results saved to: {output_path}")


def print_summary(latencies):
    """Print aggregate statistics including percentiles."""
    arr = np.array(latencies)
    avg_latency = arr.mean()
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    med_latency = np.median(arr)
    med_fps = 1000.0 / med_latency if med_latency > 0 else 0.0
    std_latency = arr.std()
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)

    print(f"\n── Summary ──")
    print(f"  Iterations:    {len(latencies)}")
    print(f"  Avg Latency:   {avg_latency:.2f} ms  (FPS: {avg_fps:.2f})")
    print(f"  Med Latency:   {med_latency:.2f} ms  (FPS: {med_fps:.2f})")
    print(f"  Std Latency:   {std_latency:.2f} ms")
    print(f"  Min Latency:   {arr.min():.2f} ms")
    print(f"  Max Latency:   {arr.max():.2f} ms")
    print(f"  P95 Latency:   {p95:.2f} ms")
    print(f"  P99 Latency:   {p99:.2f} ms")
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"Model: {args.model_path}")
    if args.input_shape:
        print(f"Input shape override: {args.input_shape}x{args.input_shape}")
    else:
        print("Input shape: auto (using model's native resolution)")

    # 1. Create session & dummy input (need shape info before profiling)
    session = create_session(args.model_path)
    input_name, dummy_input = build_dummy_input(session, args.input_shape)

    # 2. Profile (use actual resolved shape)
    actual_shape = dummy_input.shape[-1]  # last dim = W
    total_flops, total_params = profile_model(args.model_path, actual_shape)
    print_profile(total_flops, total_params)

    # 3. Benchmark
    latencies = run_benchmark(session, input_name, dummy_input, args.num_iters)

    # 4. Save & summarize
    write_csv(latencies, args.output_csv)
    print_summary(latencies)


if __name__ == "__main__":
    main()
