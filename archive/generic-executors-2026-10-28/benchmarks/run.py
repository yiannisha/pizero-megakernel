"""BF16 pi0 latency suite. Each backend runs in a fresh subprocess."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
sys.path[:0] = [
    str(ROOT / "third_party/openpi/src"),
    str(ROOT / "third_party/realtime-vla"),
    str(ROOT),
]
BACKENDS = ("openpi-eager", "openpi-compiled", "openpi-cudagraph", "realtime-triton")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("all",) + BACKENDS, default="all")
    p.add_argument("--views", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--prompt-len", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=ROOT / "results/bf16.json")
    p.add_argument("--openpi-checkpoint", help="Optional OpenPI model.safetensors")
    p.add_argument(
        "--triton-checkpoint",
        help="Optional torch.save weights dict, including language_embeds",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.prompt_len < 0 or a.chunk_size < 1 or a.warmup < 1 or a.iterations < 1:
        p.error(
            "Lengths must be nonnegative; chunk size, warmup and iterations must be positive"
        )
    return a


def percentile(xs, q):
    s = sorted(xs)
    index = (len(s) - 1) * q
    lo = int(index)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (index - lo)


def worker(a):
    import torch
    import triton

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting BF16 is required")
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    images = (
        torch.rand(a.views, 224, 224, 3, device="cuda", dtype=torch.bfloat16) * 2 - 1
    )
    state = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    noise = torch.randn(a.chunk_size, 32, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randint(0, 1000, (1, a.prompt_len), device="cuda")
    setup_start = time.perf_counter()
    validation = {}
    execution = {}
    with torch.inference_mode():
        if a.backend.startswith("openpi"):
            from benchmarks.openpi_backend import build

            model, eager, inputs = build(a, images, state, noise, tokens)
            reference = eager().clone()
            fn = eager
            if a.backend == "openpi-compiled":
                # Disable implicit graphs so this measures compilation independently.
                fn = torch.compile(
                    eager, backend="inductor", mode="max-autotune-no-cudagraphs"
                )
            elif a.backend == "openpi-cudagraph":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(a.warmup):
                        eager()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    static_out = eager()

                def fn():
                    graph.replay()
                    return static_out

            actual = fn()
            torch.cuda.synchronize()
            if a.backend == "openpi-compiled":
                from torch._dynamo.utils import counters
                from torch._inductor import metrics

                execution["compile_stats"] = dict(counters["stats"])
                execution["graph_breaks"] = dict(counters["graph_break"])
                execution["inductor_generated_kernels"] = metrics.generated_kernel_count
                if counters["stats"]["unique_graphs"] < 1:
                    raise RuntimeError("Compiled baseline did not compile any graphs")
            error = (actual - reference).float()
            validation["vs_fixed_eager_max_abs"] = error.abs().max().item()
            if a.backend == "openpi-compiled":
                # Inductor elides intermediate BF16 downcast/upcast pairs across
                # fused operations. Preserve its default precision policy and
                # report that drift explicitly using scale-normalized errors.
                scale = reference.float().square().mean().sqrt().clamp_min(1e-12)
                nrmse = (error.square().mean().sqrt() / scale).item()
                normalized_max = (error.abs().max() / scale).item()
                cosine = torch.nn.functional.cosine_similarity(
                    actual.float().flatten(), reference.float().flatten(), dim=0
                ).item()
                epsilon = torch.finfo(torch.bfloat16).eps
                limits = {
                    "nrmse": 2 * epsilon,
                    "normalized_max_abs": 8 * epsilon,
                    "cosine_min": 0.999,
                }
                validation["compiled_numerics"] = {
                    "nrmse": nrmse,
                    "normalized_max_abs": normalized_max,
                    "cosine": cosine,
                    "reference_rms": scale.item(),
                    "limits": limits,
                    "pointwise_close_rtol_0_02_atol_0_02": torch.isclose(
                        actual, reference, rtol=0.02, atol=0.02
                    )
                    .all()
                    .item(),
                }
                if not (
                    nrmse <= limits["nrmse"]
                    and normalized_max <= limits["normalized_max_abs"]
                    and cosine >= limits["cosine_min"]
                ):
                    raise RuntimeError(
                        f"Compiled BF16 numerical gate failed: {validation}"
                    )
            else:
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
            # Verify fixed loop against the upstream sample_actions implementation.
            # Feed the same prepared tensors to the original upstream loop.
            original_preprocess = model._preprocess_observation

            def prepared(observation, train=False):
                return inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

            model._preprocess_observation = prepared
            from types import SimpleNamespace

            adapted_prefix, adapted_suffix = model.embed_prefix, model.embed_suffix
            model.embed_prefix = model.upstream_embed_prefix
            model.embed_suffix = model.upstream_embed_suffix
            upstream = model.sample_actions(
                torch.device("cuda"), SimpleNamespace(state=inputs[4]), noise=inputs[5]
            )
            model.embed_prefix, model.embed_suffix = adapted_prefix, adapted_suffix
            model._preprocess_observation = original_preprocess
            torch.testing.assert_close(reference, upstream, rtol=1e-5, atol=1e-5)
            validation["vs_upstream_max_abs"] = (
                (reference - upstream).abs().max().item()
            )
        else:
            from pi0_infer import Pi0Inference

            # Upstream's benchmark leaves almost all torch.empty weights uninitialized.
            # Initialize every weight BEFORE constructor warmup and graph capture.
            class InitializedPi0(Pi0Inference):
                def record_infer_graph(self):
                    derived = {"encoder_ffn_packed_w", "decoder_attn_o_packed_w"}
                    for name, weight in self.weights.items():
                        if name in derived:
                            continue
                        if name == "language_embeds":
                            continue
                        if "norm_w" in name:
                            weight.fill_(1)
                        elif name.endswith("_b") or "biases" in name:
                            weight.zero_()
                        else:
                            weight.normal_(0, 0.01)
                    if a.triton_checkpoint:
                        weights = torch.load(
                            a.triton_checkpoint, map_location="cuda", weights_only=True
                        )
                        if set(weights) != set(self.weights) - derived:
                            raise ValueError(
                                "Triton checkpoint must contain every weight and no unknown keys"
                            )
                        for name, value in weights.items():
                            if value.shape != self.weights[name].shape:
                                raise ValueError(f"Shape mismatch: {name}")
                            self.weights[name].copy_(value)
                    if self.optimized:
                        self.weights["encoder_ffn_packed_w"] = torch.stack(
                            (
                                self.weights["encoder_ffn_gate_w"],
                                self.weights["encoder_ffn_up_w"],
                            ),
                            dim=-1,
                        ).reshape(18, 2048, 32768)
                        self.weights["decoder_attn_o_packed_w"] = (
                            self.weights["decoder_attn_o_w"]
                            .transpose(1, 2)
                            .contiguous()
                        )
                    for name in (
                        "observation_images_normalized",
                        "observation_state_normalized",
                        "diffusion_noise",
                    ):
                        self.buffers[name].zero_()
                    super().record_infer_graph()

            checkpoint = {
                "language_embeds": torch.randn(a.prompt_len, 2048, dtype=torch.bfloat16)
            }
            infer = InitializedPi0(
                checkpoint, num_views=a.views, chunk_size=a.chunk_size
            )

            execution["upstream_sm120_optimized"] = infer.optimized

            def fn():
                return infer.forward(images, state, noise)

            reference = fn().clone()
            torch.cuda.synchronize()
            validation["cross_backend_parity"] = "not_checked_independent_weights"
        for _ in range(a.warmup):
            out = fn()
        torch.cuda.synchronize()
        if tuple(out.shape) not in ((1, a.chunk_size, 32), (a.chunk_size, 32)):
            raise RuntimeError(f"Unexpected output shape {out.shape}")
        if not torch.isfinite(out).all().item():
            raise RuntimeError("Nonfinite model output")
        setup_s = time.perf_counter() - setup_start
        torch.cuda.reset_peak_memory_stats()
        wall, gpu = [], []
        for _ in range(a.iterations):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            wall.append((time.perf_counter_ns() - t0) / 1e6)
            gpu.append(start.elapsed_time(end))
        if not torch.isfinite(out).all().item():
            raise RuntimeError("Nonfinite timed output")

    def stats(values):
        return {
            "median_ms": statistics.median(values),
            "p10_ms": percentile(values, 0.1),
            "p90_ms": percentile(values, 0.9),
            "min_ms": min(values),
            "samples_ms": values,
        }

    revisions = {}
    for name in ("openpi", "realtime-vla"):
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party" / name), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    result = {
        "backend": a.backend,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "execution": execution,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "benchmarks").glob("*.py"))
        },
        "workload": {
            "model": "pi0",
            "batch": 1,
            "views": a.views,
            "image_size": [224, 224],
            "prompt_len": a.prompt_len,
            "chunk_size": a.chunk_size,
            "action_dim": 32,
            "flow_steps": 10,
            "seed": a.seed,
        },
        "weights": {
            "openpi": a.openpi_checkpoint or "random_initialized",
            "triton": a.triton_checkpoint or "random_initialized",
        },
        "scope": "GPU-resident normalized images/state/noise to action chunk; excludes tokenization, transfers, policy transforms",
        "precision": "upstream BF16; OpenPI FP32 norms/patch embedding/action heads; Triton BF16 with FP32 accumulations",
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "upstreams": revisions,
        },
        "warmup": a.warmup,
        "iterations": a.iterations,
        "setup_seconds": setup_s,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "validation": validation,
        "wall": stats(wall),
        "cuda_event": stats(gpu),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{a.backend}: wall {statistics.median(wall):.3f} ms; GPU {statistics.median(gpu):.3f} ms",
        flush=True,
    )


def main():
    a = parse()
    if a.worker:
        worker(a)
        return
    results = []
    for backend in BACKENDS if a.backend == "all" else (a.backend,):
        dest = a.output.with_name(a.output.stem + "." + backend + ".json")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--backend",
            backend,
            "--worker",
            "--output",
            str(dest),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        results.append(json.loads(dest.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
