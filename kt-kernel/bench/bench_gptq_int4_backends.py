#!/usr/bin/env python
"""Benchmark one GPTQ INT4 MoE backend with reproducible routing and shapes."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import select
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

KT_KERNEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KT_KERNEL_ROOT / "python"))
if os.environ.get("KT_KERNEL_EXT_DIR"):
    sys.path.insert(0, os.environ["KT_KERNEL_EXT_DIR"])

import kt_kernel_ext  # noqa: E402

BACKENDS = {
    "avx2": "AVX2GPTQInt4_MOE",
    "vnni": "AVXVNNI256GPTQInt4_MOE",
    "packed-vnni": "AVXVNNI256PackedGPTQInt4_MOE",
    "sycl": "SYCLGPTQInt4_MOE",
    "hybrid": "CPUiGPUGPTQInt4_MOE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--qlen", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--igpu-ratio", type=float, default=0.5)
    parser.add_argument("--dynamic", action="store_true", help="Enable the CPU-load-aware hybrid scheduler")
    parser.add_argument("--background-load", choices=("none", "compute", "memory"), default="none")
    parser.add_argument("--background-workers", type=int, default=0, help="Zero uses --threads")
    parser.add_argument("--background-cpus", help="Defaults to the CPUInfer cores, e.g. 0-7")
    parser.add_argument("--background-memory-mib", type=int, default=64)
    parser.add_argument("--background-warmup-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Append the result as one JSONL record")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.experts <= 0 or args.topk <= 0 or args.topk > args.experts:
        raise ValueError("expected 0 < topk <= experts")
    if args.hidden_size % 8 or args.intermediate_size % 8:
        raise ValueError("hidden and intermediate sizes must be divisible by 8")
    if args.hidden_size % args.group_size or args.intermediate_size % args.group_size:
        raise ValueError("hidden and intermediate sizes must be divisible by group size")
    if args.threads <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("threads and iterations must be positive; warmup must be non-negative")
    if any(qlen <= 0 for qlen in args.qlen):
        raise ValueError("qlen values must be positive")
    if not 0.0 <= args.igpu_ratio <= 1.0:
        raise ValueError("igpu-ratio must be between 0 and 1")
    if args.background_workers < 0 or args.background_memory_mib <= 0 or args.background_warmup_seconds < 0:
        raise ValueError("background load parameters must be non-negative and memory-mib must be positive")


def read_cpu_times(cpus: list[int]) -> dict[int, tuple[int, int]]:
    wanted = {f"cpu{cpu}": cpu for cpu in cpus}
    result: dict[int, tuple[int, int]] = {}
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields or fields[0] not in wanted:
                continue
            values = [int(value) for value in fields[1:]]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[wanted[fields[0]]] = (total, idle)
    except OSError:
        pass
    return result


def cpu_busy_fraction(before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]) -> float | None:
    total_delta = 0
    idle_delta = 0
    for cpu, (before_total, before_idle) in before.items():
        if cpu not in after:
            continue
        after_total, after_idle = after[cpu]
        total_delta += after_total - before_total
        idle_delta += after_idle - before_idle
    if total_delta <= 0:
        return None
    return 1.0 - idle_delta / total_delta


def read_pressure(resource: str) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    try:
        for line in Path(f"/proc/pressure/{resource}").read_text(encoding="utf-8").splitlines():
            category, *raw_fields = line.split()
            fields: dict[str, float | int] = {}
            for raw_field in raw_fields:
                key, value = raw_field.split("=", 1)
                fields[key] = int(value) if key == "total" else float(value)
            result[category] = fields
    except OSError:
        pass
    return result


@contextlib.contextmanager
def run_background_load(args: argparse.Namespace):
    if args.background_load == "none":
        yield {"kind": "none", "workers": 0, "cpus": []}
        return

    workers = args.background_workers or args.threads
    cpus = args.background_cpus or f"0-{args.threads - 1}"
    command = [
        sys.executable,
        str(KT_KERNEL_ROOT / "bench" / "cpu_background_load.py"),
        "--kind",
        args.background_load,
        "--workers",
        str(workers),
        "--cpus",
        cpus,
        "--memory-mib",
        str(args.background_memory_mib),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        readable, _, _ = select.select([process.stdout], [], [], 35.0)
        if not readable:
            raise RuntimeError("background load did not become ready within 35 seconds")
        line = process.stdout.readline()
        if not line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"background load exited before readiness: {stderr.strip()}")
        metadata = json.loads(line)
        time.sleep(args.background_warmup_seconds)
        yield metadata
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, 9)
            process.communicate(timeout=5.0)


def read_memory_kib() -> dict[str, int | None]:
    values: dict[str, int | None] = {"rss_kib": None, "pss_kib": None, "private_kib": None}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                values["rss_kib"] = int(line.split()[1])
                break
    except OSError:
        pass

    try:
        fields = {}
        for line in Path("/proc/self/smaps_rollup").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, raw_value = line.split(":", 1)
                parts = raw_value.split()
                if parts and parts[0].isdigit():
                    fields[key] = int(parts[0])
        values["pss_kib"] = fields.get("Pss")
        private_clean = fields.get("Private_Clean", 0)
        private_dirty = fields.get("Private_Dirty", 0)
        values["private_kib"] = private_clean + private_dirty
    except OSError:
        pass
    return values


def percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def git_metadata() -> dict[str, object]:
    repo_root = KT_KERNEL_ROOT.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty_files = subprocess.check_output(
            ["git", "status", "--short"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def make_source_weights(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def qweight(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32, generator=generator).contiguous()

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (torch.rand(shape, dtype=torch.float32, generator=generator) * 0.02 + 0.001).contiguous()

    gate_up_qweight_shape = (args.experts, args.hidden_size // 8, args.intermediate_size)
    gate_up_scale_shape = (args.experts, args.hidden_size // args.group_size, args.intermediate_size)
    down_qweight_shape = (args.experts, args.intermediate_size // 8, args.hidden_size)
    down_scale_shape = (args.experts, args.intermediate_size // args.group_size, args.hidden_size)
    return {
        "gate": qweight(gate_up_qweight_shape),
        "up": qweight(gate_up_qweight_shape),
        "down": qweight(down_qweight_shape),
        "gate_scale": scales(gate_up_scale_shape),
        "up_scale": scales(gate_up_scale_shape),
        "down_scale": scales(down_scale_shape),
    }


def make_cpu_infer(threads: int):
    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [0]
    worker_config.subpool_thread_count = [threads]
    return kt_kernel_ext.CPUInfer(worker_config)


def make_backend(args: argparse.Namespace, source: dict[str, torch.Tensor], cpu_infer):
    extension_name = BACKENDS[args.backend]
    if not hasattr(kt_kernel_ext.moe, extension_name):
        raise RuntimeError(f"current kt_kernel_ext does not provide {extension_name}")

    config = kt_kernel_ext.moe.MOEConfig(
        args.experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        0,
    )
    config.layer_idx = 0
    config.max_len = max(args.qlen)
    config.pool = cpu_infer.backend_
    config.gate_proj = source["gate"].data_ptr()
    config.up_proj = source["up"].data_ptr()
    config.down_proj = source["down"].data_ptr()
    config.gate_scale = source["gate_scale"].data_ptr()
    config.up_scale = source["up_scale"].data_ptr()
    config.down_scale = source["down_scale"].data_ptr()
    config.quant_config.bits = 4
    config.quant_config.group_size = args.group_size
    config.quant_config.zero_point = False
    config.cpu_igpu_igpu_ratio = args.igpu_ratio
    config.cpu_igpu_dynamic = args.dynamic

    backend = getattr(kt_kernel_ext.moe, extension_name)(config)
    physical_to_logical = torch.arange(args.experts, dtype=torch.int64).contiguous()
    before = read_memory_kib()
    start = time.perf_counter()
    cpu_infer.submit(backend.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    load_ms = (time.perf_counter() - start) * 1000.0
    after = read_memory_kib()
    return backend, physical_to_logical, load_ms, before, after


def make_workload(args: argparse.Namespace, qlen: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + qlen)

    if qlen == 1:
        expert_ids = torch.arange(args.topk, dtype=torch.int64).view(1, -1)
    else:
        expert_ids = torch.stack(
            [torch.randperm(args.experts, generator=generator)[: args.topk] for _ in range(qlen)]
        ).contiguous()
    weights = torch.rand((qlen, args.topk), dtype=torch.float32, generator=generator).contiguous()
    hidden_states = torch.randn((qlen, args.hidden_size), dtype=torch.bfloat16, generator=generator).contiguous()
    output = torch.empty((qlen, args.hidden_size), dtype=torch.bfloat16).contiguous()
    qlen_tensor = torch.tensor([qlen], dtype=torch.int32)
    return {
        "expert_ids": expert_ids,
        "weights": weights,
        "hidden_states": hidden_states,
        "output": output,
        "qlen": qlen_tensor,
    }


def run_once(cpu_infer, backend, workload: dict[str, torch.Tensor], topk: int) -> None:
    cpu_infer.submit(
        backend.forward_task(
            workload["qlen"].data_ptr(),
            topk,
            workload["expert_ids"].data_ptr(),
            workload["weights"].data_ptr(),
            workload["hidden_states"].data_ptr(),
            workload["output"].data_ptr(),
            False,
        )
    )
    cpu_infer.sync()


def benchmark_case(args: argparse.Namespace, cpu_infer, backend, qlen: int) -> dict[str, object]:
    workload = make_workload(args, qlen)
    for _ in range(args.warmup):
        run_once(cpu_infer, backend, workload, args.topk)

    samples_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        run_once(cpu_infer, backend, workload, args.topk)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    mean_ms = statistics.fmean(samples_ms)
    result = {
        "qlen": qlen,
        "iterations": args.iterations,
        "mean_ms": mean_ms,
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "p99_ms": percentile(samples_ms, 0.99),
        "tokens_per_second": qlen * 1000.0 / mean_ms,
        "output_checksum": float(workload["output"].float().sum().item()),
    }
    if hasattr(backend, "scheduler_igpu_ratio"):
        result["scheduler_final_igpu_ratio"] = backend.scheduler_igpu_ratio()
        result["scheduler_final_cpu_load"] = backend.scheduler_cpu_load()
        if hasattr(backend, "scheduler_debug"):
            result["scheduler_debug"] = backend.scheduler_debug(qlen == 1)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(1)

    source = make_source_weights(args)
    source_bytes = sum(tensor.numel() * tensor.element_size() for tensor in source.values())
    cpu_infer = make_cpu_infer(args.threads)
    backend, physical_to_logical, load_ms, memory_before, memory_after = make_backend(args, source, cpu_infer)

    observed_cpus = list(range(args.threads))
    with run_background_load(args) as background_load:
        cpu_times_before = read_cpu_times(observed_cpus)
        cpu_pressure_before = read_pressure("cpu")
        memory_pressure_before = read_pressure("memory")
        observation_start = time.perf_counter()
        cases = [benchmark_case(args, cpu_infer, backend, qlen) for qlen in args.qlen]
        observation_seconds = time.perf_counter() - observation_start
        cpu_times_after = read_cpu_times(observed_cpus)
        cpu_pressure_after = read_pressure("cpu")
        memory_pressure_after = read_pressure("memory")

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": args.backend,
        "extension": BACKENDS[args.backend],
        "shape": {
            "experts": args.experts,
            "topk": args.topk,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "group_size": args.group_size,
        },
        "threads": args.threads,
        "igpu_ratio": args.igpu_ratio,
        "dynamic": args.dynamic,
        "warmup": args.warmup,
        "seed": args.seed,
        "source_weight_bytes": source_bytes,
        "load_ms": load_ms,
        "memory_before_load": memory_before,
        "memory_after_load": memory_after,
        "memory_load_delta_kib": {
            key: (
                memory_after[key] - memory_before[key]
                if memory_after[key] is not None and memory_before[key] is not None
                else None
            )
            for key in memory_before
        },
        "cases": cases,
        "background_load": background_load,
        "load_observation": {
            "cpuinfer_cpus": observed_cpus,
            "cpu_busy_fraction": cpu_busy_fraction(cpu_times_before, cpu_times_after),
            "observation_seconds": observation_seconds,
            "cpu_pressure_some_fraction": (
                (cpu_pressure_after["some"]["total"] - cpu_pressure_before["some"]["total"])
                / (observation_seconds * 1_000_000.0)
                if observation_seconds > 0 and "some" in cpu_pressure_before and "some" in cpu_pressure_after
                else None
            ),
            "cpu_pressure_before": cpu_pressure_before,
            "cpu_pressure_after": cpu_pressure_after,
            "memory_pressure_before": memory_pressure_before,
            "memory_pressure_after": memory_pressure_after,
        },
        "scheduler": (
            {
                "final_igpu_ratio": backend.scheduler_igpu_ratio(),
                "final_cpu_load": backend.scheduler_cpu_load(),
            }
            if hasattr(backend, "scheduler_igpu_ratio")
            else None
        ),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
        },
        "git": git_metadata(),
    }

    # Keep source-owned objects alive until all native calls have completed.
    _ = physical_to_logical
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output_file:
            output_file.write(encoded + "\n")


if __name__ == "__main__":
    main()
