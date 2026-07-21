#!/usr/bin/env python3
"""End-to-end VNNI-only versus CPU-iGPU dynamic scheduling experiment.

The runner launches an SGLang server for each backend, applies reproducible
CPU-pinned background loads, sends streaming completion requests, and writes
request-level JSONL plus aggregate CSV files suitable for paper figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import select
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
KT_KERNEL_ROOT = REPO_ROOT / "kt-kernel"
BACKGROUND_LOAD_SCRIPT = KT_KERNEL_ROOT / "bench" / "cpu_background_load.py"
DEFAULT_MODEL = Path("/home/wy/Work/models/Qwen3.5-35B-A3B-GPTQ-Int4")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "artifacts" / "cpu-igpu-e2e"
DEFAULT_ONEAPI_SETVARS = Path("/opt/intel/oneapi/setvars.sh")


@dataclass(frozen=True)
class BackendSpec:
    name: str
    method: str
    requires_oneapi: bool


@dataclass(frozen=True)
class LoadSpec:
    kind: str
    workers: int

    @property
    def label(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}-{self.workers}"


@dataclass(frozen=True)
class WorkloadSpec:
    prompt_tokens: int
    output_tokens: int

    @property
    def label(self) -> str:
        return f"p{self.prompt_tokens}-o{self.output_tokens}"


BACKENDS = {
    "vnni-only": BackendSpec("vnni-only", "GPTQ_INT4", False),
    "vnni-sycl-dynamic": BackendSpec("vnni-sycl-dynamic", "CPU_IGPU_GPTQ_INT4", True),
}

METRICS = ("prefill_tps", "decode_tps", "ttft_ms", "topt_ms", "e2e_ms")
THROUGHPUT_METRICS = {"prefill_tps", "decode_tps"}


def discover_local_package_root() -> Path | None:
    candidates = sorted(
        (KT_KERNEL_ROOT / "build").glob("lib.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        package = candidate / "kt_kernel"
        if package.joinpath("__init__.py").is_file() and any(package.glob("kt_kernel_ext*.so")):
            return candidate
    return None


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid CPU range: {part}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    if not cpus or min(cpus) < 0:
        raise argparse.ArgumentTypeError("CPU list must contain non-negative CPU ids")
    return sorted(set(cpus))


def parse_loads(value: str) -> list[LoadSpec]:
    loads: list[LoadSpec] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item == "none":
            load = LoadSpec("none", 0)
        else:
            try:
                kind, worker_text = item.split(":", 1)
                workers = int(worker_text)
            except ValueError as error:
                raise argparse.ArgumentTypeError(f"invalid load specification: {item}") from error
            if kind not in {"compute", "memory"} or workers <= 0:
                raise argparse.ArgumentTypeError(f"invalid load specification: {item}")
            load = LoadSpec(kind, workers)
        if load not in loads:
            loads.append(load)
    if not loads:
        raise argparse.ArgumentTypeError("at least one load is required")
    return loads


def parse_workloads(value: str) -> list[WorkloadSpec]:
    workloads: list[WorkloadSpec] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower().replace("x", ":")
        if not item:
            continue
        try:
            prompt_text, output_text = item.split(":", 1)
            workload = WorkloadSpec(int(prompt_text), int(output_text))
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid workload specification: {item}") from error
        if workload.prompt_tokens <= 0 or workload.output_tokens <= 0:
            raise argparse.ArgumentTypeError("workload token counts must be positive")
        if workload not in workloads:
            workloads.append(workload)
    if not workloads:
        raise argparse.ArgumentTypeError("at least one workload is required")
    return workloads


def parse_backends(value: str) -> list[BackendSpec]:
    result: list[BackendSpec] = []
    for raw_name in value.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name not in BACKENDS:
            raise argparse.ArgumentTypeError(f"unknown backend: {name}")
        if BACKENDS[name] not in result:
            result.append(BACKENDS[name])
    if not result:
        raise argparse.ArgumentTypeError("at least one backend is required")
    return result


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_server_args(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--server-arg must be KEY=VALUE, got: {value}")
        key, raw_value = value.split("=", 1)
        key = key.strip().lstrip("-")
        if not key:
            raise ValueError("--server-arg key must not be empty")
        result[key] = parse_scalar(raw_value.strip())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weight-path", type=Path, help="Defaults to --model")
    parser.add_argument("--served-name", help="Defaults to the model directory name")
    parser.add_argument("--python-path", default=sys.executable)
    parser.add_argument("--oneapi-setvars", type=Path, default=DEFAULT_ONEAPI_SETVARS)
    parser.add_argument(
        "--kt-kernel-package-root",
        type=Path,
        default=discover_local_package_root(),
        help="Directory containing the built kt_kernel package (defaults to kt-kernel/build/lib.*)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backends", type=parse_backends, default=parse_backends("vnni-only,vnni-sycl-dynamic"))
    parser.add_argument("--loads", type=parse_loads, default=parse_loads("none,compute:4,compute:8,memory:1,memory:4"))
    parser.add_argument(
        "--workloads", type=parse_workloads, default=parse_workloads("1:128,1024:300,4096:300,8192:300")
    )
    parser.add_argument("--load-affinity", choices=("pinned", "free"), default="pinned")
    parser.add_argument(
        "--load-nice",
        type=int,
        default=0,
        help="Background worker nice value (-20 is highest priority; negative values need permission)",
    )
    parser.add_argument(
        "--load-cpus",
        type=parse_cpu_list,
        default=parse_cpu_list("0-7"),
        help="Pinned load CPUs; in free mode this is the target CPU subset reported separately",
    )
    parser.add_argument("--cpuinfer-threads", type=int, default=8)
    parser.add_argument("--threadpool-count", type=int, default=1)
    parser.add_argument("--num-gpu-experts", type=int, default=24)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--request-repetitions", type=int, default=3)
    parser.add_argument("--server-repetitions", type=int, default=1)
    parser.add_argument("--scenario-warmups", type=int, default=15)
    parser.add_argument("--warmup-prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--load-warmup-seconds", type=float, default=2.0)
    parser.add_argument("--scenario-cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--memory-mib-per-buffer", type=int, default=64)
    parser.add_argument("--oneapi-device-selector", default="level_zero:gpu")
    parser.add_argument("--show-server-log", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override or add an SGLang launch argument; may be repeated",
    )
    args = parser.parse_args()
    args.weight_path = args.weight_path or args.model
    args.served_name = args.served_name or args.model.name
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_RESULTS_ROOT / timestamp
    positive_values = {
        "cpuinfer-threads": args.cpuinfer_threads,
        "threadpool-count": args.threadpool_count,
        "request-repetitions": args.request_repetitions,
        "server-repetitions": args.server_repetitions,
        "bootstrap-samples": args.bootstrap_samples,
        "memory-mib-per-buffer": args.memory_mib_per_buffer,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        parser.error(f"these values must be positive: {', '.join(invalid)}")
    if args.scenario_warmups < 0 or args.load_warmup_seconds < 0 or args.scenario_cooldown_seconds < 0:
        parser.error("warmup counts and durations must be non-negative")
    if not -20 <= args.load_nice <= 19:
        parser.error("--load-nice must be between -20 and 19")
    return args


def stable_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty_files = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def command_output(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout + result.stderr).strip()
        return output if output else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def hardware_metadata() -> dict[str, Any]:
    memory_total_kib = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total_kib = int(line.split()[1])
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_kib": memory_total_kib,
        "lscpu_json": command_output(["lscpu", "--json"]),
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "sycl_ls": command_output(["sycl-ls"]),
    }


_ONEAPI_ENV_CACHE: dict[str, str] | None = None


def oneapi_environment(setvars_path: Path) -> dict[str, str]:
    global _ONEAPI_ENV_CACHE
    if _ONEAPI_ENV_CACHE is not None:
        return dict(_ONEAPI_ENV_CACHE)
    if not setvars_path.is_file():
        raise RuntimeError(f"oneAPI setvars script does not exist: {setvars_path}")
    result = subprocess.run(
        ["bash", "-c", 'set +u; source "$1" --force >/dev/null && env -0', "bash", str(setvars_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"failed to load oneAPI environment: {error}")
    environment: dict[str, str] = {}
    for item in result.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            environment[os.fsdecode(key)] = os.fsdecode(value)
    _ONEAPI_ENV_CACHE = environment
    return dict(environment)


def common_server_arguments(args: argparse.Namespace) -> dict[str, Any]:
    server_args: dict[str, Any] = {
        "model": str(args.model),
        "served-model-name": args.served_name,
        "kt-weight-path": str(args.weight_path),
        "kt-cpuinfer": args.cpuinfer_threads,
        "kt-threadpool-count": args.threadpool_count,
        "kt-num-gpu-experts": args.num_gpu_experts,
        "attention-backend": "triton",
        "trust-remote-code": True,
        "mem-fraction-static": 0.85,
        "chunked-prefill-size": args.chunked_prefill_size,
        "max-running-requests": 1,
        "max-total-tokens": 32000,
        "enable-mixed-chunk": True,
        "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
        "disable-radix-cache": True,
        "disable-cuda-graph": True,
        "dtype": "float16",
    }
    server_args.update(parse_server_args(args.server_arg))
    return server_args


def backend_server_arguments(args: argparse.Namespace, backend: BackendSpec) -> dict[str, Any]:
    server_args = common_server_arguments(args)
    server_args["kt-method"] = backend.method
    return server_args


def build_server_command(args: argparse.Namespace, backend: BackendSpec, port: int) -> list[str]:
    command = [args.python_path, "-m", "sglang.launch_server", "--host", "127.0.0.1", "--port", str(port)]
    for key, value in backend_server_arguments(args, backend).items():
        if value is None or value is False:
            continue
        command.append(f"--{key}")
        if value is not True:
            command.append(str(value))
    return command


def build_server_environment(args: argparse.Namespace, backend: BackendSpec) -> dict[str, str]:
    environment = oneapi_environment(args.oneapi_setvars)
    python_paths = []
    if args.kt_kernel_package_root is not None:
        python_paths.append(str(args.kt_kernel_package_root))
    python_paths.append(str(REPO_ROOT / "third_party" / "sglang" / "python"))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    environment.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    environment.setdefault("SGLANG_MAMBA_CONV_DTYPE", "float16")

    for key in (
        "KT_GPTQ_INT4_BACKEND",
        "KT_AVXVNNI_FUSED_MOE",
        "KT_CPU_IGPU_POLICY",
        "KT_CPU_IGPU_RATIO",
        "ONEAPI_DEVICE_SELECTOR",
    ):
        environment.pop(key, None)
    if backend.name == "vnni-only":
        environment["KT_GPTQ_INT4_BACKEND"] = "avxvnni"
        environment["KT_AVXVNNI_FUSED_MOE"] = "1"
    else:
        environment["KT_CPU_IGPU_POLICY"] = "dynamic"
        environment["KT_CPU_IGPU_RATIO"] = "0"
        environment["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector
    return environment


def validate_runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.kt_kernel_package_root
    if package_root is None or not (package_root / "kt_kernel" / "__init__.py").is_file():
        raise RuntimeError(
            "built kt_kernel package not found; run `cd kt-kernel && python setup.py build_py` "
            "after building the extension, or pass --kt-kernel-package-root"
        )
    environment = build_server_environment(args, BACKENDS["vnni-sycl-dynamic"])
    probe = """
import json
import kt_kernel

extension = kt_kernel.kt_kernel_ext
print(json.dumps({
    "package_file": kt_kernel.__file__,
    "extension_file": extension.__file__,
    "cpu_variant": kt_kernel.__cpu_variant__,
    "has_cpu_igpu_gptq_int4": hasattr(extension.moe, "CPUiGPUGPTQInt4_MOE"),
}))
"""
    result = subprocess.run(
        [args.python_path, "-c", probe],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"kt_kernel runtime preflight failed: {detail}")
    try:
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid kt_kernel runtime preflight output: {result.stdout!r}") from error
    if not metadata["has_cpu_igpu_gptq_int4"]:
        raise RuntimeError("built kt_kernel extension does not expose CPUiGPUGPTQInt4_MOE; rebuild with SYCL support")
    package_file = Path(metadata["package_file"]).resolve()
    try:
        package_file.relative_to(package_root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"runtime imported kt_kernel from {package_file}, outside {package_root.resolve()}"
        ) from error
    return metadata


def public_environment(environment: dict[str, str]) -> dict[str, str]:
    keys = (
        "KT_GPTQ_INT4_BACKEND",
        "KT_AVXVNNI_FUSED_MOE",
        "KT_CPU_IGPU_POLICY",
        "KT_CPU_IGPU_RATIO",
        "ONEAPI_DEVICE_SELECTOR",
    )
    return {key: environment[key] for key in keys if key in environment}


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


def pressure_delta(
    before: dict[str, dict[str, float | int]],
    after: dict[str, dict[str, float | int]],
    elapsed_seconds: float,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for category in ("some", "full"):
        try:
            delta_us = int(after[category]["total"]) - int(before[category]["total"])
            result[f"{category}_fraction"] = delta_us / (elapsed_seconds * 1_000_000.0)
        except (KeyError, TypeError, ZeroDivisionError):
            result[f"{category}_fraction"] = None
    return result


def read_cpu_times(cpus: list[int]) -> dict[int, tuple[int, int]]:
    names = {f"cpu{cpu}": cpu for cpu in cpus}
    result: dict[int, tuple[int, int]] = {}
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields or fields[0] not in names:
                continue
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[names[fields[0]]] = (sum(values), idle)
    except OSError:
        pass
    return result


def load_observation_cpus(args: argparse.Namespace) -> list[int]:
    if args.load_affinity == "pinned":
        return list(args.load_cpus)
    return sorted(os.sched_getaffinity(0))


def cpu_busy_fraction(before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]) -> float | None:
    total_delta = 0
    idle_delta = 0
    for cpu, (before_total, before_idle) in before.items():
        if cpu not in after:
            continue
        after_total, after_idle = after[cpu]
        total_delta += after_total - before_total
        idle_delta += after_idle - before_idle
    return None if total_delta <= 0 else 1.0 - idle_delta / total_delta


def cpu_busy_by_cpu(before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]) -> dict[str, float]:
    result = {}
    for cpu, (before_total, before_idle) in before.items():
        if cpu not in after:
            continue
        after_total, after_idle = after[cpu]
        total_delta = after_total - before_total
        idle_delta = after_idle - before_idle
        if total_delta > 0:
            result[str(cpu)] = 1.0 - idle_delta / total_delta
    return result


def process_group_memory_kib(process_group: int) -> tuple[int, int]:
    rss_kib = 0
    pss_kib = 0
    for process_path in Path("/proc").iterdir():
        if not process_path.name.isdigit():
            continue
        try:
            stat_line = (process_path / "stat").read_text(encoding="utf-8")
            command_end = stat_line.rfind(")")
            if command_end < 0:
                continue
            fields = stat_line[command_end + 2 :].split()
            if len(fields) < 3 or int(fields[2]) != process_group:
                continue
            for line in (process_path / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
                    break
            for line in (process_path / "smaps_rollup").read_text(encoding="utf-8").splitlines():
                if line.startswith("Pss:"):
                    pss_kib += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return rss_kib, pss_kib


class ProcessMemoryMonitor:
    def __init__(self, process_group: int, interval_seconds: float = 1.0):
        self.process_group = process_group
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._scope = "server"
        self._peaks: dict[str, dict[str, int]] = {}
        self._thread = threading.Thread(target=self._run, name="e2e-memory-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_scope(self, scope: str) -> None:
        with self._lock:
            self._scope = scope
            self._peaks.setdefault(scope, {"rss_kib": 0, "pss_kib": 0})

    def peak(self, scope: str) -> dict[str, int]:
        with self._lock:
            return dict(self._peaks.get(scope, {"rss_kib": 0, "pss_kib": 0}))

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss_kib, pss_kib = process_group_memory_kib(self.process_group)
            with self._lock:
                for scope in {"server", self._scope}:
                    peak = self._peaks.setdefault(scope, {"rss_kib": 0, "pss_kib": 0})
                    peak["rss_kib"] = max(peak["rss_kib"], rss_kib)
                    peak["pss_kib"] = max(peak["pss_kib"], pss_kib)
            self._stop.wait(self.interval_seconds)


class ServerSession:
    def __init__(
        self,
        command: list[str],
        environment: dict[str, str],
        log_path: Path,
        port: int,
        startup_timeout: float,
        show_log: bool,
    ):
        self.command = command
        self.environment = environment
        self.log_path = log_path
        self.port = port
        self.startup_timeout = startup_timeout
        self.show_log = show_log
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file = None
        self.tee_thread: threading.Thread | None = None
        self.memory_monitor: ProcessMemoryMonitor | None = None
        self.final_memory_peak_kib = {"rss_kib": 0, "pss_kib": 0}

    def __enter__(self) -> "ServerSession":
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_path.open("w", encoding="utf-8")
            self.log_file.write("# Command: " + " ".join(self.command) + "\n")
            self.log_file.flush()
            self.process = subprocess.Popen(
                self.command,
                cwd=REPO_ROOT,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            assert self.process.stdout is not None
            self.tee_thread = threading.Thread(target=self._tee_output, args=(self.process.stdout,), daemon=True)
            self.tee_thread.start()
            memory_monitor = ProcessMemoryMonitor(self.process.pid)
            memory_monitor.start()
            self.memory_monitor = memory_monitor
            self._wait_until_ready()
            return self
        except BaseException:
            self._cleanup(suppress_errors=True)
            raise

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._cleanup(suppress_errors=_exc_type is not None)

    def _cleanup(self, suppress_errors: bool) -> None:
        cleanup_errors: list[BaseException] = []

        if self.process is not None:
            try:
                terminate_process_group(self.process, timeout=20.0)
            except BaseException as error:
                cleanup_errors.append(error)
            self.process = None
        if self.memory_monitor is not None:
            try:
                self.memory_monitor.stop()
                self.final_memory_peak_kib = self.memory_monitor.peak("server")
            except BaseException as error:
                cleanup_errors.append(error)
            self.memory_monitor = None
        if self.tee_thread is not None:
            try:
                self.tee_thread.join(timeout=5.0)
            except BaseException as error:
                cleanup_errors.append(error)
            self.tee_thread = None
        if self.log_file is not None:
            try:
                self.log_file.close()
            except BaseException as error:
                cleanup_errors.append(error)
            self.log_file = None

        if cleanup_errors and not suppress_errors:
            raise cleanup_errors[0]

    def validate_backend(self, backend: BackendSpec) -> None:
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        marker = (
            "KT_SELECTED_MOE_BACKEND=CPU_IGPU_GPTQ_INT4"
            if backend.requires_oneapi
            else "KT_SELECTED_MOE_BACKEND=GPTQ_INT4:AVXVNNI256GPTQInt4_MOE"
        )
        if marker not in text:
            raise RuntimeError(f"server log does not contain expected backend marker: {marker}")

    def _tee_output(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                if self.log_file is not None:
                    self.log_file.write(line)
                    self.log_file.flush()
                if self.show_log:
                    print(f"[server] {line}", end="", flush=True)
        finally:
            pipe.close()

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout
        health_url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup with code {self.process.returncode}")
            try:
                if requests.get(health_url, timeout=5.0).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"server did not become healthy within {self.startup_timeout:.0f} seconds")


def build_background_load_command(args: argparse.Namespace, load: LoadSpec, duration: float | None = None) -> list[str]:
    command = [
        sys.executable,
        str(BACKGROUND_LOAD_SCRIPT),
        "--kind",
        load.kind,
        "--workers",
        str(load.workers),
        "--affinity",
        args.load_affinity,
        "--nice",
        str(args.load_nice),
        "--memory-mib",
        str(args.memory_mib_per_buffer),
    ]
    if args.load_affinity == "pinned":
        command.extend(["--cpus", ",".join(str(cpu) for cpu in args.load_cpus)])
    if duration is not None:
        command.extend(["--duration", str(duration)])
    return command


def preflight_background_load(args: argparse.Namespace) -> dict[str, Any] | None:
    if not any(load.kind != "none" for load in args.loads):
        return None
    command = build_background_load_command(args, LoadSpec("compute", 1), duration=0.2)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"background load preflight failed: {detail}")
    try:
        metadata = json.loads(result.stdout.strip().splitlines()[0])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid background load preflight output: {result.stdout!r}") from error
    if metadata.get("status") != "ready":
        raise RuntimeError(f"background load preflight was not ready: {metadata}")
    if metadata.get("effective_nice_values") != [args.load_nice]:
        raise RuntimeError(
            "background load preflight did not apply the requested nice value: "
            f"{metadata.get('effective_nice_values')}"
        )
    return metadata


class BackgroundLoad:
    def __init__(self, args: argparse.Namespace, load: LoadSpec):
        self.args = args
        self.load = load
        self.process: subprocess.Popen[str] | None = None
        self.metadata: dict[str, Any] = {
            "kind": "none",
            "workers": 0,
            "affinity": args.load_affinity,
            "requested_nice": args.load_nice,
            "effective_nice_values": [],
            "cpus": load_observation_cpus(args),
        }

    def __enter__(self) -> dict[str, Any]:
        if self.load.kind == "none":
            return self.metadata
        command = build_background_load_command(self.args, self.load)
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            assert self.process.stdout is not None
            readable, _, _ = select.select([self.process.stdout], [], [], 40.0)
            if not readable:
                raise TimeoutError("background load did not become ready within 40 seconds")
            line = self.process.stdout.readline()
            if not line:
                return_code = self.process.poll()
                raise RuntimeError(f"background load exited before readiness (return code {return_code})")
            self.metadata = json.loads(line)
            if self.metadata.get("status") != "ready":
                raise RuntimeError(f"background load was not ready: {self.metadata}")
            if self.metadata.get("requested_nice") != self.args.load_nice:
                raise RuntimeError(
                    "background load reported an unexpected requested nice value: "
                    f"{self.metadata.get('requested_nice')}"
                )
            if self.metadata.get("effective_nice_values") != [self.args.load_nice]:
                raise RuntimeError(
                    "background load did not apply the requested nice value: "
                    f"{self.metadata.get('effective_nice_values')}"
                )
            return self.metadata
        except BaseException:
            self._cleanup(suppress_errors=True)
            raise

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._cleanup(suppress_errors=_exc_type is not None)

    def _cleanup(self, suppress_errors: bool) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        try:
            terminate_process_group(process, timeout=10.0)
        except BaseException:
            if not suppress_errors:
                raise

    def check(self) -> None:
        if self.process is not None:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"background load exited unexpectedly with code {return_code}")


def terminate_process_group(process: subprocess.Popen, timeout: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5.0)


def generate_prompt(approximate_tokens: int, nonce: str) -> str:
    if approximate_tokens <= 2:
        return f"hi {nonce}"
    prefix = f"Experiment {nonce}. Repeat the following word sequence exactly. "
    return prefix + "word " * max(1, approximate_tokens - 10)


def sample_nonce(server_repetition: int, load: str, workload: str, repetition: int) -> str:
    return f"sample-{server_repetition}-{load}-{workload}-{repetition}"


def extract_usage(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    candidates = [chunk.get("usage"), chunk.get("meta_info")]
    choices = chunk.get("choices") or []
    if choices and isinstance(choices[0], dict):
        candidates.extend([choices[0].get("usage"), choices[0].get("meta_info")])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        prompt_tokens = candidate.get("prompt_tokens") or candidate.get("input_tokens")
        completion_tokens = candidate.get("completion_tokens") or candidate.get("output_tokens")
        if prompt_tokens is not None or completion_tokens is not None:
            return (
                int(prompt_tokens) if prompt_tokens is not None else None,
                int(completion_tokens) if completion_tokens is not None else None,
            )
    return None, None


def run_streaming_request(
    port: int,
    served_name: str,
    prompt: str,
    requested_prompt_tokens: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": served_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first_token_time = None
    last_token_time = None
    chunk_count = 0
    output_parts: list[str] = []
    prompt_tokens = None
    completion_tokens = None
    start = time.perf_counter()
    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_text = line[6:].strip()
            if data_text == "[DONE]":
                break
            try:
                chunk = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            usage_prompt, usage_completion = extract_usage(chunk)
            prompt_tokens = usage_prompt if usage_prompt is not None else prompt_tokens
            completion_tokens = usage_completion if usage_completion is not None else completion_tokens
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text", "")
            if not text:
                continue
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            chunk_count += 1
            output_parts.append(text)
    end = time.perf_counter()
    if first_token_time is None:
        raise RuntimeError("stream completed without a generated token")

    actual_prompt_tokens = prompt_tokens or requested_prompt_tokens
    actual_completion_tokens = completion_tokens or chunk_count
    usage_source = "server" if prompt_tokens is not None and completion_tokens is not None else "estimated"
    decode_intervals = max(actual_completion_tokens - 1, 0)
    decode_seconds = 0.0
    if decode_intervals > 0:
        decode_seconds = max((last_token_time or end) - first_token_time, 0.0)
        if decode_seconds == 0.0:
            decode_seconds = max(end - first_token_time, 0.0)
    ttft_seconds = first_token_time - start
    e2e_seconds = end - start
    output_text = "".join(output_parts)
    return {
        "requested_prompt_tokens": requested_prompt_tokens,
        "requested_output_tokens": max_tokens,
        "prompt_tokens": actual_prompt_tokens,
        "completion_tokens": actual_completion_tokens,
        "usage_source": usage_source,
        "stream_chunk_count": chunk_count,
        "ttft_ms": ttft_seconds * 1000.0,
        "e2e_ms": e2e_seconds * 1000.0,
        "decode_ms": decode_seconds * 1000.0 if decode_intervals > 0 else None,
        "prefill_tps": actual_prompt_tokens / ttft_seconds if ttft_seconds > 0 else None,
        "decode_tps": decode_intervals / decode_seconds if decode_seconds > 0 else None,
        "topt_ms": decode_seconds * 1000.0 / decode_intervals if decode_intervals > 0 else None,
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "output_characters": len(output_text),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    random_generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(random_generator.choice(values) for _ in values))
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_samples(samples: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for sample in samples:
        if sample.get("status") != "ok":
            continue
        key = (
            sample["backend"],
            sample.get("load_affinity", "pinned"),
            int(sample.get("load_nice", 0)),
            sample["load"],
            sample["workload"],
        )
        groups.setdefault(key, []).append(sample)

    rows: list[dict[str, Any]] = []
    for (backend, load_affinity, load_nice, load, workload), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "backend": backend,
            "load_affinity": load_affinity,
            "load_nice": load_nice,
            "load": load,
            "workload": workload,
            "n": len(group),
        }
        for metric in METRICS:
            values = [float(sample[metric]) for sample in group if sample.get(metric) is not None]
            if not values:
                continue
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                bootstrap_samples,
                stable_seed(
                    seed,
                    backend,
                    load_affinity,
                    load_nice,
                    load,
                    workload,
                    metric,
                ),
            )
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_p50"] = percentile(values, 0.50)
            row[f"{metric}_p95"] = percentile(values, 0.95)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return rows


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return comparison_rows_with_samples(summary, [], bootstrap_samples=0, seed=0)


def paired_successful_samples(
    samples: list[dict[str, Any]],
) -> dict[tuple[str, int, str, str], list[tuple[dict[str, Any], dict[str, Any]]]]:
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for sample in samples:
        backend = sample.get("backend")
        if sample.get("status") != "ok" or backend not in BACKENDS:
            continue
        identity = (
            sample.get("server_repetition"),
            sample.get("load_affinity", "pinned"),
            int(sample.get("load_nice", 0)),
            sample.get("load"),
            sample.get("workload"),
            sample.get("repetition"),
        )
        indexed.setdefault(identity, {})[backend] = sample

    result: dict[tuple[str, int, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for identity, backends in indexed.items():
        vnni = backends.get("vnni-only")
        dynamic = backends.get("vnni-sycl-dynamic")
        if vnni is None or dynamic is None:
            continue
        condition = (
            str(identity[1]),
            int(identity[2]),
            str(identity[3]),
            str(identity[4]),
        )
        result.setdefault(condition, []).append((vnni, dynamic))
    return result


def speedup_from_values(vnni_values: list[float], dynamic_values: list[float], throughput: bool) -> float:
    vnni_mean = statistics.fmean(vnni_values)
    dynamic_mean = statistics.fmean(dynamic_values)
    return dynamic_mean / vnni_mean if throughput else vnni_mean / dynamic_mean


def bootstrap_speedup_ci(
    paired_values: list[tuple[float, float]],
    throughput: bool,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    vnni_values = [pair[0] for pair in paired_values]
    dynamic_values = [pair[1] for pair in paired_values]
    point = speedup_from_values(vnni_values, dynamic_values, throughput)
    if len(paired_values) == 1 or samples <= 0:
        return point, point

    generator = random.Random(seed)
    speedups = []
    for _ in range(samples):
        resampled = [generator.choice(paired_values) for _ in paired_values]
        speedups.append(
            speedup_from_values(
                [pair[0] for pair in resampled],
                [pair[1] for pair in resampled],
                throughput,
            )
        )
    return percentile(speedups, 0.025), percentile(speedups, 0.975)


def comparison_rows_with_samples(
    summary: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["backend"],
            row.get("load_affinity", "pinned"),
            int(row.get("load_nice", 0)),
            row["load"],
            row["workload"],
        ): row
        for row in summary
    }
    conditions = sorted(
        {
            (
                row.get("load_affinity", "pinned"),
                int(row.get("load_nice", 0)),
                row["load"],
                row["workload"],
            )
            for row in summary
        }
    )
    paired_by_condition = paired_successful_samples(samples)
    rows = []
    for load_affinity, load_nice, load, workload in conditions:
        vnni = indexed.get(("vnni-only", load_affinity, load_nice, load, workload))
        dynamic = indexed.get(("vnni-sycl-dynamic", load_affinity, load_nice, load, workload))
        if vnni is None or dynamic is None:
            continue
        pairs = paired_by_condition.get((load_affinity, load_nice, load, workload), [])
        row: dict[str, Any] = {
            "load_affinity": load_affinity,
            "load_nice": load_nice,
            "load": load,
            "workload": workload,
            "vnni_n": vnni["n"],
            "dynamic_n": dynamic["n"],
            "paired_n": len(pairs),
        }
        for metric in METRICS:
            vnni_value = vnni.get(f"{metric}_mean")
            dynamic_value = dynamic.get(f"{metric}_mean")
            if vnni_value is None or dynamic_value is None or vnni_value == 0 or dynamic_value == 0:
                continue
            row[f"vnni_{metric}"] = vnni_value
            row[f"dynamic_{metric}"] = dynamic_value
            row[f"{metric}_speedup"] = (
                dynamic_value / vnni_value if metric in THROUGHPUT_METRICS else vnni_value / dynamic_value
            )
            paired_values = [
                (float(pair[0][metric]), float(pair[1][metric]))
                for pair in pairs
                if pair[0].get(metric) is not None
                and pair[1].get(metric) is not None
                and float(pair[0][metric]) > 0
                and float(pair[1][metric]) > 0
            ]
            if paired_values:
                paired_vnni = [pair[0] for pair in paired_values]
                paired_dynamic = [pair[1] for pair in paired_values]
                row[f"{metric}_paired_n"] = len(paired_values)
                row[f"vnni_{metric}"] = statistics.fmean(paired_vnni)
                row[f"dynamic_{metric}"] = statistics.fmean(paired_dynamic)
                row[f"{metric}_speedup"] = speedup_from_values(
                    paired_vnni, paired_dynamic, metric in THROUGHPUT_METRICS
                )
                ci_low, ci_high = bootstrap_speedup_ci(
                    paired_values,
                    metric in THROUGHPUT_METRICS,
                    bootstrap_samples,
                    stable_seed(
                        seed,
                        load_affinity,
                        load_nice,
                        load,
                        workload,
                        metric,
                        "speedup",
                    ),
                )
                row[f"{metric}_speedup_ci95_low"] = ci_low
                row[f"{metric}_speedup_ci95_high"] = ci_high

        output_pairs = [
            pair
            for pair in pairs
            if pair[0].get("output_sha256") is not None and pair[1].get("output_sha256") is not None
        ]
        output_matches = sum(pair[0]["output_sha256"] == pair[1]["output_sha256"] for pair in output_pairs)
        row["output_pairs"] = len(output_pairs)
        row["output_match_count"] = output_matches
        row["output_mismatch_count"] = len(output_pairs) - output_matches
        row["output_match_rate"] = output_matches / len(output_pairs) if output_pairs else None
        rows.append(row)
    return rows


def format_result(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def format_mean_ci(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    if mean is None:
        return "NA"
    low = row.get(f"{metric}_ci95_low")
    high = row.get(f"{metric}_ci95_high")
    if low is None or high is None:
        return format_result(mean)
    return f"{format_result(mean)} [{format_result(low)}, {format_result(high)}]"


def format_speedup(row: dict[str, Any], metric: str) -> str:
    point = row.get(f"{metric}_speedup")
    if point is None:
        return "NA"
    low = row.get(f"{metric}_speedup_ci95_low")
    high = row.get(f"{metric}_speedup_ci95_high")
    if low is None or high is None:
        return f"{format_result(point)}x"
    return f"{format_result(point)}x [{format_result(low)}, {format_result(high)}]"


def write_markdown_report(
    path: Path,
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    indexed = {
        (
            row["backend"],
            row.get("load_affinity", "pinned"),
            int(row.get("load_nice", 0)),
            row["load"],
            row["workload"],
        ): row
        for row in summary
    }
    lines = [
        "# CPU-iGPU MoE End-to-End Experiment",
        "",
        "This report is generated from the raw request samples. Values are arithmetic means; "
        "brackets contain bootstrap 95% confidence intervals.",
        "",
        f"- Status: {manifest.get('status', 'unknown')}",
        f"- Started: {manifest.get('started_at', 'unknown')}",
        f"- Finished: {manifest.get('finished_at', 'unknown')}",
        f"- Successful samples: {manifest.get('successful_sample_count', 0)} / " f"{manifest.get('sample_count', 0)}",
        "",
        "## Result Files",
        "",
        "- [Manifest](manifest.json)",
        "- [Request samples](samples.jsonl)",
        "- [Backend summaries](summary.csv)",
        "- [Paired comparisons](comparisons.csv)",
        "",
        "## Performance",
        "",
        "Speedup is dynamic/VNNI for throughput and VNNI/dynamic for latency, so values above "
        "1.0 always favor dynamic scheduling. Speedup intervals use paired bootstrap resampling.",
        "",
        "| Affinity | Nice | Load | Workload | Metric | VNNI mean [95% CI] | Dynamic mean [95% CI] | Speedup [95% CI] |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for comparison in comparisons:
        load_affinity = comparison.get("load_affinity", "pinned")
        load_nice = int(comparison.get("load_nice", 0))
        load = comparison["load"]
        workload = comparison["workload"]
        vnni = indexed[("vnni-only", load_affinity, load_nice, load, workload)]
        dynamic = indexed[("vnni-sycl-dynamic", load_affinity, load_nice, load, workload)]
        for metric in METRICS:
            if comparison.get(f"{metric}_speedup") is None:
                continue
            lines.append(
                f"| {load_affinity} | {load_nice} | {load} | {workload} | {metric} | "
                f"{format_mean_ci(vnni, metric)} | "
                f"{format_mean_ci(dynamic, metric)} | {format_speedup(comparison, metric)} |"
            )

    lines.extend(
        [
            "",
            "## Output Agreement",
            "",
            "Output agreement is exact SHA-256 equality for byte-identical prompts and deterministic "
            "decoding parameters.",
            "",
            "| Affinity | Nice | Load | Workload | Matched | Paired outputs | Match rate |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for comparison in comparisons:
        match_rate = comparison.get("output_match_rate")
        lines.append(
            f"| {comparison.get('load_affinity', 'pinned')} | "
            f"{comparison.get('load_nice', 0)} | {comparison['load']} | "
            f"{comparison['workload']} | "
            f"{comparison.get('output_match_count', 0)} | {comparison.get('output_pairs', 0)} | "
            f"{format_result(match_rate, digits=4)} |"
        )
    if not comparisons:
        lines.append("| NA | 0 | NA | NA | 0 | 0 | NA |")

    lines.extend(
        [
            "",
            "## Process-Group Memory",
            "",
            "Peak values cover the full server lifetime, including model loading and weight packing.",
            "",
            "| Server run | Backend | Peak RSS (GiB) | Peak PSS (GiB) | Status |",
            "|---|---|---:|---:|---|",
        ]
    )
    for server_run in manifest.get("server_runs", []):
        peak = server_run.get("server_memory_peak_kib", {})
        rss_gib = peak.get("rss_kib", 0) / (1024.0 * 1024.0)
        pss_gib = peak.get("pss_kib", 0) / (1024.0 * 1024.0)
        lines.append(
            f"| {server_run.get('name', 'unknown')} | {server_run.get('backend', 'unknown')} | "
            f"{format_result(rss_gib)} | {format_result(pss_gib)} | "
            f"{server_run.get('status', 'unknown')} |"
        )
    if not manifest.get("server_runs"):
        lines.append("| NA | NA | 0.000 | 0.000 | unknown |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        if not fieldnames:
            return
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (BackendSpec, LoadSpec, WorkloadSpec)):
        return asdict(value)
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    return value


def make_request_schedule(workloads: list[WorkloadSpec], repetitions: int, seed: int) -> list[tuple[WorkloadSpec, int]]:
    schedule = [(workload, repetition) for workload in workloads for repetition in range(repetitions)]
    random.Random(seed).shuffle(schedule)
    return schedule


def print_plan(args: argparse.Namespace) -> None:
    print("End-to-end experiment plan")
    print(f"  Model: {args.model}")
    print(f"  Weight path: {args.weight_path}")
    print(f"  kt_kernel package root: {args.kt_kernel_package_root}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Backends: {', '.join(backend.name for backend in args.backends)}")
    observed_cpus = load_observation_cpus(args)
    placement = (
        f"pinned to CPUs {args.load_cpus}"
        if args.load_affinity == "pinned"
        else f"freely scheduled over CPUs {observed_cpus}"
    )
    print(f"  Loads: {', '.join(load.label for load in args.loads)}; {placement}")
    print(f"  Background priority: nice={args.load_nice}")
    print(f"  Workloads: {', '.join(workload.label for workload in args.workloads)}")
    measured = len(args.backends) * len(args.loads) * len(args.workloads) * args.request_repetitions
    measured *= args.server_repetitions
    warmups = len(args.backends) * len(args.loads) * args.scenario_warmups * args.server_repetitions
    print(f"  Measured requests: {measured}; scenario warmups: {warmups}")
    for backend in args.backends:
        command = build_server_command(args, backend, args.port)
        if args.dry_run:
            environment = (
                {
                    "KT_GPTQ_INT4_BACKEND": "avxvnni",
                    "KT_AVXVNNI_FUSED_MOE": "1",
                }
                if backend.name == "vnni-only"
                else {
                    "KT_CPU_IGPU_POLICY": "dynamic",
                    "KT_CPU_IGPU_RATIO": "0",
                    "ONEAPI_DEVICE_SELECTOR": args.oneapi_device_selector,
                }
            )
        else:
            environment = build_server_environment(args, backend)
        print(f"  [{backend.name}] env={public_environment(environment)}")
        print("    " + " ".join(command))


def run_scenario_requests(
    args: argparse.Namespace,
    backend: BackendSpec,
    server: ServerSession,
    server_repetition: int,
    port: int,
    load: LoadSpec,
    scenario_name: str,
    sample_file: Any,
    samples: list[dict[str, Any]],
) -> int:
    assert server.memory_monitor is not None
    request_error_count = 0
    observed_cpus = load_observation_cpus(args)
    target_cpus = list(args.load_cpus)
    load_controller = BackgroundLoad(args, load)
    with load_controller as load_metadata:
        if args.load_warmup_seconds:
            time.sleep(args.load_warmup_seconds)
        for warmup_index in range(args.scenario_warmups):
            load_controller.check()
            nonce = f"warmup-{server_repetition}-{backend.name}-{args.load_affinity}-" f"{load.label}-{warmup_index}"
            run_streaming_request(
                port,
                args.served_name,
                generate_prompt(args.warmup_prompt_tokens, nonce),
                args.warmup_prompt_tokens,
                args.warmup_output_tokens,
                args.request_timeout,
            )

        schedule = make_request_schedule(
            args.workloads,
            args.request_repetitions,
            stable_seed(args.seed, server_repetition, args.load_affinity, load.label),
        )
        for request_index, (workload, repetition) in enumerate(schedule):
            load_controller.check()
            nonce = sample_nonce(server_repetition, load.label, workload.label, repetition)
            prompt = generate_prompt(workload.prompt_tokens, nonce)
            cpu_before = read_cpu_times(observed_cpus)
            target_cpu_before = read_cpu_times(target_cpus)
            cpu_pressure_before = read_pressure("cpu")
            memory_pressure_before = read_pressure("memory")
            request_start = time.perf_counter()
            request_error: Exception | None = None
            base_record: dict[str, Any] = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "server_repetition": server_repetition,
                "backend": backend.name,
                "method": backend.method,
                "load": load.label,
                "load_kind": load.kind,
                "load_workers": load.workers,
                "load_affinity": args.load_affinity,
                "load_nice": args.load_nice,
                "load_cpus": observed_cpus,
                "observed_cpus": observed_cpus,
                "target_cpus": target_cpus,
                "load_metadata": load_metadata,
                "workload": workload.label,
                "repetition": repetition,
                "request_index": request_index,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            try:
                request_result = run_streaming_request(
                    port,
                    args.served_name,
                    prompt,
                    workload.prompt_tokens,
                    workload.output_tokens,
                    args.request_timeout,
                )
                base_record.update(request_result)
                base_record["status"] = "ok"
            except Exception as error:
                request_error = error
                request_error_count += 1
                base_record["status"] = "error"
                base_record["error"] = repr(error)

            elapsed = time.perf_counter() - request_start
            cpu_after = read_cpu_times(observed_cpus)
            target_cpu_after = read_cpu_times(target_cpus)
            base_record["cpu_busy_fraction"] = cpu_busy_fraction(cpu_before, cpu_after)
            base_record["cpu_busy_by_cpu"] = cpu_busy_by_cpu(cpu_before, cpu_after)
            base_record["target_cpu_busy_fraction"] = cpu_busy_fraction(target_cpu_before, target_cpu_after)
            base_record["cpu_pressure"] = pressure_delta(cpu_pressure_before, read_pressure("cpu"), elapsed)
            base_record["memory_pressure"] = pressure_delta(memory_pressure_before, read_pressure("memory"), elapsed)
            base_record["server_memory_peak_kib"] = server.memory_monitor.peak(scenario_name)
            sample_file.write(json.dumps(base_record, sort_keys=True) + "\n")
            samples.append(base_record)
            if request_error is None:
                print(
                    f"    {workload.label} r{repetition}: "
                    f"TTFT={base_record['ttft_ms']:.1f} ms, "
                    f"decode={base_record.get('decode_tps') or 0:.2f} tok/s"
                )
            else:
                print(f"    {workload.label} r{repetition}: ERROR {base_record['error']}")
                if args.fail_fast:
                    raise request_error
    return request_error_count


def experiment_completion_errors(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[str]:
    errors = []
    failed_servers = [run["name"] for run in manifest["server_runs"] if run.get("status") != "ok"]
    if failed_servers:
        errors.append(f"failed server runs: {', '.join(failed_servers)}")
    failed_scenarios = [run["name"] for run in manifest["scenario_runs"] if run.get("status") != "ok"]
    if failed_scenarios:
        errors.append(f"failed scenarios: {', '.join(failed_scenarios)}")
    failed_samples = sum(sample.get("status") != "ok" for sample in samples)
    if failed_samples:
        errors.append(f"failed measured requests: {failed_samples}")

    backend_names = {backend.name for backend in args.backends}
    if {"vnni-only", "vnni-sycl-dynamic"}.issubset(backend_names):
        indexed = {
            (
                row.get("load_affinity", "pinned"),
                int(row.get("load_nice", 0)),
                row["load"],
                row["workload"],
            ): row
            for row in comparisons
        }
        expected_pairs = args.server_repetitions * args.request_repetitions
        for load in args.loads:
            for workload in args.workloads:
                row = indexed.get(
                    (
                        args.load_affinity,
                        args.load_nice,
                        load.label,
                        workload.label,
                    )
                )
                actual_pairs = 0 if row is None else int(row.get("paired_n", 0))
                if actual_pairs != expected_pairs:
                    errors.append(
                        f"{load.label}/{workload.label}: expected {expected_pairs} paired samples, "
                        f"found {actual_pairs}"
                    )
    return errors


def run_experiment(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model}")
    if not args.weight_path.is_dir():
        raise FileNotFoundError(f"weight directory does not exist: {args.weight_path}")
    if not BACKGROUND_LOAD_SCRIPT.is_file():
        raise FileNotFoundError(f"background load script does not exist: {BACKGROUND_LOAD_SCRIPT}")

    print_plan(args)
    if args.dry_run:
        return

    background_preflight = preflight_background_load(args)
    if background_preflight is not None:
        print(
            "Background load preflight: "
            f"affinity={background_preflight['affinity']}, "
            f"nice={background_preflight['effective_nice_values']}"
        )
    runtime_metadata = validate_runtime_environment(args)
    print(f"Runtime preflight: {runtime_metadata['package_file']} " f"({runtime_metadata['cpu_variant']})")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    logs_directory = args.output_dir / "logs"
    samples_path = args.output_dir / "samples.jsonl"
    summary_path = args.output_dir / "summary.csv"
    comparisons_path = args.output_dir / "comparisons.csv"
    report_path = args.output_dir / "report.md"
    manifest_path = args.output_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "arguments": json_compatible(vars(args)),
        "backends": [asdict(backend) for backend in args.backends],
        "loads": [asdict(load) | {"label": load.label} for load in args.loads],
        "load_placement": {
            "affinity": args.load_affinity,
            "requested_nice": args.load_nice,
            "configured_cpus": args.load_cpus,
            "observed_cpus": load_observation_cpus(args),
            "target_cpus": args.load_cpus,
        },
        "workloads": [asdict(workload) | {"label": workload.label} for workload in args.workloads],
        "git": git_metadata(),
        "hardware": hardware_metadata(),
        "runtime": runtime_metadata,
        "background_load_preflight": background_preflight,
        "server_runs": [],
        "scenario_runs": [],
    }
    write_json(manifest_path, manifest)

    samples: list[dict[str, Any]] = []
    with samples_path.open("a", encoding="utf-8", buffering=1) as sample_file:
        try:
            for server_repetition in range(args.server_repetitions):
                backend_order = list(args.backends)
                if server_repetition % 2 == 1:
                    backend_order.reverse()
                scenario_order = list(args.loads)
                random.Random(stable_seed(args.seed, "loads", server_repetition)).shuffle(scenario_order)

                for backend_index, backend in enumerate(backend_order):
                    port = args.port + backend_index
                    run_name = f"server-{server_repetition:02d}-{backend.name}"
                    log_path = logs_directory / f"{run_name}.log"
                    command = build_server_command(args, backend, port)
                    environment = build_server_environment(args, backend)
                    server_record = {
                        "name": run_name,
                        "server_repetition": server_repetition,
                        "backend": backend.name,
                        "port": port,
                        "command": command,
                        "environment": public_environment(environment),
                        "log": str(log_path.relative_to(args.output_dir)),
                        "started_at": datetime.now().astimezone().isoformat(),
                    }
                    manifest["server_runs"].append(server_record)
                    write_json(manifest_path, manifest)
                    print(f"\nStarting {run_name} on port {port}")

                    server_session = ServerSession(
                        command,
                        environment,
                        log_path,
                        port,
                        args.startup_timeout,
                        args.show_server_log,
                    )
                    try:
                        with server_session as server:
                            server.validate_backend(backend)
                            assert server.memory_monitor is not None
                            for scenario_index, load in enumerate(scenario_order):
                                scenario_name = f"{run_name}-{load.label}"
                                print(f"  Scenario {scenario_index + 1}/{len(scenario_order)}: {load.label}")
                                if args.scenario_cooldown_seconds:
                                    time.sleep(args.scenario_cooldown_seconds)
                                server.memory_monitor.set_scope(scenario_name)
                                scenario_record: dict[str, Any] = {
                                    "name": scenario_name,
                                    "backend": backend.name,
                                    "load": load.label,
                                    "load_affinity": args.load_affinity,
                                    "load_nice": args.load_nice,
                                    "observed_cpus": load_observation_cpus(args),
                                    "status": "running",
                                    "started_at": datetime.now().astimezone().isoformat(),
                                }
                                manifest["scenario_runs"].append(scenario_record)
                                write_json(manifest_path, manifest)
                                try:
                                    request_errors = run_scenario_requests(
                                        args,
                                        backend,
                                        server,
                                        server_repetition,
                                        port,
                                        load,
                                        scenario_name,
                                        sample_file,
                                        samples,
                                    )
                                    scenario_record["request_error_count"] = request_errors
                                    if request_errors:
                                        scenario_record["status"] = "error"
                                        scenario_record["error"] = f"{request_errors} measured request(s) failed"
                                    else:
                                        scenario_record["status"] = "ok"
                                except Exception as error:
                                    scenario_record["status"] = "error"
                                    scenario_record["error"] = repr(error)
                                    if args.fail_fast:
                                        raise
                                    print(f"  Scenario failed: {error}", file=sys.stderr)
                                finally:
                                    if scenario_record["status"] == "running":
                                        scenario_record["status"] = "interrupted"
                                    scenario_record["finished_at"] = datetime.now().astimezone().isoformat()
                                    scenario_record["server_memory_peak_kib"] = server.memory_monitor.peak(
                                        scenario_name
                                    )
                                    write_json(manifest_path, manifest)
                    except Exception as error:
                        server_record["status"] = "error"
                        server_record["error"] = repr(error)
                        write_json(manifest_path, manifest)
                        if args.fail_fast:
                            raise
                        print(f"  Server run failed: {error}", file=sys.stderr)
                    else:
                        server_record["status"] = "ok"
                    finally:
                        server_record["finished_at"] = datetime.now().astimezone().isoformat()
                        server_record["server_memory_peak_kib"] = server_session.final_memory_peak_kib
                        write_json(manifest_path, manifest)
        except BaseException as error:
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            manifest["error"] = repr(error)
            manifest["finished_at"] = datetime.now().astimezone().isoformat()
            write_json(manifest_path, manifest)
            raise

    summary = summarize_samples(samples, args.bootstrap_samples, args.seed)
    comparisons = comparison_rows_with_samples(summary, samples, args.bootstrap_samples, args.seed)
    write_csv(summary_path, summary)
    write_csv(comparisons_path, comparisons)
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["sample_count"] = len(samples)
    manifest["successful_sample_count"] = sum(sample.get("status") == "ok" for sample in samples)
    manifest["outputs"] = {
        "samples": str(samples_path.name),
        "summary": str(summary_path.name),
        "comparisons": str(comparisons_path.name),
        "report": str(report_path.name),
    }
    completion_errors = experiment_completion_errors(args, manifest, samples, comparisons)
    if completion_errors:
        manifest["status"] = "complete_with_errors"
        manifest["completion_errors"] = completion_errors
    write_markdown_report(report_path, manifest, summary, comparisons)
    write_json(manifest_path, manifest)
    outcome = "finished with errors" if completion_errors else "complete"
    print(f"\nExperiment {outcome}: {args.output_dir}")
    print(f"  Samples: {samples_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Comparisons: {comparisons_path}")
    print(f"  Report: {report_path}")
    if completion_errors:
        raise RuntimeError("experiment incomplete: " + "; ".join(completion_errors))


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
