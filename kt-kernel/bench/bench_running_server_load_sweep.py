#!/usr/bin/env python3
"""Run a reproducible steady CPU-load sweep against an existing server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import select
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "kt-kernel" / "bench" / "bench_running_server.py"
BACKGROUND_LOAD_SCRIPT = REPO_ROOT / "kt-kernel" / "bench" / "cpu_background_load.py"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "artifacts" / "running-server-sweeps"
PROTOCOL_VERSION = "cpu-igpu-steady-v1"
SOURCE_FILES = (
    "kt-kernel/operators/common.hpp",
    "kt-kernel/operators/cpu_igpu_service_scheduler.hpp",
    "kt-kernel/operators/cpu_load_monitor.hpp",
    "kt-kernel/operators/avx2/gptq_int4_packed_avxvnni-moe.hpp",
    "kt-kernel/operators/sycl/gptq_int4_cpu_igpu-moe.hpp",
    "kt-kernel/ext_bindings.cpp",
    "kt-kernel/python/utils/amx.py",
    "kt-kernel/bench/bench_running_server.py",
    "kt-kernel/bench/cpu_background_load.py",
    "kt-kernel/bench/bench_running_server_load_sweep.py",
    "perf-log/35b-test-cpu-igpu.sh",
)


@dataclass(frozen=True)
class BackendExpectation:
    policy: str
    ratio: str
    prefill_ratio: str | None = None
    decode_ratio: str | None = None


BACKEND_EXPECTATIONS = {
    "dynamic": BackendExpectation("dynamic", "0"),
    "phase-fixed": BackendExpectation("phase-fixed", "0", "0", "1"),
    "packed-cpu-fixed": BackendExpectation("fixed", "0"),
    "igpu-fixed": BackendExpectation("fixed", "1"),
}


def parse_worker_counts(value: str) -> list[int]:
    try:
        workers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("load workers must be comma-separated integers") from error
    if not workers:
        raise argparse.ArgumentTypeError("at least one load worker count is required")
    if any(worker < 0 for worker in workers):
        raise argparse.ArgumentTypeError("load worker counts must be non-negative")
    if len(set(workers)) != len(workers):
        raise argparse.ArgumentTypeError("load worker counts must be unique")
    return workers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30100")
    parser.add_argument("--backend", choices=sorted(BACKEND_EXPECTATIONS), required=True)
    parser.add_argument("--block-label", required=True)
    parser.add_argument(
        "--load-workers",
        type=parse_worker_counts,
        default=parse_worker_counts("0,4,8,12,16,20"),
    )
    parser.add_argument("--load-order", choices=("listed", "shuffled"), default="shuffled")
    parser.add_argument("--load-affinity", choices=("free", "pinned"), default="free")
    parser.add_argument("--load-cpus", help="CPU list for pinned load, such as 0-7")
    parser.add_argument("--load-nice", type=int, default=0)
    parser.add_argument("--load-stabilization-seconds", type=float, default=3.0)
    parser.add_argument("--scenario-cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--workloads", default="1024:600")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--warmup-prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup-output-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--server-wait-timeout", type=float, default=30.0)
    parser.add_argument("--background-ready-timeout", type=float, default=40.0)
    parser.add_argument("--expected-server-nice", type=int, default=5)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--equivalence-margin-percent", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.block_label):
        parser.error("--block-label must contain only letters, digits, '.', '_' or '-'")
    if not -20 <= args.load_nice <= 19:
        parser.error("--load-nice must be between -20 and 19")
    if not -20 <= args.expected_server_nice <= 19:
        parser.error("--expected-server-nice must be between -20 and 19")
    if args.load_affinity == "pinned" and not args.load_cpus:
        parser.error("--load-cpus is required when --load-affinity=pinned")
    if args.load_affinity == "free" and args.load_cpus:
        parser.error("--load-cpus is only valid when --load-affinity=pinned")
    non_negative = {
        "warmups": args.warmups,
        "load-stabilization-seconds": args.load_stabilization_seconds,
        "scenario-cooldown-seconds": args.scenario_cooldown_seconds,
    }
    invalid_non_negative = [name for name, value in non_negative.items() if value < 0]
    if invalid_non_negative:
        parser.error(f"these values must be non-negative: {', '.join(invalid_non_negative)}")
    positive = {
        "warmup-prompt-tokens": args.warmup_prompt_tokens,
        "warmup-output-tokens": args.warmup_output_tokens,
        "repetitions": args.repetitions,
        "bootstrap-samples": args.bootstrap_samples,
        "request-timeout": args.request_timeout,
        "server-wait-timeout": args.server_wait_timeout,
        "background-ready-timeout": args.background_ready_timeout,
        "equivalence-margin-percent": args.equivalence_margin_percent,
    }
    invalid_positive = [name for name, value in positive.items() if value <= 0]
    if invalid_positive:
        parser.error(f"these values must be positive: {', '.join(invalid_positive)}")
    args.base_url = args.base_url.rstrip("/")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_RESULTS_ROOT / f"{timestamp}-{args.backend}-{args.block_label}"
    return args


def make_load_schedule(workers: list[int], order: str, seed: int) -> list[int]:
    schedule = list(workers)
    if order == "shuffled":
        random.Random(seed).shuffle(schedule)
    return schedule


def command_output(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout + result.stderr).strip()
    return output or None


def git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        status = git("status", "--short").splitlines()
        tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT)
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "dirty_files": status,
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": repr(error)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints() -> dict[str, str | None]:
    return {
        relative: sha256_file(REPO_ROOT / relative) if (REPO_ROOT / relative).is_file() else None
        for relative in SOURCE_FILES
    }


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
        "lspci": command_output(["lspci", "-nn"]),
        "sycl_ls": command_output(["sycl-ls"]),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(f"process {pid} does not exist") from error
    return [os.fsdecode(item) for item in raw.split(b"\0") if item]


def read_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(f"process {pid} does not exist") from error
    environment = {}
    for item in raw.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            environment[os.fsdecode(key)] = os.fsdecode(value)
    return environment


def option_value(command: list[str], option: str) -> str | None:
    for index, argument in enumerate(command):
        if argument == option and index + 1 < len(command):
            return command[index + 1]
        if argument.startswith(option + "="):
            return argument.split("=", 1)[1]
    return None


def local_server_port(base_url: str) -> int:
    parsed = urlsplit(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("formal sweep server process validation requires a localhost URL")
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def find_server_pids(port: int) -> list[int]:
    matches = []
    for process_path in Path("/proc").iterdir():
        if not process_path.name.isdigit():
            continue
        try:
            command = read_cmdline(int(process_path.name))
        except (PermissionError, ProcessLookupError, RuntimeError):
            continue
        if "sglang.launch_server" not in " ".join(command):
            continue
        if option_value(command, "--port") == str(port):
            matches.append(int(process_path.name))
    return sorted(matches)


def resolve_server_pid(base_url: str, requested_pid: int | None) -> int:
    port = local_server_port(base_url)
    if requested_pid is not None:
        command = read_cmdline(requested_pid)
        if "sglang.launch_server" not in " ".join(command):
            raise RuntimeError(f"PID {requested_pid} is not an SGLang launch_server process")
        if option_value(command, "--port") != str(port):
            raise RuntimeError(f"PID {requested_pid} does not serve port {port}")
        return requested_pid
    matches = find_server_pids(port)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one SGLang server on port {port}, found PIDs {matches}; "
            "pass --server-pid to disambiguate"
        )
    return matches[0]


def extension_metadata(environment: dict[str, str]) -> dict[str, Any] | None:
    package_root = environment.get("KT_KERNEL_PACKAGE_ROOT")
    roots = [Path(package_root)] if package_root else []
    roots.extend(Path(item) for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item)
    root = next(
        (candidate for candidate in roots if candidate.joinpath("kt_kernel").is_dir()),
        None,
    )
    if root is None:
        return None
    candidates = sorted(root.glob("kt_kernel/kt_kernel_ext*.so"))
    return {
        "package_root": str(root),
        "extensions": [{"path": str(path), "sha256": sha256_file(path)} for path in candidates],
    }


def server_metadata(pid: int) -> dict[str, Any]:
    command = read_cmdline(pid)
    environment = read_environ(pid)
    public_keys = (
        "KT_CPU_IGPU_POLICY",
        "KT_CPU_IGPU_RATIO",
        "KT_CPU_IGPU_PREFILL_RATIO",
        "KT_CPU_IGPU_DECODE_RATIO",
        "SCHEDULER_TELEMETRY_FILE",
        "SCHEDULER_TELEMETRY_LAYER",
        "ONEAPI_DEVICE_SELECTOR",
        "KT_KERNEL_PACKAGE_ROOT",
    )
    model_path = option_value(command, "--model")
    model_config = Path(model_path, "config.json") if model_path else None
    return {
        "pid": pid,
        "command": command,
        "nice": os.getpriority(os.PRIO_PROCESS, pid),
        "affinity": sorted(os.sched_getaffinity(pid)),
        "environment": {key: environment.get(key) for key in public_keys},
        "model_config": (
            {
                "path": str(model_config),
                "sha256": sha256_file(model_config),
            }
            if model_config is not None and model_config.is_file()
            else None
        ),
        "kt_kernel_extension": extension_metadata(environment),
    }


def validate_server(metadata: dict[str, Any], backend: str, expected_nice: int) -> None:
    expectation = BACKEND_EXPECTATIONS[backend]
    environment = metadata["environment"]
    errors = []
    expected_environment = {
        "KT_CPU_IGPU_POLICY": expectation.policy,
        "KT_CPU_IGPU_RATIO": expectation.ratio,
        "KT_CPU_IGPU_PREFILL_RATIO": expectation.prefill_ratio,
        "KT_CPU_IGPU_DECODE_RATIO": expectation.decode_ratio,
    }
    for key, expected in expected_environment.items():
        actual = environment.get(key)
        normalized_actual = actual if actual not in {None, ""} else None
        if normalized_actual != expected:
            errors.append(f"{key} expected {expected!r}, got {actual!r}")
    telemetry_file = environment.get("SCHEDULER_TELEMETRY_FILE")
    if telemetry_file not in {None, ""}:
        errors.append("SCHEDULER_TELEMETRY_FILE must be empty for formal performance runs")
    if int(metadata["nice"]) != expected_nice:
        errors.append(f"server nice expected {expected_nice}, got {metadata['nice']}")
    if option_value(metadata["command"], "--kt-method") != "CPU_IGPU_GPTQ_INT4":
        errors.append("server --kt-method must be CPU_IGPU_GPTQ_INT4")
    if errors:
        raise RuntimeError("server preflight failed: " + "; ".join(errors))


def validate_server_identity(base_url: str, expected_pid: int, backend: str, expected_nice: int) -> None:
    actual_pid = resolve_server_pid(base_url, None)
    if actual_pid != expected_pid:
        raise RuntimeError(f"server PID changed during the sweep: expected {expected_pid}, got {actual_pid}")
    validate_server(server_metadata(actual_pid), backend, expected_nice)


def find_background_load_pids() -> list[int]:
    matches = []
    for process_path in Path("/proc").iterdir():
        if not process_path.name.isdigit():
            continue
        try:
            command = process_path.joinpath("cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"cpu_background_load.py" in command:
            matches.append(int(process_path.name))
    return sorted(matches)


def terminate_process_group(process: subprocess.Popen[Any], timeout: float) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    process.wait(timeout=5.0)
    return False


class ManagedBackgroundLoad:
    def __init__(self, args: argparse.Namespace, workers: int):
        self.args = args
        self.workers = workers
        self.process: subprocess.Popen[str] | None = None
        self.metadata: dict[str, Any] | None = None
        self.command = self.build_command()

    def build_command(self) -> list[str]:
        command = [
            sys.executable,
            str(BACKGROUND_LOAD_SCRIPT),
            "--kind",
            "compute",
            "--workers",
            str(self.workers),
            "--affinity",
            self.args.load_affinity,
            "--nice",
            str(self.args.load_nice),
        ]
        if self.args.load_affinity == "pinned":
            command.extend(["--cpus", self.args.load_cpus])
        return command

    def __enter__(self) -> dict[str, Any]:
        if self.workers <= 0:
            raise RuntimeError("ManagedBackgroundLoad requires a positive worker count")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            assert self.process.stdout is not None
            readable, _, _ = select.select([self.process.stdout], [], [], self.args.background_ready_timeout)
            if not readable:
                raise TimeoutError("background load did not become ready in time")
            line = self.process.stdout.readline()
            if not line:
                detail = self.process.stderr.read().strip() if self.process.stderr else ""
                raise RuntimeError(
                    f"background load exited before readiness with code " f"{self.process.poll()}: {detail}"
                )
            self.metadata = json.loads(line)
            expected = {
                "status": "ready",
                "workers": self.workers,
                "affinity": self.args.load_affinity,
                "requested_nice": self.args.load_nice,
                "effective_nice_values": [self.args.load_nice],
            }
            mismatches = {
                key: (value, self.metadata.get(key))
                for key, value in expected.items()
                if self.metadata.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"background load readiness mismatch: {mismatches}")
            return self.metadata
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        terminate_process_group(process, timeout=10.0)

    def check(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"background load exited unexpectedly with code {self.process.returncode}")


def benchmark_command(args: argparse.Namespace, workers: int, run_label: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--base-url",
        args.base_url,
        "--run-label",
        run_label,
        "--workloads",
        args.workloads,
        "--warmups",
        str(args.warmups),
        "--warmup-prompt-tokens",
        str(args.warmup_prompt_tokens),
        "--warmup-output-tokens",
        str(args.warmup_output_tokens),
        "--repetitions",
        str(args.repetitions),
        "--seed",
        str(args.seed),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--timeout",
        str(args.request_timeout),
        "--server-wait-timeout",
        str(args.server_wait_timeout),
        "--fail-fast",
        "--output-dir",
        str(output_dir),
    ]


def load_label(workers: int) -> str:
    return "none" if workers == 0 else f"compute{workers}"


def run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as output:
        output.write("# Command: " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
            return process.wait()
        except BaseException:
            terminate_process_group(process, timeout=10.0)
            raise


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_compatible(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_point_summary(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_sweep_summary(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        for summary in record.get("summary", []):
            rows.append(
                {
                    "backend": record["backend"],
                    "block_label": record["block_label"],
                    "order_index": record["order_index"],
                    "load_workers": record["load_workers"],
                    "load": record["load"],
                    "status": record["status"],
                    **summary,
                }
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Running-Server CPU Load Sweep",
        "",
        f"- Status: {manifest['status']}",
        f"- Protocol: {manifest['protocol_version']}",
        f"- Backend: {manifest['backend']}",
        f"- Block: {manifest['block_label']}",
        f"- Load order: {manifest['realized_load_order']}",
        f"- Practical-equivalence margin: " f"{manifest['equivalence_margin_percent']:.2f}%",
        "",
        "| Order | Load | Status | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for record in manifest.get("runs", []):
        summary = record.get("summary", [])
        row = summary[0] if len(summary) == 1 else {}
        lines.append(
            f"| {record['order_index']} | {record['load']} | {record['status']} | "
            f"{row.get('prefill_tps_mean', 'NA')} | {row.get('decode_tps_mean', 'NA')} | "
            f"{row.get('ttft_ms_mean', 'NA')} | {row.get('tpot_ms_mean', 'NA')} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [Manifest](manifest.json)",
            "- [Combined summary](summary.csv)",
            "- Per-load artifacts are under `runs/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5.0)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def print_plan(args: argparse.Namespace, schedule: list[int]) -> None:
    print("Formal steady-load sweep plan")
    print(f"  Protocol: {PROTOCOL_VERSION}")
    print(f"  Backend: {args.backend}")
    print(f"  Block: {args.block_label}")
    print(f"  Requested loads: {args.load_workers}")
    print(f"  Realized order: {schedule}")
    print(f"  Workloads: {args.workloads}; repetitions={args.repetitions}")
    print(f"  Telemetry: disabled (enforced)")
    for order_index, workers in enumerate(schedule, start=1):
        point = args.output_dir / "runs" / f"{order_index:02d}-{load_label(workers)}"
        label = f"{args.backend}__engine-low__{load_label(workers)}__{args.block_label}"
        print("  " + " ".join(benchmark_command(args, workers, label, point)))


def run_sweep(args: argparse.Namespace) -> int:
    schedule = make_load_schedule(args.load_workers, args.load_order, args.seed)
    print_plan(args, schedule)
    if args.dry_run:
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    existing_loads = find_background_load_pids()
    if existing_loads:
        raise RuntimeError(
            "formal sweep requires no existing cpu_background_load.py processes; " f"found {existing_loads}"
        )
    wait_for_server(args.base_url, args.server_wait_timeout)
    server_pid = resolve_server_pid(args.base_url, args.server_pid)
    server = server_metadata(server_pid)
    validate_server(server, args.backend, args.expected_server_nice)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = args.output_dir / "runs"
    logs_dir = args.output_dir / "logs"
    runs_dir.mkdir()
    logs_dir.mkdir()
    manifest_path = args.output_dir / "manifest.json"
    report_path = args.output_dir / "report.md"
    summary_path = args.output_dir / "summary.csv"
    manifest: dict[str, Any] = {
        "status": "running",
        "protocol_version": PROTOCOL_VERSION,
        "started_at": datetime.now().astimezone().isoformat(),
        "backend": args.backend,
        "block_label": args.block_label,
        "requested_load_workers": args.load_workers,
        "realized_load_order": schedule,
        "equivalence_margin_percent": args.equivalence_margin_percent,
        "arguments": {key: value for key, value in vars(args).items() if key != "dry_run"},
        "git": git_metadata(),
        "source_fingerprints": source_fingerprints(),
        "hardware": hardware_metadata(),
        "server": server,
        "runs": [],
    }
    write_json(manifest_path, manifest)
    fatal_error: BaseException | None = None
    interrupted = False
    try:
        for order_index, workers in enumerate(schedule, start=1):
            wait_for_server(args.base_url, args.server_wait_timeout)
            validate_server_identity(args.base_url, server_pid, args.backend, args.expected_server_nice)
            unexpected_loads = find_background_load_pids()
            if unexpected_loads:
                raise RuntimeError(f"unexpected background load processes before scenario: {unexpected_loads}")
            label = f"{args.backend}__engine-low__{load_label(workers)}__{args.block_label}"
            output_dir = runs_dir / f"{order_index:02d}-{load_label(workers)}"
            log_path = logs_dir / f"{order_index:02d}-{load_label(workers)}.log"
            command = benchmark_command(args, workers, label, output_dir)
            record: dict[str, Any] = {
                "backend": args.backend,
                "block_label": args.block_label,
                "order_index": order_index,
                "load": load_label(workers),
                "load_workers": workers,
                "started_at": datetime.now().astimezone().isoformat(),
                "status": "running",
                "benchmark_command": command,
                "benchmark_output_dir": str(output_dir),
                "benchmark_log": str(log_path),
            }
            manifest["runs"].append(record)
            write_json(manifest_path, manifest)
            try:
                if workers == 0:
                    record["background"] = {
                        "kind": "none",
                        "workers": 0,
                        "affinity": args.load_affinity,
                        "requested_nice": args.load_nice,
                    }
                    return_code = run_logged(command, log_path)
                else:
                    controller = ManagedBackgroundLoad(args, workers)
                    record["background_command"] = controller.command
                    with controller as metadata:
                        record["background"] = metadata
                        write_json(manifest_path, manifest)
                        if args.load_stabilization_seconds:
                            time.sleep(args.load_stabilization_seconds)
                        controller.check()
                        return_code = run_logged(command, log_path)
                        controller.check()
                record["return_code"] = return_code
                record["summary"] = read_point_summary(output_dir)
                record["status"] = "complete" if return_code == 0 else "failed"
                if return_code != 0 and not args.continue_on_error:
                    raise RuntimeError(f"benchmark failed for {load_label(workers)} with code {return_code}")
            except Exception as error:
                record["status"] = "failed"
                record["error"] = repr(error)
                if not args.continue_on_error:
                    raise
            finally:
                record["finished_at"] = datetime.now().astimezone().isoformat()
                write_json(manifest_path, manifest)
                write_sweep_summary(summary_path, manifest["runs"])
                write_report(report_path, manifest)
            if order_index < len(schedule) and args.scenario_cooldown_seconds:
                time.sleep(args.scenario_cooldown_seconds)
    except KeyboardInterrupt:
        interrupted = True
        print("\nSweep interrupted; completed artifacts were preserved.", file=sys.stderr)
    except Exception as error:
        fatal_error = error
        print(f"Sweep failed: {error}", file=sys.stderr)
    finally:
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        if interrupted:
            manifest["status"] = "interrupted"
        elif fatal_error is not None:
            manifest["status"] = "failed"
            manifest["error"] = repr(fatal_error)
        elif all(run.get("status") == "complete" for run in manifest["runs"]):
            manifest["status"] = "complete"
        else:
            manifest["status"] = "complete_with_errors"
        write_json(manifest_path, manifest)
        write_sweep_summary(summary_path, manifest["runs"])
        write_report(report_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_sweep(args)
    except (FileExistsError, OSError, requests.RequestException, RuntimeError, TimeoutError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def terminate_as_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    raise SystemExit(main())
