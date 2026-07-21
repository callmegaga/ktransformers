#!/usr/bin/env python3
"""Benchmark an already-running OpenAI-compatible SGLang server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "artifacts" / "running-server-bench"
BACKGROUND_LOAD_SCRIPT = REPO_ROOT / "kt-kernel" / "bench" / "cpu_background_load.py"
DEFAULT_WORKLOADS = "1:300,1024:300,4096:300,8192:300"
METRICS = (
    "prompt_tokens",
    "completion_tokens",
    "prefill_tps",
    "decode_tps",
    "ttft_ms",
    "tpot_ms",
    "output_phase_ms",
    "ttlt_ms",
    "e2e_ms",
    "cpu_busy_fraction",
    "cpu_user_fraction",
    "cpu_nice_fraction",
    "cpu_system_fraction",
    "cpu_psi_some_fraction",
    "cpu_psi_full_fraction",
    "scheduler_prefill_igpu_ratio",
    "scheduler_prefill_cpu_load",
    "scheduler_prefill_exploration_fraction",
    "scheduler_decode_igpu_ratio",
    "scheduler_decode_cpu_load",
    "scheduler_decode_exploration_fraction",
    "background_ready_delay_ms",
    "transition_client_pre_tps",
    "transition_client_post_tps",
)
SHORT_PROMPTS = (
    "red",
    "blue",
    "green",
    "black",
    "white",
    "north",
    "south",
    "east",
    "west",
    "spring",
    "summer",
    "autumn",
    "winter",
    "alpha",
    "bravo",
    "delta",
    "river",
    "field",
    "stone",
    "cloud",
    "light",
    "sound",
    "paper",
    "metal",
    "circle",
    "square",
    "table",
    "chair",
    "glass",
    "clock",
    "plant",
    "house",
)


@dataclass(frozen=True)
class WorkloadSpec:
    target_prompt_tokens: int
    output_tokens: int

    @property
    def label(self) -> str:
        return f"p{self.target_prompt_tokens}-o{self.output_tokens}"


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
        if workload.target_prompt_tokens <= 0 or workload.output_tokens <= 0:
            raise argparse.ArgumentTypeError("workload token counts must be positive")
        if workload not in workloads:
            workloads.append(workload)
    if not workloads:
        raise argparse.ArgumentTypeError("at least one workload is required")
    return workloads


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30100")
    parser.add_argument("--model", help="Defaults to the first model from /v1/models")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--run-label",
        default="manual",
        help="Backend/load label stored with every sample",
    )
    parser.add_argument(
        "--workloads",
        type=parse_workloads,
        default=parse_workloads(DEFAULT_WORKLOADS),
        help="Comma-separated prompt:output token targets",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--warmups",
        type=int,
        default=15,
        help="Warmups before measurement; dynamic CPU-iGPU policy needs 15 for calibration",
    )
    parser.add_argument("--warmup-prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--server-wait-timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scheduler-telemetry-file",
        type=Path,
        help="JSONL written by SCHEDULER_TELEMETRY_FILE in the server launcher",
    )
    parser.add_argument(
        "--stop-background-pid",
        type=int,
        help="PID of cpu_background_load.py to stop during the measured decode",
    )
    parser.add_argument(
        "--stop-background-after-output-tokens",
        type=int,
        help="Send SIGTERM to --stop-background-pid after this many streamed output chunks",
    )
    parser.add_argument(
        "--start-compute-background-workers",
        type=int,
        help="Start this many free-affinity nice=0 compute workers during measured decode",
    )
    parser.add_argument(
        "--start-background-after-output-tokens",
        type=int,
        help="Start managed compute workers after this many streamed output chunks",
    )
    parser.add_argument(
        "--transition-static-baseline",
        action="store_true",
        help="Allow transition timing without scheduler telemetry for a fixed backend",
    )
    parser.add_argument(
        "--disable-cpu-telemetry",
        action="store_true",
        help="Disable request-window /proc/stat and CPU PSI measurements",
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--allow-estimated-usage",
        action="store_true",
        help="Allow approximate token counts if the server omits streaming usage",
    )
    parser.add_argument(
        "--allow-short-output",
        action="store_true",
        help="Accept fewer output tokens than requested despite ignore_eos=true",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    positive = {
        "repetitions": args.repetitions,
        "warmup-prompt-tokens": args.warmup_prompt_tokens,
        "warmup-output-tokens": args.warmup_output_tokens,
        "timeout": args.timeout,
        "server-wait-timeout": args.server_wait_timeout,
        "bootstrap-samples": args.bootstrap_samples,
    }
    invalid = [name for name, item in positive.items() if item <= 0]
    if invalid:
        parser.error(f"these values must be positive: {', '.join(invalid)}")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    stop_transition_values = (
        args.stop_background_pid,
        args.stop_background_after_output_tokens,
    )
    start_transition_values = (
        args.start_compute_background_workers,
        args.start_background_after_output_tokens,
    )
    if any(value is not None for value in stop_transition_values) and not all(
        value is not None for value in stop_transition_values
    ):
        parser.error("--stop-background-pid and --stop-background-after-output-tokens must be used together")
    if any(value is not None for value in start_transition_values) and not all(
        value is not None for value in start_transition_values
    ):
        parser.error(
            "--start-compute-background-workers and --start-background-after-output-tokens must be used together"
        )
    stop_transition = args.stop_background_pid is not None
    start_transition = args.start_compute_background_workers is not None
    if stop_transition and start_transition:
        parser.error("stop-background and start-background transition modes are mutually exclusive")
    if args.transition_static_baseline and not (stop_transition or start_transition):
        parser.error("--transition-static-baseline requires a background transition mode")
    if stop_transition or start_transition:
        if start_transition and args.start_compute_background_workers <= 0:
            parser.error("--start-compute-background-workers must be positive")
        if stop_transition and args.stop_background_pid <= 1:
            parser.error("--stop-background-pid must be greater than 1")
        repeated_managed_static = start_transition and args.transition_static_baseline
        if len(args.workloads) != 1 or (args.repetitions != 1 and not repeated_managed_static):
            parser.error(
                "background transition measurement requires one workload and one repetition; "
                "managed start-background static baselines may use multiple repetitions"
            )
        if args.scheduler_telemetry_file is None and not args.transition_static_baseline:
            parser.error("background transition measurement requires --scheduler-telemetry-file")
        if start_transition and args.warmups == 0:
            parser.error("start-background transition requires at least one warmup")
        transition_tokens = (
            args.stop_background_after_output_tokens if stop_transition else args.start_background_after_output_tokens
        )
        assert transition_tokens is not None
        if transition_tokens <= 0 or transition_tokens >= args.workloads[0].output_tokens:
            parser.error("--stop-background-after-output-tokens must be between 1 and output_tokens - 1")
    args.base_url = args.base_url.rstrip("/")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_RESULTS_ROOT / f"{timestamp}-{slugify(args.run_label)}"
    return args


def slugify(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "-_." else "-" for character in value.strip())
    result = "-".join(part for part in result.split("-") if part)
    return result[:80] or "manual"


def stable_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def wait_for_server(base_url: str, headers: dict[str, str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", headers=headers, timeout=5.0)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            time.sleep(1.0)
    raise TimeoutError(f"server at {base_url} did not become healthy within {timeout:.0f} seconds: " f"{last_error}")


def discover_model(base_url: str, headers: dict[str, str], timeout: float) -> str:
    response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    models = response.json().get("data", [])
    if not models:
        raise RuntimeError("the server returned no models from /v1/models")
    return str(models[0]["id"])


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
            result[category] = delta_us / (elapsed_seconds * 1_000_000.0)
        except (KeyError, TypeError, ZeroDivisionError):
            result[category] = None
    return result


def read_cpu_times() -> dict[int, tuple[int, int, int, int, int]]:
    """Return total, idle, user, nice, and system-like ticks for every CPU."""
    result: dict[int, tuple[int, int, int, int, int]] = {}
    try:
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
                continue
            values = [int(value) for value in fields[1:]]
            values.extend([0] * (10 - len(values)))
            user = max(values[0] - values[8], 0)
            nice = max(values[1] - values[9], 0)
            system = values[2] + values[5] + values[6]
            idle = values[3] + values[4]
            total = user + nice + system + idle + values[7]
            result[int(fields[0][3:])] = (total, idle, user, nice, system)
    except OSError:
        pass
    return result


def cpu_utilization_delta(
    before: dict[int, tuple[int, int, int, int, int]],
    after: dict[int, tuple[int, int, int, int, int]],
) -> dict[str, Any]:
    totals = {"total": 0, "idle": 0, "user": 0, "nice": 0, "system": 0}
    by_cpu = {"busy": {}, "user": {}, "nice": {}, "system": {}}
    for cpu, before_values in before.items():
        if cpu not in after:
            continue
        deltas = [after[cpu][index] - before_values[index] for index in range(5)]
        total, idle, user, nice, system = deltas
        if total <= 0:
            continue
        totals["total"] += total
        totals["idle"] += idle
        totals["user"] += user
        totals["nice"] += nice
        totals["system"] += system
        by_cpu["busy"][str(cpu)] = 1.0 - idle / total
        by_cpu["user"][str(cpu)] = user / total
        by_cpu["nice"][str(cpu)] = nice / total
        by_cpu["system"][str(cpu)] = system / total
    total = totals["total"]
    if total <= 0:
        return {}
    return {
        "cpu_busy_fraction": 1.0 - totals["idle"] / total,
        "cpu_user_fraction": totals["user"] / total,
        "cpu_nice_fraction": totals["nice"] / total,
        "cpu_system_fraction": totals["system"] / total,
        "cpu_busy_by_cpu": by_cpu["busy"],
        "cpu_user_by_cpu": by_cpu["user"],
        "cpu_nice_by_cpu": by_cpu["nice"],
        "cpu_system_by_cpu": by_cpu["system"],
    }


class SchedulerTelemetryTail:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()

    def mark(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def read_since(self, offset: int) -> list[dict[str, Any]]:
        if not self.path.is_file():
            raise RuntimeError(f"scheduler telemetry file does not exist: {self.path}")
        with self.path.open("rb") as source:
            source.seek(offset)
            payload = source.read()
        events = []
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid scheduler telemetry JSON after offset {offset}, line {line_number}: {error}"
                ) from error
            if isinstance(event, dict):
                events.append(event)
        return events


def validate_background_load_pid(pid: int) -> str:
    try:
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(f"background load PID {pid} does not exist") from error
    command = " ".join(part.decode("utf-8", errors="replace") for part in raw_command.split(b"\0") if part)
    if "cpu_background_load.py" not in command:
        raise RuntimeError(f"PID {pid} is not cpu_background_load.py; refusing to signal it: {command}")
    return command


def find_background_load_pids() -> list[int]:
    pids = []
    for process_path in Path("/proc").iterdir():
        if not process_path.name.isdigit():
            continue
        try:
            command = process_path.joinpath("cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"cpu_background_load.py" in command:
            pids.append(int(process_path.name))
    return sorted(pids)


def wait_for_process_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.05)
    return not Path(f"/proc/{pid}").exists()


class ManagedComputeBackground:
    def __init__(self, workers: int):
        self.workers = workers
        self.process: subprocess.Popen[str] | None = None
        self.launch_ns: int | None = None
        self.ready_ns: int | None = None
        self.metadata: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self._ready_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("managed background load was already started")
        self.launch_ns = None
        self.ready_ns = None
        self.metadata = None
        self.error = None
        self._ready_thread = None
        command = [
            sys.executable,
            str(BACKGROUND_LOAD_SCRIPT),
            "--kind",
            "compute",
            "--workers",
            str(self.workers),
            "--affinity",
            "free",
            "--nice",
            "0",
        ]
        self.launch_ns = time.monotonic_ns()
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._ready_thread = threading.Thread(target=self._capture_ready, daemon=True)
        self._ready_thread.start()

    def _capture_ready(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            line = self.process.stdout.readline()
            if not line:
                detail = self.process.stderr.read().strip() if self.process.stderr is not None else ""
                raise RuntimeError(
                    f"managed background exited before readiness with code {self.process.poll()}: {detail}"
                )
            metadata = json.loads(line)
            if metadata.get("status") != "ready":
                raise RuntimeError(f"managed background was not ready: {metadata}")
            if metadata.get("workers") != self.workers:
                raise RuntimeError(f"managed background reported wrong worker count: {metadata}")
            if metadata.get("effective_nice_values") != [0] or metadata.get("affinity") != "free":
                raise RuntimeError(f"managed background reported wrong scheduling policy: {metadata}")
            self.metadata = metadata
            self.ready_ns = time.monotonic_ns()
        except BaseException as error:
            self.error = error

    def wait_ready(self, timeout: float = 40.0) -> dict[str, Any]:
        if self._ready_thread is None:
            raise RuntimeError("managed background load was not started")
        self._ready_thread.join(timeout)
        if self._ready_thread.is_alive():
            raise TimeoutError("managed background load did not become ready within 40 seconds")
        if self.error is not None:
            raise RuntimeError(f"managed background load failed: {self.error}") from self.error
        assert self.metadata is not None
        return self.metadata

    def stop(self) -> bool:
        if self.process is None:
            return True
        process = self.process
        self.process = None
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
            return True
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)
            return False


def summarize_scheduler_transition(
    events: list[dict[str, Any]], transition_ns: int, target: str = "cpu"
) -> dict[str, Any]:
    if target not in {"cpu", "igpu"}:
        raise ValueError("transition target must be 'cpu' or 'igpu'")
    decode_events = sorted(
        (
            event
            for event in events
            if event.get("phase") == "decode"
            and event.get("igpu_ratio") is not None
            and int(event.get("execution_calls_delta", 1)) > 0
        ),
        key=lambda event: int(event["monotonic_ns"]),
    )
    before = [event for event in decode_events if int(event["monotonic_ns"]) < transition_ns]
    after = [event for event in decode_events if int(event["monotonic_ns"]) >= transition_ns]
    result: dict[str, Any] = {
        "transition_decode_calls_before_signal": sum(int(event.get("execution_calls_delta", 1)) for event in before),
        "transition_decode_calls_after_signal": sum(int(event.get("execution_calls_delta", 1)) for event in after),
    }

    def add_response(prefix: str, predicate: Callable[[dict[str, Any]], bool]) -> None:
        calls_before = 0
        for event in after:
            if predicate(event):
                event_ns = int(event["monotonic_ns"])
                result[f"transition_{prefix}_delay_calls"] = calls_before
                result[f"transition_{prefix}_delay_ms"] = (event_ns - transition_ns) / 1_000_000.0
                result[f"transition_{prefix}_sequence"] = int(event.get("sequence", -1))
                result[f"transition_{prefix}_event_ns"] = event_ns
                return
            calls_before += int(event.get("execution_calls_delta", 1))
        result[f"transition_{prefix}_delay_calls"] = None
        result[f"transition_{prefix}_delay_ms"] = None
        result[f"transition_{prefix}_sequence"] = None
        result[f"transition_{prefix}_event_ns"] = None

    target_is_igpu = target == "igpu"

    def is_target_ratio(value: float) -> bool:
        return value >= 0.5 if target_is_igpu else value < 0.5

    add_response(
        f"first_{target}_execution",
        lambda event: is_target_ratio(float(event["igpu_ratio"])),
    )
    add_response(
        f"settled_{target}",
        lambda event: is_target_ratio(float(event["igpu_ratio"]))
        and is_target_ratio(float(event.get("policy_igpu_ratio", event["igpu_ratio"])))
        and not bool(event.get("exploration", False)),
    )
    if decode_events:
        result["transition_final_igpu_ratio"] = float(decode_events[-1]["igpu_ratio"])
        result["transition_final_policy_igpu_ratio"] = float(
            decode_events[-1].get("policy_igpu_ratio", decode_events[-1]["igpu_ratio"])
        )
    return result


def summarize_client_transition_times(output_token_ns: list[int], transition_output_token: int) -> dict[str, float]:
    if not 1 < transition_output_token < len(output_token_ns):
        raise RuntimeError("transition token does not split the streamed output into two intervals")
    transition_index = transition_output_token - 1
    pre_seconds = (output_token_ns[transition_index] - output_token_ns[0]) / 1_000_000_000.0
    post_seconds = (output_token_ns[-1] - output_token_ns[transition_index]) / 1_000_000_000.0
    if pre_seconds <= 0 or post_seconds <= 0:
        raise RuntimeError("streamed output timestamps are not increasing")
    return {
        "transition_client_pre_tps": (transition_output_token - 1) / pre_seconds,
        "transition_client_post_tps": (len(output_token_ns) - transition_output_token) / post_seconds,
    }


def summarize_scheduler_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"scheduler_event_count": len(events)}
    for phase in ("prefill", "decode"):
        phase_events = [event for event in events if event.get("phase") == phase]
        result[f"scheduler_{phase}_events"] = len(phase_events)
        if not phase_events:
            continue
        ratio_events = [
            event
            for event in phase_events
            if event.get("igpu_ratio") is not None
            and ("execution_calls_delta" not in event or int(event["execution_calls_delta"]) > 0)
        ]
        ratios = [float(event["igpu_ratio"]) for event in ratio_events]
        loads = [float(event["cpu_load"]) for event in phase_events]
        switches = [int(event["switch_count"]) for event in phase_events]
        high_load = [bool(event.get("high_load_epoch", False)) for event in phase_events]
        exploration = [bool(event.get("exploration", False)) for event in phase_events]
        result.update(
            {
                f"scheduler_{phase}_cpu_load": statistics.fmean(loads),
                f"scheduler_{phase}_cpu_load_max": max(loads),
                f"scheduler_{phase}_cpu_load_final": loads[-1],
                f"scheduler_{phase}_switch_count_first": switches[0],
                f"scheduler_{phase}_switch_count_final": switches[-1],
                f"scheduler_{phase}_switch_count_delta": max(switches) - min(switches),
                f"scheduler_{phase}_high_load_fraction": statistics.fmean(high_load),
                f"scheduler_{phase}_exploration_fraction": statistics.fmean(exploration),
            }
        )
        execution_calls = sum(max(0, int(event.get("execution_calls_delta", 0))) for event in phase_events)
        if any("execution_calls_delta" in event for event in phase_events):
            result[f"scheduler_{phase}_execution_calls"] = execution_calls
        if ratios:
            if all("execution_calls_delta" in event for event in ratio_events):
                ratio_mean = sum(
                    float(event["igpu_ratio"]) * int(event["execution_calls_delta"]) for event in ratio_events
                ) / sum(int(event["execution_calls_delta"]) for event in ratio_events)
            else:
                ratio_mean = statistics.fmean(ratios)
            result.update(
                {
                    f"scheduler_{phase}_igpu_ratio": ratio_mean,
                    f"scheduler_{phase}_igpu_ratio_min": min(ratios),
                    f"scheduler_{phase}_igpu_ratio_max": max(ratios),
                    f"scheduler_{phase}_igpu_ratio_final": ratios[-1],
                }
            )
    return result


def short_prompt_marker(nonce: str) -> str:
    prefix, separator, suffix = nonce.rpartition("-")
    try:
        offset = int(suffix) if separator else 0
    except ValueError:
        prefix, offset = nonce, 0
    return SHORT_PROMPTS[(stable_seed(prefix) + offset) % len(SHORT_PROMPTS)]


def make_prompt(target_tokens: int, nonce: str) -> str:
    marker = short_prompt_marker(nonce)
    if target_tokens <= 2:
        return marker
    prefix = f"{marker} {nonce} Read and retain the following sequence. "
    filler_words = max(target_tokens - len(prefix.split()), 1)
    return prefix + "word " * filler_words


def extract_usage(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    candidates = [chunk.get("usage"), chunk.get("meta_info")]
    choices = chunk.get("choices") or []
    if choices and isinstance(choices[0], dict):
        candidates.extend([choices[0].get("usage"), choices[0].get("meta_info")])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        prompt_tokens = candidate.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = candidate.get("input_tokens")
        completion_tokens = candidate.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = candidate.get("output_tokens")
        if prompt_tokens is not None or completion_tokens is not None:
            return (
                int(prompt_tokens) if prompt_tokens is not None else None,
                int(completion_tokens) if completion_tokens is not None else None,
            )
    return None, None


def run_streaming_request(
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    requested_prompt_tokens: int,
    requested_output_tokens: int,
    seed: int,
    timeout: float,
    allow_estimated_usage: bool,
    allow_short_output: bool,
    clock=time.perf_counter,
    on_output_token: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": requested_output_tokens,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    }
    first_token_time = None
    last_token_time = None
    prompt_tokens = None
    completion_tokens = None
    output_parts: list[str] = []
    stream_chunks = 0
    stream_tokens = 0
    start = clock()
    with requests.post(
        f"{base_url}/v1/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
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
            if usage_prompt is not None:
                prompt_tokens = usage_prompt
            if usage_completion is not None:
                completion_tokens = usage_completion
            choices = chunk.get("choices") or []
            if not choices:
                continue
            if usage_completion is not None:
                if usage_completion < stream_tokens:
                    raise RuntimeError("streamed completion token count moved backwards")
                if on_output_token is not None:
                    for output_token in range(stream_tokens + 1, usage_completion + 1):
                        on_output_token(output_token)
                stream_tokens = usage_completion
            text = choices[0].get("text", "")
            if not text:
                continue
            now = clock()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            stream_chunks += 1
            output_parts.append(str(text))
    end = clock()

    if first_token_time is None or last_token_time is None:
        raise RuntimeError("stream completed without a generated token")
    if prompt_tokens is None or completion_tokens is None:
        if not allow_estimated_usage:
            raise RuntimeError(
                "server omitted prompt/completion token usage; rerun with "
                "--allow-estimated-usage only for non-paper diagnostics"
            )
        prompt_tokens = prompt_tokens or requested_prompt_tokens
        completion_tokens = completion_tokens or stream_chunks
        usage_source = "estimated"
    else:
        usage_source = "server"
    if completion_tokens <= 0:
        raise RuntimeError("server reported no completion tokens")
    if not allow_short_output and completion_tokens != requested_output_tokens:
        raise RuntimeError(
            f"server produced {completion_tokens} tokens, expected exactly "
            f"{requested_output_tokens} with ignore_eos=true"
        )

    ttft_seconds = first_token_time - start
    output_phase_seconds = max(last_token_time - first_token_time, 0.0)
    ttlt_seconds = last_token_time - start
    e2e_seconds = end - start
    decode_intervals = completion_tokens - 1
    output_text = "".join(output_parts)
    return {
        "requested_prompt_tokens": requested_prompt_tokens,
        "requested_output_tokens": requested_output_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_source": usage_source,
        "stream_chunks": stream_chunks,
        "stream_tokens": stream_tokens,
        "ttft_ms": ttft_seconds * 1000.0,
        "output_phase_ms": output_phase_seconds * 1000.0,
        "ttlt_ms": ttlt_seconds * 1000.0,
        "e2e_ms": e2e_seconds * 1000.0,
        "stream_tail_ms": max(end - last_token_time, 0.0) * 1000.0,
        "prefill_tps": prompt_tokens / ttft_seconds if ttft_seconds > 0 else None,
        "decode_tps": (
            decode_intervals / output_phase_seconds if decode_intervals > 0 and output_phase_seconds > 0 else None
        ),
        "tpot_ms": (output_phase_seconds * 1000.0 / decode_intervals if decode_intervals > 0 else None),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "output_characters": len(output_text),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = [statistics.fmean(generator.choice(values) for _ in values) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_samples(samples: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample["workload"]), []).append(sample)

    rows = []
    for workload, group in sorted(groups.items()):
        successful = [sample for sample in group if sample.get("status") == "ok"]
        row: dict[str, Any] = {
            "workload": workload,
            "target_prompt_tokens": group[0]["target_prompt_tokens"],
            "requested_output_tokens": group[0]["requested_output_tokens"],
            "n": len(successful),
            "error_count": len(group) - len(successful),
        }
        for metric in METRICS:
            values = [float(sample[metric]) for sample in successful if sample.get(metric) is not None]
            if not values:
                continue
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                bootstrap_samples,
                stable_seed(seed, workload, metric),
            )
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_p95"] = percentile(values, 0.95)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return rows


def format_value(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def format_mean_ci(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    low = row.get(f"{metric}_ci95_low")
    high = row.get(f"{metric}_ci95_high")
    if mean is None:
        return "NA"
    return f"{format_value(mean)} [{format_value(low)}, {format_value(high)}]"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        if not fieldnames:
            return
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, WorkloadSpec):
        return asdict(value)
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    return value


def write_report(path: Path, manifest: dict[str, Any], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Running Server Benchmark",
        "",
        f"- Status: {manifest['status']}",
        f"- Run label: {manifest['run_label']}",
        f"- Server: {manifest['server']['base_url']}",
        f"- Model: {manifest['server']['model']}",
        f"- Successful samples: {manifest.get('successful_sample_count', 0)} / " f"{manifest.get('sample_count', 0)}",
        "",
        "Prefill throughput is prompt tokens divided by TTFT. Decode throughput and TPOT "
        "use the intervals between the first and last output token. TTLT is time to last token; "
        "E2E additionally includes the final stream completion overhead.",
        "",
        "| Workload | N | Prompt tokens | Output tokens | Prefill tok/s [95% CI] | "
        "Decode tok/s [95% CI] | TTFT ms [95% CI] | TPOT ms/tok [95% CI] | "
        "Output phase ms | TTLT ms | E2E ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['workload']} | {row['n']} | "
            f"{format_value(row.get('prompt_tokens_mean'), 1)} | "
            f"{format_value(row.get('completion_tokens_mean'), 1)} | "
            f"{format_mean_ci(row, 'prefill_tps')} | "
            f"{format_mean_ci(row, 'decode_tps')} | "
            f"{format_mean_ci(row, 'ttft_ms')} | "
            f"{format_mean_ci(row, 'tpot_ms')} | "
            f"{format_value(row.get('output_phase_ms_mean'))} | "
            f"{format_value(row.get('ttlt_ms_mean'))} | "
            f"{format_value(row.get('e2e_ms_mean'))} |"
        )
    if not summary:
        lines.append("| NA | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    if any(row.get("cpu_busy_fraction_mean") is not None for row in summary):
        lines.extend(
            [
                "",
                "## CPU Telemetry",
                "",
                "Fractions cover each complete request window. `user` includes normal-priority work; "
                "`nice` includes the low-priority inference scope.",
                "",
                "| Workload | CPU busy | CPU user | CPU nice | CPU system | CPU PSI some |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary:
            lines.append(
                f"| {row['workload']} | {format_value(row.get('cpu_busy_fraction_mean'), 4)} | "
                f"{format_value(row.get('cpu_user_fraction_mean'), 4)} | "
                f"{format_value(row.get('cpu_nice_fraction_mean'), 4)} | "
                f"{format_value(row.get('cpu_system_fraction_mean'), 4)} | "
                f"{format_value(row.get('cpu_psi_some_fraction_mean'), 4)} |"
            )
    if any(row.get("scheduler_decode_igpu_ratio_mean") is not None for row in summary):
        lines.extend(
            [
                "",
                "## Scheduler Telemetry",
                "",
                "Values are request means from the MoE layer or layers selected at server launch.",
                "",
                "| Workload | Prefill iGPU ratio | Prefill CPU load | Prefill exploration | Decode iGPU ratio | Decode CPU load | Decode exploration |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary:
            lines.append(
                f"| {row['workload']} | "
                f"{format_value(row.get('scheduler_prefill_igpu_ratio_mean'), 4)} | "
                f"{format_value(row.get('scheduler_prefill_cpu_load_mean'), 4)} | "
                f"{format_value(row.get('scheduler_prefill_exploration_fraction_mean'), 4)} | "
                f"{format_value(row.get('scheduler_decode_igpu_ratio_mean'), 4)} | "
                f"{format_value(row.get('scheduler_decode_cpu_load_mean'), 4)} | "
                f"{format_value(row.get('scheduler_decode_exploration_fraction_mean'), 4)} |"
            )
    transition = manifest.get("background_transition")
    if transition:
        transition_result = transition.get("result", {})
        transition_summary = summary[0] if len(summary) == 1 else {}

        def format_transition_metric(metric: str, digits: int = 2) -> str:
            if transition_summary.get(f"{metric}_mean") is not None:
                return format_mean_ci(transition_summary, metric)
            return format_value(transition_result.get(metric), digits)

        direction = transition.get("direction", "high-to-low")
        lines.extend(
            [
                "",
                "## Background Load Transition",
                "",
                f"- Direction: {direction}",
                f"- Transition samples: {transition_summary.get('n', 1)}",
            ]
        )
        if direction == "high-to-low":
            lines.extend(
                [
                    f"- Signalled PID: {transition['pid']}",
                    f"- Stop after output token: {transition['stop_after_output_tokens']}",
                    f"- Background stopped: {transition_result.get('background_stopped', False)}",
                    f"- Decode calls before signal: {format_value(transition_result.get('transition_decode_calls_before_signal'), 0)}",
                    f"- First CPU execution delay: {format_value(transition_result.get('transition_first_cpu_execution_delay_calls'), 0)} calls / "
                    f"{format_value(transition_result.get('transition_first_cpu_execution_delay_ms'))} ms",
                    f"- Settled CPU delay: {format_value(transition_result.get('transition_settled_cpu_delay_calls'), 0)} calls / "
                    f"{format_value(transition_result.get('transition_settled_cpu_delay_ms'))} ms",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Managed workers: {transition['workers']}",
                    f"- Start after output token: {transition['start_after_output_tokens']}",
                    f"- Background ready delay: {format_transition_metric('background_ready_delay_ms')} ms",
                    f"- Background stopped after request: {transition_result.get('background_stopped', False)}",
                    f"- Decode calls before launch: {format_value(transition_result.get('transition_decode_calls_before_signal'), 0)}",
                    f"- First iGPU execution delay: {format_value(transition_result.get('transition_first_igpu_execution_delay_calls'), 0)} calls / "
                    f"{format_value(transition_result.get('transition_first_igpu_execution_delay_ms'))} ms",
                    f"- Settled iGPU delay: {format_value(transition_result.get('transition_settled_igpu_delay_calls'), 0)} calls / "
                    f"{format_value(transition_result.get('transition_settled_igpu_delay_ms'))} ms",
                ]
            )
        lines.extend(
            [
                f"- Client pre-transition throughput: {format_transition_metric('transition_client_pre_tps')} token/s",
                f"- Client post-transition throughput: {format_transition_metric('transition_client_post_tps')} token/s",
            ]
        )
        lines.append(
            f"- Final iGPU execution ratio: {format_value(transition_result.get('transition_final_igpu_ratio'), 4)}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [Manifest](manifest.json)",
            "- [Request samples](samples.jsonl)",
            "- [Summary CSV](summary.csv)",
        ]
    )
    if manifest.get("telemetry", {}).get("scheduler_events_file"):
        lines.append("- [Per-request scheduler events](scheduler-telemetry.jsonl)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_sample(sample: dict[str, Any]) -> None:
    if sample["status"] != "ok":
        print(f"  {sample['workload']} r{sample['repetition'] + 1}: " f"ERROR {sample['error']}")
        return
    print(
        f"  {sample['workload']} r{sample['repetition'] + 1}: "
        f"prefill={format_value(sample.get('prefill_tps'))} tok/s, "
        f"decode={format_value(sample.get('decode_tps'))} tok/s, "
        f"TTFT={format_value(sample.get('ttft_ms'))} ms, "
        f"TPOT={format_value(sample.get('tpot_ms'))} ms/tok, "
        f"output={format_value(sample.get('output_phase_ms'))} ms, "
        f"E2E={format_value(sample.get('e2e_ms'))} ms"
    )


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nSummary (mean [bootstrap 95% CI])")
    print(f"{'Workload':<16} {'N':>3} {'Prefill tok/s':>29} " f"{'Decode tok/s':>29} {'TTFT ms':>29} {'TPOT ms':>29}")
    for row in summary:
        print(
            f"{row['workload']:<16} {row['n']:>3} "
            f"{format_mean_ci(row, 'prefill_tps'):>29} "
            f"{format_mean_ci(row, 'decode_tps'):>29} "
            f"{format_mean_ci(row, 'ttft_ms'):>29} "
            f"{format_mean_ci(row, 'tpot_ms'):>29}"
        )


def run_benchmark(args: argparse.Namespace) -> int:
    headers = auth_headers(args.api_key)
    print(f"Waiting for server: {args.base_url}")
    wait_for_server(args.base_url, headers, args.server_wait_timeout)
    model = args.model or discover_model(args.base_url, headers, args.timeout)
    print(f"Model: {model}")
    print(f"Run label: {args.run_label}")
    print(f"Output: {args.output_dir}")
    scheduler_telemetry = (
        SchedulerTelemetryTail(args.scheduler_telemetry_file) if args.scheduler_telemetry_file else None
    )
    if scheduler_telemetry is not None:
        print(f"Scheduler telemetry: {scheduler_telemetry.path}")
    background_transition: dict[str, Any] | None = None
    managed_background: ManagedComputeBackground | None = None
    if args.stop_background_pid is not None:
        command = validate_background_load_pid(args.stop_background_pid)
        background_transition = {
            "direction": "high-to-low",
            "pid": args.stop_background_pid,
            "command": command,
            "stop_after_output_tokens": args.stop_background_after_output_tokens,
        }
        print(
            "Background transition: "
            f"stop PID {args.stop_background_pid} after "
            f"{args.stop_background_after_output_tokens} output tokens"
        )
    elif args.start_compute_background_workers is not None:
        existing_background_pids = find_background_load_pids()
        if existing_background_pids:
            raise RuntimeError(
                "start-background transition requires no existing cpu_background_load.py processes; "
                f"found {existing_background_pids}"
            )
        managed_background = ManagedComputeBackground(args.start_compute_background_workers)
        background_transition = {
            "direction": "low-to-high",
            "workers": args.start_compute_background_workers,
            "start_after_output_tokens": args.start_background_after_output_tokens,
        }
        print(
            "Background transition: "
            f"start {args.start_compute_background_workers} compute workers after "
            f"{args.start_background_after_output_tokens} output tokens"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = args.output_dir / "samples.jsonl"
    summary_path = args.output_dir / "summary.csv"
    manifest_path = args.output_dir / "manifest.json"
    report_path = args.output_dir / "report.md"
    scheduler_events_path = args.output_dir / "scheduler-telemetry.jsonl"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "run_label": args.run_label,
        "server": {"base_url": args.base_url, "model": model},
        "arguments": json_compatible({key: value for key, value in vars(args).items() if key != "api_key"}),
        "workloads": [asdict(workload) | {"label": workload.label} for workload in args.workloads],
        "telemetry": {
            "cpu_enabled": not args.disable_cpu_telemetry,
            "scheduler_source": str(scheduler_telemetry.path) if scheduler_telemetry is not None else None,
            "scheduler_events_file": scheduler_events_path.name if scheduler_telemetry is not None else None,
        },
        "background_transition": background_transition,
    }
    write_json(manifest_path, manifest)

    samples: list[dict[str, Any]] = []
    fatal_error: BaseException | None = None
    interrupted = False
    scheduler_event_count = 0
    scheduler_output = (
        scheduler_events_path.open("a", encoding="utf-8", buffering=1) if scheduler_telemetry is not None else None
    )
    scheduler_warmup_offset = scheduler_telemetry.mark() if scheduler_telemetry is not None else 0
    try:
        if args.warmups:
            print(f"Warmups: {args.warmups}")
        for warmup_index in range(args.warmups):
            nonce = f"warmup-{args.seed}-{warmup_index}"
            run_streaming_request(
                args.base_url,
                headers,
                model,
                make_prompt(args.warmup_prompt_tokens, nonce),
                args.warmup_prompt_tokens,
                args.warmup_output_tokens,
                args.seed,
                args.timeout,
                args.allow_estimated_usage,
                args.allow_short_output,
            )
            print(f"  warmup {warmup_index + 1}/{args.warmups}", end="\r", flush=True)
        if args.warmups:
            print()
        if scheduler_telemetry is not None:
            if not scheduler_telemetry.path.is_file():
                raise RuntimeError(
                    "the server did not create scheduler telemetry during warmup; ensure it was launched with "
                    f"SCHEDULER_TELEMETRY_FILE={scheduler_telemetry.path} and the rebuilt Python package"
                )
            warmup_events = scheduler_telemetry.read_since(scheduler_warmup_offset)
            if args.warmups and not warmup_events:
                raise RuntimeError(
                    "scheduler telemetry did not advance during warmup; the selected file may belong to a "
                    f"different server run: {scheduler_telemetry.path}"
                )
            if managed_background is not None:
                warmup_decode = [
                    event
                    for event in warmup_events
                    if event.get("phase") == "decode"
                    and event.get("igpu_ratio") is not None
                    and int(event.get("execution_calls_delta", 1)) > 0
                ]
                if not warmup_decode:
                    raise RuntimeError("start-background transition warmup produced no decode telemetry")
                final_warmup = warmup_decode[-1]
                if (
                    float(final_warmup["igpu_ratio"]) >= 0.5
                    or float(final_warmup.get("policy_igpu_ratio", 1.0)) >= 0.5
                    or bool(final_warmup.get("exploration", False))
                ):
                    raise RuntimeError(
                        "start-background transition requires a stable CPU warmup endpoint; "
                        f"got execution_igpu_ratio={float(final_warmup['igpu_ratio']):.3f}, "
                        f"policy_igpu_ratio={float(final_warmup.get('policy_igpu_ratio', 1.0)):.3f}, "
                        f"exploration={bool(final_warmup.get('exploration', False))}, "
                        f"cpu_samples={final_warmup.get('cpu_samples')}, "
                        f"igpu_samples={final_warmup.get('igpu_samples')}, "
                        f"cpu_ms_per_row={final_warmup.get('cpu_ms_per_row')}, "
                        f"igpu_ms_per_row={final_warmup.get('igpu_ms_per_row')}"
                    )

        schedule = [(workload, repetition) for workload in args.workloads for repetition in range(args.repetitions)]
        if not args.no_shuffle:
            random.Random(args.seed).shuffle(schedule)
        print(f"Measured requests: {len(schedule)}")
        with samples_path.open("a", encoding="utf-8", buffering=1) as output:
            for request_index, (workload, repetition) in enumerate(schedule):
                nonce = f"sample-{args.seed}-{workload.label}-{repetition}"
                prompt = make_prompt(workload.target_prompt_tokens, nonce)
                sample: dict[str, Any] = {
                    "status": "running",
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "run_label": args.run_label,
                    "workload": workload.label,
                    "target_prompt_tokens": workload.target_prompt_tokens,
                    "requested_output_tokens": workload.output_tokens,
                    "repetition": repetition,
                    "request_index": request_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
                cpu_before = read_cpu_times() if not args.disable_cpu_telemetry else {}
                cpu_pressure_before = read_pressure("cpu") if not args.disable_cpu_telemetry else {}
                scheduler_offset = scheduler_telemetry.mark() if scheduler_telemetry is not None else 0
                request_start = time.perf_counter()
                transition_signal: dict[str, Any] = {}
                output_token_ns: list[int] = []

                def on_output_token(output_token: int) -> None:
                    output_token_ns.append(time.monotonic_ns())
                    if background_transition is None or transition_signal:
                        return
                    direction = str(background_transition["direction"])
                    token_key = (
                        "stop_after_output_tokens" if direction == "high-to-low" else "start_after_output_tokens"
                    )
                    if output_token < int(background_transition[token_key]):
                        return
                    if direction == "high-to-low":
                        signal_ns = time.monotonic_ns()
                        try:
                            os.kill(int(background_transition["pid"]), signal.SIGTERM)
                        except ProcessLookupError as error:
                            raise RuntimeError(
                                f"background load PID {background_transition['pid']} exited before transition"
                            ) from error
                        transition_signal.update(
                            {
                                "background_stop_signal_ns": signal_ns,
                                "background_stop_output_token": output_token,
                            }
                        )
                    else:
                        assert managed_background is not None
                        managed_background.start()
                        assert managed_background.process is not None
                        assert managed_background.launch_ns is not None
                        transition_signal.update(
                            {
                                "background_start_launch_ns": managed_background.launch_ns,
                                "background_start_output_token": output_token,
                                "background_start_pid": managed_background.process.pid,
                            }
                        )

                try:
                    sample.update(
                        run_streaming_request(
                            args.base_url,
                            headers,
                            model,
                            prompt,
                            workload.target_prompt_tokens,
                            workload.output_tokens,
                            args.seed,
                            args.timeout,
                            args.allow_estimated_usage,
                            args.allow_short_output,
                            on_output_token=(on_output_token if background_transition is not None else None),
                        )
                    )
                    if background_transition is not None and not transition_signal:
                        raise RuntimeError("background transition was not triggered")
                    sample.update(transition_signal)
                    if background_transition is not None:
                        if len(output_token_ns) != int(sample["completion_tokens"]):
                            raise RuntimeError(
                                "continuous streamed usage did not provide one timestamp per output token"
                            )
                        transition_token = int(
                            background_transition[
                                (
                                    "stop_after_output_tokens"
                                    if background_transition["direction"] == "high-to-low"
                                    else "start_after_output_tokens"
                                )
                            ]
                        )
                        sample.update(summarize_client_transition_times(output_token_ns, transition_token))
                    sample["status"] = "ok"
                except Exception as error:
                    sample["status"] = "error"
                    sample["error"] = repr(error)
                elapsed = time.perf_counter() - request_start
                if not args.disable_cpu_telemetry:
                    sample["cpu_observation_seconds"] = elapsed
                    sample.update(cpu_utilization_delta(cpu_before, read_cpu_times()))
                    cpu_pressure = pressure_delta(cpu_pressure_before, read_pressure("cpu"), elapsed)
                    sample["cpu_psi_some_fraction"] = cpu_pressure["some"]
                    sample["cpu_psi_full_fraction"] = cpu_pressure["full"]
                if scheduler_telemetry is not None:
                    events = scheduler_telemetry.read_since(scheduler_offset)
                    if sample["status"] == "ok" and not events:
                        raise RuntimeError(
                            "scheduler telemetry did not advance during measured request "
                            f"{request_index}; verify that the benchmark and server use the same file: "
                            f"{scheduler_telemetry.path}"
                        )
                    sample.update(summarize_scheduler_events(events))
                    if transition_signal:
                        direction = str(background_transition["direction"])
                        transition_ns = int(
                            transition_signal[
                                (
                                    "background_stop_signal_ns"
                                    if direction == "high-to-low"
                                    else "background_start_launch_ns"
                                )
                            ]
                        )
                        sample.update(
                            summarize_scheduler_transition(
                                events,
                                transition_ns,
                                target="cpu" if direction == "high-to-low" else "igpu",
                            )
                        )
                    assert scheduler_output is not None
                    for event in events:
                        scheduler_output.write(
                            json.dumps(
                                event
                                | {
                                    "request_index": request_index,
                                    "workload": workload.label,
                                    "repetition": repetition,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    scheduler_event_count += len(events)
                if transition_signal and background_transition is not None:
                    if background_transition["direction"] == "high-to-low":
                        sample["background_stopped"] = wait_for_process_exit(int(background_transition["pid"]), 10.0)
                    else:
                        assert managed_background is not None
                        metadata = managed_background.wait_ready()
                        assert managed_background.ready_ns is not None
                        ready_ns = managed_background.ready_ns
                        launch_ns = int(transition_signal["background_start_launch_ns"])
                        sample["background_ready_ns"] = ready_ns
                        sample["background_ready_delay_ms"] = (ready_ns - launch_ns) / 1_000_000.0
                        sample["background_metadata"] = metadata
                        for prefix in ("first_igpu_execution", "settled_igpu"):
                            event_ns = sample.get(f"transition_{prefix}_event_ns")
                            sample[f"transition_{prefix}_ready_delay_ms"] = (
                                (int(event_ns) - ready_ns) / 1_000_000.0 if event_ns is not None else None
                            )
                        sample["background_stopped"] = managed_background.stop()
                output.write(json.dumps(sample, sort_keys=True) + "\n")
                samples.append(sample)
                print_sample(sample)
                if sample["status"] != "ok" and args.fail_fast:
                    raise RuntimeError(sample["error"])
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; partial samples have been preserved.", file=sys.stderr)
    except BaseException as error:
        fatal_error = error
        print(f"Benchmark failed: {error}", file=sys.stderr)
    finally:
        if managed_background is not None and managed_background.process is not None:
            managed_background.stop()
        if scheduler_output is not None:
            scheduler_output.close()
        summary = summarize_samples(samples, args.bootstrap_samples, args.seed)
        write_csv(summary_path, summary)
        successful = sum(sample.get("status") == "ok" for sample in samples)
        expected = len(args.workloads) * args.repetitions
        if interrupted:
            status = "interrupted"
        elif fatal_error is not None:
            status = "failed"
        elif successful != expected:
            status = "complete_with_errors"
        else:
            status = "complete"
        manifest.update(
            {
                "status": status,
                "finished_at": datetime.now().astimezone().isoformat(),
                "sample_count": len(samples),
                "successful_sample_count": successful,
                "expected_sample_count": expected,
                "error": repr(fatal_error) if fatal_error is not None else None,
            }
        )
        if background_transition is not None:
            transition_sample = next(
                (
                    sample
                    for sample in samples
                    if "background_stop_signal_ns" in sample or "background_start_launch_ns" in sample
                ),
                None,
            )
            if transition_sample is not None:
                manifest["background_transition"]["result"] = {
                    key: value
                    for key, value in transition_sample.items()
                    if key.startswith("transition_")
                    or key
                    in {
                        "background_stopped",
                        "background_stop_output_token",
                        "background_start_output_token",
                        "background_start_pid",
                        "background_ready_delay_ms",
                    }
                }
        manifest["telemetry"]["scheduler_event_count"] = scheduler_event_count
        write_json(manifest_path, manifest)
        write_report(report_path, manifest, summary)
        print_summary(summary)
        print(f"\nResults: {args.output_dir}")
    return 0 if manifest["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_benchmark(args)
    except (OSError, requests.RequestException, RuntimeError, TimeoutError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
