"""Optional JSONL telemetry for the CPU/iGPU MoE scheduler."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

TELEMETRY_FILE_ENV = "KT_CPU_IGPU_TELEMETRY_FILE"
TELEMETRY_LAYER_ENV = "KT_CPU_IGPU_TELEMETRY_LAYER"


class SchedulerTelemetryWriter:
    """Append selected-layer scheduler snapshots without buffering."""

    def __init__(self, path: Path, layer_idx: int, policy: str, configured_ratio: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        self.path = path
        self.layer_idx = layer_idx
        self.policy = policy
        self.configured_ratio = configured_ratio
        self._fd = os.open(path, flags, 0o644)
        self._lock = threading.Lock()
        self._sequence = 0
        self._execution_counters = {False: (0, 0), True: (0, 0)}

    @classmethod
    def from_environment(cls, layer_idx: int, method: str) -> "SchedulerTelemetryWriter | None":
        raw_path = os.getenv(TELEMETRY_FILE_ENV, "").strip()
        if method != "CPU_IGPU_GPTQ_INT4" or not raw_path:
            return None
        raw_layer = os.getenv(TELEMETRY_LAYER_ENV, "0").strip().lower()
        if raw_layer != "all":
            try:
                selected_layer = int(raw_layer)
            except ValueError as error:
                raise ValueError(f"{TELEMETRY_LAYER_ENV} must be a non-negative integer or 'all'") from error
            if selected_layer < 0:
                raise ValueError(f"{TELEMETRY_LAYER_ENV} must be a non-negative integer or 'all'")
            if layer_idx != selected_layer:
                return None
        policy = os.getenv("KT_CPU_IGPU_POLICY", "dynamic").strip().lower()
        configured_ratio = float(os.getenv("KT_CPU_IGPU_RATIO", "0"))
        return cls(Path(raw_path).expanduser(), layer_idx, policy, configured_ratio)

    def close(self) -> None:
        with self._lock:
            if self._fd < 0:
                return
            os.close(self._fd)
            self._fd = -1

    def record(self, moe: Any, qlen: int) -> None:
        if self._fd < 0 or not hasattr(moe, "scheduler_debug"):
            return
        decode = qlen == 1
        debug = list(moe.scheduler_debug(decode))
        if len(debug) not in {7, 8, 10, 12}:
            raise RuntimeError(f"unexpected CPU/iGPU scheduler_debug payload: {debug!r}")
        ratio_snapshot = float(moe.scheduler_igpu_ratio())
        execution_calls_delta = 0
        actual_ratio: float | None = ratio_snapshot
        if hasattr(moe, "scheduler_execution_debug"):
            actual_ratio = None
            execution = list(moe.scheduler_execution_debug(decode))
            if len(execution) != 2:
                raise RuntimeError(f"unexpected CPU/iGPU execution payload: {execution!r}")
            calls, ratio_units = int(execution[0]), int(execution[1])
            previous_calls, previous_ratio_units = self._execution_counters[decode]
            execution_calls_delta = calls - previous_calls
            ratio_units_delta = ratio_units - previous_ratio_units
            if execution_calls_delta > 0 and ratio_units_delta >= 0:
                actual_ratio = ratio_units_delta / (execution_calls_delta * 1_000_000.0)
            self._execution_counters[decode] = (calls, ratio_units)

        event = {
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "layer": self.layer_idx,
            "qlen": int(qlen),
            "phase": "decode" if decode else "prefill",
            "policy": self.policy,
            "configured_igpu_ratio": self.configured_ratio,
            "igpu_ratio": actual_ratio,
            "igpu_ratio_snapshot": ratio_snapshot,
            "execution_calls_delta": execution_calls_delta,
            "policy_igpu_ratio": float(debug[0]),
            "cpu_load": float(moe.scheduler_cpu_load()),
            "cpu_ms_per_row": float(debug[1]),
            "igpu_ms_per_row": float(debug[2]),
            "cpu_samples": int(debug[3]),
            "igpu_samples": int(debug[4]),
            "switch_count": int(debug[5]),
            "high_load_epoch": bool(debug[6]),
            "exploration": bool(debug[7]) if len(debug) >= 8 else False,
        }
        if len(debug) >= 10:
            event["cpu_sample_load"] = float(debug[8])
            event["igpu_sample_load"] = float(debug[9])
        if len(debug) >= 12:
            event["igpu_reference_load"] = float(debug[10])
            event["reprobe_reason"] = int(debug[11])
        with self._lock:
            event["sequence"] = self._sequence
            self._sequence += 1
            encoded = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            view = memoryview(encoded)
            while view:
                written = os.write(self._fd, view)
                view = view[written:]

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass
