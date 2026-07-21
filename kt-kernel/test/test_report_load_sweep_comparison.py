"""Tests for steady-load sweep comparisons."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "report_load_sweep_comparison.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("report_load_sweep_comparison", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def make_samples(path: Path, decode_tps: float, e2e_ms: float) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "status": "ok",
            "prefill_tps": 100.0,
            "decode_tps": decode_tps,
            "ttft_ms": 1000.0,
            "tpot_ms": 1000.0 / decode_tps,
            "e2e_ms": e2e_ms,
            "cpu_busy_fraction": 0.5,
            "cpu_psi_some_fraction": 0.1,
            "completion_tokens": 10,
            "stream_tokens": 10,
        }
        for _ in range(2)
    ]
    (path / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_sweep(path: Path, backend: str, decode_by_workers: dict[int, float], fingerprint: str = "same") -> None:
    runs = []
    for order_index, (workers, decode_tps) in enumerate(decode_by_workers.items(), start=1):
        artifact = path / "runs" / f"{order_index:02d}"
        make_samples(artifact, decode_tps, 10_000.0 / decode_tps)
        runs.append(
            {
                "status": "complete",
                "load_workers": workers,
                "load": "none" if workers == 0 else f"compute{workers}",
                "benchmark_output_dir": str(artifact),
            }
        )
    manifest = {
        "status": "complete",
        "protocol_version": "test-v1",
        "backend": backend,
        "block_label": "b1",
        "requested_load_workers": list(decode_by_workers),
        "realized_load_order": list(decode_by_workers),
        "arguments": {"workloads": "1024:600"},
        "source_fingerprints": {"scheduler": fingerprint},
        "server": {"kt_kernel_extension": {"sha256": fingerprint}},
        "runs": runs,
    }
    (path / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_compares_matching_sweeps_and_reports_static_oracle(tmp_path):
    dynamic = tmp_path / "dynamic"
    cpu = tmp_path / "cpu"
    igpu = tmp_path / "igpu"
    output = tmp_path / "report"
    make_sweep(dynamic, "dynamic", {0: 20.0, 8: 30.0})
    make_sweep(cpu, "packed-cpu-fixed", {0: 20.0, 8: 10.0})
    make_sweep(igpu, "igpu-fixed", {0: 10.0, 8: 20.0})

    assert (
        reporter.main(
            [
                "--sweep",
                f"dynamic={dynamic}",
                "--sweep",
                f"packed-cpu-fixed={cpu}",
                "--sweep",
                f"igpu-fixed={igpu}",
                "--candidate",
                "dynamic",
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    with (output / "oracle.csv").open(encoding="utf-8") as source:
        oracle = {(int(row["load_workers"]), row["metric"]): row for row in csv.DictReader(source)}
    assert oracle[(0, "decode_tps")]["static_oracle_strategy"] == "packed-cpu-fixed"
    assert float(oracle[(0, "decode_tps")]["static_oracle_attainment_pct"]) == 100.0
    assert oracle[(8, "decode_tps")]["static_oracle_strategy"] == "igpu-fixed"
    assert float(oracle[(8, "decode_tps")]["static_oracle_attainment_pct"]) == 150.0
    assert (output / "effects.csv").is_file()
    assert (output / "report.md").is_file()


def test_rejects_mismatched_source_identity(tmp_path):
    dynamic = tmp_path / "dynamic"
    fixed = tmp_path / "fixed"
    make_sweep(dynamic, "dynamic", {0: 20.0}, fingerprint="a")
    make_sweep(fixed, "packed-cpu-fixed", {0: 20.0}, fingerprint="b")

    with pytest.raises(RuntimeError, match="source identity differs"):
        reporter.load_sweeps([("dynamic", dynamic), ("fixed", fixed)])
