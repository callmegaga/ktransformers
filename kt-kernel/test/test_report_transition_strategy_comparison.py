"""Tests for fixed/dynamic transition strategy comparison."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "report_transition_strategy_comparison.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("report_transition_strategy_comparison", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def make_static(path: Path, decode_values: list[float]) -> None:
    path.mkdir()
    rows = []
    for decode_tps in decode_values:
        rows.append(
            {
                "status": "ok",
                "prefill_tps": 100.0,
                "decode_tps": decode_tps,
                "ttft_ms": 1000.0,
                "tpot_ms": 1000.0 / decode_tps,
                "e2e_ms": 2000.0,
                "transition_client_pre_tps": 20.0,
                "transition_client_post_tps": decode_tps,
                "background_ready_delay_ms": 50.0,
                "completion_tokens": 10,
                "stream_tokens": 10,
            }
        )
    (path / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_dynamic(path: Path, decode_values: list[float]) -> None:
    path.mkdir()
    rows = []
    for decode_tps in decode_values:
        rows.append(
            {
                "prefill_tps": 100.0,
                "decode_tps": decode_tps,
                "ttft_ms": 1000.0,
                "tpot_ms": 1000.0 / decode_tps,
                "e2e_ms": 1500.0,
                "client_pre_transition_tps": 20.0,
                "client_post_transition_tps": decode_tps,
                "background_ready_ms": 50.0,
                "completion_tokens": 10,
                "stream_tokens": 10,
                "token_timestamps_exact": True,
            }
        )
    with (path / "cycles.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compares_static_and_dynamic_artifacts(tmp_path):
    static = tmp_path / "static"
    dynamic = tmp_path / "dynamic"
    output = tmp_path / "comparison"
    make_static(static, [10.0, 10.0])
    make_dynamic(dynamic, [20.0, 20.0])

    assert (
        reporter.main(
            [
                "--strategy",
                f"static={static}",
                "--strategy",
                f"dynamic={dynamic}",
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

    with (output / "effects.csv").open(encoding="utf-8") as source:
        effects = {(row["baseline"], row["metric"]): row for row in csv.DictReader(source)}
    with (output / "summary.csv").open(encoding="utf-8") as source:
        summary = {(row["strategy"], row["metric"]): row for row in csv.DictReader(source)}

    assert float(summary[("dynamic", "decode_tps")]["mean"]) == 20.0
    assert float(effects[("static", "decode_tps")]["effect_pct"]) == 100.0
    assert float(effects[("static", "e2e_ms")]["effect_pct"]) == 25.0
    assert (output / "report.md").is_file()
    assert (output / "manifest.json").is_file()


def test_rejects_inexact_static_token_timestamps(tmp_path):
    static = tmp_path / "static"
    make_static(static, [10.0])
    sample_path = static / "samples.jsonl"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["stream_tokens"] = 9
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stream token timestamps do not match"):
        reporter.load_strategy("static", static)
