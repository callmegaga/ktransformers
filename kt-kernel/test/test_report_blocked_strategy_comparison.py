"""Tests for blocked running-server strategy comparisons."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "report_blocked_strategy_comparison.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("report_blocked_strategy_comparison", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def make_block(path: Path, decode_values: list[float], e2e_ms: float) -> None:
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
                "e2e_ms": e2e_ms,
                "cpu_busy_fraction": 0.9,
                "cpu_psi_some_fraction": 0.1,
                "completion_tokens": 10,
                "stream_tokens": 10,
            }
        )
    (path / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compares_repeated_strategy_blocks(tmp_path):
    fixed_a1 = tmp_path / "fixed-a1"
    fixed_a2 = tmp_path / "fixed-a2"
    dynamic_b = tmp_path / "dynamic-b"
    output = tmp_path / "comparison"
    make_block(fixed_a1, [10.0, 10.0], 2000.0)
    make_block(fixed_a2, [12.0, 12.0], 2000.0)
    make_block(dynamic_b, [22.0, 22.0], 1500.0)

    assert (
        reporter.main(
            [
                "--block",
                f"fixed={fixed_a1}",
                "--block",
                f"dynamic={dynamic_b}",
                "--block",
                f"fixed={fixed_a2}",
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

    with (output / "summary.csv").open(encoding="utf-8") as source:
        summary = {(row["strategy"], row["metric"]): row for row in csv.DictReader(source)}
    with (output / "effects.csv").open(encoding="utf-8") as source:
        effects = {(row["baseline"], row["metric"]): row for row in csv.DictReader(source)}

    assert float(summary[("fixed", "decode_tps")]["mean"]) == 11.0
    assert float(effects[("fixed", "decode_tps")]["effect_pct"]) == 100.0
    assert float(effects[("fixed", "e2e_ms")]["effect_pct"]) == 25.0
    assert (output / "blocks.csv").is_file()
    assert (output / "report.md").is_file()


def test_rejects_inexact_block_samples(tmp_path):
    block = tmp_path / "block"
    make_block(block, [10.0], 2000.0)
    sample_path = block / "samples.jsonl"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["stream_tokens"] = 9
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="inexact token timestamps"):
        reporter.load_block("fixed", block, 1)
