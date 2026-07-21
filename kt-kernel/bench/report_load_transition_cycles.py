#!/usr/bin/env python3
"""Aggregate repeated dynamic-load transition benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from bench_running_server import bootstrap_mean_ci

COMMON_METRICS = {
    "prefill_tps": "token/s",
    "decode_tps": "token/s",
    "ttft_ms": "ms",
    "tpot_ms": "ms/token",
    "e2e_ms": "ms",
    "first_target_delay_calls": "calls",
    "first_target_delay_ms": "ms",
    "settle_target_delay_calls": "calls",
    "settle_target_delay_ms": "ms",
    "pre_transition_calls_per_s": "calls/s",
    "post_transition_calls_per_s": "calls/s",
    "post_vs_pre_rate": "x",
}

LOW_TO_HIGH_METRICS = {
    "background_ready_ms": "ms",
    "ready_to_first_target_calls": "calls",
    "ready_to_first_target_ms": "ms",
    "ready_to_settle_target_calls": "calls",
    "ready_to_settle_target_ms": "ms",
}

CLIENT_TOKEN_METRICS = {
    "client_pre_transition_tps": "token/s",
    "client_post_transition_tps": "token/s",
    "client_post_vs_pre_tps": "x",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Transition artifact directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def read_single_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one sample in {path}, found {len(rows)}")
    if rows[0].get("status") != "ok":
        raise RuntimeError(f"sample is not successful in {path}: {rows[0].get('status')}")
    return rows[0]


def execution_calls(event: dict[str, Any]) -> int:
    return max(0, int(event.get("execution_calls_delta", 1)))


def calls_per_second(events: list[dict[str, Any]], skip_first_interval: bool) -> float:
    start_index = 1 if skip_first_interval else 0
    if len(events) <= start_index + 1:
        raise RuntimeError("not enough scheduler events to calculate a segmented rate")
    elapsed_seconds = (int(events[-1]["monotonic_ns"]) - int(events[start_index]["monotonic_ns"])) / 1_000_000_000.0
    if elapsed_seconds <= 0:
        raise RuntimeError("scheduler event timestamps are not increasing")
    calls = sum(execution_calls(event) for event in events[start_index + 1 :])
    return calls / elapsed_seconds


def load_cycle(path: Path, cycle: int) -> dict[str, Any]:
    sample = read_single_jsonl(path / "samples.jsonl")
    completion_tokens = int(sample["completion_tokens"])
    stream_tokens_value = sample.get("stream_tokens")
    stream_tokens = int(stream_tokens_value) if stream_tokens_value is not None else None
    if stream_tokens is not None and stream_tokens != completion_tokens:
        raise RuntimeError(
            f"stream token timestamps do not match completion tokens in {path}: "
            f"{stream_tokens} != {completion_tokens}"
        )
    client_pre_value = sample.get("transition_client_pre_tps")
    client_post_value = sample.get("transition_client_post_tps")
    client_pre_tps = float(client_pre_value) if client_pre_value is not None else None
    client_post_tps = float(client_post_value) if client_post_value is not None else None
    events = [
        json.loads(line)
        for line in (path / "scheduler-telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decode = sorted(
        (event for event in events if event.get("phase") == "decode" and execution_calls(event) > 0),
        key=lambda event: int(event["monotonic_ns"]),
    )
    if not decode:
        raise RuntimeError(f"no decode telemetry in {path}")
    if "background_stop_signal_ns" in sample:
        direction = "high-to-low"
        target = "cpu"
        transition_ns = int(sample["background_stop_signal_ns"])
        ready_ms = None
        ready_to_first_calls = None
        ready_to_first_ms = None
        ready_to_settle_calls = None
        ready_to_settle_ms = None
    elif "background_start_launch_ns" in sample:
        direction = "low-to-high"
        target = "igpu"
        transition_ns = int(sample["background_start_launch_ns"])
        ready_ms = float(sample["background_ready_delay_ms"])
        ready_ns = int(sample["background_ready_ns"])
        first_sequence = int(sample["transition_first_igpu_execution_sequence"])
        settled_sequence_for_ready = int(sample["transition_settled_igpu_sequence"])
        ready_to_first_calls = sum(
            execution_calls(event)
            for event in decode
            if int(event["monotonic_ns"]) >= ready_ns and int(event.get("sequence", -1)) < first_sequence
        )
        ready_to_first_ms = float(sample["transition_first_igpu_execution_ready_delay_ms"])
        ready_to_settle_calls = sum(
            execution_calls(event)
            for event in decode
            if int(event["monotonic_ns"]) >= ready_ns and int(event.get("sequence", -1)) < settled_sequence_for_ready
        )
        ready_to_settle_ms = float(sample["transition_settled_igpu_ready_delay_ms"])
    else:
        raise RuntimeError(f"sample has no supported transition timestamp in {path}")
    first_prefix = f"transition_first_{target}_execution"
    settle_prefix = f"transition_settled_{target}"
    settled_sequence = int(sample[f"{settle_prefix}_sequence"])
    before_transition = [event for event in decode if int(event["monotonic_ns"]) < transition_ns]
    settled_target = [event for event in decode if int(event.get("sequence", -1)) >= settled_sequence]
    pre_rate = calls_per_second(before_transition, skip_first_interval=True)
    post_rate = calls_per_second(settled_target, skip_first_interval=False)
    total_calls = sum(execution_calls(event) for event in decode)
    igpu_calls = sum(
        float(event["igpu_ratio"]) * execution_calls(event) for event in decode if event.get("igpu_ratio") is not None
    )
    exploration_calls = sum(execution_calls(event) for event in decode if bool(event.get("exploration", False)))
    return {
        "cycle": cycle,
        "artifact": str(path),
        "direction": direction,
        "target": target,
        "prefill_tps": float(sample["prefill_tps"]),
        "decode_tps": float(sample["decode_tps"]),
        "ttft_ms": float(sample["ttft_ms"]),
        "tpot_ms": float(sample["tpot_ms"]),
        "e2e_ms": float(sample["e2e_ms"]),
        "first_target_delay_calls": int(sample[f"{first_prefix}_delay_calls"]),
        "first_target_delay_ms": float(sample[f"{first_prefix}_delay_ms"]),
        "settle_target_delay_calls": int(sample[f"{settle_prefix}_delay_calls"]),
        "settle_target_delay_ms": float(sample[f"{settle_prefix}_delay_ms"]),
        "background_ready_ms": ready_ms,
        "ready_to_first_target_calls": ready_to_first_calls,
        "ready_to_first_target_ms": ready_to_first_ms,
        "ready_to_settle_target_calls": ready_to_settle_calls,
        "ready_to_settle_target_ms": ready_to_settle_ms,
        "pre_transition_calls_per_s": pre_rate,
        "post_transition_calls_per_s": post_rate,
        "post_vs_pre_rate": post_rate / pre_rate,
        "client_pre_transition_tps": client_pre_tps,
        "client_post_transition_tps": client_post_tps,
        "client_post_vs_pre_tps": (
            client_post_tps / client_pre_tps if client_pre_tps is not None and client_post_tps is not None else None
        ),
        "completion_tokens": completion_tokens,
        "stream_tokens": stream_tokens,
        "stream_chunks": sample.get("stream_chunks"),
        "token_timestamps_exact": stream_tokens == completion_tokens if stream_tokens is not None else None,
        "execution_calls": total_calls,
        "igpu_calls": igpu_calls,
        "cpu_calls": total_calls - igpu_calls,
        "exploration_calls": exploration_calls,
        "switch_delta": int(decode[-1]["switch_count"]) - int(decode[0]["switch_count"]),
        "final_igpu_ratio": float(decode[-1]["igpu_ratio"]),
        "background_stopped": bool(sample.get("background_stopped", False)),
    }


def summarize(cycles: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    directions = {str(cycle["direction"]) for cycle in cycles}
    if len(directions) != 1:
        raise RuntimeError(f"cannot aggregate mixed transition directions: {sorted(directions)}")
    metrics = dict(COMMON_METRICS)
    if directions == {"low-to-high"}:
        metrics.update(LOW_TO_HIGH_METRICS)
    if all(cycle["client_pre_transition_tps"] is not None for cycle in cycles) and all(
        cycle["client_post_transition_tps"] is not None for cycle in cycles
    ):
        metrics.update(CLIENT_TOKEN_METRICS)
    summary = []
    for metric_index, (metric, unit) in enumerate(metrics.items()):
        values = [float(cycle[metric]) for cycle in cycles]
        low, high = bootstrap_mean_ci(values, bootstrap_samples, seed + metric_index)
        summary.append(
            {
                "metric": metric,
                "unit": unit,
                "n": len(values),
                "mean": statistics.fmean(values),
                "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, cycles: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    direction = str(cycles[0]["direction"])
    target = str(cycles[0]["target"])
    lines = [
        "# Dynamic Load Transition Cycles",
        "",
        f"- Cycles: {len(cycles)}",
        f"- Direction: {direction}",
        f"- Target: {target}",
        "- Pre-transition rate excludes the first prefill/decode boundary interval.",
        "- Confidence intervals are percentile bootstrap intervals of the cycle mean.",
        "",
        "## Cycle Results",
        "",
        "| Cycle | Decode tok/s | First target calls | First target ms | Settle calls | Settle ms | Pre calls/s | Post calls/s | Client pre tok/s | Client post tok/s | Exact token timestamps | Ready ms | Exploration calls | Switches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for cycle in cycles:
        ready = f"{cycle['background_ready_ms']:.2f}" if cycle["background_ready_ms"] is not None else "NA"
        client_pre = (
            f"{cycle['client_pre_transition_tps']:.3f}" if cycle["client_pre_transition_tps"] is not None else "NA"
        )
        client_post = (
            f"{cycle['client_post_transition_tps']:.3f}" if cycle["client_post_transition_tps"] is not None else "NA"
        )
        token_exact = (
            "yes"
            if cycle["token_timestamps_exact"] is True
            else "no" if cycle["token_timestamps_exact"] is False else "NA"
        )
        lines.append(
            f"| {cycle['cycle']} | {cycle['decode_tps']:.3f} | "
            f"{cycle['first_target_delay_calls']} | {cycle['first_target_delay_ms']:.2f} | "
            f"{cycle['settle_target_delay_calls']} | {cycle['settle_target_delay_ms']:.2f} | "
            f"{cycle['pre_transition_calls_per_s']:.3f} | "
            f"{cycle['post_transition_calls_per_s']:.3f} | {client_pre} | {client_post} | "
            f"{token_exact} | {ready} | "
            f"{cycle['exploration_calls']} | {cycle['switch_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Metric | N | Mean | Sample stdev | Bootstrap 95% CI | Unit |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['metric']} | {row['n']} | {row['mean']:.4f} | "
            f"{row['sample_stdev']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['unit']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [Cycle data](cycles.csv)",
            "- [Aggregate statistics](summary.csv)",
            "- [Manifest](manifest.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cycles = [load_cycle(path.resolve(), index) for index, path in enumerate(args.inputs, start=1)]
    summary = summarize(cycles, args.bootstrap_samples, args.seed)
    write_csv(args.output_dir / "cycles.csv", cycles)
    write_csv(args.output_dir / "summary.csv", summary)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "output_dir": str(args.output_dir),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "cycle_count": len(cycles),
        "direction": cycles[0]["direction"],
        "target": cycles[0]["target"],
        "exact_token_timestamp_cycles": sum(cycle["token_timestamps_exact"] is True for cycle in cycles),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", cycles, summary)
    print(f"Cycles: {len(cycles)}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
