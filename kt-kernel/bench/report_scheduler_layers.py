#!/usr/bin/env python3
"""Summarize per-layer CPU/iGPU scheduler telemetry from a benchmark artifact."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def execution_calls(event: dict[str, Any]) -> int:
    return max(0, int(event.get("execution_calls_delta", 1)))


def weighted_mean(events: list[dict[str, Any]], key: str) -> float | None:
    weighted = [
        (float(event[key]), execution_calls(event))
        for event in events
        if event.get(key) is not None and execution_calls(event) > 0
    ]
    calls = sum(weight for _value, weight in weighted)
    if calls == 0:
        return None
    return sum(value * weight for value, weight in weighted) / calls


def last_value(events: list[dict[str, Any]], key: str) -> float | None:
    for event in reversed(events):
        if event.get(key) is not None:
            return float(event[key])
    return None


def format_optional(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def switch_delta(events: list[dict[str, Any]]) -> int:
    by_request: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_request[int(event.get("request_index", 0))].append(event)
    return sum(
        int(request_events[-1]["switch_count"]) - int(request_events[0]["switch_count"])
        for request_events in by_request.values()
        if request_events
    )


def load_events(artifact: Path) -> list[dict[str, Any]]:
    path = artifact / "scheduler-telemetry.jsonl"
    if not path.is_file():
        raise RuntimeError(f"artifact has no scheduler-telemetry.jsonl: {artifact}")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events:
        raise RuntimeError(f"artifact has no scheduler events: {artifact}")
    missing = [event for event in events if event.get("layer") is None]
    if missing:
        raise RuntimeError("scheduler events are missing layer identifiers")
    return events


def summarize_layers(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[int(event["layer"])].append(event)
    rows = []
    for layer, layer_events in sorted(grouped.items()):
        ordered = sorted(layer_events, key=lambda event: int(event["monotonic_ns"]))
        prefill = [event for event in ordered if event.get("phase") == "prefill"]
        decode = [event for event in ordered if event.get("phase") == "decode" and execution_calls(event) > 0]
        if not decode:
            raise RuntimeError(f"layer {layer} has no decode execution events")
        final = decode[-1]
        prefill_policy_igpu_ratio = weighted_mean(prefill, "policy_igpu_ratio")
        if prefill_policy_igpu_ratio is None:
            prefill_policy_igpu_ratio = last_value(prefill, "policy_igpu_ratio")
        rows.append(
            {
                "layer": layer,
                "prefill_calls": sum(execution_calls(event) for event in prefill),
                "prefill_igpu_ratio": weighted_mean(prefill, "igpu_ratio"),
                "prefill_policy_igpu_ratio": prefill_policy_igpu_ratio,
                "decode_events": len(decode),
                "decode_calls": sum(execution_calls(event) for event in decode),
                "decode_igpu_ratio": weighted_mean(decode, "igpu_ratio"),
                "decode_policy_igpu_ratio": weighted_mean(decode, "policy_igpu_ratio"),
                "decode_cpu_load": weighted_mean(decode, "cpu_load"),
                "exploration_calls": sum(
                    execution_calls(event) for event in decode if bool(event.get("exploration", False))
                ),
                "load_drop_reprobe_calls": sum(
                    execution_calls(event) for event in decode if int(event.get("reprobe_reason", 0)) == 1
                ),
                "periodic_reprobe_calls": sum(
                    execution_calls(event) for event in decode if int(event.get("reprobe_reason", 0)) == 2
                ),
                "switch_delta": switch_delta(decode),
                "final_cpu_ms_per_row": float(final["cpu_ms_per_row"]),
                "final_igpu_ms_per_row": float(final["igpu_ms_per_row"]),
                "final_cpu_load": float(final["cpu_load"]),
                "final_igpu_ratio": float(final["igpu_ratio"]),
                "final_policy_igpu_ratio": float(final["policy_igpu_ratio"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fully_igpu = sum(
        float(row["decode_igpu_ratio"]) >= 0.999 and float(row["decode_policy_igpu_ratio"]) >= 0.999 for row in rows
    )
    fully_cpu_prefill = sum(
        row["prefill_igpu_ratio"] is not None and float(row["prefill_igpu_ratio"]) <= 0.001 for row in rows
    )
    observed_prefill = sum(row["prefill_igpu_ratio"] is not None for row in rows)
    cpu_policy_prefill = sum(
        row["prefill_policy_igpu_ratio"] is not None and float(row["prefill_policy_igpu_ratio"]) <= 0.001
        for row in rows
    )
    fully_cpu_decode = sum(
        float(row["decode_igpu_ratio"]) <= 0.001 and float(row["decode_policy_igpu_ratio"]) <= 0.001 for row in rows
    )
    exploration_calls = sum(int(row["exploration_calls"]) for row in rows)
    reprobe_calls = sum(int(row["load_drop_reprobe_calls"]) + int(row["periodic_reprobe_calls"]) for row in rows)
    switches = sum(int(row["switch_delta"]) for row in rows)
    lines = [
        "# Per-Layer Scheduler Telemetry",
        "",
        f"- Layers: {len(rows)}",
        f"- Decode fully iGPU layers: {fully_igpu} / {len(rows)}",
        f"- Decode fully CPU layers: {fully_cpu_decode} / {len(rows)}",
        f"- Prefill fully CPU layers: {fully_cpu_prefill} / {observed_prefill} "
        f"observed (coverage {observed_prefill} / {len(rows)})",
        f"- Prefill CPU-policy layers: {cpu_policy_prefill} / {len(rows)}",
        f"- Decode exploration calls: {exploration_calls}",
        f"- Decode reprobe calls: {reprobe_calls}",
        f"- Request-window switch delta across layers: {switches}",
        "",
        "## Layer Results",
        "",
        "| Layer | Prefill iGPU | Decode calls | Decode iGPU | Policy iGPU | CPU load | "
        "Explore | Reprobe | Switches | Final CPU ms/row | Final iGPU ms/row |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reprobes = int(row["load_drop_reprobe_calls"]) + int(row["periodic_reprobe_calls"])
        lines.append(
            f"| {row['layer']} | {format_optional(row['prefill_igpu_ratio'])} | "
            f"{row['decode_calls']} | {float(row['decode_igpu_ratio']):.3f} | "
            f"{float(row['decode_policy_igpu_ratio']):.3f} | "
            f"{float(row['decode_cpu_load']):.3f} | {row['exploration_calls']} | "
            f"{reprobes} | {row['switch_delta']} | "
            f"{float(row['final_cpu_ms_per_row']):.5f} | "
            f"{float(row['final_igpu_ms_per_row']):.5f} |"
        )
    cpu_costs = [float(row["final_cpu_ms_per_row"]) for row in rows]
    igpu_costs = [float(row["final_igpu_ms_per_row"]) for row in rows]
    lines.extend(
        [
            "",
            "## Cost Snapshot",
            "",
            f"- Mean final CPU service cost: {statistics.fmean(cpu_costs):.5f} ms/row",
            f"- Mean final iGPU service cost: {statistics.fmean(igpu_costs):.5f} ms/row",
            f"- Layers with lower final iGPU cost: "
            f"{sum(igpu < cpu for cpu, igpu in zip(cpu_costs, igpu_costs))} / {len(rows)}",
            "",
            "## Files",
            "",
            "- [Layer data](layers.csv)",
            "- [Manifest](manifest.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = args.artifact.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    events = load_events(artifact)
    rows = summarize_layers(events)
    write_csv(args.output_dir / "layers.csv", rows)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "artifact": str(artifact),
        "event_count": len(events),
        "layer_count": len(rows),
        "request_indices": sorted({int(event.get("request_index", 0)) for event in events}),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", rows)
    print(f"Layers: {len(rows)}")
    print(f"Events: {len(events)}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
