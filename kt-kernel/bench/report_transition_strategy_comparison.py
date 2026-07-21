#!/usr/bin/env python3
"""Compare fixed and dynamic load-transition benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from bench_running_server import bootstrap_mean_ci, percentile, stable_seed

METRICS = {
    "prefill_tps": ("token/s", "higher"),
    "decode_tps": ("token/s", "higher"),
    "ttft_ms": ("ms", "lower"),
    "tpot_ms": ("ms/token", "lower"),
    "e2e_ms": ("ms", "lower"),
    "client_pre_transition_tps": ("token/s", "higher"),
    "client_post_transition_tps": ("token/s", "higher"),
    "background_ready_ms": ("ms", "none"),
}


def strategy_argument(value: str) -> tuple[str, Path]:
    try:
        label, path_text = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("strategy must use LABEL=ARTIFACT_DIR") from error
    if not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("strategy label and artifact directory must be non-empty")
    return label.strip(), Path(path_text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        action="append",
        type=strategy_argument,
        required=True,
        help="Strategy and artifact directory as LABEL=ARTIFACT_DIR",
    )
    parser.add_argument("--candidate", required=True, help="Strategy used for effect estimates")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)
    labels = [label for label, _path in args.strategy]
    if len(labels) < 2:
        parser.error("at least two --strategy arguments are required")
    if len(labels) != len(set(labels)):
        parser.error("strategy labels must be unique")
    if args.candidate not in labels:
        parser.error("--candidate must match a --strategy label")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def require_float(row: dict[str, Any], key: str, path: Path) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise RuntimeError(f"missing {key} in {path}")
    return float(value)


def normalized_row(row: dict[str, Any], path: Path, source: str) -> dict[str, Any]:
    completion_tokens = int(row["completion_tokens"])
    stream_tokens_value = row.get("stream_tokens")
    if stream_tokens_value in (None, ""):
        raise RuntimeError(f"missing stream_tokens in {path}")
    stream_tokens = int(stream_tokens_value)
    if stream_tokens != completion_tokens:
        raise RuntimeError(f"stream token timestamps do not match completion tokens in {path}")
    ready_key = "background_ready_ms" if source == "cycles" else "background_ready_delay_ms"
    return {
        "prefill_tps": require_float(row, "prefill_tps", path),
        "decode_tps": require_float(row, "decode_tps", path),
        "ttft_ms": require_float(row, "ttft_ms", path),
        "tpot_ms": require_float(row, "tpot_ms", path),
        "e2e_ms": require_float(row, "e2e_ms", path),
        "client_pre_transition_tps": require_float(
            row,
            "client_pre_transition_tps" if source == "cycles" else "transition_client_pre_tps",
            path,
        ),
        "client_post_transition_tps": require_float(
            row,
            "client_post_transition_tps" if source == "cycles" else "transition_client_post_tps",
            path,
        ),
        "background_ready_ms": require_float(row, ready_key, path),
        "completion_tokens": completion_tokens,
        "token_timestamps_exact": (
            row.get("token_timestamps_exact") in (True, "True")
            if source == "cycles"
            else stream_tokens == completion_tokens
        ),
    }


def load_strategy(label: str, path: Path) -> list[dict[str, Any]]:
    cycles_path = path / "cycles.csv"
    samples_path = path / "samples.jsonl"
    if cycles_path.is_file():
        with cycles_path.open(encoding="utf-8", newline="") as source:
            rows = [normalized_row(row, path, "cycles") for row in csv.DictReader(source)]
    elif samples_path.is_file():
        raw_rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failed = [row for row in raw_rows if row.get("status") != "ok"]
        if failed:
            raise RuntimeError(f"strategy {label} contains unsuccessful samples in {path}")
        rows = [normalized_row(row, path, "samples") for row in raw_rows]
    else:
        raise RuntimeError(f"strategy {label} has no cycles.csv or samples.jsonl in {path}")
    if not rows:
        raise RuntimeError(f"strategy {label} has no samples in {path}")
    if not all(row["token_timestamps_exact"] for row in rows):
        raise RuntimeError(f"strategy {label} contains inexact token timestamps in {path}")
    return rows


def summarize_strategies(
    strategies: dict[str, list[dict[str, Any]]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    summary = []
    for label, rows in strategies.items():
        for metric, (unit, _direction) in METRICS.items():
            values = [float(row[metric]) for row in rows]
            low, high = bootstrap_mean_ci(
                values,
                bootstrap_samples,
                stable_seed(seed, label, metric),
            )
            summary.append(
                {
                    "strategy": label,
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


def bootstrap_effect_ci(
    candidate: list[float],
    baseline: list[float],
    effect: Callable[[float, float], float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    effects = []
    for _ in range(samples):
        candidate_mean = statistics.fmean(generator.choice(candidate) for _ in candidate)
        baseline_mean = statistics.fmean(generator.choice(baseline) for _ in baseline)
        effects.append(effect(candidate_mean, baseline_mean))
    return percentile(effects, 0.025), percentile(effects, 0.975)


def compare_candidate(
    strategies: dict[str, list[dict[str, Any]]],
    candidate_label: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidate_rows = strategies[candidate_label]
    effects = []
    for baseline_label, baseline_rows in strategies.items():
        if baseline_label == candidate_label:
            continue
        for metric, (unit, direction) in METRICS.items():
            if direction == "none":
                continue
            candidate_values = [float(row[metric]) for row in candidate_rows]
            baseline_values = [float(row[metric]) for row in baseline_rows]
            if direction == "higher":
                effect = lambda candidate, baseline: (candidate / baseline - 1.0) * 100.0
                effect_name = "improvement"
            else:
                effect = lambda candidate, baseline: (1.0 - candidate / baseline) * 100.0
                effect_name = "reduction"
            point = effect(statistics.fmean(candidate_values), statistics.fmean(baseline_values))
            low, high = bootstrap_effect_ci(
                candidate_values,
                baseline_values,
                effect,
                bootstrap_samples,
                stable_seed(seed, candidate_label, baseline_label, metric),
            )
            effects.append(
                {
                    "candidate": candidate_label,
                    "baseline": baseline_label,
                    "metric": metric,
                    "unit": unit,
                    "effect": effect_name,
                    "effect_pct": point,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return effects


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_ci(row: dict[str, Any]) -> str:
    return f"{row['mean']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"


def write_report(
    path: Path,
    labels: list[str],
    summary: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    candidate: str,
) -> None:
    indexed = {(row["strategy"], row["metric"]): row for row in summary}
    lines = [
        "# Load Transition Strategy Comparison",
        "",
        f"- Candidate: {candidate}",
        "- Confidence intervals use independent percentile bootstrap resampling of strategy means.",
        "- Positive effect percentages favor the candidate.",
        "",
        "## Strategy Results",
        "",
        "| Strategy | N | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms/token | "
        "E2E ms | Client pre tok/s | Client post tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        decode = indexed[(label, "decode_tps")]
        lines.append(
            f"| {label} | {decode['n']} | {format_ci(indexed[(label, 'prefill_tps')])} | "
            f"{format_ci(decode)} | {format_ci(indexed[(label, 'ttft_ms')])} | "
            f"{format_ci(indexed[(label, 'tpot_ms')])} | {format_ci(indexed[(label, 'e2e_ms')])} | "
            f"{format_ci(indexed[(label, 'client_pre_transition_tps')])} | "
            f"{format_ci(indexed[(label, 'client_post_transition_tps')])} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Effects",
            "",
            "| Baseline | Metric | Effect | Bootstrap 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['baseline']} | {row['metric']} {row['effect']} | "
            f"{row['effect_pct']:+.2f}% | [{row['ci95_low']:+.2f}%, {row['ci95_high']:+.2f}%] |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [Normalized samples](samples.csv)",
            "- [Strategy statistics](summary.csv)",
            "- [Candidate effects](effects.csv)",
            "- [Manifest](manifest.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {label: path.resolve() for label, path in args.strategy}
    strategies = {label: load_strategy(label, path) for label, path in inputs.items()}
    summary = summarize_strategies(strategies, args.bootstrap_samples, args.seed)
    effects = compare_candidate(strategies, args.candidate, args.bootstrap_samples, args.seed)
    normalized = [
        {"strategy": label, "sample": index, **row}
        for label, rows in strategies.items()
        for index, row in enumerate(rows, start=1)
    ]
    write_csv(args.output_dir / "samples.csv", normalized)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "effects.csv", effects)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "inputs": {label: str(path) for label, path in inputs.items()},
        "candidate": args.candidate,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "sample_counts": {label: len(rows) for label, rows in strategies.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(
        args.output_dir / "report.md",
        list(strategies),
        summary,
        effects,
        args.candidate,
    )
    print(f"Strategies: {len(strategies)}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
