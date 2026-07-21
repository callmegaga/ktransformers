#!/usr/bin/env python3
"""Compare repeated running-server strategy blocks with hierarchical bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from bench_running_server import percentile, stable_seed

METRICS = {
    "prefill_tps": ("token/s", "higher"),
    "decode_tps": ("token/s", "higher"),
    "ttft_ms": ("ms", "lower"),
    "tpot_ms": ("ms/token", "lower"),
    "e2e_ms": ("ms", "lower"),
}


def block_argument(value: str) -> tuple[str, Path]:
    try:
        strategy, path_text = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("block must use STRATEGY=ARTIFACT_DIR") from error
    if not strategy.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("strategy and artifact directory must be non-empty")
    return strategy.strip(), Path(path_text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        action="append",
        type=block_argument,
        required=True,
        help="Repeatable strategy block as STRATEGY=ARTIFACT_DIR",
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)
    strategies = {strategy for strategy, _path in args.block}
    if len(strategies) < 2:
        parser.error("blocks must contain at least two strategies")
    if args.candidate not in strategies:
        parser.error("--candidate must match a block strategy")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def load_block(strategy: str, path: Path, block_index: int) -> list[dict[str, Any]]:
    samples_path = path / "samples.jsonl"
    if not samples_path.is_file():
        raise RuntimeError(f"block has no samples.jsonl: {path}")
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise RuntimeError(f"block contains no samples or unsuccessful samples: {path}")
    normalized = []
    for sample_index, row in enumerate(rows, start=1):
        completion_tokens = int(row["completion_tokens"])
        stream_tokens = int(row["stream_tokens"])
        if stream_tokens != completion_tokens:
            raise RuntimeError(f"inexact token timestamps in {path}")
        normalized.append(
            {
                "strategy": strategy,
                "block": block_index,
                "sample": sample_index,
                "artifact": str(path),
                **{metric: float(row[metric]) for metric in METRICS},
                "cpu_busy_fraction": float(row["cpu_busy_fraction"]),
                "cpu_psi_some_fraction": float(row["cpu_psi_some_fraction"]),
                "completion_tokens": completion_tokens,
            }
        )
    return normalized


def resample_strategy_mean(blocks: list[list[float]], generator: random.Random) -> float:
    selected_blocks = [generator.choice(blocks) for _ in blocks]
    return statistics.fmean(statistics.fmean(generator.choice(block) for _ in block) for block in selected_blocks)


def hierarchical_mean_ci(blocks: list[list[float]], samples: int, seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [resample_strategy_mean(blocks, generator) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def group_metric_blocks(grouped: dict[str, list[list[dict[str, Any]]]], metric: str) -> dict[str, list[list[float]]]:
    return {
        strategy: [[float(sample[metric]) for sample in block] for block in blocks]
        for strategy, blocks in grouped.items()
    }


def summarize(
    grouped: dict[str, list[list[dict[str, Any]]]], samples: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block_rows = []
    summary_rows = []
    for strategy, blocks in grouped.items():
        for block_index, block in enumerate(blocks, start=1):
            block_rows.append(
                {
                    "strategy": strategy,
                    "block": block_index,
                    "n": len(block),
                    **{metric: statistics.fmean(float(row[metric]) for row in block) for metric in METRICS},
                    "cpu_busy_fraction": statistics.fmean(float(row["cpu_busy_fraction"]) for row in block),
                    "cpu_psi_some_fraction": statistics.fmean(float(row["cpu_psi_some_fraction"]) for row in block),
                }
            )
        for metric_index, (metric, (unit, _direction)) in enumerate(METRICS.items()):
            metric_blocks = [[float(row[metric]) for row in block] for block in blocks]
            block_means = [statistics.fmean(block) for block in metric_blocks]
            low, high = hierarchical_mean_ci(
                metric_blocks,
                samples,
                stable_seed(seed, strategy, metric, metric_index),
            )
            summary_rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "unit": unit,
                    "blocks": len(blocks),
                    "samples": sum(len(block) for block in blocks),
                    "mean": statistics.fmean(block_means),
                    "block_stdev": (statistics.stdev(block_means) if len(block_means) > 1 else 0.0),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return block_rows, summary_rows


def compare(
    grouped: dict[str, list[list[dict[str, Any]]]],
    candidate: str,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    effects = []
    for baseline in grouped:
        if baseline == candidate:
            continue
        for metric_index, (metric, (unit, direction)) in enumerate(METRICS.items()):
            metric_blocks = group_metric_blocks(grouped, metric)
            candidate_blocks = metric_blocks[candidate]
            baseline_blocks = metric_blocks[baseline]
            candidate_mean = statistics.fmean(statistics.fmean(block) for block in candidate_blocks)
            baseline_mean = statistics.fmean(statistics.fmean(block) for block in baseline_blocks)
            if direction == "higher":
                effect: Callable[[float, float], float] = (
                    lambda candidate_value, baseline_value: (candidate_value / baseline_value - 1.0) * 100.0
                )
                effect_name = "improvement"
            else:
                effect = lambda candidate_value, baseline_value: (1.0 - candidate_value / baseline_value) * 100.0
                effect_name = "reduction"
            generator = random.Random(stable_seed(seed, candidate, baseline, metric, metric_index))
            bootstrap_effects = []
            for _ in range(samples):
                candidate_sample = resample_strategy_mean(candidate_blocks, generator)
                baseline_sample = resample_strategy_mean(baseline_blocks, generator)
                bootstrap_effects.append(effect(candidate_sample, baseline_sample))
            effects.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "unit": unit,
                    "effect": effect_name,
                    "effect_pct": effect(candidate_mean, baseline_mean),
                    "ci95_low": percentile(bootstrap_effects, 0.025),
                    "ci95_high": percentile(bootstrap_effects, 0.975),
                }
            )
    return effects


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    block_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> None:
    lines = [
        "# Blocked Strategy Comparison",
        "",
        "- Strategy means give equal weight to each block.",
        "- Confidence intervals resample blocks and then requests within selected blocks.",
        "- Positive effect percentages favor the candidate.",
        "",
        "## Block Means",
        "",
        "| Strategy | Block | N | Prefill tok/s | Decode tok/s | TTFT ms | "
        "TPOT ms | E2E ms | CPU busy | CPU PSI some |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block_rows:
        lines.append(
            f"| {row['strategy']} | {row['block']} | {row['n']} | "
            f"{row['prefill_tps']:.3f} | {row['decode_tps']:.3f} | "
            f"{row['ttft_ms']:.2f} | {row['tpot_ms']:.2f} | {row['e2e_ms']:.2f} | "
            f"{row['cpu_busy_fraction']:.4f} | {row['cpu_psi_some_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Strategy Statistics",
            "",
            "| Strategy | Metric | Blocks | Samples | Mean | Block stdev | " "Hierarchical bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['strategy']} | {row['metric']} | {row['blocks']} | "
            f"{row['samples']} | {row['mean']:.4f} | {row['block_stdev']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Candidate Effects",
            "",
            "| Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['baseline']} | {row['metric']} {row['effect']} | "
            f"{row['effect_pct']:+.2f}% | "
            f"[{row['ci95_low']:+.2f}%, {row['ci95_high']:+.2f}%] |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [Normalized samples](samples.csv)",
            "- [Block means](blocks.csv)",
            "- [Strategy statistics](summary.csv)",
            "- [Candidate effects](effects.csv)",
            "- [Manifest](manifest.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    grouped: dict[str, list[list[dict[str, Any]]]] = {}
    inputs = []
    for strategy, raw_path in args.block:
        path = raw_path.resolve()
        block_index = len(grouped.setdefault(strategy, [])) + 1
        block = load_block(strategy, path, block_index)
        grouped[strategy].append(block)
        inputs.append({"strategy": strategy, "artifact": str(path)})
    block_rows, summary_rows = summarize(grouped, args.bootstrap_samples, args.seed)
    effects = compare(grouped, args.candidate, args.bootstrap_samples, args.seed)
    sample_rows = [sample for blocks in grouped.values() for block in blocks for sample in block]
    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "blocks.csv", block_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "effects.csv", effects)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "inputs": inputs,
        "candidate": args.candidate,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "block_counts": {strategy: len(blocks) for strategy, blocks in grouped.items()},
        "sample_counts": {strategy: sum(len(block) for block in blocks) for strategy, blocks in grouped.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", block_rows, summary_rows, effects)
    print(f"Strategies: {len(grouped)}")
    print(f"Blocks: {sum(len(blocks) for blocks in grouped.values())}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
