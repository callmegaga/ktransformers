#!/usr/bin/env python3
"""Compare repeated steady-load sweep blocks with hierarchical bootstrap."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from report_blocked_strategy_comparison import (
    METRICS,
    compare,
    load_block,
    summarize,
    write_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def sweep_argument(value: str) -> tuple[str, Path]:
    try:
        strategy, path_text = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sweep must use STRATEGY=SWEEP_ARTIFACT_DIR") from error
    if not strategy.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("strategy and sweep artifact directory must be non-empty")
    return strategy.strip(), Path(path_text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        action="append",
        type=sweep_argument,
        required=True,
        help="Repeatable strategy sweep as STRATEGY=SWEEP_ARTIFACT_DIR",
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args(argv)
    strategies = {strategy for strategy, _path in args.sweep}
    if len(strategies) < 2:
        parser.error("sweeps must contain at least two strategies")
    if args.candidate not in strategies:
        parser.error("--candidate must match a sweep strategy")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def resolve_artifact_path(path_text: str, sweep_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    repo_relative = (REPO_ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    return (sweep_root / path).resolve()


def load_sweeps(
    arguments: list[tuple[str, Path]],
) -> tuple[
    dict[int, dict[str, list[list[dict[str, Any]]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    grouped: dict[int, dict[str, list[list[dict[str, Any]]]]] = {}
    inputs = []
    reference: dict[str, Any] | None = None
    strategy_blocks: dict[str, int] = {}
    for strategy, raw_root in arguments:
        root = raw_root.resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"sweep has no manifest.json: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"sweep is not complete: {root}")
        identity = {
            "protocol_version": manifest.get("protocol_version"),
            "requested_load_workers": manifest.get("requested_load_workers"),
            "workloads": manifest.get("arguments", {}).get("workloads"),
            "source_fingerprints": manifest.get("source_fingerprints"),
            "extension": manifest.get("server", {}).get("kt_kernel_extension"),
        }
        if reference is None:
            reference = identity
        elif identity != reference:
            raise RuntimeError(f"sweep protocol or source identity differs from the first input: {root}")
        block_index = strategy_blocks.get(strategy, 0) + 1
        strategy_blocks[strategy] = block_index
        seen_workers = set()
        for run in manifest.get("runs", []):
            if run.get("status") != "complete":
                raise RuntimeError(f"sweep contains an incomplete run: {root}")
            workers = int(run["load_workers"])
            if workers in seen_workers:
                raise RuntimeError(f"sweep repeats load_workers={workers}: {root}")
            seen_workers.add(workers)
            artifact = resolve_artifact_path(str(run["benchmark_output_dir"]), root)
            block = load_block(strategy, artifact, block_index)
            for sample in block:
                sample["load_workers"] = workers
                sample["load"] = str(run["load"])
                sample["sweep_artifact"] = str(root)
            grouped.setdefault(workers, {}).setdefault(strategy, []).append(block)
        expected_workers = set(int(item) for item in identity["requested_load_workers"])
        if seen_workers != expected_workers:
            raise RuntimeError(
                f"sweep worker set {sorted(seen_workers)} does not match "
                f"manifest {sorted(expected_workers)}: {root}"
            )
        inputs.append(
            {
                "strategy": strategy,
                "block": block_index,
                "artifact": str(root),
                "backend": manifest.get("backend"),
                "block_label": manifest.get("block_label"),
                "realized_load_order": manifest.get("realized_load_order"),
            }
        )
    assert reference is not None
    expected_strategies = set(strategy_blocks)
    for workers, by_strategy in grouped.items():
        if set(by_strategy) != expected_strategies:
            raise RuntimeError(f"load_workers={workers} does not contain all strategies: " f"{sorted(by_strategy)}")
    return grouped, inputs, reference


def analyze(
    grouped: dict[int, dict[str, list[list[dict[str, Any]]]]],
    candidate: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    normalized_samples = []
    block_rows = []
    summary_rows = []
    effect_rows = []
    oracle_rows = []
    for workers, by_strategy in sorted(grouped.items()):
        for blocks in by_strategy.values():
            for block in blocks:
                normalized_samples.extend(block)
        load_blocks, load_summary = summarize(by_strategy, bootstrap_samples, seed + workers)
        load_effects = compare(by_strategy, candidate, bootstrap_samples, seed + workers)
        block_rows.extend({"load_workers": workers, **row} for row in load_blocks)
        summary_rows.extend({"load_workers": workers, **row} for row in load_summary)
        effect_rows.extend({"load_workers": workers, **row} for row in load_effects)

        means = {(row["strategy"], row["metric"]): float(row["mean"]) for row in load_summary}
        baselines = [strategy for strategy in by_strategy if strategy != candidate]
        for metric, (unit, direction) in METRICS.items():
            if direction == "higher":
                oracle_strategy = max(baselines, key=lambda strategy: means[(strategy, metric)])
                oracle_value = means[(oracle_strategy, metric)]
                attainment = means[(candidate, metric)] / oracle_value * 100.0
            else:
                oracle_strategy = min(baselines, key=lambda strategy: means[(strategy, metric)])
                oracle_value = means[(oracle_strategy, metric)]
                attainment = oracle_value / means[(candidate, metric)] * 100.0
            oracle_rows.append(
                {
                    "load_workers": workers,
                    "candidate": candidate,
                    "metric": metric,
                    "unit": unit,
                    "candidate_mean": means[(candidate, metric)],
                    "static_oracle_strategy": oracle_strategy,
                    "static_oracle_mean": oracle_value,
                    "static_oracle_attainment_pct": attainment,
                }
            )
    return normalized_samples, block_rows, summary_rows, effect_rows, oracle_rows


def write_report(
    path: Path,
    candidate: str,
    block_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Steady-Load Sweep Comparison",
        "",
        "- Strategy means give equal weight to each sweep block.",
        "- Confidence intervals resample sweep blocks and then requests.",
        "- Positive effect percentages favor the candidate.",
        "- Static oracle means the better observed fixed baseline, not a theoretical oracle.",
        "",
        "## Block Means",
        "",
        "| Workers | Strategy | Block | N | Prefill tok/s | Decode tok/s | " "E2E ms | CPU busy | PSI some |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block_rows:
        lines.append(
            f"| {row['load_workers']} | {row['strategy']} | {row['block']} | "
            f"{row['n']} | {row['prefill_tps']:.3f} | {row['decode_tps']:.3f} | "
            f"{row['e2e_ms']:.2f} | {row['cpu_busy_fraction']:.4f} | "
            f"{row['cpu_psi_some_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Effects",
            "",
            "| Workers | Baseline | Metric | Effect | Hierarchical bootstrap 95% CI |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in effect_rows:
        if row["metric"] not in {"prefill_tps", "decode_tps", "e2e_ms"}:
            continue
        lines.append(
            f"| {row['load_workers']} | {row['baseline']} | {row['metric']} "
            f"{row['effect']} | {row['effect_pct']:+.2f}% | "
            f"[{row['ci95_low']:+.2f}%, {row['ci95_high']:+.2f}%] |"
        )
    lines.extend(
        [
            "",
            "## Static Oracle Attainment",
            "",
            "| Workers | Metric | Candidate | Static oracle | Oracle strategy | Attainment |",
            "|---:|---|---:|---:|---|---:|",
        ]
    )
    for row in oracle_rows:
        if row["metric"] not in {"prefill_tps", "decode_tps", "e2e_ms"}:
            continue
        lines.append(
            f"| {row['load_workers']} | {row['metric']} | "
            f"{row['candidate_mean']:.3f} | {row['static_oracle_mean']:.3f} | "
            f"{row['static_oracle_strategy']} | "
            f"{row['static_oracle_attainment_pct']:.2f}% |"
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
            "- [Static oracle attainment](oracle.csv)",
            "- [Manifest](manifest.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    inputs: list[dict[str, Any]],
    reference: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "candidate": args.candidate,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "inputs": inputs,
                "validated_identity": reference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    grouped, inputs, reference = load_sweeps(args.sweep)
    samples, blocks, summary_rows, effects, oracle = analyze(grouped, args.candidate, args.bootstrap_samples, args.seed)
    write_csv(args.output_dir / "samples.csv", samples)
    write_csv(args.output_dir / "blocks.csv", blocks)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "effects.csv", effects)
    write_csv(args.output_dir / "oracle.csv", oracle)
    write_manifest(args.output_dir / "manifest.json", args, inputs, reference)
    write_report(args.output_dir / "report.md", args.candidate, blocks, effects, oracle)
    print(f"Loads: {len(grouped)}")
    print(f"Strategies: {len({item['strategy'] for item in inputs})}")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
