#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.compare import compare_runs
from kt_kernel.eval.result_schema import EvaluationRun


def load_run(path: str) -> EvaluationRun:
    with open(path, "r", encoding="utf-8") as f:
        return EvaluationRun.from_dict(json.load(f))


def write_json(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = ["task", "metric", "direction", "baseline", "candidate", "abs_delta", "rel_delta_pct"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args(argv)

    rows = compare_runs(load_run(args.baseline), load_run(args.candidate))
    for row in rows:
        print(
            f"{row['task']}\t{row['metric']}\t{row['direction']}\t"
            f"{row['baseline']:.6f}\t{row['candidate']:.6f}\t"
            f"{row['abs_delta']:+.6f}\t{row['rel_delta_pct']:+.2f}%"
        )
    if args.output_json:
        write_json(args.output_json, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
