#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.eval.result_schema import EvaluationRun, MetricResult, RunMeta
from kt_kernel.eval.task_registry import TASK_SPECS, resolve_tasks


def get_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def run_selected_tasks(
    base_url: str,
    model_name: str,
    tasks: list[str],
    limit: int | None = None,
    max_samples_per_task: int | None = None,
    timeout: int = 120,
) -> dict:
    ppl_tasks = [task for task in tasks if TASK_SPECS[task].kind == "ppl"]
    lm_eval_tasks = [task for task in tasks if TASK_SPECS[task].kind == "lm_eval"]
    results = {}

    if ppl_tasks:
        from kt_kernel.eval.ppl import run_ppl_tasks

        results.update(
            run_ppl_tasks(
                base_url=base_url,
                model_name=model_name,
                tasks=ppl_tasks,
                max_samples_per_task=max_samples_per_task,
                timeout=timeout,
            )
        )

    if lm_eval_tasks:
        from kt_kernel.eval.lm_eval_runner import run_lm_eval_tasks

        results.update(
            run_lm_eval_tasks(
                base_url=base_url,
                model_name=model_name,
                tasks=lm_eval_tasks,
                limit=limit,
            )
        )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--task-group", default="core")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-samples-per-task", type=int)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    tasks = resolve_tasks(task_group=args.task_group, tasks_arg=args.tasks)
    raw_results = run_selected_tasks(
        base_url=args.base_url,
        model_name=args.model_name,
        tasks=tasks,
        limit=args.limit,
        max_samples_per_task=args.max_samples_per_task,
        timeout=args.timeout,
    )
    run = EvaluationRun(
        schema_version=1,
        run_meta=RunMeta(
            run_id=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.label}",
            label=args.label,
            model_name=args.model_name,
            base_url=args.base_url,
            created_at=datetime.now().astimezone().isoformat(),
            git_commit=get_git_commit(),
            tasks=tasks,
            task_group=args.task_group,
        ),
        results={name: MetricResult(**result) for name, result in raw_results.items()},
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
