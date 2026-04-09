from __future__ import annotations

from kt_kernel.eval.result_schema import EvaluationRun
from kt_kernel.eval.task_registry import TASK_SPECS


def compare_runs(baseline: EvaluationRun, candidate: EvaluationRun) -> list[dict]:
    rows = []
    for task in baseline.results:
        if task not in candidate.results:
            raise ValueError(f"Missing task in candidate results: {task}")
        base_result = baseline.results[task]
        cand_result = candidate.results[task]
        if base_result.metric != cand_result.metric:
            raise ValueError(f"Mismatched metric for task {task}")

        baseline_value = float(base_result.value)
        candidate_value = float(cand_result.value)
        abs_delta = round(candidate_value - baseline_value, 10)
        rel_delta_pct = 0.0 if baseline_value == 0 else round(abs_delta / baseline_value * 100.0, 10)

        rows.append(
            {
                "task": task,
                "metric": base_result.metric,
                "direction": TASK_SPECS[task].direction,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "abs_delta": abs_delta,
                "rel_delta_pct": rel_delta_pct,
            }
        )
    return rows
