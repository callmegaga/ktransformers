from __future__ import annotations

from kt_kernel.eval.task_registry import TASK_SPECS

LM_EVAL_API_MAX_PREFILL_TOKENS = 16000
LM_EVAL_DEFAULT_MAX_GEN_TOKS = 256
LM_EVAL_TASK_MAX_GEN_TOKS = {
    "ifeval": 1280,
    "mmlu_pro": 2048,
}


def _is_numeric_metric(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _task_max_gen_toks(task: str) -> int:
    return LM_EVAL_TASK_MAX_GEN_TOKS.get(task, LM_EVAL_DEFAULT_MAX_GEN_TOKS)


def _build_model_args(
    *,
    base_url: str,
    model_name: str,
    task: str,
    num_concurrent: int,
) -> dict:
    max_gen_toks = _task_max_gen_toks(task)
    return {
        "model": model_name,
        "base_url": base_url.rstrip("/") + "/completions",
        "num_concurrent": num_concurrent,
        "max_gen_toks": max_gen_toks,
        # lm-eval's API model defaults to max_length=2048, which truncates long tasks
        # before they reach the server. Match the 30b build scripts' prefill budget instead.
        "max_length": LM_EVAL_API_MAX_PREFILL_TOKENS + max_gen_toks + 1,
    }


def _select_metric(metrics: dict, task: str) -> tuple[str, float]:
    preferred_metric = TASK_SPECS[task].metric
    metric_names = [name for name, value in metrics.items() if "stderr" not in name and _is_numeric_metric(value)]

    if preferred_metric == "primary":
        metric_name = metric_names[0]
        return metric_name, float(metrics[metric_name])

    prefixes = {
        "accuracy": ("acc", "accuracy"),
        "exact_match": ("exact_match", "em"),
    }.get(preferred_metric, (preferred_metric,))

    for metric_name in metric_names:
        if metric_name == preferred_metric or any(metric_name.startswith(prefix) for prefix in prefixes):
            return metric_name, float(metrics[metric_name])

    metric_name = metric_names[0]
    return metric_name, float(metrics[metric_name])


def normalize_lm_eval_results(raw_results: dict, requested_tasks: list[str]) -> dict:
    normalized = {}
    for task in requested_tasks:
        metrics = raw_results["results"][task]
        metric_name, metric_value = _select_metric(metrics, task)
        normalized[task] = {
            "metric": metric_name,
            "value": float(metric_value),
            "samples": raw_results.get("n-samples", {}).get(task, 0),
            "notes": {},
        }
    return normalized


def run_lm_eval_tasks(
    base_url: str,
    model_name: str,
    tasks: list[str],
    limit: int | None = None,
    num_concurrent: int = 1,
) -> dict:
    import lm_eval

    results = {}
    for task in tasks:
        raw_results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=_build_model_args(
                base_url=base_url,
                model_name=model_name,
                task=task,
                num_concurrent=num_concurrent,
            ),
            tasks=[task],
            num_fewshot=TASK_SPECS[task].fewshot,
            limit=limit,
            batch_size="auto",
            gen_kwargs={"temperature": 0, "top_p": 1},
        )
        results.update(normalize_lm_eval_results(raw_results, [task]))
    return results
