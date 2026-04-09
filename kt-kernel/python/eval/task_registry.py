from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    metric: str
    direction: str
    fewshot: int


TASK_SPECS = {
    "wikitext": TaskSpec("wikitext", "ppl", "perplexity", "lower_is_better", 0),
    "c4": TaskSpec("c4", "ppl", "perplexity", "lower_is_better", 0),
    "hellaswag": TaskSpec("hellaswag", "lm_eval", "accuracy", "higher_is_better", 0),
    "arc_challenge": TaskSpec("arc_challenge", "lm_eval", "accuracy", "higher_is_better", 0),
    "winogrande": TaskSpec("winogrande", "lm_eval", "accuracy", "higher_is_better", 0),
    "piqa": TaskSpec("piqa", "lm_eval", "accuracy", "higher_is_better", 0),
    "mmlu_pro": TaskSpec("mmlu_pro", "lm_eval", "accuracy", "higher_is_better", 5),
    "gsm8k": TaskSpec("gsm8k", "lm_eval", "exact_match", "higher_is_better", 8),
    "ifeval": TaskSpec("ifeval", "lm_eval", "primary", "higher_is_better", 0),
}

CORE_TASK_GROUPS = {
    "core": [
        "wikitext",
        "c4",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "piqa",
        "mmlu_pro",
        "gsm8k",
        "ifeval",
    ]
}


def resolve_tasks(task_group: str | None, tasks_arg: str | None) -> list[str]:
    if tasks_arg:
        return [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if task_group:
        return CORE_TASK_GROUPS[task_group]
    return CORE_TASK_GROUPS["core"]
