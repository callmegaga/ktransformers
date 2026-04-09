from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(eq=True)
class MetricResult:
    metric: str
    value: float
    samples: int
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=True)
class RunMeta:
    run_id: str
    label: str
    model_name: str
    base_url: str
    created_at: str
    git_commit: str
    tasks: list[str]
    task_group: str | None = None


@dataclass(eq=True)
class EvaluationRun:
    schema_version: int
    run_meta: RunMeta
    results: dict[str, MetricResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationRun":
        run_meta = RunMeta(**payload["run_meta"])
        results = {name: MetricResult(**result) for name, result in payload["results"].items()}
        return cls(
            schema_version=payload["schema_version"],
            run_meta=run_meta,
            results=results,
        )
