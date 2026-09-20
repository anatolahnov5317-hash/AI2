"""Predeclared Block 1 quality gate for open mention/coreference learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GATE_SCHEMA = "ai2-block1-quality-gate-v1"


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class Block1GateConfig:
    max_unsupported_gold_rate: float = 0.05
    min_mention_f1_delta_from_surface_baseline: float = 0.0
    require_validation_link_gate: bool = True
    min_validation_link_precision: float = 0.90
    min_validation_evaluable_links: int = 10
    min_test_accepted_link_precision: float = 0.80
    min_test_evaluable_links: int = 10
    min_oracle_coreference_f1_delta_from_baseline: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_unsupported_gold_rate",
            "min_validation_link_precision",
            "min_test_accepted_link_precision",
        ):
            _number(getattr(self, name), name)
        for name in (
            "min_mention_f1_delta_from_surface_baseline",
            "min_oracle_coreference_f1_delta_from_baseline",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        for name in (
            "min_validation_evaluable_links",
            "min_test_evaluable_links",
        ):
            _integer(getattr(self, name), name)
        if type(self.require_validation_link_gate) is not bool:
            raise ValueError("require_validation_link_gate must be boolean")


def evaluate_block1_gate(
    policy: dict[str, Any],
    metrics: dict[str, Any],
    config: Block1GateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate predeclared criteria without changing model or thresholds."""

    config = config or Block1GateConfig()
    mentions = metrics["mentions"]
    baseline = metrics["baseline"]
    link_gate = policy["link_gate"]
    accepted = metrics["accepted_links"]
    oracle = metrics["coreference_oracle_mentions"]
    oracle_baseline = baseline["coreference_oracle_mentions"]

    gold_count = _integer(mentions["gold_count"], "mention gold_count")
    unsupported = _integer(mentions["unsupported_gold_spans"], "unsupported_gold_spans")
    unsupported_rate = unsupported / gold_count if gold_count else 1.0
    mention_delta = float(metrics["comparison"]["mention_f1_delta_from_baseline"])
    oracle_delta = float(oracle["f1"]) - float(oracle_baseline["f1"])
    validation_precision = link_gate.get("validation_empirical_precision")
    validation_evaluable = link_gate.get("validation_evaluable_decisions", 0)
    test_precision = accepted.get("precision")
    test_evaluable = _integer(
        accepted.get("evaluable_accepted_count", 0),
        "test evaluable accepted links",
    )

    criteria = {
        "candidate_representability": {
            "passed": unsupported_rate <= config.max_unsupported_gold_rate,
            "value": unsupported_rate,
            "maximum": config.max_unsupported_gold_rate,
        },
        "mention_beats_surface_baseline": {
            "passed": mention_delta
            >= config.min_mention_f1_delta_from_surface_baseline,
            "value": mention_delta,
            "minimum": config.min_mention_f1_delta_from_surface_baseline,
        },
        "validation_link_gate": {
            "passed": (
                (
                    not config.require_validation_link_gate
                    or link_gate.get("enabled") is True
                )
                and validation_precision is not None
                and float(validation_precision) >= config.min_validation_link_precision
                and _integer(validation_evaluable, "validation evaluable links")
                >= config.min_validation_evaluable_links
            ),
            "enabled": link_gate.get("enabled") is True,
            "precision": validation_precision,
            "evaluable_links": validation_evaluable,
            "minimum_precision": config.min_validation_link_precision,
            "minimum_evaluable_links": config.min_validation_evaluable_links,
        },
        "test_accepted_links": {
            "passed": (
                test_precision is not None
                and float(test_precision) >= config.min_test_accepted_link_precision
                and test_evaluable >= config.min_test_evaluable_links
            ),
            "precision": test_precision,
            "evaluable_links": test_evaluable,
            "minimum_precision": config.min_test_accepted_link_precision,
            "minimum_evaluable_links": config.min_test_evaluable_links,
        },
        "oracle_coreference_beats_surface_baseline": {
            "passed": oracle_delta
            >= config.min_oracle_coreference_f1_delta_from_baseline,
            "value": oracle_delta,
            "minimum": config.min_oracle_coreference_f1_delta_from_baseline,
        },
    }
    return {
        "schema": GATE_SCHEMA,
        "passed": all(item["passed"] for item in criteria.values()),
        "criteria": criteria,
        "test_threshold_tuning": False,
        "interpretation": (
            "research quality gate for Block 1; passing is not a production-readiness "
            "claim"
        ),
    }
