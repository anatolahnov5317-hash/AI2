"""Check the P01 pilot contract without accessing private corpus material.

The .yaml file intentionally uses the JSON subset of YAML 1.2, so this
validation runs with the standard library on every supported Python version.
Passing a draft check establishes contract consistency, not pilot readiness.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SPLITS = ["train", "development", "calibration", "sealed_test", "future_stream"]
METRICS = {
    "unsupported_answer_risk": (
        "accepted_substantive_answers_with_unsupported_or_materially_wrong_claim",
        "all_accepted_substantive_answers",
    ),
    "useful_coverage": (
        "adjudicated_answerable_queries_with_accepted_useful_supported_answer",
        "all_preselected_adjudicated_answerable_queries",
    ),
    "answer_attribution": (
        "accepted_answers_with_valid_authorized_source_and_exact_version_receipt",
        "all_accepted_substantive_answers",
    ),
}
REQUIRED_DECISIONS = (
    ("scope", "decision_owner"),
    ("scope", "data_controller"),
    ("data", "language_evidence_reference"),
    ("data", "annotation_policy_reference"),
    ("data", "privacy", "authorized_access_policy_reference"),
    ("independence", "sealed_test_source_reference"),
    ("evaluation", "block1_quality_gates_reference"),
    ("operations", "target_hardware_reference"),
    ("operations", "ram_limit_gib"),
    ("operations", "disk_limit_gib"),
    ("operations", "latency_p95_seconds"),
    ("operations", "revision_job_deadline_seconds"),
    ("operations", "error_cost_policy_reference"),
)


def field(document: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def is_filled(value: Any) -> bool:
    return value is not None and value != "" and value != []


def validate_contract(document: Any) -> tuple[list[str], list[str]]:
    """Return schema errors and unresolved decisions; do not attest their truth."""
    errors: list[str] = []
    blockers: list[str] = []
    if not isinstance(document, dict):
        return ["contract must be a mapping"], blockers

    if document.get("schema_version") != "ai2-pilot-contract-v1":
        errors.append("unsupported schema_version")
    if document.get("stage") != "P01":
        errors.append("stage must be P01")
    if document.get("status") not in ("draft", "finalized"):
        errors.append("status must be draft or finalized")
    if not re.fullmatch(r"[0-9a-f]{40}", str(document.get("based_on_commit", ""))):
        errors.append("based_on_commit must be a full SHA-1")
    if field(document, "scope", "target_language") != "ru":
        errors.append("this pilot must explicitly evaluate Russian")
    if not field(document, "scope", "supported_questions"):
        errors.append("supported_questions must be nonempty")
    if not field(document, "scope", "out_of_scope"):
        errors.append("out_of_scope must be nonempty")

    sources = field(document, "data", "sources")
    if not isinstance(sources, list) or not sources:
        errors.append("data.sources must be a nonempty list")
        sources = []
    ids = [source.get("source_id") for source in sources if isinstance(source, dict)]
    if len(ids) != len(sources) or any(
        not isinstance(source_id, str) or not source_id for source_id in ids
    ):
        errors.append("every source needs a source_id")
    elif len(set(ids)) != len(ids):
        errors.append("source_id values must be unique")
    for source in sources:
        if not isinstance(source, dict):
            continue
        identifier = source.get("source_id") or "unknown"
        if source.get("status") not in ("not_provided", "available"):
            errors.append(f"source {identifier}: invalid status")
        if source.get("status") != "available":
            blockers.append(f"data.sources.{identifier}.status")
        for key in (
            "location_reference",
            "rights_evidence_reference",
            "collection_owner",
        ):
            if not is_filled(source.get(key)):
                blockers.append(f"data.sources.{identifier}.{key}")

    privacy = field(document, "data", "privacy")
    if not isinstance(privacy, dict) or any(
        privacy.get(key) is not expected
        for key, expected in (
            ("raw_text_in_repository", False),
            ("weights_from_restricted_data_in_repository", False),
            ("revocation_and_deletion_required", True),
        )
    ):
        errors.append("privacy protections must be explicit")
    budget = field(document, "data", "discovery_budget_not_acceptance_sample")
    if not isinstance(budget, dict) or any(
        type(budget.get(key)) is not int or budget[key] <= 0
        for key in (
            "independent_documents",
            "correction_chains",
            "observed_transitions",
            "reestimate_after_documents",
        )
    ):
        errors.append("discovery budget must have positive integer counts")

    independence = field(document, "independence")
    if field(independence, "splits") != SPLITS:
        errors.append("all five ordered evaluation splits are required")
    if field(independence, "unit") != "document_family_and_task_history":
        errors.append("independence unit must include source family and task history")
    disclosed = field(independence, "already_disclosed")
    if not isinstance(disclosed, list) or not any(
        isinstance(x, str) and "GUM" in x for x in disclosed
    ):
        errors.append("already opened GUM test must be disclosed")
    if not is_filled(field(independence, "sealed_test_policy")):
        errors.append("sealed_test_policy is required")
    sealed = field(independence, "sealed_test_source_reference")
    if isinstance(sealed, str) and "gum" in sealed.lower():
        errors.append("already disclosed GUM cannot be a new sealed_test")

    evaluation = field(document, "evaluation")
    if not is_filled(field(evaluation, "question_adjudication")):
        errors.append("question_adjudication is required")
    for name, (numerator, denominator) in METRICS.items():
        metric = field(evaluation, name)
        if field(metric, "numerator") != numerator:
            errors.append(f"{name}: unexpected numerator")
        if field(metric, "denominator") != denominator:
            errors.append(f"{name}: unexpected denominator")
    risk = field(evaluation, "unsupported_answer_risk")
    coverage = field(evaluation, "useful_coverage")
    attribution = field(evaluation, "answer_attribution")
    for name, metric in (("risk", risk), ("coverage", coverage)):
        if field(metric, "trial_unit") != (
            "one_preselected_query_from_each_independent_group"
        ):
            errors.append(f"{name}: correlated queries cannot be binomial trials")
    if field(risk, "bound") != "one_sided_exact_binomial_clopper_pearson":
        errors.append("risk requires an exact one-sided bound")
    if field(risk, "confidence") != 0.95 or field(risk, "upper_bound_max") != 0.01:
        errors.append("risk gate must be 95% one-sided upper bound <= 1%")
    if field(risk, "zero_denominator") != "undefined_not_zero_risk":
        errors.append("zero accepted answers cannot imply zero risk")
    if field(coverage, "minimum") != 0.70:
        errors.append("useful coverage gate must be >= 70%")
    if field(coverage, "abstention") != "counts_as_not_covered":
        errors.append("abstention must reduce useful coverage")
    if field(attribution, "minimum") != 1.0:
        errors.append("every accepted answer must have a receipt")
    if not field(evaluation, "critical_errors"):
        errors.append("critical_errors must be listed")
    if not is_filled(field(evaluation, "calibration_rule")):
        errors.append("calibration_rule is required")

    baseline = field(document, "baseline")
    if field(baseline, "language") != "en" or field(baseline, "status") != (
        "research_baseline_only_not_russian_pilot_evidence"
    ):
        errors.append("English baseline cannot be Russian pilot evidence")
    operations = field(document, "operations")
    if field(operations, "rollback_required") is not True or field(
        operations, "checkpoint_and_progress_required"
    ) is not True:
        errors.append("rollback and resumable progress are required")
    for path in REQUIRED_DECISIONS:
        if not is_filled(field(document, *path)):
            blockers.append(".".join(path))
    for key in (
        "ram_limit_gib",
        "disk_limit_gib",
        "latency_p95_seconds",
        "revision_job_deadline_seconds",
    ):
        value = field(operations, key)
        if value is not None and (type(value) not in (float, int) or value <= 0):
            errors.append(f"operations.{key} must be positive when set")
    if document.get("status") == "finalized" and blockers:
        errors.append("finalized contract still has unresolved decisions")
    return errors, sorted(set(blockers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--require-finalized", action="store_true")
    args = parser.parse_args()
    try:
        document = json.loads(args.contract.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(json.dumps({"errors": [str(exc)]}, ensure_ascii=False))
        return 1
    errors, blockers = validate_contract(document)
    finalized = document.get("status") == "finalized" and not blockers and not errors
    print(
        json.dumps(
            {
                "stage": "P01",
                "contract_valid": not errors,
                "contract_finalized": finalized,
                "errors": errors,
                "unresolved_decisions": blockers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        return 1
    return 0 if finalized or not args.require_finalized else 2


if __name__ == "__main__":
    raise SystemExit(main())
