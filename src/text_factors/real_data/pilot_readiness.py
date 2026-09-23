"""Conservative P24 scoring of independently selected pilot questions.

These are measurement primitives, not an authorization to expose answers to
users. The P01 owner must certify group independence, adjudication, access,
and the frozen selection before a shadow or limited real-data run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .pilot import PilotGateConfig, _exact_binomial_upper


@dataclass(frozen=True, slots=True)
class PilotTrial:
    """One question preselected from one P01 document/task-history group.

    None denotes an unknown adjudication. In particular, unknown support is
    never treated as evidence that an accepted answer is safe.
    """

    group_id: str
    query_id: str
    answerable: bool | None
    accepted: bool
    useful_supported: bool | None = None
    unsupported_or_wrong: bool | None = None
    authorized_source_and_version: bool | None = None
    critical_error: bool = False
    preselected: bool = True

    def __post_init__(self) -> None:
        for name in ("group_id", "query_id"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "answerable",
            "useful_supported",
            "unsupported_or_wrong",
            "authorized_source_and_version",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean or unknown")
        for name in ("accepted", "critical_error", "preselected"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not self.accepted and any(
            value is not None
            for value in (
                self.useful_supported,
                self.unsupported_or_wrong,
                self.authorized_source_and_version,
            )
        ):
            raise ValueError("abstentions must not contain answer assessments")


@dataclass(frozen=True, slots=True)
class PilotGateResult:
    phase: Literal["shadow", "limited"]
    independent_groups: int
    answerable_queries: int
    accepted_answers: int
    useful_supported_answers: int
    unsupported_answers: int
    attributed_answers: int
    critical_errors: int
    useful_coverage: float | None
    unsupported_upper_bound: float | None
    attribution_rate: float | None
    blockers: tuple[str, ...]
    numeric_gate_passed: bool


def score_p24_trials(
    trials: Sequence[PilotTrial],
    *,
    phase: Literal["shadow", "limited"] = "shadow",
    config: PilotGateConfig | None = None,
) -> PilotGateResult:
    """Score one phase; never count multiple questions from the same group.

    The caller is responsible for preserving the selection and adjudication
    records and for verifying the P01 contract. This method fails closed on
    incomplete labels and never combines shadow and limited measurements.
    """
    if phase not in ("shadow", "limited"):
        raise ValueError("phase must be shadow or limited")
    config = config or PilotGateConfig()
    groups: set[str] = set()
    queries: set[str] = set()
    blockers: set[str] = set()
    for trial in trials:
        if not isinstance(trial, PilotTrial):
            raise TypeError("each trial must be a PilotTrial")
        if not trial.preselected:
            raise ValueError("every question must be selected before model output")
        if trial.group_id in groups:
            raise ValueError("one preselected query per independent group")
        if trial.query_id in queries:
            raise ValueError("query IDs must be unique")
        groups.add(trial.group_id)
        queries.add(trial.query_id)
        if trial.answerable is None:
            blockers.add("unknown_answerability")
        if trial.accepted:
            if trial.unsupported_or_wrong is None and trial.answerable is not False:
                blockers.add("unknown_accepted_answer_support")
            if trial.authorized_source_and_version is None:
                blockers.add("unknown_accepted_answer_attribution")
            if trial.answerable and trial.useful_supported is None:
                blockers.add("unknown_accepted_answer_usefulness")
            if trial.answerable is False and trial.unsupported_or_wrong is False:
                blockers.add("accepted_unanswerable_label_conflict")
            if trial.useful_supported is True and (
                trial.answerable is not True
                or trial.unsupported_or_wrong is not False
                or trial.authorized_source_and_version is not True
            ):
                blockers.add("contradictory_usefulness_label")
        if trial.critical_error:
            blockers.add("critical_error")

    answerable = sum(trial.answerable is True for trial in trials)
    accepted = sum(trial.accepted for trial in trials)
    useful = sum(
        trial.answerable is True
        and trial.accepted
        and trial.useful_supported is True
        and trial.unsupported_or_wrong is False
        and trial.authorized_source_and_version is True
        for trial in trials
    )
    unsupported = sum(
        trial.accepted
        and (
            trial.answerable is False
            or trial.unsupported_or_wrong is True
            or trial.authorized_source_and_version is False
        )
        for trial in trials
    )
    attributed = sum(
        trial.accepted and trial.authorized_source_and_version is True
        for trial in trials
    )
    critical = sum(trial.critical_error for trial in trials)
    if not answerable:
        blockers.add("no_adjudicated_answerable_queries")
    if not accepted:
        blockers.add("no_accepted_answers")

    coverage = (
        useful / answerable
        if answerable
        and "unknown_answerability" not in blockers
        and "unknown_accepted_answer_usefulness" not in blockers
        else None
    )
    upper = (
        _exact_binomial_upper(unsupported, accepted, config.confidence)
        if accepted and "unknown_accepted_answer_support" not in blockers
        else None
    )
    attribution = (
        attributed / accepted
        if accepted and "unknown_accepted_answer_attribution" not in blockers
        else None
    )
    if coverage is not None and coverage < config.min_resolvable_coverage:
        blockers.add("useful_coverage_below_70_percent")
    if upper is not None and upper > config.max_ungrounded_upper:
        blockers.add("unsupported_risk_upper_bound_above_1_percent")
    if attribution is not None and attribution < 1.0:
        blockers.add("missing_authorized_source_or_exact_version")
    return PilotGateResult(
        phase=phase,
        independent_groups=len(groups),
        answerable_queries=answerable,
        accepted_answers=accepted,
        useful_supported_answers=useful,
        unsupported_answers=unsupported,
        attributed_answers=attributed,
        critical_errors=critical,
        useful_coverage=coverage,
        unsupported_upper_bound=upper,
        attribution_rate=attribution,
        blockers=tuple(sorted(blockers)),
        numeric_gate_passed=not blockers,
    )
