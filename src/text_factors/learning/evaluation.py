"""Independent bounded component and dialogue evaluation with a freeze gate.

Do not use sealed results to tune production sources. Only development may run
without a matching explicit source/model freeze. Controls consume published
training data, never evaluation annotations. The exact-phrase memorization
baseline is intentionally confined to this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from . import evaluation_cases as corpus
from .schema import DialogueContext, Interpretation, Meaning

REPORT_SCHEMA = "ai2-learned-evaluation-v1"
FREEZE_SCHEMA = "ai2-learned-source-freeze-v1"
_COMPONENTS = ("understanding", "dynamics", "policy", "generation", "dialogue")
_ATTRIBUTES = {"generation": "generator"}
_CONTROLS = ("trained", "untrained", "shuffled", "memorization")


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    else:
        serializer = getattr(value, "to_dict", None)
        if callable(serializer):
            value = serializer()
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _plain(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def corpus_manifest(
    split: str = "development", *, include_cases: bool = False
) -> dict[str, Any]:
    groups = corpus.all_cases(split)
    serialized = {
        name: [corpus.serialize_case(case) for case in cases]
        for name, cases in groups.items()
    }
    result = {
        "version": corpus.DATA_VERSION,
        "split": split,
        "sha256": _digest(serialized),
        "counts": {name: len(cases) for name, cases in groups.items()},
        "dialogue_turns": sum(len(case.turns) for case in groups["dialogue"]),
        "training_examples_imported_by_case_builder": False,
        "new_phrasal_families": split != "development",
        "new_entity_morphology_claim": False,
    }
    if include_cases:
        result["cases"] = serialized
    return _plain(result)


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if "evaluation" not in path.relative_to(root).parts
        and path.name not in {"evaluation.py", "evaluation_cases.py"}
    }


def _component_fingerprints(bundle: Any) -> dict[str, Any]:
    result = {}
    for component in _COMPONENTS[:-1]:
        model = getattr(bundle, _ATTRIBUTES.get(component, component), None)
        if model is None:
            result[component] = {"available": False}
            continue
        serialized = _plain(model.to_dict()) if hasattr(model, "to_dict") else {}
        result[component] = {
            "available": True,
            "training_fingerprint": _training_fingerprint(model, serialized),
            "numeric_checkpoint_sha256": _digest(serialized),
        }
    return result


def _training_fingerprint(model: Any, serialized: Mapping[str, Any]) -> Any:
    return (
        getattr(model, "fingerprint", None)
        or getattr(model, "training_fingerprint", None)
        or serialized.get("fingerprint")
        or serialized.get("training_fingerprint")
        or serialized.get("training", {}).get("fingerprint")
    )


def capture_source_freeze(bundle: Any) -> dict[str, Any]:
    """Create a no-inference freeze token after an explicit source-freeze decision.

    Calling this does not run or reveal individual held-out cases. Preserve the
    token alongside the source/checkpoint artifact before evaluating a reserve.
    """
    sources = _source_manifest()
    return {
        "schema": FREEZE_SCHEMA,
        "source_sha256": sources,
        "source_digest": _digest(sources),
        "evaluation_source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("evaluation.py", "evaluation_cases.py")
        },
        "bundle_sha256": _digest(bundle.to_dict()),
        "bundle_fingerprint": getattr(bundle, "fingerprint", None),
        "components": _component_fingerprints(bundle),
        "corpora": {split: corpus_manifest(split)["sha256"] for split in corpus.SPLITS},
    }


def _check_freeze(
    bundle: Any, split: str, supplied: Mapping[str, Any] | None
) -> dict[str, Any]:
    current = capture_source_freeze(bundle)
    if supplied is None:
        if split != "development":
            raise ValueError(
                "sealed evaluation requires an explicit source/model freeze"
            )
        return {"required": False, "validated": False, "current": current}
    if type(supplied) is not dict or supplied.get("schema") != FREEZE_SCHEMA:
        raise ValueError("invalid evaluation source freeze")
    if _digest(supplied) != _digest(current):
        raise ValueError(
            "stale source/model/corpus freeze; do not tune on consumed holdout"
        )
    return {"required": split != "development", "validated": True, "current": current}


def meaning_core(value: Any) -> dict[str, Any] | None:
    """Entity inventory metadata is not the meaning tree's role/scope content."""
    if value is None:
        return None
    if isinstance(value, Meaning):
        value = value.to_dict()
    if type(value) is not dict:
        raise ValueError("meaning must be a meaning object or JSON object")
    return _plain({key: value.get(key) for key in ("act", "event", "query")})


def _leaves(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_leaves(child, f"{path}.{key}" if path else key))
        return result
    return {path: value}


def _fact_key(value: Mapping[str, Any]) -> corpus.Fact:
    return (
        value["subject"],
        value["relation"],
        value["value"],
        value.get("negated", False),
        value.get("spatial", "in"),
    )


def _effect_key(value: Mapping[str, Any]) -> corpus.Effect:
    return (
        value["op"],
        value["subject"],
        value["relation"],
        value["value"],
        value.get("spatial", "in"),
    )


def _score_understanding(case: corpus.UnderstandingCase, result: Any) -> dict[str, Any]:
    if isinstance(result, Interpretation):
        predicted, score, reason = result.meaning, result.score, result.reason
    elif isinstance(result, dict):
        predicted, score, reason = (
            result.get("meaning"),
            result.get("score", 0),
            result.get("reason", ""),
        )
    else:
        raise ValueError("invalid understanding result")
    actual = meaning_core(predicted)
    expected = meaning_core(case.expected)
    abstained = actual is None or actual.get("act") == "unknown"
    exact = abstained if expected is None else actual == expected
    gold_paths, paths = _leaves(expected), _leaves(actual)
    role_paths = [
        key
        for key in gold_paths
        if key.rsplit(".", 1)[-1]
        in {
            "actor",
            "object",
            "recipient",
            "place",
            "subject",
            "value",
            "relation",
            "spatial",
        }
    ]
    scope_paths = [
        key
        for key in gold_paths
        if key.rsplit(".", 1)[-1]
        in {
            "predicate",
            "modality",
            "time",
            "negated",
            "content",
            "condition",
        }
    ]
    return {
        "correct": bool(exact),
        "expected": expected,
        "predicted": actual,
        "abstained": abstained,
        "expected_abstention": expected is None,
        "role_paths_correct": sum(
            key in paths and paths[key] == gold_paths[key] for key in role_paths
        ),
        "role_paths_total": len(role_paths),
        "scope_paths_correct": sum(
            key in paths and paths[key] == gold_paths[key] for key in scope_paths
        ),
        "scope_paths_total": len(scope_paths),
        "score": score,
        "score_is_probability": False,
        "reason": reason,
        "result": _plain(result),
    }


def _score_dynamics(case: corpus.DynamicsCase, result: Any) -> dict[str, Any]:
    raw = _plain(result)
    actual = set(map(_effect_key, raw.get("effects", [])))
    supported = bool(raw.get("supported", False))
    return {
        "correct": supported == case.expected_supported
        and actual == set(case.expected_effects),
        "supported": supported,
        "expected_supported": case.expected_supported,
        "expected_effects": sorted(case.expected_effects),
        "predicted_effects": sorted(actual),
        "empty_supported_transition": supported and not actual,
        "abstained": not supported,
        "result": raw,
    }


def _independent_slot_support(case: corpus.GenerationCase) -> bool:
    slots = dict(case.slots)
    if case.action not in {"answer", "ack", "corrected", "explain"}:
        return True
    obj = slots.get("object", "")
    relevant = [fact for fact in case.evidence if fact[0] == obj]
    if slots.get("truth") in {"yes", "no"}:
        negative = slots["truth"] == "no"
        relevant = [fact for fact in relevant if fact[3] == negative]
    if not relevant:
        return False
    if "place" in slots:
        spatial = "on" if slots.get("prep") in {"on", "на"} else "in"
        return any(
            fact[1] == "location" and fact[2] == slots["place"] and fact[4] == spatial
            for fact in relevant
        )
    if "holder" in slots:
        return any(
            fact[1] == "holder" and fact[2] == slots["holder"] for fact in relevant
        )
    return True


def _score_generation(
    case: corpus.GenerationCase, result: Any, verifier: Any
) -> dict[str, Any]:
    raw = _plain(result)
    text = raw.get("text", "")
    slots = dict(case.slots)
    evidence = [corpus.fact_dict(fact, evidence=True) for fact in case.evidence]
    verified = (
        bool(verifier(case.action, slots, evidence, result))
        if callable(verifier)
        else False
    )
    support = _independent_slot_support(case)
    forbidden = [
        name for name in case.forbidden_names if name.casefold() in text.casefold()
    ]
    claimed_grounded = bool(raw.get("grounded", False))
    if case.expected_grounded:
        correct = (
            bool(text) and claimed_grounded and verified and support and not forbidden
        )
    else:
        correct = not claimed_grounded and not forbidden
    return {
        "correct": bool(correct),
        "result": raw,
        "independent_slot_support": support,
        "production_copy_verifier": verified,
        "claimed_grounded": claimed_grounded,
        "expected_grounded": case.expected_grounded,
        "forbidden_names_in_text": forbidden,
        "text_truth_scope": (
            "copy/evidence integrity, not independent unrestricted semantic proof"
        ),
    }


class _TrainingPhraseMemorizer:
    """Exact complete-phrase baseline; no entity replacement or hidden data."""

    def __init__(self, examples: list[Any]) -> None:
        self._targets: dict[str, list[Meaning]] = {}
        for example in examples:
            key = self._normalize(example.text)
            self._targets.setdefault(key, []).append(example.meaning)
        self.fingerprint = _digest(
            [
                {"text": example.text, "meaning": example.meaning.to_dict()}
                for example in examples
            ]
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().replace("ё", "е").strip(" .!?").split())

    def interpret(
        self, text: str, context: DialogueContext | None = None
    ) -> Interpretation:
        del context
        candidates = self._targets.get(self._normalize(text), ())
        meanings = {
            json.dumps(meaning_core(item), sort_keys=True): item for item in candidates
        }
        selected = next(iter(meanings.values())) if len(meanings) == 1 else None
        return Interpretation(
            selected,
            1.0 if selected else 0.0,
            reason="training_only_exact_complete_phrase_baseline",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "evaluation-only-phrase-memorization",
            "fingerprint": self.fingerprint,
        }


def _training_language(seed: int = 42) -> list[Any]:
    from .language_data import training_examples

    return list(training_examples(seed=seed))


def _control_provenance(model: Any, training: Any = None) -> dict[str, Any]:
    serialized = _plain(model.to_dict())
    return {
        "training_data_only": True,
        "seed": getattr(model, "seed", None),
        "training_fingerprint": _training_fingerprint(model, serialized),
        "numeric_checkpoint_sha256": _digest(serialized),
        "training_metrics": (
            _plain(training)
            if type(training) is dict
            else _plain(getattr(model, "training_metrics", {}))
        ),
        "training_turns": getattr(model, "training_turns", None),
    }


def _make_control(
    bundle: Any, component: str, control: str, seconds: float
) -> tuple[Any, dict[str, Any]]:
    model = getattr(bundle, _ATTRIBUTES.get(component, component), None)
    if model is None:
        raise NotImplementedError("component not available in bundle")
    if control == "trained":
        return model, {"training_performed_in_evaluation": False}
    seed = getattr(model, "seed", 42)
    if (
        control != "untrained"
        and getattr(bundle, "metadata", {}).get(
            "dataset_kind", "bundled_synthetic_training_only"
        )
        != "bundled_synthetic_training_only"
    ):
        raise NotImplementedError("matching custom training corpus is not available")
    if control == "memorization":
        if component != "understanding":
            raise NotImplementedError(
                "exact phrase baseline applies to understanding only"
            )
        baseline = _TrainingPhraseMemorizer(_training_language(seed))
        return baseline, {
            "training_data_only": True,
            "fingerprint": baseline.fingerprint,
        }
    if control == "untrained":
        return type(model)(seed=seed), {"unfitted_numeric_model": True, "seed": seed}
    if component == "understanding":
        clone = type(model).fit(
            _training_language(seed),
            seed=seed,
            seconds=seconds,
            shuffle_targets=True,
        )
        return clone, _control_provenance(clone)
    if component == "dynamics":
        from .transition_data import training_episodes

        clone = type(model)(seed=seed, mode="shuffled")
        training = clone.fit(training_episodes(seed=seed), seconds=seconds)
        return clone, _control_provenance(clone, training)
    if component in {"policy", "generation"}:
        from .dialogue_data import training_dialogues

        clone = type(model)(seed=seed)
        clone.fit(training_dialogues(seed=seed), seconds=seconds, shuffle_targets=True)
        return clone, _control_provenance(clone)
    raise NotImplementedError("no matched control for this component")


def _new_session(bundle: Any) -> Any:
    from .session import LearnedSession

    return LearnedSession(bundle)


def _reported_execution_status(result: Mapping[str, Any]) -> str:
    """A caught runtime failure is not a successful semantic abstention."""
    reason = result.get("reason", "")
    if reason in {
        "time_budget",
        "generation_deadline",
        "turn_time_budget",
        "component_time_budget",
    }:
        return "timed_out"
    if result.get("action") in {"error", "limit"} or reason in {
        "internal_error",
        "busy",
        "incomplete_effect_recognition",
    }:
        return "failed"
    return "returned"


def _call_component(
    component: str, model: Any, case: Any, seconds: float
) -> dict[str, Any]:
    if component == "understanding":
        return _score_understanding(
            case, model.interpret(case.text, context=case.context)
        )
    if component == "dynamics":
        started = perf_counter()
        before = [corpus.fact_dict(fact) for fact in case.before]
        result = model.predict(before, case.event, seconds=min(1.0, seconds))
        scored = _score_dynamics(case, result)
        if case.rival is not None:
            ranker = getattr(model, "score_interpretations", None)
            if callable(ranker):
                left = seconds - (perf_counter() - started)
                if left <= 0:
                    raise TimeoutError("evaluation_global_deadline")
                scores = _plain(
                    ranker(before, [case.event, case.rival], seconds=min(1.0, left))
                )
                scored["counterexample_scores"] = scores
                numeric = [item.get("score") for item in scores]
                supported = len(numeric) == 2 and all(
                    type(value) in (int, float) and isfinite(value) for value in numeric
                )
                scored["counterexample_correct"] = bool(
                    supported and numeric[0] > numeric[1]
                )
                scored["correct"] = (
                    scored["correct"] and scored["counterexample_correct"]
                )
            else:
                scored["counterexample_correct"] = False
                scored["counterexample_unavailable"] = True
                scored["correct"] = False
        return scored
    if component == "policy":
        result = _plain(model.choose(dict(case.features), case.allowed_actions))
        return {
            "correct": result.get("action") == case.expected_action,
            "expected_action": case.expected_action,
            "result": result,
            "all_actions_allowed": True,
        }
    if component == "generation":
        evidence = [corpus.fact_dict(fact, evidence=True) for fact in case.evidence]
        result = model.generate(
            case.action,
            dict(case.slots),
            evidence,
            max_tokens=48,
            seconds=min(0.5, seconds),
        )
        return _score_generation(case, result, getattr(model, "verify", None))
    raise ValueError("unknown component")


def _score_dialogue_turn(
    expected: corpus.DialogueTurn, result: dict[str, Any], world: Any
) -> dict[str, Any]:
    facts = _plain(list(world.facts()))
    state = set(map(_fact_key, facts))
    assertions = set(map(_fact_key, result.get("assertions", [])))
    expected_state = set(expected.state)
    expected_assertions = (
        None if expected.assertions is None else set(expected.assertions)
    )
    unsupported = assertions - expected_state - (expected_assertions or set())
    action_correct = result.get("action") in expected.actions
    assertion_correct = expected_assertions is None or assertions == expected_assertions
    return {
        "correct": state == expected_state
        and action_correct
        and assertion_correct
        and not unsupported,
        "state_correct": state == expected_state,
        "action_correct": action_correct,
        "assertions_correct": assertion_correct,
        "unsupported_assertions": sorted(unsupported),
        "abstained": result.get("action") in {"unknown", "clarify"},
        "expected_abstention": bool(set(expected.actions) & {"unknown", "clarify"}),
        "result": result,
        "actual_state": facts,
    }


def _run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["rows"]
    planned = run["planned"]
    completed = sum(row["execution_status"] == "returned" for row in rows)
    correct = sum(bool(row.get("correct", False)) for row in rows)
    incorrect = sum(
        row["execution_status"] == "returned" and not row.get("correct", False)
        for row in rows
    )
    execution_failed = sum(row["execution_status"] == "failed" for row in rows)
    durations = sorted(
        row["elapsed_seconds"] for row in rows if "elapsed_seconds" in row
    )
    strata = {}
    for name, requested in run.get("strata_requested", {}).items():
        observed = [row for row in rows if row.get("stratum") == name]
        matched = sum(bool(row.get("correct", False)) for row in observed)
        strata[name] = {
            "requested": requested,
            "attempted": len(observed),
            "correct": matched,
            "accuracy_all_requested": (
                matched / requested
                if requested and run.get("status") != "unavailable"
                else None
            ),
        }
    return {
        "requested": planned,
        "attempted": len(rows),
        "completed": completed,
        "correct": correct,
        "failed": incorrect + execution_failed,
        "incorrect": incorrect,
        "execution_failed": execution_failed,
        "timed_out": sum(row["execution_status"] == "timed_out" for row in rows),
        "skipped": planned - len(rows),
        "not_attempted": planned - len(rows),
        "accuracy_all_requested": (
            correct / planned
            if planned and run.get("status") != "unavailable"
            else None
        ),
        "accuracy_observed_prefix": correct / len(rows) if rows else None,
        "unsupported_assertions": sum(
            len(row.get("unsupported_assertions", ())) for row in rows
        ),
        "state_correct": sum(bool(row.get("state_correct", False)) for row in rows),
        "abstentions": sum(bool(row.get("abstained", False)) for row in rows),
        "role_paths_correct": sum(row.get("role_paths_correct", 0) for row in rows),
        "role_paths_requested": run.get("role_paths_requested", 0),
        "scope_paths_correct": sum(row.get("scope_paths_correct", 0) for row in rows),
        "scope_paths_requested": run.get("scope_paths_requested", 0),
        "strata": strata,
        "latency_seconds": {
            "p95": durations[ceil(0.95 * len(durations)) - 1] if durations else None,
            "max": durations[-1] if durations else None,
        },
    }


def evaluate_learned_dialogue(
    bundle: Any,
    *,
    split: str = "development",
    source_freeze: Mapping[str, Any] | None = None,
    seconds: float = 60.0,
    progress: Callable[[dict[str, Any]], None] | None = None,
    controls: tuple[str, ...] = _CONTROLS,
) -> dict[str, Any]:
    """Run frozen component tests; sealed splits require an unchanged token.

    Failures remain in planned denominators. Controls are component ablations,
    not silently substituted end-to-end systems. The report records explicitly
    unavailable controls and never presents them as successful experiments.
    External hard-timeout supervision is still required for blocked extensions.
    """
    if (
        type(seconds) not in (int, float)
        or not isfinite(seconds)
        or not 0 < seconds <= 600
    ):
        raise ValueError("seconds must be finite and in (0, 600]")
    if (
        type(controls) is not tuple
        or not controls
        or len(controls) > 4
        or any(
            type(control) is not str or control not in _CONTROLS for control in controls
        )
        or len(set(controls)) != len(controls)
    ):
        raise ValueError(
            "controls must be a nonempty unique tuple of supported controls"
        )
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    if split not in corpus.SPLITS:
        raise ValueError("invalid evaluation split")
    started = perf_counter()
    deadline = started + seconds
    freeze = _check_freeze(bundle, split, source_freeze)
    cases = corpus.all_cases(split)
    planned_runs = {}
    for component in _COMPONENTS[:-1]:
        for control in controls:
            run: dict[str, Any] = {
                "component": component,
                "control": control,
                "planned": len(cases[component]),
                "rows": [],
                "status": "pending",
            }
            if component == "understanding":
                strata_requested: dict[str, int] = {}
                for case in cases[component]:
                    name = corpus.understanding_stratum(case)
                    strata_requested[name] = strata_requested.get(name, 0) + 1
                run["strata_requested"] = strata_requested
                gold_paths = [
                    path
                    for case in cases[component]
                    for path in _leaves(meaning_core(case.expected))
                ]
                run["role_paths_requested"] = sum(
                    path.rsplit(".", 1)[-1]
                    in {
                        "actor",
                        "object",
                        "recipient",
                        "place",
                        "subject",
                        "value",
                        "relation",
                        "spatial",
                    }
                    for path in gold_paths
                )
                run["scope_paths_requested"] = sum(
                    path.rsplit(".", 1)[-1]
                    in {
                        "predicate",
                        "modality",
                        "time",
                        "negated",
                        "content",
                        "condition",
                    }
                    for path in gold_paths
                )
            planned_runs[(component, control)] = run
    if "trained" in controls:
        planned_runs[("dialogue", "trained")] = {
            "component": "dialogue",
            "control": "trained",
            "status": "pending",
            "planned": sum(len(case.turns) for case in cases["dialogue"]),
            "rows": [],
        }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "complete": False,
        "status": "running",
        "reason": "",
        "split": split,
        "source_freeze": freeze,
        "corpus": corpus_manifest(split, include_cases=True),
        "controls_requested": list(controls),
        "seconds_budget": seconds,
        "runs": list(planned_runs.values()),
        "unavailable_controls": [],
        "errors": [],
        "attribution": {
            "explicit_scaffolds": "ontology, role basis, world projection, copy guards",
            "learned_modules": (
                "phrase interpretation, dynamics, policy, token generation"
            ),
            "end_to_end_controls": (
                "trained bundle only; component ablations are separate"
            ),
            "free_conversation_claim": False,
            "all_failures_in_requested_denominators": True,
            "hard_timeout_requires_supervisor": True,
        },
    }

    def remaining() -> float:
        left = deadline - perf_counter()
        if left <= 0:
            raise TimeoutError("evaluation_global_deadline")
        return left

    def checkpoint() -> None:
        report["elapsed_seconds"] = perf_counter() - started
        for run in report["runs"]:
            run["metrics"] = _run_metrics(run)
        if progress is not None:
            progress(_plain(report))

    checkpoint()
    try:
        for component in _COMPONENTS[:-1]:
            for control in controls:
                remaining()
                run = planned_runs[(component, control)]
                if control == "memorization" and component != "understanding":
                    run["status"] = "unavailable"
                    report["unavailable_controls"].append(
                        {
                            "component": component,
                            "control": control,
                            "reason": (
                                "exact phrase baseline applies to understanding only"
                            ),
                        }
                    )
                    checkpoint()
                    continue
                try:
                    model, provenance = _make_control(
                        bundle, component, control, min(30.0, remaining())
                    )
                except NotImplementedError as error:
                    run["status"] = "unavailable"
                    report["unavailable_controls"].append(
                        {
                            "component": component,
                            "control": control,
                            "reason": str(error),
                        }
                    )
                    checkpoint()
                    continue
                except Exception as error:
                    run.update(status="setup_failed", setup_failed=True)
                    report["errors"].append(
                        {
                            "component": component,
                            "control": control,
                            "phase": "control_setup",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    checkpoint()
                    continue
                run.update(provenance=provenance, status="running")
                for case in cases[component]:
                    remaining()
                    before = perf_counter()
                    row = {
                        "case_id": case.case_id,
                        "input_expected": corpus.serialize_case(case),
                    }
                    if component == "understanding":
                        row["stratum"] = corpus.understanding_stratum(case)
                    try:
                        row.update(_call_component(component, model, case, remaining()))
                        row["execution_status"] = _reported_execution_status(
                            row.get("result", {})
                        )
                        if row["execution_status"] != "returned":
                            row["correct"] = False
                    except Exception as error:
                        row.update(
                            correct=False,
                            execution_status="timed_out"
                            if isinstance(error, TimeoutError)
                            else "failed",
                            error={"type": type(error).__name__, "message": str(error)},
                        )
                    row["elapsed_seconds"] = perf_counter() - before
                    run["rows"].append(row)
                    checkpoint()
                    remaining()
                run["status"] = "completed"
        if "trained" in controls:
            run = planned_runs[("dialogue", "trained")]
            run["status"] = "running"
            for case in cases["dialogue"]:
                remaining()
                session = _new_session(bundle)
                for index, expected in enumerate(case.turns):
                    remaining()
                    before = perf_counter()
                    row = {
                        "case_id": case.case_id,
                        "turn_index": index,
                        "input_expected": corpus.serialize_case(expected),
                    }
                    try:
                        result = _plain(
                            session.respond(
                                expected.text, request_id=f"{case.case_id}:{index}"
                            )
                        )
                        row.update(
                            _score_dialogue_turn(expected, result, session.world)
                        )
                        row["execution_status"] = _reported_execution_status(result)
                        if row["execution_status"] != "returned":
                            row["correct"] = False
                    except Exception as error:
                        row.update(
                            correct=False,
                            execution_status="timed_out"
                            if isinstance(error, TimeoutError)
                            else "failed",
                            error={"type": type(error).__name__, "message": str(error)},
                        )
                    row["elapsed_seconds"] = perf_counter() - before
                    run["rows"].append(row)
                    checkpoint()
                    remaining()
            run["status"] = "completed"
        bad_runtime = report["errors"] or any(
            row["execution_status"] != "returned"
            for run in report["runs"]
            for row in run["rows"]
        )
        available = any(run["status"] != "unavailable" for run in report["runs"])
        report["complete"] = bool(not bad_runtime and available)
        report["status"] = (
            "unavailable"
            if not available
            else "completed_with_errors"
            if bad_runtime
            else "completed"
        )
    except TimeoutError as error:
        report["status"] = "timed_out"
        report["reason"] = str(error)
    except Exception as error:
        report["status"] = "failed"
        report["reason"] = f"{type(error).__name__}: {error}"
    for run in report["runs"]:
        if run["status"] in {"pending", "running"}:
            run["status"] = "timed_out" if report["status"] == "timed_out" else "failed"
        elif run["status"] == "completed" and any(
            row["execution_status"] != "returned" for row in run["rows"]
        ):
            run["status"] = "completed_with_errors"
    # Detect accidental training/mutation of the evaluated bundle during inference.
    if _digest(bundle.to_dict()) != freeze["current"]["bundle_sha256"]:
        report.update(
            complete=False, status="failed", reason="evaluated_bundle_mutated"
        )
    if _source_manifest() != freeze["current"]["source_sha256"]:
        report.update(
            complete=False,
            status="failed",
            reason="production_sources_changed_during_evaluation",
        )
    checkpoint()
    return _plain(report)


evaluate_learned = evaluate_learned_dialogue
