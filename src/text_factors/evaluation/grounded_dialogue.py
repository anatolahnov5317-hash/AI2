"""Prospective, bounded evaluation of AI2's grounded dialogue increment.

The corpus is constructed without a model, predictions or a training seed.
Whole dialogues/entity combinations are held out from the *lexical* teaching
recipe. This does not test acquisition of new language or discovered grammar.
The explicitly labelled oracle lives here only, never in production inference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from ..conversation.bridge import FactorSemanticBridge, PredicateResolution
from ..conversation.schema import Assertion, BudgetExceeded, ConversationLimits

CORPUS_SCHEMA = "ai2-grounded-dialogue-corpus-v1"
REPORT_SCHEMA = "ai2-grounded-dialogue-evaluation-v1"
DEFAULT_SEEDS = (7, 17, 42)
DEFAULT_MODES = ("factor", "untrained", "shuffled", "nearest", "oracle")
SPLITS = ("development", "held_out", "challenge")
# topic, object, relation, value, negated, spatial qualifier.
SemanticKey = tuple[str, str, str, str, bool, str]


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    text: str
    state: tuple[SemanticKey, ...]
    actions: tuple[str, ...] = ("ack",)
    assertions: tuple[SemanticKey, ...] | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DialogueCase:
    case_id: str
    split: str
    category: str
    turns: tuple[DialogueTurn, ...]
    lexical_axis: str = "known_taught_cues_new_dialogue_combinations"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fact(
    subject: str,
    relation: str,
    value: str,
    *,
    negated: bool = False,
    topic: str = "default",
    qualifier: str = "in",
) -> SemanticKey:
    return (topic, subject, relation, value, negated, qualifier)


def _turn(
    text: str,
    state: Iterable[SemanticKey] = (),
    action: str = "ack",
    *,
    assertions: Iterable[SemanticKey] | None = None,
    reason: str = "",
) -> DialogueTurn:
    return DialogueTurn(
        text,
        tuple(sorted(state)),
        (action,),
        None if assertions is None else tuple(sorted(assertions)),
        reason,
    )


def make_dialogue_cases(split: str = "held_out") -> tuple[DialogueCase, ...]:
    """Return a frozen seed-independent corpus; never call inference here.

    The challenge set is a prospectively reserved entity/role permutation of
    the same task families, not a claim of new linguistic task generalization.
    Do not inspect challenge predictions until development is complete.
    """
    if type(split) is not str or split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    # (object, second object, its accusative, person, dative, genitive,
    # other person, other dative, other genitive). Roles and expected meanings
    # are authored in advance; no case is selected to suit a prediction.
    vocabulary = {
        "development": (
            "шар",
            "книга",
            "книгу",
            "петя",
            "пете",
            "пети",
            "катя",
            "кате",
            "кати",
        ),
        "held_out": (
            "ключ",
            "книга",
            "книгу",
            "миша",
            "мише",
            "миши",
            "маша",
            "маше",
            "маши",
        ),
        "challenge": (
            "карандаш",
            "игрушка",
            "игрушку",
            "борис",
            "борису",
            "бориса",
            "анна",
            "анне",
            "анны",
        ),
    }
    obj, other, other_acc, person, person_dat, person_gen, peer, peer_dat, peer_gen = (
        vocabulary[split]
    )
    box = _fact(obj, "location", "ящик")
    bag = _fact(obj, "location", "сумка")
    table = _fact(obj, "location", "стол", qualifier="on")
    holder = _fact(obj, "holder", person)
    receiver = _fact(obj, "holder", peer)
    other_holder = _fact(other, "holder", peer)
    other_receiver = _fact(other, "holder", person)
    not_box = _fact(obj, "location", "ящик", negated=True)
    other_table = _fact(other, "location", "стол", qualifier="on")
    at_home = _fact(obj, "location", "стол", topic="дом", qualifier="on")
    at_work = _fact(obj, "location", "ящик", topic="работа")
    cases: list[DialogueCase] = []

    def add(
        category: str,
        turns: Iterable[DialogueTurn],
        lexical_axis: str = "known_taught_cues_new_dialogue_combinations",
    ) -> None:
        cases.append(
            DialogueCase(
                f"{split}/{category}", split, category, tuple(turns), lexical_axis
            )
        )

    add(
        "move_pronoun",
        (
            _turn(f"Я положил {obj} в ящик.", (box,)),
            _turn("Я переложил его в сумку.", (bag,)),
            _turn(f"Где {obj}?", (bag,), "answer", assertions=(bag,)),
        ),
    )
    add(
        "transfer",
        (
            _turn(f"У {person_gen} есть {obj}.", (holder,)),
            _turn(f"{person} передал {obj} {peer_dat}.", (receiver,)),
            _turn(f"У кого {obj}?", (receiver,), "answer", assertions=(receiver,)),
            _turn(f"Что у {peer_gen}?", (receiver,), "answer", assertions=(receiver,)),
        ),
    )
    add(
        "role_reversal",
        (
            _turn(f"У {peer_gen} есть {other}.", (other_holder,)),
            _turn(f"{peer} отдала {other_acc} {person_dat}.", (other_receiver,)),
            _turn(
                f"У кого {other}?",
                (other_receiver,),
                "answer",
                assertions=(other_receiver,),
            ),
        ),
    )
    add(
        "negated_location",
        (
            _turn(f"{obj} не в ящике.", (not_box,)),
            _turn(
                f"{obj} в ящике?",
                (not_box,),
                "answer",
                assertions=(not_box,),
                reason="false",
            ),
            _turn(f"Где {obj}?", (not_box,), "unknown", assertions=()),
        ),
    )
    add(
        "negated_transfer",
        (
            _turn(f"У {person_gen} есть {obj}.", (holder,)),
            _turn(f"{person} не передал {obj} {peer_dat}.", (holder,)),
            _turn(f"У кого {obj}?", (holder,), "answer", assertions=(holder,)),
        ),
    )
    add(
        "negated_move",
        (
            _turn(f"{obj} в сумке.", (bag,)),
            _turn(f"{person} не переложил {obj} в ящик.", (bag,)),
            _turn(f"Где {obj}?", (bag,), "answer", assertions=(bag,)),
        ),
    )
    add(
        "correction_retraction",
        (
            _turn(f"{obj} в ящике.", (box,)),
            _turn(f"Я переложил {obj} в сумку.", (bag,)),
            _turn(f"Нет, {obj} на столе.", (table,)),
            _turn(f"Где {obj}?", (table,), "answer", assertions=(table,)),
            _turn("Отмени последнее утверждение.", (bag,), "retracted"),
            _turn(f"Где {obj}?", (bag,), "answer", assertions=(bag,)),
        ),
    )
    add(
        "retraction",
        (
            _turn(f"{obj} в ящике.", (box,)),
            _turn(f"Я переложил {obj} в сумку.", (bag,)),
            _turn("Последнее сообщение было ошибкой.", (box,), "retracted"),
            _turn(f"Где {obj}?", (box,), "answer", assertions=(box,)),
        ),
    )
    add(
        "topic_isolation",
        (
            _turn("Тема: дом.", (), "topic", assertions=()),
            _turn(f"{obj} на столе.", (at_home,)),
            _turn("Тема: работа.", (), "topic", assertions=()),
            _turn(f"Где {obj}?", (), "unknown", assertions=()),
            _turn(f"{obj} в ящике.", (at_work,)),
            _turn("Тема: дом.", (at_home,), "topic", assertions=()),
            _turn(f"Где {obj}?", (at_home,), "answer", assertions=(at_home,)),
        ),
    )
    add(
        "unknown_object",
        (
            _turn(f"{obj} в ящике.", (box,)),
            _turn("Где тессеракт?", (box,), "unknown", assertions=()),
        ),
        "unseen_entity_unknown_answer_not_new_predicate_learning",
    )
    add(
        "hypothesis",
        (
            _turn(f"Если {obj} в ящике.", (), "hypothetical", assertions=()),
            _turn(f"Где {obj}?", (), "unknown", assertions=()),
            _turn(f"Я положил {obj} в сумку.", (bag,)),
            _turn(f"Если {obj} на столе.", (bag,), "hypothetical", assertions=()),
            _turn(f"Где {obj}?", (bag,), "answer", assertions=(bag,)),
        ),
    )
    add(
        "untaught_predicate",
        (
            _turn(f"{obj} в сумке.", (bag,)),
            _turn(
                f"{person} телепортировал {obj} в ящик.",
                (bag,),
                "clarify",
                assertions=(),
            ),
            _turn(f"Где {obj}?", (bag,), "answer", assertions=(bag,)),
        ),
        "untaught_predicate_should_abstain_not_generalize",
    )
    add(
        "atomic_partial_parse",
        (
            _turn(f"{obj} в сумке.", (bag,)),
            _turn(
                f"Я положил {obj} в ящик; открой портал квантов.",
                (bag,),
                "clarify",
                assertions=(),
            ),
            _turn(f"Где {obj}?", (bag,), "answer", assertions=(bag,)),
        ),
        "mixed_supported_unsupported_turn_should_rollback",
    )
    add(
        "ambiguous_pronoun",
        (
            _turn(f"{obj} в ящике; {other} на столе.", (box, other_table)),
            _turn(
                "Я переложил его в сумку.", (box, other_table), "clarify", assertions=()
            ),
            _turn(f"Где {obj}?", (box, other_table), "answer", assertions=(box,)),
        ),
    )
    # Small public development set; held_out and challenge remain separate.
    return tuple(cases[:3] if split == "development" else cases)


def _json_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def corpus_manifest(split: str = "held_out") -> dict[str, Any]:
    cases = make_dialogue_cases(split)
    serialized = [case.to_dict() for case in cases]
    return {
        "schema": CORPUS_SCHEMA,
        "split": split,
        "sha256": _json_digest(serialized),
        "case_count": len(cases),
        "turn_count": sum(len(case.turns) for case in cases),
        "seed_independent": True,
        "frozen_before_predictions": True,
        "generalization_axis": "whole_dialogues_and_entity_combinations",
        "new_language_generalization": False,
        "training_unit": "isolated_normalized_lexical_cue_and_predicate_label",
        "holdout_caveat": (
            "If code or data are revised after inspecting this split's results, "
            "relabel the split development; use an untouched challenge split."
        ),
        "cases": serialized,
    }


def semantic_key(assertion: Assertion | Mapping[str, Any]) -> SemanticKey:
    data = assertion.to_dict() if isinstance(assertion, Assertion) else assertion
    return (
        data["topic"],
        data["subject"],
        data["relation"],
        data["value"],
        data["negated"],
        data["qualifier"],
    )


def _teaching_pairs() -> list[tuple[str, str]]:
    from ..conversation.language import DEFAULT_TEACHING_PAIRS

    return list(DEFAULT_TEACHING_PAIRS)


class _EvaluatorOnlyOracle:
    """Label-control stub, deliberately not a production-loadable bridge.

    It receives the same finite teacher pairs as every trained control. It
    cannot use expected answers, entity identities or test dialogue state.
    Unknown lexical forms remain unknown even in this oracle control.
    """

    mode = "oracle"

    def __init__(self, seed: int, pairs: list[tuple[str, str]]) -> None:
        self.seed = seed
        self._pairs = tuple(pairs)
        self._labels: dict[str, set[str]] = {}
        for cue, label in pairs:
            normalized = " ".join(cue.casefold().split())
            self._labels.setdefault(normalized, set()).add(label)

    def classify(self, cue: str) -> PredicateResolution:
        candidates = tuple(
            sorted(self._labels.get(" ".join(cue.casefold().split()), ()))
        )
        label = candidates[0] if len(candidates) == 1 else None
        return PredicateResolution(
            label,
            1.0 if label else 0.0,
            candidates,
            {
                "mode": "oracle",
                "reason": "evaluator_only_teacher_label_control",
                "factor_evidence_used": False,
                "score_is_probability": False,
                "oracle_confined_to_evaluation": True,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "evaluator-only-oracle-do-not-restore",
            "seed": self.seed,
            "teaching_sha256": _json_digest(self._pairs),
        }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    # Include underlying memories/transforms/readers, not just the adapter.
    targets = sorted(root.rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in targets
    }


def _make_bridge(
    seed: int,
    mode: str,
    pairs: list[tuple[str, str]],
    seconds: float,
) -> tuple[Any, dict[str, Any]]:
    if mode == "oracle":
        return _EvaluatorOnlyOracle(seed, pairs), {
            "complete": True,
            "mode": mode,
            "elapsed_seconds": 0.0,
            "unique_teaching_pairs": len(set(pairs)),
            "factor_evidence_used": False,
        }
    bridge = FactorSemanticBridge(seed=seed, mode=mode)
    training = bridge.fit(pairs, epochs=8, seconds=min(15.0, seconds))
    return bridge, training


def _new_session(seed: int, bridge: Any, seconds: float) -> Any:
    from ..conversation.engine import ConversationSession

    return ConversationSession(
        seed=seed,
        bridge=bridge,
        train_defaults=False,
        limits=ConversationLimits(turn_seconds=min(2.0, seconds)),
    )


def _score_turn(
    expected: DialogueTurn,
    response: dict[str, Any],
    facts: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    actual_state = set(map(semantic_key, facts))
    actual_assertions = set(map(semantic_key, response["assertions"]))
    expected_state = set(expected.state)
    expected_assertions = (
        None if expected.assertions is None else set(expected.assertions)
    )
    action_correct = response["action"] in expected.actions
    reason_correct = not expected.reason or response["reason"] == expected.reason
    exact_assertions = (
        expected_assertions is None or actual_assertions == expected_assertions
    )
    unsupported = actual_assertions - expected_state - (expected_assertions or set())
    ungrounded_in_state = actual_assertions - actual_state
    state_correct = actual_state == expected_state
    response_correct = action_correct and reason_correct and exact_assertions
    answer_scored = expected.assertions is not None and bool(
        set(expected.actions) & {"answer", "unknown"}
    )
    return {
        "input": expected.text,
        "expected": expected.to_dict(),
        "actual": response,
        "actual_state": facts,
        "actual_semantic_state": sorted(actual_state),
        "state_correct": state_correct,
        "action_correct": action_correct,
        "semantic_response_correct": response_correct,
        "semantic_answer_scored": answer_scored,
        "semantic_answer_correct": response_correct if answer_scored else None,
        "unsupported_assertions": sorted(unsupported),
        "assertions_not_in_actual_state": sorted(ungrounded_in_state),
        "abstained": response["action"] in {"unknown", "clarify"},
        "expected_abstention": bool(set(expected.actions) & {"unknown", "clarify"}),
        "success": state_correct and response_correct and not unsupported,
        "elapsed_seconds": elapsed,
    }


def _metrics(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
    traces = list(traces)
    scored = [turn for turn in traces if turn["semantic_answer_scored"]]
    latencies = sorted(turn["elapsed_seconds"] for turn in traces)

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    state_correct = sum(turn["state_correct"] for turn in traces)
    response_correct = sum(turn["semantic_response_correct"] for turn in traces)
    answer_correct = sum(bool(turn["semantic_answer_correct"]) for turn in scored)
    abstentions = sum(turn["abstained"] for turn in traces)
    return {
        "completed_turns": len(traces),
        "state_correct": state_correct,
        "state_accuracy": rate(state_correct, len(traces)),
        "semantic_responses_correct": response_correct,
        "semantic_response_accuracy": rate(response_correct, len(traces)),
        "semantic_answers_scored": len(scored),
        "semantic_answers_correct": answer_correct,
        "semantic_answer_accuracy": rate(answer_correct, len(scored)),
        "unsupported_assertions": sum(
            len(turn["unsupported_assertions"]) for turn in traces
        ),
        "turns_with_unsupported_assertions": sum(
            bool(turn["unsupported_assertions"]) for turn in traces
        ),
        "assertions_not_in_actual_state": sum(
            len(turn["assertions_not_in_actual_state"]) for turn in traces
        ),
        "abstentions": abstentions,
        "abstention_rate": rate(abstentions, len(traces)),
        "unnecessary_abstentions": sum(
            turn["abstained"] and not turn["expected_abstention"] for turn in traces
        ),
        "missed_required_abstentions": sum(
            not turn["abstained"] and turn["expected_abstention"] for turn in traces
        ),
        "latency_seconds": {
            "median": (
                (latencies[(len(latencies) - 1) // 2] + latencies[len(latencies) // 2])
                / 2
                if latencies
                else None
            ),
            "p95": latencies[ceil(0.95 * len(latencies)) - 1] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
    }


def _refresh_report(report: dict[str, Any], started: float) -> None:
    report["elapsed_seconds"] = perf_counter() - started
    for run in report["runs"]:
        run["metrics"] = _metrics(
            turn for case in run["cases"] for turn in case["turns"]
        )
    report["completed_runs"] = sum(run["complete"] for run in report["runs"])
    all_cases = [case for run in report["runs"] for case in run["cases"]]
    report["completed_dialogues"] = sum(case["complete"] for case in all_cases)
    report["successful_dialogues"] = sum(
        case["complete"] and case["success"] for case in all_cases
    )
    report["metrics"] = _metrics(turn for case in all_cases for turn in case["turns"])
    by_mode = {}
    for mode in report["modes"]:
        cases = [
            case
            for run in report["runs"]
            if run["mode"] == mode
            for case in run["cases"]
        ]
        mode_metrics = _metrics(turn for case in cases for turn in case["turns"])
        mode_metrics["completed_dialogues"] = sum(case["complete"] for case in cases)
        mode_metrics["successful_dialogues"] = sum(
            case["complete"] and case["success"] for case in cases
        )
        by_mode[mode] = mode_metrics
    report["by_mode"] = by_mode


def evaluate_grounded_dialogue(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    modes: tuple[str, ...] = DEFAULT_MODES,
    seconds: float = 60.0,
    progress: Callable[[dict[str, Any]], None] | None = None,
    split: str = "held_out",
) -> dict[str, Any]:
    """Evaluate identical frozen scenarios with complete completed-prefix traces.

    A cooperative global deadline is checked before/after calls. The CLI must
    additionally supervise this in a hard-timeout subprocess: Python cannot
    interrupt an arbitrary blocking extension with a cooperative deadline.
    ``progress`` receives independent JSON-safe report snapshots after each
    returned turn; it can atomically persist checkpoints before a hard kill.
    """
    if (
        type(seeds) is not tuple
        or not 1 <= len(seeds) <= 8
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be 1 to 8 unique uint32 integers in a tuple")
    if (
        type(modes) is not tuple
        or not 1 <= len(modes) <= len(DEFAULT_MODES)
        or any(type(mode) is not str or mode not in DEFAULT_MODES for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("modes must be a nonempty unique tuple of supported controls")
    if (
        type(seconds) not in (int, float)
        or not isfinite(seconds)
        or not 0 < seconds <= 600
    ):
        raise ValueError("seconds must be finite and in (0, 600]")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    started = perf_counter()
    deadline = started + seconds
    cases = make_dialogue_cases(split)
    pairs = _teaching_pairs()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "complete": False,
        "status": "running",
        "reason": "",
        "seeds": list(seeds),
        "modes": list(modes),
        "seconds_budget": seconds,
        "planned_runs": len(seeds) * len(modes),
        "planned_dialogues": len(seeds) * len(modes) * len(cases),
        "planned_turns": len(seeds)
        * len(modes)
        * sum(len(case.turns) for case in cases),
        "corpus": corpus_manifest(split),
        "source_sha256": _source_hashes(),
        "teaching": {
            "sha256": _json_digest(pairs),
            "pairs": pairs,
            "epochs": 8,
            "test_dialogues_used_for_fit": False,
            "test_expectations_available_to_production_bridge": False,
        },
        "limits": ConversationLimits().to_dict(),
        "runs": [],
        "failures": [],
        "measurement_scope": {
            "grammar_roles_state_policy_renderer": "explicit_engineering_scaffolds",
            "factor_contribution": (
                "learned_lexical_cue_to_predicate_with_factor_memory"
            ),
            "semantic_assertions": "exact_typed_tuples_excluding_event_ids",
            "text_fluency_scored": False,
            "safe_declines_are_not_runtime_failures": True,
            "prefix_metrics_are_not_full_run_metrics": True,
            "completion_is_not_accuracy": True,
            "hard_timeout_requires_external_supervisor": True,
            "untrained": "lexical_transform_untrained_semantic_anchors_present",
            "shuffled": "training_target_permutation_preserves_label_frequencies",
            "nearest": "exact_normalized_taught_cue_lookup_baseline",
            "oracle": "evaluator_only_teacher_predicate_control_not_learned",
        },
    }
    current_run: dict[str, Any] | None = None
    current_case: dict[str, Any] | None = None
    current_turn_index: int | None = None

    def checkpoint() -> None:
        _refresh_report(report, started)
        if progress is not None:
            progress(
                json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))
            )

    def remaining() -> float:
        left = deadline - perf_counter()
        if left <= 0:
            raise TimeoutError("evaluation_global_deadline")
        return left

    checkpoint()
    try:
        for seed in seeds:
            for mode in modes:
                remaining()
                current_run = {
                    "seed": seed,
                    "mode": mode,
                    "complete": False,
                    "cases": [],
                }
                current_case = None
                current_turn_index = None
                report["runs"].append(current_run)
                bridge, training = _make_bridge(seed, mode, pairs, remaining())
                current_run["training"] = training
                checkpoint()
                remaining()
                for case in cases:
                    current_case = {
                        "case_id": case.case_id,
                        "category": case.category,
                        "planned_turns": len(case.turns),
                        "turns": [],
                        "complete": False,
                        "success": False,
                    }
                    current_run["cases"].append(current_case)
                    current_turn_index = None
                    session = _new_session(seed, bridge, remaining())
                    remaining()
                    for index, expected in enumerate(case.turns):
                        current_turn_index = index
                        remaining()
                        turn_started = perf_counter()
                        response = session.respond(
                            expected.text, request_id=f"{case.case_id}:{index}"
                        )
                        elapsed = perf_counter() - turn_started
                        facts = [fact.to_dict() for fact in session.state.facts()]
                        scored = _score_turn(
                            expected, response.to_dict(), facts, elapsed
                        )
                        current_case["turns"].append(scored)
                        # Persist the returned turn even if it exceeded time.
                        checkpoint()
                        remaining()
                        if response.reason in {
                            "turn_time_budget",
                            "time_budget",
                            "incomplete_recognition",
                        }:
                            raise TimeoutError(
                                f"turn_runtime_timeout:{response.reason}"
                            )
                        if response.reason in {
                            "internal_error",
                            "invalid_input",
                            "busy",
                            "session_capacity",
                            "input_capacity",
                        }:
                            raise RuntimeError(
                                f"contained_turn_failure:{response.reason}"
                            )
                    current_case["complete"] = True
                    current_case["success"] = all(
                        turn["success"] for turn in current_case["turns"]
                    )
                    checkpoint()
                current_run["complete"] = True
                checkpoint()
        report["complete"] = True
        report["status"] = "completed"
    except (TimeoutError, BudgetExceeded) as error:
        report["status"] = "timed_out"
        report["reason"] = str(error)
        report["interrupted_at"] = {
            "seed": None if current_run is None else current_run["seed"],
            "mode": None if current_run is None else current_run["mode"],
            "case_id": None if current_case is None else current_case["case_id"],
            "turn_index": current_turn_index,
        }
    except Exception as error:
        report["status"] = "failed"
        report["reason"] = f"{type(error).__name__}: {error}"
        report["failures"].append(
            {
                "seed": None if current_run is None else current_run["seed"],
                "mode": None if current_run is None else current_run["mode"],
                "case_id": None if current_case is None else current_case["case_id"],
                "turn_index": current_turn_index,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )
    checkpoint()
    # dataclass/asdict semantic tuples are convenient internally, but the
    # supervised worker's strict serializer accepts dict/list JSON trees only.
    # Normalize every terminal status, matching the checkpoint contract.
    return json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))
