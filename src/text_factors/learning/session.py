"""Transactional dialogue using learned understanding, dynamics, policy and LM."""

from __future__ import annotations

import threading
from copy import deepcopy
from math import isfinite
from time import perf_counter
from typing import Any

from ..conversation.persistence import encode_json
from ..conversation.schema import Budget, BudgetExceeded, ConversationLimits
from .attention import (
    AttentionLimits,
    AttentionState,
    ReviewCue,
    memory_token,
    validate_attention_trace,
)
from .candidate_search import SearchLimits
from .candidate_selection import select, validate_trace
from .hypotheses import Observation, digest
from .model import ModelBundle
from .schema import (
    DialogueContext,
    Entity,
    Interpretation,
    Meaning,
    bounded_text,
    exact_fields,
)
from .world import ExperienceWorld, validate_assertion

SCHEMA = "ai2-learned-dialogue-session-v1"
_SAFE_ACTIONS = {
    "answer": ("answer", "clarify", "unknown"),
    "ack": ("ack", "clarify"),
    "corrected": ("corrected", "clarify"),
    "retracted": ("retracted", "unknown"),
    "nonactual": ("nonactual", "clarify"),
    "explain": ("explain", "unknown"),
    "clarify": ("clarify", "unknown"),
    "unknown": ("unknown", "clarify"),
    "greet": ("greet", "help"),
    "thanks": ("thanks", "help"),
    "help": ("help",),
}
_MUTATING = {"ack", "corrected", "retracted", "nonactual"}
_SEMANTIC_FACT_FIELDS = {"subject", "relation", "value", "negated", "spatial"}
_MAX_DIAGNOSTIC_BYTES = 262_144


def _finite_number(value: Any, name: str) -> None:
    if type(value) not in (float, int):
        raise ValueError(f"invalid {name}")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"invalid {name}")


def _same_json(left: Any, right: Any) -> bool:
    return encode_json({"value": left}) == encode_json({"value": right})


class LearnedSession:
    def __init__(
        self, bundle: ModelBundle, limits: ConversationLimits | None = None
    ) -> None:
        if not isinstance(bundle, ModelBundle):
            raise ValueError("expected ModelBundle")
        if limits is not None and not isinstance(limits, ConversationLimits):
            raise ValueError("expected ConversationLimits")
        self.bundle = bundle
        self.limits = limits or ConversationLimits(max_state_bytes=3_000_000)
        self.model_fingerprint = bundle.fingerprint
        self.world = ExperienceWorld(
            max_events=self.limits.max_events, max_entities=self.limits.max_entities
        )
        self.context = DialogueContext()
        self.attention = AttentionState(self.model_fingerprint)
        self.turn_count = 0
        self._previous_action = ""
        self._last_assertions: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        encode_json(self._snapshot(), max_bytes=self.limits.max_state_bytes)

    def _snapshot(self) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "model_fingerprint": self.model_fingerprint,
            "limits": self.limits.to_dict(),
            "world": self.world.to_dict(),
            "context": self.context.to_dict(),
            "turn_count": self.turn_count,
            "previous_action": self._previous_action,
            "last_assertions": deepcopy(self._last_assertions),
            "history": deepcopy(self._history),
        }
        # An empty default archive carries no observations or learned updates.
        # Keep the legacy compact empty session usable under small byte budgets.
        if self.attention.generation or self.attention.limits != AttentionLimits():
            value["attention"] = self.attention.to_dict()
        return value

    def to_dict(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise ValueError("session is busy")
        try:
            value = self._snapshot()
            encode_json(value, max_bytes=self.limits.max_state_bytes)
            return value
        finally:
            self._lock.release()

    @staticmethod
    def _request_id(value: Any) -> None:
        if value is not None:
            bounded_text(value, "request id", empty=False)
            if any(char.isspace() for char in value):
                raise ValueError("request id cannot contain whitespace")

    def _failure(self, action: str, reason: str, *, started: float) -> dict[str, Any]:
        # Emergency messages are disclosed fixed safety responses, not LM output.
        text = (
            "Достигнут предел обработки. Реплика не изменила память."
            if action == "limit"
            else "Не удалось безопасно обработать реплику. Память не изменена."
        )
        return {
            "turn_id": self.turn_count,
            "text": text,
            "action": action,
            "complete": False,
            "reason": reason,
            "meaning": None,
            "assertions": [],
            "evidence": [],
            "diagnostics": {"emergency_response": True},
            "elapsed_seconds": round(perf_counter() - started, 6),
        }

    def _next_context(
        self, text: str, meaning: Meaning | None, action: str
    ) -> DialogueContext:
        return self._advance_context(self.context, text, meaning, action)

    @staticmethod
    def _advance_context(
        context: DialogueContext, text: str, meaning: Meaning | None, action: str
    ) -> DialogueContext:
        entities = {e.name: e for e in context.entities}
        focus = list(context.focus)
        mentioned: list[str] = []
        if meaning:
            entities.update({e.name: e for e in meaning.entities})
            event = meaning.event
            while event and event.content:
                event = event.content
            if event:
                mentioned = [
                    name
                    for name in (
                        event.object,
                        event.actor,
                        event.recipient,
                        event.place,
                    )
                    if name
                ]
            if meaning.query and meaning.query.subject:
                mentioned = [meaning.query.subject]
            for name in mentioned:
                entities.setdefault(name, Entity(name))
        focus = list(dict.fromkeys([*mentioned, *focus]))[:16]
        # Eviction is context-only. Evidence remains in the bounded event ledger.
        order = list(dict.fromkeys([*focus, *reversed(entities)]))[:64]
        return DialogueContext(
            (*context.turns, text)[-16:],
            tuple(entities[name] for name in order if name in entities),
            tuple(focus),
            "clarify" if action == "clarify" else "",
        )

    @staticmethod
    def _slots(fact: dict[str, Any] | None, truth: str = "") -> dict[str, str]:
        if not fact:
            return {}
        slots = {"object": fact["subject"], "source": fact["source"]}
        if fact["relation"] == "location":
            slots.update(
                place=fact["value"], prep="на" if fact["spatial"] == "on" else "в"
            )
        else:
            slots["holder"] = fact["value"]
        if truth:
            slots["truth"] = truth
        return slots

    @staticmethod
    def _remaining(budget: Budget, cap: float) -> float:
        budget.check()
        remaining = budget.deadline - perf_counter()
        if remaining <= 0:
            raise BudgetExceeded("turn_time_budget")
        return min(cap, remaining)

    def respond(self, text: str, request_id: str | None = None) -> dict[str, Any]:
        started = perf_counter()
        self._request_id(request_id)
        if not self._lock.acquire(blocking=False):
            return self._failure("error", "session_busy", started=started)
        processing = False
        try:
            if type(text) is not str:
                raise ValueError("utterance must be a string")
            if len(text) > min(self.limits.max_chars, 2048) or len(text.split()) > min(
                self.limits.max_tokens, 96
            ):
                return self._failure("limit", "input_capacity", started=started)
            bounded_text(text, "utterance", cap=self.limits.max_chars)
            for receipt in self._history:
                if request_id is not None and receipt["request_id"] == request_id:
                    if receipt["input"] != text:
                        raise ValueError("request id was reused for different input")
                    return deepcopy(receipt["response"])
            if self.turn_count >= 2**53 - 1:
                return self._failure("limit", "turn_counter_capacity", started=started)
            budget = Budget(self.limits.turn_seconds)
            processing = True
            turn_id = self.turn_count + 1
            candidate_world = self.world.clone()
            candidate_attention = self.attention.clone()
            interpretation = self.bundle.understanding.interpret(
                text, context=self.context
            )
            budget.check()
            semantic_facts = [
                {key: fact[key] for key in _SEMANTIC_FACT_FIELDS}
                for fact in self.world.facts()
            ]
            proposals = self.bundle.understanding.propose(
                text,
                self.context,
                observation=Observation(
                    f"turn:{turn_id}", text, turn_id, f"сообщение {turn_id}"
                ),
                initial=interpretation,
                limits=SearchLimits(seconds=self._remaining(budget, 0.4)),
            )
            interpretation = select(
                proposals,
                interpretation,
                self.bundle.dynamics,
                semantic_facts,
                model_fingerprint=self.model_fingerprint,
                seconds=self._remaining(budget, 0.8),
                dependency_event_ids=tuple(
                    sorted({fact["event_id"] for fact in self.world.facts()})
                ),
            )
            budget.check()
            meaning = interpretation.meaning
            attention_trace = candidate_attention.process(
                proposals,
                meaning,
                self.bundle.dynamics,
                self.bundle.understanding,
                seconds=self._remaining(budget, candidate_attention.limits.seconds),
            )
            if not attention_trace["complete"]:
                raise BudgetExceeded(attention_trace["reason"])
            assert interpretation.diagnostics is not None
            interpretation.diagnostics["attention"] = attention_trace
            candidate_attention.remember(
                interpretation.diagnostics["hypotheses"], self.context, semantic_facts
            )
            diagnostics: dict[str, Any] = {
                "understanding": {
                    "score": float(interpretation.score),
                    "reason": interpretation.reason,
                    "alternatives": [m.to_dict() for m in interpretation.alternatives],
                    "details": interpretation.diagnostics or {},
                }
            }
            outcome: dict[str, Any] = {
                "action": "unknown",
                "reason": interpretation.reason,
                "assertions": [],
                "evidence": [],
            }
            if meaning is None:
                outcome["action"] = (
                    "clarify"
                    if interpretation.alternatives
                    or "ambig" in interpretation.reason
                    or "reference" in interpretation.reason
                    else "unknown"
                )
            elif meaning.event is not None:
                prediction = self.bundle.dynamics.predict(
                    semantic_facts, meaning.event, seconds=self._remaining(budget, 1.0)
                )
                diagnostics["dynamics"] = prediction.to_dict()
                budget.check()
                if not prediction.supported and prediction.reason == "time_budget":
                    raise TimeoutError("dynamics time budget")
                if prediction.supported:
                    outcome = candidate_world.apply(
                        meaning,
                        prediction.effects,
                        turn_id=turn_id,
                        source=f"сообщение {turn_id}",
                        budget=budget,
                    )
                else:
                    outcome.update(action="clarify", reason=prediction.reason)
            elif meaning.query is not None:
                outcome = (
                    self.world.explain(
                        self._last_assertions, subject=meaning.query.subject
                    )
                    if meaning.query.kind == "why"
                    else self.world.query(meaning.query)
                )
            elif meaning.act == "retract":
                outcome = candidate_world.apply(
                    meaning,
                    (),
                    turn_id=turn_id,
                    source=f"сообщение {turn_id}",
                    budget=budget,
                )
            elif meaning.act in {"greet", "thanks", "help"}:
                outcome.update(action=meaning.act, reason="")
            budget.check()
            expected = outcome["action"]
            features = {
                "act": meaning.act if meaning else "unknown",
                "has_answer": bool(outcome["assertions"]),
                "ambiguous": expected == "clarify",
                "unsupported": meaning is None,
                "nonactual": expected == "nonactual",
                "corrected": expected == "corrected",
                "retracted": expected == "retracted",
                "pending": self.context.pending,
                "task": "facts",
                "prev_action": self._previous_action,
                "evidence_count": len(outcome["assertions"]),
                "failed": False,
            }
            decision = self.bundle.policy.choose(features, _SAFE_ACTIONS[expected])
            action = decision.action
            if type(action) is not str or action not in _SAFE_ACTIONS[expected]:
                return self._failure(
                    "error", "policy_action_not_eligible", started=started
                )
            diagnostics["policy"] = {"action": action, "scores": dict(decision.scores)}
            self._validate_policy(diagnostics["policy"], _SAFE_ACTIONS[expected])
            if action != expected:
                candidate_world = self.world.clone()
                outcome = {
                    "action": action,
                    "reason": "policy_requested_clarification"
                    if action == "clarify"
                    else "policy_abstained",
                    "assertions": [],
                    "evidence": [],
                }
            if action not in _MUTATING:
                candidate_world = self.world.clone()
            assertions = outcome["assertions"]
            if len(assertions) > self.limits.max_candidates:
                action = "clarify"
                outcome = {
                    "action": action,
                    "reason": "answer_capacity",
                    "assertions": [],
                    "evidence": [],
                }
                assertions = []
                candidate_world = self.world.clone()
            segments: list[dict[str, Any]] = []
            for fact in assertions or [None]:
                budget.check()
                slots = self._slots(fact, outcome.get("truth", ""))
                evidence = [fact] if fact else []
                generated = self.bundle.generator.generate(
                    action,
                    slots,
                    evidence,
                    max_tokens=48,
                    seconds=self._remaining(budget, 0.5),
                )
                if generated.reason == "generation_deadline":
                    raise TimeoutError("generation time budget")
                if (
                    not generated.grounded
                    or not self.bundle.generator.verify(
                        action, slots, evidence, generated
                    )
                    or not 1 <= len(generated.tokens) <= 48
                ):
                    return self._failure(
                        "error", "generation_grounding_failed", started=started
                    )
                segments.append(
                    {
                        "text": generated.text,
                        "tokens": list(generated.tokens),
                        "slots": slots,
                        "assertions": evidence,
                    }
                )
            output = " ".join(s["text"] for s in segments)
            if len(output) > self.limits.max_chars:
                raise BudgetExceeded("response_capacity")
            bounded_text(
                output, "generated response", cap=self.limits.max_chars, empty=False
            )
            diagnostics["generation"] = {"segments": segments, "grounded": True}
            response = {
                "turn_id": turn_id,
                "text": output,
                "action": action,
                "complete": action not in {"clarify", "unknown"},
                "reason": outcome["reason"],
                "meaning": meaning.to_dict() if meaning else None,
                "assertions": assertions,
                "evidence": outcome["evidence"],
                "diagnostics": diagnostics,
                "elapsed_seconds": round(perf_counter() - started, 6),
            }
            self._diagnostics(response, meaning)
            context = self._next_context(text, meaning, action)
            last_assertions = (
                deepcopy(assertions) if action in {"answer", "explain"} else []
            )
            history = [
                *self._history,
                {
                    "input": text,
                    "request_id": request_id,
                    "response": deepcopy(response),
                },
            ][-self.limits.max_history :]
            snapshot = {
                "schema": SCHEMA,
                "model_fingerprint": self.model_fingerprint,
                "limits": self.limits.to_dict(),
                "world": candidate_world.to_dict(),
                "context": context.to_dict(),
                "turn_count": turn_id,
                "previous_action": action,
                "last_assertions": last_assertions,
                "history": history,
                "attention": candidate_attention.to_dict(),
            }
            try:
                encode_json(snapshot, max_bytes=self.limits.max_state_bytes)
            except ValueError as exc:
                if "budget" in str(exc):
                    raise BudgetExceeded("state_capacity") from exc
                raise
            budget.check()
            self.world, self.context, self.turn_count = (
                candidate_world,
                context,
                turn_id,
            )
            self._previous_action, self._last_assertions, self._history = (
                action,
                last_assertions,
                history,
            )
            self.attention = candidate_attention
            return deepcopy(response)
        except BudgetExceeded as exc:
            return self._failure("limit", str(exc), started=started)
        except TimeoutError:
            return self._failure("limit", "component_time_budget", started=started)
        except Exception as exc:
            if not processing:
                raise
            return self._failure(
                "error",
                f"component_contract_failed:{type(exc).__name__}",
                started=started,
            )
        finally:
            self._lock.release()

    @staticmethod
    def _validate_policy(policy: dict[str, Any], eligible: tuple[str, ...]) -> None:
        scores = policy["scores"]
        if type(scores) is not dict or set(scores) != set(eligible):
            raise ValueError("policy scores do not match eligible actions")
        for score in scores.values():
            _finite_number(score, "policy score")
        if policy["action"] != max(
            sorted(eligible), key=lambda candidate: scores[candidate]
        ):
            raise ValueError("policy choice is inconsistent with its eligible scores")

    @staticmethod
    def _diagnostics(
        response: dict[str, Any], meaning: Meaning | None
    ) -> dict[str, Any]:
        diagnostics = response["diagnostics"]
        encode_json(diagnostics, max_bytes=_MAX_DIAGNOSTIC_BYTES)
        expected = {"understanding", "policy", "generation"}
        if meaning is not None and meaning.event is not None:
            expected.add("dynamics")
        exact_fields(diagnostics, expected, "response diagnostics")
        understanding = exact_fields(
            diagnostics["understanding"],
            {"score", "reason", "alternatives", "details"},
            "understanding diagnostics",
        )
        alternatives = understanding["alternatives"]
        if type(alternatives) is not list or len(alternatives) > 8:
            raise ValueError("invalid interpretation alternatives")
        Interpretation(
            meaning,
            understanding["score"],
            tuple(Meaning.from_dict(item) for item in alternatives),
            understanding["reason"],
            understanding["details"],
        )
        if type(understanding["details"]) is not dict:
            raise ValueError("invalid understanding detail metadata")
        if "hypotheses" in understanding["details"]:
            batch = validate_trace(understanding["details"]["hypotheses"], meaning)
            if "attention" in understanding["details"]:
                validate_attention_trace(understanding["details"]["attention"], batch)
        elif "attention" in understanding["details"]:
            raise ValueError("attention receipt has no observation")
        policy = exact_fields(
            diagnostics["policy"], {"action", "scores"}, "policy receipt"
        )
        if type(policy["action"]) is not str or policy["action"] not in _SAFE_ACTIONS:
            raise ValueError("invalid policy action")
        scores = policy["scores"]
        if type(scores) is not dict or not 1 <= len(scores) <= len(_SAFE_ACTIONS):
            raise ValueError("invalid policy scores")
        for action, score in scores.items():
            if type(action) is not str or action not in _SAFE_ACTIONS:
                raise ValueError("invalid scored policy action")
            _finite_number(score, "policy score")
        if "dynamics" in diagnostics:
            prediction = exact_fields(
                diagnostics["dynamics"],
                {
                    "effects",
                    "supported",
                    "score",
                    "reason",
                    "context_scores",
                    "evidence",
                },
                "dynamics receipt",
            )
            if (
                type(prediction["supported"]) is not bool
                or type(prediction["effects"]) is not list
                or len(prediction["effects"]) > 1
                or type(prediction["context_scores"]) is not list
                or len(prediction["context_scores"]) > 2
                or (
                    prediction["evidence"] is not None
                    and type(prediction["evidence"]) is not dict
                )
            ):
                raise ValueError("invalid dynamics metadata")
            bounded_text(prediction["reason"], "prediction reason", cap=256)
            _finite_number(prediction["score"], "prediction score")
        generation = exact_fields(
            diagnostics["generation"], {"segments", "grounded"}, "generation receipt"
        )
        if generation["grounded"] is not True:
            raise ValueError("cached generation is not verified")
        return diagnostics

    @staticmethod
    def _receipt_outcome(
        response: dict[str, Any],
        meaning: Meaning | None,
        historical: ExperienceWorld,
        previous_answer: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        diagnostics = response["diagnostics"]
        reason = diagnostics["understanding"]["reason"]
        outcome: dict[str, Any] = {
            "action": "unknown",
            "reason": reason,
            "assertions": [],
            "evidence": [],
        }
        if meaning is None:
            if (
                diagnostics["understanding"]["alternatives"]
                or "ambig" in reason
                or "reference" in reason
            ):
                outcome["action"] = "clarify"
        elif meaning.event is not None:
            prediction = diagnostics["dynamics"]
            if not prediction["supported"]:
                if prediction["effects"]:
                    raise ValueError("unsupported prediction contains asserted effects")
                outcome.update(action="clarify", reason=prediction["reason"])
            else:
                outcome = historical.clone().apply(
                    meaning,
                    prediction["effects"],
                    turn_id=response["turn_id"],
                    source=f"сообщение {response['turn_id']}",
                )
        elif meaning.query is not None:
            if meaning.query.kind == "why":
                # A truncated window may not include the preceding answer.
                # Even then its claimed facts must have historical evidence.
                reference = (
                    response["assertions"]
                    if previous_answer is None
                    else previous_answer
                )
                outcome = historical.explain(reference, subject=meaning.query.subject)
            else:
                outcome = historical.query(meaning.query)
        elif meaning.act == "retract":
            outcome = historical.clone().apply(
                meaning,
                (),
                turn_id=response["turn_id"],
                source=f"сообщение {response['turn_id']}",
            )
        elif meaning.act in {"greet", "thanks", "help"}:
            outcome.update(action=meaning.act, reason="")
        return outcome

    @classmethod
    def from_dict(cls, value: Any, bundle: ModelBundle) -> LearnedSession:
        from .dialogue_learning import GeneratedReply

        if not isinstance(bundle, ModelBundle):
            raise ValueError("expected ModelBundle")
        fields = {
            "schema",
            "model_fingerprint",
            "limits",
            "world",
            "context",
            "turn_count",
            "previous_action",
            "last_assertions",
            "history",
        }
        if type(value) is dict and "attention" in value:
            fields.add("attention")
        value = exact_fields(
            value,
            fields,
            "learned session",
        )
        limits = ConversationLimits.from_dict(value["limits"])
        encode_json(value, max_bytes=limits.max_state_bytes)
        if (
            value["schema"] != SCHEMA
            or value["model_fingerprint"] != bundle.fingerprint
        ):
            raise ValueError("session and trained model do not match")
        count = value["turn_count"]
        if type(count) is not int or not 0 <= count < 2**53:
            raise ValueError("invalid session turn count")
        history = value["history"]
        if type(history) is not list or len(history) != min(count, limits.max_history):
            raise ValueError("invalid session receipt count")
        if type(value["previous_action"]) is not str or value[
            "previous_action"
        ] not in {*_SAFE_ACTIONS, ""}:
            raise ValueError("invalid previous action")
        if (
            type(value["last_assertions"]) is not list
            or len(value["last_assertions"]) > limits.max_candidates
        ):
            raise ValueError("invalid last answer")
        world = ExperienceWorld.from_dict(value["world"])
        events = world.events
        if (
            world.max_events != limits.max_events
            or world.max_entities != limits.max_entities
            or (events and events[-1]["turn_id"] > count)
        ):
            raise ValueError("world and session limits/turns disagree")
        if any(
            record["source"] != f"сообщение {record['turn_id']}" for record in events
        ):
            raise ValueError("world evidence source does not match its session turn")
        context = DialogueContext.from_dict(value["context"])
        names = [entity.name for entity in context.entities]
        if (
            len(names) != len(set(names))
            or len(context.focus) != len(set(context.focus))
            or not set(context.focus) <= set(names)
            or len(context.turns) != min(count, 16)
        ):
            raise ValueError("invalid saved context references or turn count")
        historical = ExperienceWorld(
            max_events=limits.max_events, max_entities=limits.max_entities
        )
        cursor = 0
        seen: set[str] = set()
        previous_answer: list[dict[str, Any]] | None = None
        recent_context = DialogueContext()
        latest_entities: dict[str, Entity] = {}
        for index, receipt in enumerate(history):
            exact_fields(receipt, {"input", "request_id", "response"}, "receipt")
            bounded_text(
                receipt["input"], "saved input", cap=min(limits.max_chars, 2048)
            )
            if len(receipt["input"].split()) > min(limits.max_tokens, 96):
                raise ValueError("saved input exceeds token capacity")
            cls._request_id(receipt["request_id"])
            if receipt["request_id"] is not None:
                if receipt["request_id"] in seen:
                    raise ValueError("duplicate request id")
                seen.add(receipt["request_id"])
            response = exact_fields(
                receipt["response"],
                {
                    "turn_id",
                    "text",
                    "action",
                    "complete",
                    "reason",
                    "meaning",
                    "assertions",
                    "evidence",
                    "diagnostics",
                    "elapsed_seconds",
                },
                "saved response",
            )
            turn_id = response["turn_id"]
            if type(turn_id) is not int or turn_id != count - len(history) + index + 1:
                raise ValueError("receipts must form the contiguous most recent window")
            bounded_text(
                response["text"],
                "saved response text",
                cap=limits.max_chars,
                empty=False,
            )
            bounded_text(response["reason"], "saved response reason", cap=256)
            _finite_number(response["elapsed_seconds"], "saved response elapsed time")
            if response["elapsed_seconds"] < 0:
                raise ValueError("negative saved response elapsed time")
            action = response["action"]
            if type(action) is not str or action not in _SAFE_ACTIONS:
                raise ValueError("invalid saved action")
            if type(response["complete"]) is not bool or response["complete"] != (
                action not in {"clarify", "unknown"}
            ):
                raise ValueError("saved completion status does not match action")
            meaning = (
                Meaning.from_dict(response["meaning"])
                if response["meaning"] is not None
                else None
            )
            diagnostics = cls._diagnostics(response, meaning)
            if (
                type(response["assertions"]) is not list
                or len(response["assertions"]) > limits.max_candidates
            ):
                raise ValueError("invalid cached assertions")
            assertions = [validate_assertion(fact) for fact in response["assertions"]]
            if (
                type(response["evidence"]) is not list
                or len(response["evidence"]) > limits.max_candidates
            ):
                raise ValueError("invalid cached evidence")
            mutation_record = None
            while cursor < len(events) and events[cursor]["turn_id"] < turn_id:
                record = events[cursor]
                historical.apply(
                    Meaning.from_dict(record["meaning"]),
                    record["effects"],
                    turn_id=record["turn_id"],
                    source=record["source"],
                )
                cursor += 1
            if cursor < len(events) and events[cursor]["turn_id"] == turn_id:
                mutation_record = events[cursor]
            trace = diagnostics["understanding"]["details"].get("hypotheses")
            if trace is not None:
                batch = validate_trace(trace, meaning)
                historical_facts = [
                    {key: fact[key] for key in _SEMANTIC_FACT_FIELDS}
                    for fact in historical.facts()
                ]
                if (
                    batch.observation
                    != Observation(
                        f"turn:{turn_id}",
                        receipt["input"],
                        turn_id,
                        f"сообщение {turn_id}",
                    )
                    or trace["snapshot"]["model_fingerprint"]
                    != value["model_fingerprint"]
                    or trace["snapshot"]["before_digest"] != digest(historical_facts)
                    or trace["snapshot"]["dependency_event_ids"]
                    != sorted({fact["event_id"] for fact in historical.facts()})
                    or (
                        count == len(history)
                        and batch.context_digest != digest(recent_context.to_dict())
                    )
                ):
                    raise ValueError("candidate observation or world snapshot mismatch")
            try:
                expected = cls._receipt_outcome(
                    response, meaning, historical, previous_answer
                )
            except BudgetExceeded as exc:
                raise ValueError(
                    "cached proposed mutation exceeds world capacity"
                ) from exc
            policy = diagnostics["policy"]
            eligible = _SAFE_ACTIONS[expected["action"]]
            cls._validate_policy(policy, eligible)
            selected = policy["action"]
            if selected != expected["action"]:
                expected = {
                    "action": selected,
                    "assertions": [],
                    "evidence": [],
                    "reason": "policy_requested_clarification"
                    if selected == "clarify"
                    else "policy_abstained",
                }
            if len(expected["assertions"]) > limits.max_candidates:
                expected = {
                    "action": "clarify",
                    "reason": "answer_capacity",
                    "assertions": [],
                    "evidence": [],
                }
            if (
                action != expected["action"]
                or response["reason"] != expected["reason"]
                or not _same_json(assertions, expected["assertions"])
                or not _same_json(response["evidence"], expected["evidence"])
            ):
                raise ValueError(
                    "cached response does not match historical meaning and evidence"
                )
            if action in _MUTATING:
                if mutation_record is None or not _same_json(
                    expected["evidence"], [mutation_record]
                ):
                    raise ValueError("cached mutation has no matching world event")
                historical.apply(
                    Meaning.from_dict(mutation_record["meaning"]),
                    mutation_record["effects"],
                    turn_id=turn_id,
                    source=mutation_record["source"],
                )
                cursor += 1
            elif mutation_record is not None:
                raise ValueError("nonmutating response has a world mutation")
            if any(fact not in historical.facts() for fact in assertions):
                raise ValueError("cached answer has no historical evidence")
            segments = diagnostics["generation"]["segments"]
            if type(segments) is not list or len(segments) != max(1, len(assertions)):
                raise ValueError("invalid generation receipt")
            for segment, fact in zip(segments, assertions or [None], strict=True):
                exact_fields(
                    segment,
                    {"text", "tokens", "slots", "assertions"},
                    "generation segment",
                )
                segment_facts = [fact] if fact is not None else []
                slots = cls._slots(fact, expected.get("truth", ""))
                if not _same_json(segment["slots"], slots) or not _same_json(
                    segment["assertions"], segment_facts
                ):
                    raise ValueError(
                        "cached generation slots do not match query and evidence"
                    )
                reply = GeneratedReply.from_dict(
                    {
                        "text": segment["text"],
                        "tokens": segment["tokens"],
                        "grounded": True,
                        "reason": "",
                    }
                )
                if not 1 <= len(reply.tokens) <= 48 or not bundle.generator.verify(
                    action, slots, segment_facts, reply
                ):
                    raise ValueError("cached decoder tokens violate evidence grounding")
            if " ".join(segment["text"] for segment in segments) != response["text"]:
                raise ValueError("cached text and generation segments disagree")
            previous_answer = assertions if action in {"answer", "explain"} else []
            recent_context = cls._advance_context(
                recent_context, receipt["input"], meaning, action
            )
            latest_entities = (
                {entity.name: entity for entity in meaning.entities} if meaning else {}
            )
        last = [validate_assertion(fact) for fact in value["last_assertions"]]
        expected_previous = history[-1]["response"]["action"] if history else ""
        expected_pending = "clarify" if expected_previous == "clarify" else ""
        suffix = tuple(receipt["input"] for receipt in history[-16:])
        if (
            value["previous_action"] != expected_previous
            or context.pending != expected_pending
            or (suffix and context.turns[-len(suffix) :] != suffix)
            or not _same_json(last, previous_answer or [])
            or (count == len(history) and context != recent_context)
            or context.focus[: len(recent_context.focus)] != recent_context.focus
            or any(
                entity != latest_entities[entity.name]
                for entity in context.entities
                if entity.name in latest_entities
            )
        ):
            raise ValueError("saved context or previous answer disagrees with receipts")
        session = cls(bundle, limits)
        if "attention" in value:
            session.attention = AttentionState.from_dict(
                value["attention"], bundle.fingerprint
            )
            for record in session.attention.records:
                observation = Observation.from_dict(record["observation"])
                if (
                    observation.turn_id > count
                    or observation.observation_id != f"turn:{observation.turn_id}"
                    or observation.source != f"сообщение {observation.turn_id}"
                ):
                    raise ValueError("archive turn does not match session")
                if record["review"] is not None and record["review"][
                    "memory_version"
                ] != list(memory_token(bundle.dynamics)):
                    raise ValueError("archive review uses another memory version")
                for raw_cue in record["cues"]:
                    cue = ReviewCue.from_dict(raw_cue)
                    if (
                        cue.observation.turn_id > count
                        or cue.observation.observation_id
                        != f"turn:{cue.observation.turn_id}"
                        or cue.observation.source
                        != f"сообщение {cue.observation.turn_id}"
                    ):
                        raise ValueError(
                            "archive cue is not a prior session observation"
                        )
                    origin = session.attention.get(cue.observation.observation_id)
                    if origin is not None and (
                        origin["observation"] != cue.observation.to_dict()
                        or origin["original_meaning"] != cue.meaning.to_dict()
                    ):
                        raise ValueError(
                            "clarification does not match original observation"
                        )
                receipt = next(
                    (
                        r
                        for r in history
                        if r["response"]["turn_id"] == observation.turn_id
                    ),
                    None,
                )
                if receipt is not None:
                    trace = receipt["response"]["diagnostics"]["understanding"][
                        "details"
                    ].get("hypotheses")
                    if (
                        trace is None
                        or trace["batch"]["observation"] != record["observation"]
                        or trace["snapshot"] != record["original_snapshot"]
                    ):
                        raise ValueError("archive origin disagrees with receipt")
        elif any(
            "attention" in r["response"]["diagnostics"]["understanding"]["details"]
            for r in history
        ):
            raise ValueError("attention state missing from new session")
        session.world, session.context, session.turn_count = world, context, count
        session._previous_action, session._last_assertions, session._history = (
            expected_previous,
            deepcopy(last),
            deepcopy(history),
        )
        return session
