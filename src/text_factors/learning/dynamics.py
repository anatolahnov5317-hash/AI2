"""Learned role-bound transitions and shared-experience interpretation scores.

The ontology, role binding, structural feature basis and two context transforms
are explicit engineering scaffolds. Numeric local clusters learn which effect
template follows an encoded situation; a separate common factor memory compares
interpretations with experienced situations. This is not autonomous discovery
of contexts or a full implementation of Redozubov's theory. Entity names never
enter learned features; outputs copy only validated event roles.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from ..config import ModelConfig
from ..conversation.persistence import _encode_json
from ..memory import Cluster, ClusterStatus, CombinatorialMemory
from ..recognition import ClusterEvidence, ContextView, RecognitionLimits
from ..scene_recognition import (
    FactorPortrait,
    FactorSceneReader,
    SceneRecognitionConfig,
)
from ..transforms import LearnedSDRTransform
from .schema import Event, bounded_text, exact_fields

SCHEMA = "ai2-learned-dynamics-v1"
MAX_EPISODES = 512
MAX_STRUCTURES = 160
MAX_FACTS = 128
MAX_BYTES = 8_000_000
ROLES = ("actor", "object", "recipient", "place")


def _json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _integer(value: Any, lower: int, upper: int, name: str) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"invalid {name}")
    return value


def _budget(seconds: Any) -> tuple[float, Callable[[], None]]:
    if (
        type(seconds) not in (int, float)
        or not isfinite(seconds)
        or not 0 < seconds <= 60
    ):
        raise ValueError("seconds must be finite and in (0, 60]")
    started = perf_counter()

    def check() -> None:
        if perf_counter() - started >= seconds:
            raise TimeoutError("dynamics time budget exceeded")

    return started, check


def checked_facts(value: Any) -> list[dict[str, Any]]:
    if type(value) not in (list, tuple) or len(value) > MAX_FACTS:
        raise ValueError("before/after must be bounded fact lists")
    result = []
    for item in value:
        if (
            type(item) is not dict
            or not {"subject", "relation", "value"} <= set(item)
            or set(item) - {"subject", "relation", "value", "negated", "spatial"}
        ):
            raise ValueError("invalid transition fact fields")
        fact = {"negated": False, "spatial": "in", **item}
        for key in ("subject", "value"):
            bounded_text(fact[key], key, empty=False)
        if (
            type(fact["relation"]) is not str
            or type(fact["spatial"]) is not str
            or fact["relation"] not in {"location", "holder"}
            or fact["spatial"] not in {"in", "on"}
            or type(fact["negated"]) is not bool
        ):
            raise ValueError("invalid transition fact category")
        if fact["relation"] == "holder" and fact["spatial"] != "in":
            raise ValueError("holder facts do not have a spatial qualifier")
        result.append(fact)
    keys = [_json(fact) for fact in result]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate transition facts")
    return sorted(result, key=_json)


def _event(value: Any) -> Event:
    return Event.from_dict(value.to_dict() if isinstance(value, Event) else value)


@dataclass(frozen=True, slots=True)
class TransitionEpisode:
    before: list[dict[str, Any]]
    event: Event
    after: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": checked_facts(self.before),
            "event": _event(self.event).to_dict(),
            "after": checked_facts(self.after),
        }

    @classmethod
    def from_dict(cls, value: Any) -> TransitionEpisode:
        exact_fields(value, {"before", "event", "after"}, "transition episode")
        return cls(
            checked_facts(value["before"]),
            _event(value["event"]),
            checked_facts(value["after"]),
        )


@dataclass(frozen=True, slots=True)
class TransitionPrediction:
    effects: tuple[dict[str, Any], ...] = ()
    supported: bool = False
    score: float = 0.0
    reason: str = "untrained"
    context_scores: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] | None = None

    @property
    def confidence(self) -> float:
        """An uncalibrated support score, not a probability."""
        return self.score

    def to_dict(self) -> dict[str, Any]:
        return json.loads(_json(asdict(self)))


def _template(
    before: list[dict[str, Any]], event: Event, after: list[dict[str, Any]]
) -> dict[str, Any]:
    """Infer a role-copy effect from the observed delta, only during teaching."""
    old = {_json(fact) for fact in before}
    additions = [fact for fact in after if _json(fact) not in old]
    if not additions:
        if before != after:
            raise ValueError("unsupported removal-only teaching delta")
        return {"op": "noop"}
    if len(additions) != 1:
        raise ValueError("episodes must teach at most one fact effect")
    fact = additions[0]
    if not event.object or fact["subject"] != event.object:
        raise ValueError("effect subject is not the event object")
    value_roles = [role for role in ROLES if getattr(event, role) == fact["value"]]
    if len(value_roles) != 1:
        raise ValueError("effect value lacks an unambiguous event-role binding")
    template = {
        "op": "exclude" if fact["negated"] else "set",
        "subject_role": "object",
        "relation": fact["relation"],
        "value_role": value_roles[0],
        "spatial_source": "event" if fact["relation"] == "location" else "in",
    }
    if fact["relation"] == "location" and fact["spatial"] != event.spatial:
        raise ValueError("spatial effect is not supported by the event")
    if not fact["negated"]:
        expected = [
            f
            for f in before
            if not (
                f["subject"] == fact["subject"]
                and (
                    not f["negated"]
                    or all(f[k] == fact[k] for k in ("relation", "value", "spatial"))
                )
            )
        ] + [fact]
    else:
        expected = [
            f
            for f in before
            if not all(
                f[k] == fact[k] for k in ("subject", "relation", "value", "spatial")
            )
        ] + [fact]
    if checked_facts(expected) != after:
        raise ValueError("observed after-state contains unbound or unexplained changes")
    return template


def _features(before: list[dict[str, Any]], event: Event) -> dict[str, Any]:
    """Fixed role-coordinate transform; no names, targets or word tables."""
    where = []
    for fact in before:
        if fact["subject"] != event.object or not event.object or not event.actual:
            continue
        role = tuple(r for r in ROLES if getattr(event, r) == fact["value"])
        where.append(
            (fact["relation"], role or ("other",), fact["negated"], fact["spatial"])
        )
    return {
        "predicate": event.predicate,
        "scope": "actual" if event.actual else "nonactual",
        "negated": event.negated,
        "present_roles": tuple(role for role in ROLES if getattr(event, role)),
        "spatial": event.spatial if event.place else "in",
        "aliases": tuple(
            (left, right)
            for i, left in enumerate(ROLES)
            for right in ROLES[i + 1 :]
            if getattr(event, left) and getattr(event, left) == getattr(event, right)
        ),
        "before": sorted(where),
    }


def _sdr(features: dict[str, Any], seed: int) -> np.ndarray:
    material = _json({"basis": SCHEMA, "seed": seed, "features": features})
    order = sorted(
        range(256), key=lambda i: hashlib.sha256(material + bytes([i])).digest()
    )
    bits = np.zeros(256, dtype=np.bool_)
    bits[order[:32]] = True
    bits.flags.writeable = False
    return bits


def _config(seed: int, bank: str) -> ModelConfig:
    if bank == "output":
        return ModelConfig(
            input_bits=32,
            output_bits=16,
            point_count=64,
            receptive_bits=24,
            create_threshold=3,
            activation_threshold=3,
            min_active_points=2,
            probation_after=2,
            stable_after=3,
            prune_keep_ratio=1.0,
            max_clusters_per_point=8,
            seed=seed,
        )
    return ModelConfig(
        input_bits=256,
        output_bits=32 if bank == "transform" else 8,
        point_count=64 if bank == "transform" else 4,
        receptive_bits=160,
        create_threshold=12,
        activation_threshold=12,
        min_active_points=2,
        probation_after=2,
        stable_after=3,
        prune_keep_ratio=1.0,
        max_clusters_per_point=MAX_STRUCTURES,
        prediction_vote_threshold=2,
        seed=seed,
    )


def _reader(memory: CombinatorialMemory, *, output: bool) -> FactorSceneReader:
    return FactorSceneReader(
        memory,
        config=SceneRecognitionConfig(
            coverage=0.95,
            min_atoms=4,
            min_points=2,
            min_shared_atoms=3,
            max_portraits=8 if output else MAX_STRUCTURES,
            max_portrait_evidence=4096,
            max_atom_visits=262_144,
            seconds=1.0,
        ),
    )


class LearnedDynamics:
    """Atomic learning, source-only proposals and actual numeric checkpoints.

    Structural configurations, not lexical forms or named identities, receive
    independent SDRs. Generalization is exact role-renaming equivariance; new
    structural feature combinations can abstain. Compatibility scores use one
    shared experience memory, with event-role and actor/recipient-exchange views.
    An exchanged view is diagnostic, never an instruction to exchange the
    user's participants. Candidate ranking uses its unmodified role view.
    """

    def __init__(self, seed: int = 42, mode: str = "factor") -> None:
        self.seed = _integer(seed, 0, 2**32 - 1, "seed")
        if type(mode) is not str or mode not in {"factor", "untrained", "shuffled"}:
            raise ValueError("invalid dynamics mode")
        self.mode = mode
        self._templates: tuple[dict[str, Any], ...] = ()
        self._codes: tuple[np.ndarray, ...] = ()
        self._transform: LearnedSDRTransform | None = None
        self._output: FactorSceneReader | None = None
        self._experience: FactorSceneReader | None = None
        self._training: dict[str, Any] = {
            "fingerprint": "",
            "episodes": 0,
            "structures": 0,
            "epochs": 0,
        }

    def fit(
        self,
        episodes: Sequence[TransitionEpisode | dict[str, Any]],
        *,
        epochs: int = 8,
        seconds: float = 20.0,
    ) -> dict[str, Any]:
        started, check = _budget(seconds)
        _integer(epochs, 1, 16, "epochs")
        if (
            type(episodes) not in (list, tuple)
            or not 1 <= len(episodes) <= MAX_EPISODES
        ):
            raise ValueError("episodes must be a bounded nonempty list")
        rows = []
        associations: dict[bytes, tuple[np.ndarray, dict[str, Any]]] = {}
        for raw in episodes:
            check()
            item = raw.to_dict() if isinstance(raw, TransitionEpisode) else raw
            exact_fields(item, {"before", "event", "after"}, "transition episode")
            before, event, after = (
                checked_facts(item["before"]),
                _event(item["event"]),
                checked_facts(item["after"]),
            )
            template = _template(before, event, after)
            features = _features(before, event)
            signature = _json(features)
            if signature in associations and associations[signature][1] != template:
                raise ValueError("conflicting effects for one structural situation")
            associations[signature] = (_sdr(features, self.seed), template)
            rows.append({"before": before, "event": event.to_dict(), "after": after})
        if len(associations) > MAX_STRUCTURES:
            raise ValueError("structural situation capacity exceeded")
        templates = tuple(
            json.loads(raw)
            for raw in sorted({_json(t) for _, t in associations.values()})
        )
        if len(templates) > 8:
            raise ValueError("effect template capacity exceeded")
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, 951])
        ).permutation(32)
        codes = []
        for i in range(len(templates)):
            code = np.zeros(32, dtype=np.bool_)
            code[order[i * 4 : (i + 1) * 4]] = True
            codes.append(code)
        index = {_json(t): i for i, t in enumerate(templates)}
        ordered = [associations[key] for key in sorted(associations)]
        targets = [index[_json(t)] for _, t in ordered]
        if self.mode == "shuffled" and len(set(targets)) > 1:
            permutation = np.random.default_rng(
                np.random.SeedSequence([self.seed, 953])
            ).permutation(len(targets))
            shuffled = [targets[int(i)] for i in permutation]
            if shuffled == targets:
                offset = next(
                    i for i, target in enumerate(targets) if target != targets[0]
                )
                shuffled = targets[offset:] + targets[:offset]
            targets = shuffled
        transform = LearnedSDRTransform(_config(self.seed, "transform"))
        transform.memory = CombinatorialMemory(
            transform.memory.config,
            receptors=transform.memory.receptors,
            output_map=np.tile(np.arange(32, dtype=np.int32), 2),
        )
        common = CombinatorialMemory(_config(self.seed, "output"))
        experience = CombinatorialMemory(_config(self.seed, "experience"))
        for _ in range(4):
            for code in codes:
                check()
                common.observe(code)
        if self.mode != "untrained":
            for epoch in range(epochs):
                for (source, _), target in zip(ordered, targets, strict=True):
                    check()
                    transform.observe(source, codes[target])
                    if epoch < 4:
                        experience.observe(source)
                    check()
        output_reader, experience_reader = (
            _reader(common, output=True),
            _reader(experience, output=False),
        )
        for i, code in enumerate(codes):
            check()
            if not output_reader.observe_portrait(str(i), code):
                raise ValueError("insufficient common effect-anchor evidence")
        if self.mode != "untrained":
            for i, (source, _) in enumerate(ordered):
                check()
                if not experience_reader.observe_portrait(str(i), source):
                    # With too few epochs an explicitly unready model may be
                    # saved, but it must not manufacture learned compatibility.
                    continue
        check()
        training = {
            "fingerprint": hashlib.sha256(
                b"\n".join(sorted({_json(row) for row in rows}))
            ).hexdigest(),
            "episodes": len({_json(row) for row in rows}),
            "structures": len(ordered),
            "epochs": epochs,
        }
        check()
        self._templates, self._codes = templates, tuple(codes)
        self._transform, self._output, self._experience = (
            transform,
            output_reader,
            experience_reader,
        )
        self._training = training
        return {
            "complete": True,
            **training,
            "mode": self.mode,
            "effect_templates": len(templates),
            "presentations": transform.memory.step,
            "elapsed_seconds": perf_counter() - started,
            "independent_presentations": False,
            "basis": "fixed structural roles and scope; entity names excluded",
            "context_mechanism": (
                "explicit transforms compared against shared learned experience"
            ),
        }

    def _compatibility(
        self, before: list[dict[str, Any]], event: Event, check: Callable[[], None]
    ) -> dict[str, Any]:
        if self._experience is None:
            return {
                "score": 0.0,
                "supported": False,
                "contexts": [],
                "reason": "untrained",
            }
        contexts = [("event_roles", event)]
        if event.actor and event.recipient and event.actor != event.recipient:
            contexts.append(
                (
                    "actor_recipient_exchange",
                    replace(event, actor=event.recipient, recipient=event.actor),
                )
            )
        scored = []
        for name, transformed in contexts:
            check()
            bits = _sdr(_features(before, transformed), self.seed)
            result = self._experience.recognize_views(
                [ContextView(name, name, bits, ())],
                total_views=1,
                limits=RecognitionLimits(
                    max_views=1,
                    max_candidates=min(128, MAX_STRUCTURES),
                    max_evidence=8192,
                    seconds=1.0,
                ),
            )
            check()
            if not result.complete:
                raise TimeoutError("incomplete common experience read")
            score = max((p.coverage for p in result.proposals), default=0.0)
            scored.append(
                {
                    "context": name,
                    "score": score,
                    "matches": len(result.proposals),
                    "memory_namespace": self._experience.encoding_id,
                    "input_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                    "kind": "fixed role-coordinate transform; shared factor experience",
                }
            )
        return {
            "score": scored[0]["score"],
            "supported": bool(scored[0]["matches"]),
            "contexts": scored,
            "reason": "shared_experience_comparison",
            "score_is_probability": False,
        }

    def compatibility(
        self, before: list[dict[str, Any]], event: Event, *, seconds: float = 1.0
    ) -> dict[str, Any]:
        _, check = _budget(seconds)
        return self._compatibility(checked_facts(before), _event(event), check)

    def score_interpretations(
        self, before: list[dict[str, Any]], events: list[Event], *, seconds: float = 2.0
    ) -> tuple[dict[str, Any], ...]:
        _, check = _budget(seconds)
        if type(events) is not list or not 1 <= len(events) <= 8:
            raise ValueError("one to eight interpretation candidates required")
        checked = checked_facts(before)
        return tuple(
            {"candidate": i, **self._compatibility(checked, _event(event), check)}
            for i, event in enumerate(events)
        )

    def predict(
        self, before: list[dict[str, Any]], event: Event, *, seconds: float = 1.0
    ) -> TransitionPrediction:
        _, check = _budget(seconds)
        before, event = checked_facts(before), _event(event)
        if self._transform is None or self._output is None:
            return TransitionPrediction()
        try:
            compatibility = self._compatibility(before, event, check)
            contexts = tuple(compatibility["contexts"])
            if not compatibility["supported"]:
                return TransitionPrediction(
                    reason="unfamiliar_structural_context", context_scores=contexts
                )
            source = _sdr(_features(before, event), self.seed)
            check()
            predicted = self._transform.predict(source)
            check()
            read = self._output.recognize_views(
                [ContextView("effect", "learned_transition", predicted.output, ())],
                total_views=1,
                limits=RecognitionLimits(
                    max_views=1, max_candidates=8, max_evidence=4096, seconds=1.0
                ),
            )
            check()
            if not read.complete:
                return TransitionPrediction(
                    reason="incomplete_effect_recognition", context_scores=contexts
                )
            if len(read.proposals) != 1:
                return TransitionPrediction(
                    reason="ambiguous_or_missing_effect", context_scores=contexts
                )
            proposal = read.proposals[0]
            template = self._templates[int(proposal.portrait_id)]
            effects: tuple[dict[str, Any], ...] = ()
            if template["op"] != "noop":
                subject, value = (
                    getattr(event, template["subject_role"]),
                    getattr(event, template["value_role"]),
                )
                if not subject or not value:
                    return TransitionPrediction(
                        reason="missing_role_binding", context_scores=contexts
                    )
                effects = (
                    {
                        "op": template["op"],
                        "subject": subject,
                        "relation": template["relation"],
                        "value": value,
                        "spatial": event.spatial
                        if template["spatial_source"] == "event"
                        else "in",
                    },
                )
            return TransitionPrediction(
                effects,
                True,
                proposal.coverage,
                "learned_no_effect" if not effects else "learned_role_bound_effect",
                contexts,
                {
                    "predicted_bits": predicted.active_output_bits,
                    "template": dict(template),
                    "factor_atoms": proposal.support,
                    "score_is_probability": False,
                    "projection_authority": "proposal only; world must validate",
                },
            )
        except TimeoutError:
            return TransitionPrediction(reason="time_budget")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "seed": self.seed,
            "mode": self.mode,
            "training": dict(self._training),
            "templates": [dict(t) for t in self._templates],
            "codes": [np.flatnonzero(code).tolist() for code in self._codes],
            "transform": _memory_dump(self._transform.memory)
            if self._transform
            else None,
            "output": _reader_dump(self._output) if self._output else None,
            "experience": _reader_dump(self._experience) if self._experience else None,
        }

    @classmethod
    def from_dict(cls, value: Any) -> LearnedDynamics:
        # Bound the complete tree/bytes and reject NaN/cycles before numerical
        # arrays or cluster objects can be allocated.
        _encode_json(value, max_bytes=MAX_BYTES)
        exact_fields(
            value,
            {
                "schema",
                "seed",
                "mode",
                "training",
                "templates",
                "codes",
                "transform",
                "output",
                "experience",
            },
            "dynamics checkpoint",
        )
        if type(value["schema"]) is not str or value["schema"] != SCHEMA:
            raise ValueError("unsupported dynamics checkpoint")
        model = cls(value["seed"], value["mode"])
        training = exact_fields(
            value["training"],
            {"fingerprint", "episodes", "structures", "epochs"},
            "training metadata",
        )
        fingerprint = bounded_text(
            training["fingerprint"], "training fingerprint", cap=64
        )
        if fingerprint and (
            len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
        ):
            raise ValueError("invalid training fingerprint")
        for key, cap in (
            ("episodes", MAX_EPISODES),
            ("structures", MAX_STRUCTURES),
            ("epochs", 16),
        ):
            _integer(training[key], 0, cap, key)
        templates, codes = value["templates"], value["codes"]
        if (
            type(templates) is not list
            or type(codes) is not list
            or len(templates) != len(codes)
            or len(templates) > 8
        ):
            raise ValueError("invalid template/code counts")
        checked_codes = []
        occupied: set[int] = set()
        for template, indices in zip(templates, codes, strict=True):
            _validate_template(template)
            bits = _indices(indices, 32, 4)
            if len(bits) != 4 or occupied.intersection(bits):
                raise ValueError("effect codes must be disjoint four-bit SDRs")
            occupied.update(bits)
            code = np.zeros(32, dtype=np.bool_)
            code[bits] = True
            checked_codes.append(code)
        if len({_json(t) for t in templates}) != len(templates):
            raise ValueError("duplicate effect templates")
        if bool(templates) != bool(training["episodes"]) or bool(templates) != bool(
            fingerprint
        ):
            raise ValueError("inconsistent trained checkpoint metadata")
        if not templates:
            if any(
                value[key] is not None for key in ("transform", "output", "experience")
            ) or any(training[k] for k in ("epochs", "structures")):
                raise ValueError("invalid empty dynamics checkpoint")
            return model
        if not training["epochs"] or not training["structures"]:
            raise ValueError("trained checkpoint requires nonzero training metadata")
        if training["structures"] > training["episodes"]:
            raise ValueError("more structural examples than independent episodes")
        transform = LearnedSDRTransform(_config(model.seed, "transform"))
        transform.memory = _memory_load(
            value["transform"], _config(model.seed, "transform")
        )
        model._output = _reader_load(
            value["output"], _config(model.seed, "output"), output=True
        )
        model._experience = _reader_load(
            value["experience"], _config(model.seed, "experience"), output=False
        )
        expected = (
            0
            if model.mode == "untrained"
            else training["structures"] * training["epochs"]
        )
        expected_experience = (
            0
            if model.mode == "untrained"
            else training["structures"] * min(4, training["epochs"])
        )
        if (
            transform.memory.step != expected
            or model._experience.memory.step != expected_experience
            or model._output.memory.step != 4 * len(templates)
        ):
            raise ValueError("numeric checkpoint steps do not match training metadata")
        if {p.portrait_id for p in model._output.portraits} != {
            str(i) for i in range(len(templates))
        }:
            raise ValueError("missing/unknown effect portraits")
        for i, code in enumerate(checked_codes):
            if not model._output.observe_portrait(str(i), code):
                raise ValueError("effect code has no matching saved portrait")
        if len(_json(value)) > MAX_BYTES:
            raise ValueError("dynamics checkpoint byte capacity exceeded")
        model._templates, model._codes, model._transform = (
            tuple(dict(t) for t in templates),
            tuple(checked_codes),
            transform,
        )
        model._training = dict(training)
        return model


def _validate_template(value: Any) -> None:
    if type(value) is not dict:
        raise ValueError("invalid effect template")
    if value == {"op": "noop"}:
        return
    exact_fields(
        value,
        {"op", "subject_role", "relation", "value_role", "spatial_source"},
        "effect template",
    )
    if (
        any(type(item) is not str for item in value.values())
        or value["op"] not in {"set", "exclude"}
        or value["subject_role"] != "object"
        or value["relation"] not in {"location", "holder"}
        or value["value_role"] not in ROLES
        or value["spatial_source"] not in {"event", "in"}
    ):
        raise ValueError("invalid effect template values")


def _indices(value: Any, width: int, cap: int) -> list[int]:
    if (
        type(value) is not list
        or len(value) > cap
        or any(type(i) is not int or not 0 <= i < width for i in value)
    ):
        raise ValueError("invalid checkpoint indices")
    if value != sorted(set(value)):
        raise ValueError("indices must be unique and sorted")
    return value


def _memory_dump(memory: CombinatorialMemory) -> dict[str, Any]:
    return {
        "config": memory.config.to_dict(),
        "receptors": memory.receptors.tolist(),
        "output_map": memory.output_map.tolist(),
        "step": memory.step,
        "clusters": [
            {
                "point": point,
                "bits": cluster.bits.tolist(),
                "bit_hits": cluster.bit_hits.tolist(),
                "created_at": cluster.created_at,
                "last_seen": cluster.last_seen,
                "status": int(cluster.status),
                "partial_hits": cluster.partial_hits,
                "exact_hits": cluster.exact_hits,
                "partial_errors": cluster.partial_errors,
                "complete_errors": cluster.complete_errors,
            }
            for point, cluster in memory.iter_clusters()
        ],
    }


def _memory_load(value: Any, config: ModelConfig) -> CombinatorialMemory:
    exact_fields(
        value,
        {"config", "receptors", "output_map", "step", "clusters"},
        "factor memory",
    )
    if (
        type(value["config"]) is not dict
        or ModelConfig.from_dict(value["config"]) != config
    ):
        raise ValueError("checkpoint memory configuration mismatch")
    receptors, outputs, clusters = (
        value["receptors"],
        value["output_map"],
        value["clusters"],
    )
    if type(receptors) is not list or len(receptors) != config.point_count:
        raise ValueError("invalid receptor count")
    for row in receptors:
        if (
            len(_indices(row, config.input_bits, config.receptive_bits))
            != config.receptive_bits
        ):
            raise ValueError("invalid receptive field width")
    if type(outputs) is not list or len(outputs) != config.point_count:
        raise ValueError("invalid output-map shape")
    for output in outputs:
        _integer(output, 0, config.output_bits - 1, "output index")
    _integer(value["step"], 0, MAX_EPISODES * 16, "memory step")
    if (
        type(clusters) is not list
        or len(clusters) > config.point_count * config.max_clusters_per_point
    ):
        raise ValueError("invalid cluster count")
    memory = CombinatorialMemory(
        config,
        receptors=np.asarray(receptors, dtype=np.int32),
        output_map=np.asarray(outputs, dtype=np.int32),
    )
    memory.step = value["step"]
    fields = {
        "point",
        "bits",
        "bit_hits",
        "created_at",
        "last_seen",
        "status",
        "partial_hits",
        "exact_hits",
        "partial_errors",
        "complete_errors",
    }
    for raw in clusters:
        exact_fields(raw, fields, "cluster")
        point = _integer(raw["point"], 0, config.point_count - 1, "cluster point")
        bits = _indices(raw["bits"], config.input_bits, config.receptive_bits)
        if type(raw["bit_hits"]) is not list or len(raw["bit_hits"]) != len(bits):
            raise ValueError("invalid cluster bit-hit shape")
        for hit in raw["bit_hits"]:
            _integer(hit, 0, MAX_EPISODES * 16, "bit-hit count")
        for key in fields - {"point", "bits", "bit_hits"}:
            _integer(raw[key], 0, 2 if key == "status" else MAX_EPISODES * 16, key)
        memory.add_loaded_cluster(
            point,
            Cluster(
                bits=np.asarray(bits, dtype=np.int32),
                bit_hits=np.asarray(raw["bit_hits"], dtype=np.int64),
                created_at=raw["created_at"],
                last_seen=raw["last_seen"],
                status=ClusterStatus(raw["status"]),
                partial_hits=raw["partial_hits"],
                exact_hits=raw["exact_hits"],
                partial_errors=raw["partial_errors"],
                complete_errors=raw["complete_errors"],
            ),
        )
    return memory


def _reader_dump(reader: FactorSceneReader) -> dict[str, Any]:
    return json.loads(
        _json(
            {
                "memory": _memory_dump(reader.memory),
                "portraits": [
                    {
                        "id": p.portrait_id,
                        "step": p.memory_step,
                        "evidence": [e.to_dict() for e in p.evidence],
                    }
                    for p in reader.portraits
                ],
            }
        )
    )


def _reader_load(value: Any, config: ModelConfig, *, output: bool) -> FactorSceneReader:
    exact_fields(value, {"memory", "portraits"}, "factor reader")
    memory = _memory_load(value["memory"], config)
    reader = _reader(memory, output=output)
    portraits = value["portraits"]
    if type(portraits) is not list or len(portraits) > reader.config.max_portraits:
        raise ValueError("invalid portrait count")
    learned = {
        (point, cluster.signature): cluster for point, cluster in memory.iter_clusters()
    }
    total = 0
    for raw in portraits:
        exact_fields(raw, {"id", "step", "evidence"}, "factor portrait")
        name = bounded_text(raw["id"], "portrait ID", empty=False)
        if name in reader._portraits:
            raise ValueError("duplicate portrait ID")
        step = _integer(raw["step"], 0, memory.step, "portrait step")
        if type(raw["evidence"]) is not list:
            raise ValueError("invalid portrait evidence")
        total += len(raw["evidence"])
        if total > reader.config.max_portrait_evidence:
            raise ValueError("portrait evidence capacity exceeded")
        evidence = []
        seen = set()
        for item in raw["evidence"]:
            exact_fields(
                item,
                {
                    "point_index",
                    "signature",
                    "matched_bits",
                    "observations",
                    "output_bit",
                },
                "cluster evidence",
            )
            point = _integer(
                item["point_index"], 0, config.point_count - 1, "evidence point"
            )
            signature = tuple(
                _indices(item["signature"], config.input_bits, config.receptive_bits)
            )
            matched = tuple(
                _indices(item["matched_bits"], config.input_bits, config.receptive_bits)
            )
            key = point, signature
            cluster = learned.get(key)
            if (
                key in seen
                or cluster is None
                or cluster.status != ClusterStatus.STABLE
                or not set(matched) <= set(signature)
                or len(matched) < config.activation_threshold
            ):
                raise ValueError("portrait does not match stable cluster evidence")
            seen.add(key)
            observations = _integer(
                item["observations"], 1, cluster.partial_hits, "observations"
            )
            output_bit = _integer(
                item["output_bit"], 0, config.output_bits - 1, "evidence output"
            )
            if output_bit != int(memory.output_map[point]):
                raise ValueError("portrait output coordinate mismatch")
            evidence.append(
                ClusterEvidence(point, signature, matched, observations, output_bit)
            )
        atoms = tuple(
            sorted({(e.point_index, bit) for e in evidence for bit in e.matched_bits})
        )
        if (
            len(atoms) < reader.config.min_atoms
            or len({p for p, _ in atoms}) < reader.config.min_points
        ):
            raise ValueError("insufficient saved portrait evidence")
        reader._portraits[name] = FactorPortrait(name, tuple(evidence), atoms, step)
    reader._stored_evidence = total
    return reader


def default_dynamics(seed: int = 42, *, seconds: float = 20.0) -> LearnedDynamics:
    from .transition_data import training_episodes

    model = LearnedDynamics(seed)
    model.fit(training_episodes(seed), seconds=seconds)
    return model
