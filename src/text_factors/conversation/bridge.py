"""Bounded, taught lexical predicates carried by AI2's actual factor memories.

The adapter supplies a *raw lexical cue*, not its semantic label. Cue coding,
the four-label ontology, grammar/roles and the novelty gate are explicit
scaffolding. Supervised local clusters learn the cue-to-semantic SDR mapping;
an independently trained common memory recognizes its predicted semantic code.
No training-label lookup supplies a factor-mode inference result.

This first bridge deliberately requires a previously taught normalized cue.
It therefore tests learned lexical associations and reuse across new entity
combinations, not unsupervised grammar or unseen-word generalization. State is
persisted as a versioned, bounded teaching recipe and deterministically replayed,
not presented as a binary trained-memory checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from ..config import ModelConfig
from ..memory import CombinatorialMemory
from ..recognition import ContextView, RecognitionLimits
from ..scene_recognition import FactorSceneReader, SceneRecognitionConfig
from ..transforms import LearnedSDRTransform

LABELS = ("locate", "move", "give", "have")
MODES = ("factor", "untrained", "shuffled", "nearest")
MAX_CUE_CHARS = 64
MAX_PAIRS = 64
MAX_EPOCHS = 16
MAX_STATE_BYTES = 32_768
SCHEMA = "ai2-factor-semantic-bridge-v1"
INPUT_BITS = 256
OUTPUT_BITS = 32
POINTS = 128


def _cue(value: Any) -> str:
    if type(value) is not str or not 0 < len(value) <= MAX_CUE_CHARS:
        raise ValueError(f"cue must contain 1 to {MAX_CUE_CHARS} characters")
    normalized = " ".join(unicodedata.normalize("NFC", value).casefold().split())
    if not normalized or len(normalized) > MAX_CUE_CHARS:
        raise ValueError("cue must contain a bounded nonempty lexical form")
    if not all(char.isalpha() or char in " -'" for char in normalized):
        raise ValueError("cue must contain letters, spaces, hyphens or apostrophes")
    return normalized


def _seconds(value: Any) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 60:
        raise ValueError("seconds must be finite and in (0, 60]")
    return float(value)


@dataclass(frozen=True, slots=True)
class PredicateResolution:
    label: str | None
    score: float
    candidates: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-safe data; score is not a probability."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))


class FactorSemanticBridge:
    """Small single-threaded supervised bridge with atomic bounded retraining.

    ``fit`` merges unique teaching pairs with existing teaching, then rebuilds
    off to the side and commits only after the complete bounded run. Repeated
    presentations count as optimization epochs, not independent experiences.
    A conflicting cue is excluded from learning and always asks for clarification;
    replacing conflicting teaching requires an explicit fresh bridge/recipe.

    Modes ``untrained`` and ``shuffled`` are experimental controls. ``nearest``
    is a disclosed exact-normalized-string memorization baseline, not factor AI2.
    Inference never updates teaching or either factor memory. Concurrent calls
    that mutate a bridge are unsupported.
    """

    def __init__(self, seed: int = 42, mode: str = "factor") -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(mode) is not str or mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.seed = seed
        self.mode = mode
        self._pairs: tuple[tuple[str, str], ...] = ()
        self._epochs = 0
        self._known: frozenset[str] = frozenset()
        self._conflicts: dict[str, tuple[str, ...]] = {}
        self._transform: LearnedSDRTransform | None = None
        self._reader: FactorSceneReader | None = None
        self._codes = self._semantic_codes()

    def _semantic_codes(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 9201]))
        order = rng.permutation(OUTPUT_BITS)
        result = {}
        for index, label in enumerate(LABELS):
            bits = np.zeros(OUTPUT_BITS, dtype=np.bool_)
            bits[order[index * 4 : (index + 1) * 4]] = True
            bits.flags.writeable = False
            result[label] = bits
        return result

    def _encode(self, cue: str) -> np.ndarray:
        digest = hashlib.sha256(f"{SCHEMA}:source:{self.seed}:{cue}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
        bits = np.zeros(INPUT_BITS, dtype=np.bool_)
        bits[rng.choice(INPUT_BITS, 32, replace=False)] = True
        bits.flags.writeable = False
        return bits

    def _new_transform(self) -> LearnedSDRTransform:
        config = ModelConfig(
            input_bits=INPUT_BITS,
            output_bits=OUTPUT_BITS,
            point_count=POINTS,
            receptive_bits=160,
            create_threshold=12,
            activation_threshold=12,
            min_active_points=1,
            probation_after=2,
            stable_after=3,
            prune_keep_ratio=1.0,
            max_clusters_per_point=MAX_PAIRS,
            prediction_vote_threshold=2,
            seed=self.seed,
        )
        transform = LearnedSDRTransform(config)
        # Fixed balanced output coverage is an encoder scaffold, not learned
        # receptor selection or a data/label-conditioned memory bypass.
        output_map = np.tile(
            np.arange(OUTPUT_BITS, dtype=np.int32), POINTS // OUTPUT_BITS
        )
        transform.memory = CombinatorialMemory(
            config, receptors=transform.memory.receptors, output_map=output_map
        )
        return transform

    def _new_reader(self, check: Callable[[], None]) -> FactorSceneReader:
        config = ModelConfig(
            input_bits=OUTPUT_BITS,
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
            seed=self.seed,
        )
        common = CombinatorialMemory(config)
        for _ in range(4):
            for label in LABELS:
                check()
                common.observe(self._codes[label])
                check()
        reader = FactorSceneReader(
            common,
            config=SceneRecognitionConfig(
                coverage=0.95,
                max_portraits=len(LABELS),
                max_portrait_evidence=2048,
                max_atom_visits=32_768,
                seconds=1.0,
            ),
        )
        for label in LABELS:
            check()
            if not reader.observe_portrait(label, self._codes[label]):
                raise RuntimeError("common memory could not register a semantic anchor")
            check()
        return reader

    def fit(
        self,
        examples: list[tuple[str, str]],
        *,
        epochs: int = 8,
        seconds: float = 15.0,
    ) -> dict[str, Any]:
        """Commit a complete training run or raise without changing this bridge."""
        started = perf_counter()
        budget = _seconds(seconds)
        if type(epochs) is not int or not 1 <= epochs <= MAX_EPOCHS:
            raise ValueError(f"epochs must be an integer in [1, {MAX_EPOCHS}]")
        if type(examples) is not list or not 1 <= len(examples) <= MAX_PAIRS:
            raise ValueError(f"examples must be a list of 1 to {MAX_PAIRS} pairs")
        previous_pair_count = len(self._pairs)
        checked = set(self._pairs)
        for pair in examples:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("each example must be a (cue, label) tuple")
            cue, label = _cue(pair[0]), pair[1]
            if type(label) is not str or label not in LABELS:
                raise ValueError(f"label must be one of {LABELS}")
            checked.add((cue, label))
        if len(checked) > MAX_PAIRS:
            raise ValueError("unique teaching-pair capacity reached")
        pairs = tuple(sorted(checked))
        teaching: dict[str, set[str]] = {}
        for cue, label in pairs:
            teaching.setdefault(cue, set()).add(label)
        conflicts = {
            cue: tuple(sorted(labels))
            for cue, labels in teaching.items()
            if len(labels) > 1
        }
        usable = [(cue, label) for cue, label in pairs if cue not in conflicts]

        def check() -> None:
            if perf_counter() - started >= budget:
                raise TimeoutError("semantic bridge training time budget exceeded")

        check()
        transform = self._new_transform()
        check()
        reader = self._new_reader(check)
        targets = [label for _, label in usable]
        if self.mode == "shuffled" and len(set(targets)) > 1:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 9203]))
            shuffled = [targets[int(i)] for i in rng.permutation(len(targets))]
            if shuffled == targets:
                # Finite deterministic fallback, keeping label frequencies.
                offset = next(
                    i for i, label in enumerate(targets) if label != targets[0]
                )
                shuffled = targets[offset:] + targets[:offset]
            targets = shuffled
        encoded = [self._encode(cue) for cue, _ in usable]
        if self.mode not in {"untrained", "nearest"}:
            for _ in range(epochs):
                for source, label in zip(encoded, targets, strict=True):
                    check()
                    transform.observe(source, self._codes[label])
                    check()
        check()
        self._pairs, self._epochs = pairs, epochs
        self._known = frozenset(teaching)
        self._conflicts = conflicts
        self._transform, self._reader = transform, reader
        return {
            "complete": True,
            "mode": self.mode,
            "unique_teaching_pairs": len(pairs),
            "unique_cues": len(teaching),
            "conflicted_cues": sorted(conflicts),
            "new_unique_pairs": len(pairs) - previous_pair_count,
            "epochs": epochs,
            "training_presentations": transform.memory.step,
            "presentations_are_independent_evidence": False,
            "common_memory_presentations": reader.memory.step,
            "elapsed_seconds": perf_counter() - started,
            "unknown_cue_policy": "exact normalized taught-cue coverage required",
            "shuffled_targets_changed": targets != [label for _, label in usable],
            "persistence": "versioned deterministic replay of unique teaching pairs",
        }

    def _abstain(
        self, reason: str, *, candidates: tuple[str, ...] = ()
    ) -> PredicateResolution:
        return PredicateResolution(
            None,
            0.0,
            candidates,
            {
                "mode": self.mode,
                "reason": reason,
                "score_is_probability": False,
                "source_encoding": "exact normalized lexical cue SDR",
            },
        )

    def classify(self, cue: str) -> PredicateResolution:
        """Return only factor-supported semantics; no inference-time learning."""
        normalized = _cue(cue)
        if normalized in self._conflicts:
            return self._abstain(
                "conflicting_teaching", candidates=self._conflicts[normalized]
            )
        if normalized not in self._known:
            return self._abstain("untaught_cue")
        if self.mode == "nearest":
            # Deliberately isolated lookup baseline. This path is never used as
            # a fallback by the factor or control modes.
            label = next(label for word, label in self._pairs if word == normalized)
            return PredicateResolution(
                label,
                1.0,
                (label,),
                {
                    "mode": "nearest",
                    "reason": "exact_normalized_string_lookup_baseline",
                    "score_is_probability": False,
                    "factor_evidence_used": False,
                },
            )
        if self._transform is None or self._reader is None:
            return self._abstain("not_fitted")
        started = perf_counter()
        source = self._encode(normalized)
        prediction = self._transform.predict(source)
        if perf_counter() - started >= 1.0:
            return self._abstain("time_budget")
        result = self._reader.recognize_views(
            [ContextView("predicate", "learned-lexical", prediction.output, ())],
            total_views=1,
            limits=RecognitionLimits(
                max_views=1,
                max_candidates=len(LABELS),
                max_evidence=4096,
                seconds=max(0.001, 1.0 - (perf_counter() - started)),
            ),
        )
        proposals = sorted(
            result.proposals, key=lambda item: (-item.coverage, item.portrait_id)
        )
        candidates = tuple(item.portrait_id for item in proposals)
        complete = result.complete and perf_counter() - started < 1.0
        label = candidates[0] if complete and len(candidates) == 1 else None
        score = proposals[0].coverage if proposals else 0.0
        evidence = {
            "mode": self.mode,
            "reason": (
                "factor_semantic_portrait"
                if label is not None
                else "incomplete_recognition"
                if not complete
                else "ambiguous_semantic_portraits"
                if candidates
                else "insufficient_factor_evidence"
            ),
            "score_is_probability": False,
            "factor_evidence_used": True,
            "novelty_gate": "previously taught normalized cue; not learned grammar",
            "source_sdr_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "predicted_bits": prediction.active_output_bits,
            "transform_active_points": prediction.active_points,
            "transform_training_presentations": self._transform.memory.step,
            "common_memory_step": self._reader.memory.step,
            "common_memory_namespace": self._reader.encoding_id,
            "recognition_complete": complete,
            "portraits": [
                {
                    "label": item.portrait_id,
                    "coverage": item.coverage,
                    "support_atoms": item.support,
                    "portrait_atoms": item.portrait_atoms,
                }
                for item in proposals
            ],
        }
        return PredicateResolution(label, float(score), candidates, evidence)

    def to_dict(self) -> dict[str, Any]:
        """Bounded recipe, not stored numeric memories or an LLM checkpoint."""
        return {
            "schema": SCHEMA,
            "seed": self.seed,
            "mode": self.mode,
            "epochs": self._epochs,
            "examples": [list(pair) for pair in self._pairs],
        }

    @classmethod
    def from_dict(cls, value: Any, *, seconds: float = 15.0) -> FactorSemanticBridge:
        """Validate the exact finite recipe, replay it and return only on success."""
        _seconds(seconds)
        if type(value) is not dict or set(value) != {
            "schema",
            "seed",
            "mode",
            "epochs",
            "examples",
        }:
            raise ValueError("invalid semantic bridge recipe fields")
        if type(value["schema"]) is not str or value["schema"] != SCHEMA:
            raise ValueError("unsupported semantic bridge recipe schema")
        restored = cls(value["seed"], value["mode"])
        epochs, raw = value["epochs"], value["examples"]
        if type(epochs) is not int or not 0 <= epochs <= MAX_EPOCHS:
            raise ValueError("invalid saved training epoch count")
        if type(raw) is not list or len(raw) > MAX_PAIRS:
            raise ValueError("invalid saved teaching-pair count")
        pairs = []
        for pair in raw:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("invalid saved teaching pair")
            cue, label = _cue(pair[0]), pair[1]
            if cue != pair[0] or type(label) is not str or label not in LABELS:
                raise ValueError("saved teaching pairs must be normalized and valid")
            pairs.append((cue, label))
        if pairs != sorted(set(pairs)) or bool(pairs) != bool(epochs):
            raise ValueError("saved teaching must be unique, sorted and match epochs")
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > MAX_STATE_BYTES:
            raise ValueError("semantic bridge recipe size limit exceeded")
        if pairs:
            restored.fit(pairs, epochs=epochs, seconds=seconds)
        return restored
