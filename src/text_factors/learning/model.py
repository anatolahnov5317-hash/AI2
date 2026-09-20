"""An explicit, versioned checkpoint of all four locally learned components."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..conversation.persistence import decode_json, encode_json
from ..conversation.schema import Budget, BudgetExceeded
from .schema import exact_fields

SCHEMA = "ai2-learned-dialogue-model-v1"
MAX_MODEL_BYTES = 12_000_000


@dataclass(frozen=True)
class ModelBundle:
    """Inference does not call fit. Loading restores parameters, not examples."""

    understanding: Any
    dynamics: Any
    policy: Any
    generator: Any
    metadata: dict[str, Any]

    @classmethod
    def fit(
        cls,
        *,
        seed: int = 42,
        seconds: float = 120.0,
        dataset: dict[str, Any] | None = None,
    ) -> ModelBundle:
        from .dialogue_data import training_dialogues
        from .dialogue_learning import LearnedDialoguePolicy, LearnedTokenGenerator
        from .dynamics import LearnedDynamics
        from .language_data import training_examples
        from .transition_data import training_episodes
        from .understanding import LearnedUnderstanding

        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(seconds) not in (float, int) or not 0 < seconds <= 300:
            raise ValueError("training seconds must be in (0, 300]")
        budget = Budget(seconds)

        def remaining(cap: float = 60.0) -> float:
            budget.check()
            return min(cap, budget.deadline - perf_counter())

        if dataset is None:
            utterances = training_examples(seed=seed)
            episodes = training_episodes(seed=seed)
            dialogues = training_dialogues(seed=seed)
            dataset_kind = "bundled_synthetic_training_only"
        else:
            utterances, episodes, dialogues = _custom_dataset(dataset)
            dataset_kind = "explicit_annotated_dataset"
        budget.check()
        understanding = LearnedUnderstanding.fit(
            utterances, seed=seed, seconds=remaining()
        )
        dynamics = LearnedDynamics(seed=seed)
        dynamics.fit(episodes, seconds=remaining())
        policy = LearnedDialoguePolicy(seed=seed)
        policy.fit(dialogues, seconds=remaining())
        generator = LearnedTokenGenerator(seed=seed)
        generator.fit(dialogues, seconds=remaining())
        budget.check()
        metadata = {
            "version": "0.5.0a1",
            "seed": seed,
            "dataset_kind": dataset_kind,
            "training_counts": {
                "utterances": len(utterances),
                "transitions": len(episodes),
                "dialogues": len(dialogues),
            },
            "pretrained_model": False,
            "implicit_online_learning": False,
            "scope": "bounded Russian situation dialogue; not unrestricted chat",
        }
        result = cls(understanding, dynamics, policy, generator, metadata)
        result.to_dict()  # Validate size/finite JSON before the caller can save.
        budget.check()
        return result

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "understanding": self.understanding.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "policy": self.policy.to_dict(),
            "generator": self.generator.to_dict(),
            "metadata": deepcopy(self.metadata),
        }
        return decode_json(
            encode_json(value, max_bytes=MAX_MODEL_BYTES), max_bytes=MAX_MODEL_BYTES
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            encode_json(self.to_dict(), max_bytes=MAX_MODEL_BYTES)
        ).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> ModelBundle:
        from .dialogue_learning import LearnedDialoguePolicy, LearnedTokenGenerator
        from .dynamics import LearnedDynamics
        from .understanding import LearnedUnderstanding

        value = exact_fields(
            value,
            {"schema", "understanding", "dynamics", "policy", "generator", "metadata"},
            "model checkpoint",
        )
        if value["schema"] != SCHEMA:
            raise ValueError("unsupported model checkpoint schema")
        encode_json(value, max_bytes=MAX_MODEL_BYTES)
        metadata = exact_fields(
            value["metadata"],
            {
                "version",
                "seed",
                "dataset_kind",
                "training_counts",
                "pretrained_model",
                "implicit_online_learning",
                "scope",
            },
            "model metadata",
        )
        if (
            metadata["version"] != "0.5.0a1"
            or type(metadata["seed"]) is not int
            or not 0 <= metadata["seed"] < 2**32
        ):
            raise ValueError("invalid model version or seed")
        if (
            metadata["pretrained_model"] is not False
            or metadata["implicit_online_learning"] is not False
        ):
            raise ValueError("unsupported model provenance")
        if type(metadata["dataset_kind"]) is not str or metadata[
            "dataset_kind"
        ] not in {"bundled_synthetic_training_only", "explicit_annotated_dataset"}:
            raise ValueError("invalid dataset kind")
        counts = exact_fields(
            metadata["training_counts"],
            {"utterances", "transitions", "dialogues"},
            "training counts",
        )
        if any(type(v) is not int or not 1 <= v <= 20000 for v in counts.values()):
            raise ValueError("invalid training counts")
        if type(metadata["scope"]) is not str or len(metadata["scope"]) > 256:
            raise ValueError("invalid scope metadata")
        return cls(
            LearnedUnderstanding.from_dict(value["understanding"]),
            LearnedDynamics.from_dict(value["dynamics"]),
            LearnedDialoguePolicy.from_dict(value["policy"]),
            LearnedTokenGenerator.from_dict(value["generator"]),
            deepcopy(metadata),
        )


def _custom_dataset(value: Any) -> tuple[Any, Any, Any]:
    from .dialogue_learning import dialogues_from_data
    from .dynamics import TransitionEpisode
    from .language_data import TrainingUtterance

    value = exact_fields(
        value,
        {"schema", "understanding", "transitions", "dialogues"},
        "annotated dataset",
    )
    if value["schema"] != "ai2-learning-data-v1":
        raise ValueError("unsupported annotated dataset schema")
    encode_json(value, max_bytes=12_000_000)
    capacities = {"understanding": 2048, "transitions": 512, "dialogues": 128}
    for name, capacity in capacities.items():
        if type(value[name]) is not list or not 1 <= len(value[name]) <= capacity:
            raise ValueError(f"invalid {name} dataset size")
    utterances = []
    for record in value["understanding"]:
        utterances.append(TrainingUtterance.from_dict(record))
    episodes = []
    for record in value["transitions"]:
        episodes.append(TransitionEpisode.from_dict(record))
    dialogues = dialogues_from_data(value["dialogues"])
    if not utterances or not episodes or not dialogues:
        raise BudgetExceeded("empty_training_component")
    return utterances, episodes, dialogues
