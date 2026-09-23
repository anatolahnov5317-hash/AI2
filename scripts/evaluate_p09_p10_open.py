"""Open synthetic pre-update check for repeated histories and partial contexts.

This fixture demonstrates an invariant; it is not a measured Russian pilot or
an independent held-out claim. Train and development group IDs are disjoint.
"""

from __future__ import annotations

import json

from text_factors.real_data import ContextRegistry, LearningEpisode, SparseTransform


def evaluate() -> dict[str, object]:
    # Three independent train histories. Revisions of A occur 50 times;
    # B and C each occur once. No dev group is fitted at any stage.
    train = (
        *(("train-A", (10,)) for _ in range(50)),
        ("train-B", (11,)),
        ("train-C", (11,)),
    )
    dev = (("dev-D", (11,)), ("dev-E", (11,)))
    frequency_control = SparseTransform(32)
    group_memory = SparseTransform(32)
    for group, target in train:
        frequency_control.fit((1,), target)
        group_memory.fit((1,), target, group_id=group)

    # Freeze both models. Each answer is recorded before any dev update.
    before = group_memory.to_dict()
    frequency_correct = group_correct = 0
    for _, expected in dev:
        frequency_correct += frequency_control.predict((1,), limit=1) == expected
        group_correct += group_memory.predict((1,), limit=1) == expected
    assert group_memory.to_dict() == before, "development prediction mutated memory"

    registry = ContextRegistry(width=32, assignment_threshold=0.6)
    first, _, _ = registry.learn(LearningEpisode("e1", "train-X", (2,), (20,)))
    second, created, score = registry.learn(
        LearningEpisode("e2", "train-Y", (2,), (20, 21), (20,))
    )
    other, other_created, _ = registry.learn(
        LearningEpisode("e3", "train-Z", (2,), (30,))
    )
    return {
        "schema": "ai2-p09-p10-open-synthetic-v1",
        "train_episodes": len(train) + 3,
        "train_independent_groups": 6,
        "development_independent_groups": len(dev),
        "frequency_correct_before_update": frequency_correct,
        "group_capped_correct_before_update": group_correct,
        "development_total": len(dev),
        "memory_votes_10": group_memory.support_for((1,), 10),
        "memory_votes_11": group_memory.support_for((1,), 11),
        "masked_context_reused": first == second and not created and score == 1.0,
        "distinct_context_discovered": other != first and other_created,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, sort_keys=True))
