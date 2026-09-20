"""Offline supervision for bounded structured retrieval; no language training."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from text_factors.learning.attention_ranker import FEATURES, fit
from text_factors.learning.hypotheses import digest


def training_rows():
    rows, labels = [], []
    for target, obj, predicate, people, scope, time in itertools.product(
        (0, 1), (0, 1), (0, 1), (0.0, 1 / 3, 1.0), (0, 1), (0, 1)
    ):
        rows.append([1.0, target, obj, predicate, people, scope, time])
        labels.append(int(bool(target or (obj and predicate and scope and time))))
    return rows, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, labels = training_rows()
    value = {
        "schema": "ai2-attention-ranker-v1",
        "features": list(FEATURES),
        "weights": fit(rows, labels),
        "training": {
            "examples": len(rows),
            "steps": 400,
            "learning_rate": 0.25,
            "l2": 0.002,
            "centering": "Non-bias features; exported weights use original coordinates",
            "seed": None,
            "data_sha256": digest({"rows": rows, "labels": labels}),
            "task": "Explicit target OR matching object/predicate/scope/time",
            "disclosure": (
                "Synthetic engineering labels over structured features; "
                "no independent language evaluation."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "examples": len(rows),
                "features": len(FEATURES),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
