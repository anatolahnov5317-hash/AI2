"""Small disclosed development checks for v6 block 1; no held-out claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_factors.learning.candidate_selection import select  # noqa: E402
from text_factors.learning.hypotheses import (  # noqa: E402
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from text_factors.learning.language_data import (  # noqa: E402
    ENTITY_BY_NAME,
    development_examples,
)
from text_factors.learning.model import ModelBundle  # noqa: E402
from text_factors.learning.persistence import read_artifact  # noqa: E402
from text_factors.learning.schema import (  # noqa: E402
    DialogueContext,
    Event,
    Interpretation,
    Meaning,
)
from text_factors.learning.session import LearnedSession  # noqa: E402


def run(destination: Path):
    path = ROOT / "docs/results/v05_model_42.json"
    bundle = ModelBundle.from_dict(read_artifact(path, kind="model"))
    fingerprint = bundle.fingerprint
    code_hash = hashlib.sha256()
    for source in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py")]):
        code_hash.update(
            str(source.relative_to(ROOT)).encode() + b"\0" + source.read_bytes()
        )
    report = {
        "kind": "public_development_not_independent_evaluation",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_fingerprint": fingerprint,
        "source_sha256": code_hash.hexdigest(),
        "seed": 42,
        "cases": [],
        "metrics": {},
    }
    rows = report["cases"]

    def save():
        groups = {row["group"] for row in rows}
        report["metrics"] = {
            group: {
                "passed": sum(row["passed"] for row in rows if row["group"] == group),
                "total": sum(row["group"] == group for row in rows),
            }
            for group in sorted(groups)
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(destination)

    for example in development_examples():
        started = perf_counter()
        initial = bundle.understanding.interpret(example.text, example.context)
        batch = bundle.understanding.propose(
            example.text, example.context, initial=initial
        )
        result = select(
            batch,
            initial,
            bundle.dynamics,
            [],
            model_fingerprint=fingerprint,
            seconds=1,
        )
        covered = any(
            h.complete and h.meaning == example.meaning for h in batch.candidates
        )
        rows.append(
            {
                "group": "known_development_structures",
                "text": example.text,
                "passed": covered and result.meaning == example.meaning,
                "legacy_correct": initial.meaning == example.meaning,
                "candidate_covered": covered,
                "selected_correct": result.meaning == example.meaning,
                "seconds": perf_counter() - started,
                "trace": result.diagnostics["hypotheses"],
            }
        )
        save()
    ctx = DialogueContext(
        entities=tuple(
            ENTITY_BY_NAME[n].entity for n in ["книга", "петя", "миша", "стол"]
        ),
        focus=("книга", "петя", "миша", "стол"),
    )
    for text, actor in [
        ("Он передал книгу Пете", "миша"),
        ("Он передал книгу Мише", "петя"),
    ]:
        initial = bundle.understanding.interpret(text, ctx)
        batch = bundle.understanding.propose(text, ctx, initial=initial)
        result = select(
            batch,
            initial,
            bundle.dynamics,
            [],
            model_fingerprint=fingerprint,
            seconds=1,
        )
        rows.append(
            {
                "group": "reference_examples_with_experience",
                "text": text,
                "passed": bool(
                    result.meaning
                    and result.meaning.event
                    and result.meaning.event.actor == actor
                ),
                "legacy_abstained": initial.meaning is None,
                "trace": result.diagnostics["hypotheses"],
            }
        )
        save()
    for text in ["Маша передала", "Он положил книгу на стол"]:
        session = LearnedSession(bundle)
        response = session.respond(text)
        rows.append(
            {
                "group": "incomplete_input_no_facts",
                "text": text,
                "passed": response["meaning"] is None
                and not session.world.events
                and response["action"] in {"unknown", "clarify"},
                "response": response,
            }
        )
        save()
    meanings = [
        Meaning("inform", Event("give", actor=a, object="книга", recipient=b))
        for a, b in [("петя", "миша"), ("миша", "петя")]
    ]
    batch = CandidateSet(
        Observation("structured", "неоднозначное сообщение"),
        digest(DialogueContext().to_dict()),
        tuple(Hypothesis.create("structured", m, 0.0) for m in meanings),
    )
    for owner, subject, expected in [
        ("петя", "книга", meanings[0]),
        ("миша", "книга", meanings[1]),
        ("петя", "ключ", None),
    ]:
        facts = [
            {
                "subject": subject,
                "relation": "holder",
                "value": owner,
                "negated": False,
                "spatial": "in",
            }
        ]
        result = select(
            batch,
            Interpretation(None),
            bundle.dynamics,
            facts,
            model_fingerprint=fingerprint,
            seconds=1,
        )
        rows.append(
            {
                "group": "annotated_candidate_memory_controls",
                "before": facts,
                "passed": result.meaning == expected,
                "trace": result.diagnostics["hypotheses"],
            }
        )
        save()
    if bundle.fingerprint != fingerprint:
        raise RuntimeError("inference changed trained model")
    save()
    print(json.dumps(report["metrics"], ensure_ascii=False))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
