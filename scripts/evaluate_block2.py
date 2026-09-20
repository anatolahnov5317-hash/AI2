"""Disclosed block 2 development controls, with incrementally saved traces.

Run from the repository with PYTHONPATH=src, after installing dependencies.
Structured cases isolate archive/attention behavior from the language parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from text_factors.learning.attention import (
    AttentionLimits,
    AttentionState,
    ReviewCue,
    Work,
)
from text_factors.learning.attention_ranker import AttentionRanker
from text_factors.learning.candidate_selection import select
from text_factors.learning.hypotheses import (
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from text_factors.learning.language_data import ENTITY_BY_NAME
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact
from text_factors.learning.schema import DialogueContext, Event, Interpretation, Meaning
from text_factors.learning.session import LearnedSession

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = ROOT / "docs/results/v05_model_42.json"
    bundle = ModelBundle.from_dict(read_artifact(checkpoint, kind="model"))
    fingerprint = bundle.fingerprint
    ranker = AttentionRanker.load()
    source_hash = hashlib.sha256()
    for path in sorted(
        [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("src/**/*.json"),
            *ROOT.glob("tests/**/*.py"),
        ]
    ):
        source_hash.update(
            str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes()
        )
    data = {
        "schema": "ai2-v6-block2-development-v1",
        "disclosure": (
            "Open development controls, some used in debugging; "
            "not held-out language or end-to-end revision evaluation."
        ),
        "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_fingerprint": fingerprint,
        "ranker_fingerprint": ranker.fingerprint,
        "source_sha256": source_hash.hexdigest(),
        "python": sys.version,
        "numpy": version("numpy"),
        "cases": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)

    def record(group, name, passed, **details):
        data["cases"].append(
            {"group": group, "name": name, "passed": bool(passed), **details}
        )
        save()

    names = ("книга", "ключ", "петя", "миша", "маша", "стол")
    ctx = DialogueContext(
        entities=tuple(ENTITY_BY_NAME[n].entity for n in names), focus=names
    )

    def transfer(actor, obj="книга"):
        return Meaning(
            "inform",
            Event("give", actor=actor, object=obj, recipient="маша", time="past"),
        )

    def add(state, number, meanings, text="Он передал книгу Маше", *, real=False):
        observation = Observation(f"turn:{number}", text, number)
        if real:
            batch = bundle.understanding.propose(text, ctx, observation=observation)
        else:
            batch = CandidateSet(
                observation,
                digest(ctx.to_dict()),
                tuple(
                    Hypothesis.create(observation.observation_id, m, i * 0.08)
                    for i, m in enumerate(meanings)
                ),
            )
        result = select(
            batch,
            Interpretation(None),
            bundle.dynamics,
            [],
            model_fingerprint=fingerprint,
            seconds=1,
        )
        state.remember(result.diagnostics["hypotheses"], ctx, [])
        return batch

    def review(state, meaning, number=100):
        value = ReviewCue(
            "turn:1",
            Observation(f"turn:{number}", "Уточнение исходного эпизода", number),
            meaning,
        )
        started = perf_counter()
        proposal = state.prepare_review(value, bundle.dynamics, bundle.understanding)
        if proposal["complete"]:
            state.commit_reviews([proposal], bundle.dynamics)
        return proposal, perf_counter() - started

    for obj in ("книга", "ключ"):
        for favorite, alternative in (("петя", "миша"), ("миша", "петя")):
            for mode in ("recover", "confirm", "distract"):
                state = AttentionState(fingerprint)
                add(state, 1, [transfer(favorite, obj), transfer(alternative, obj)])
                previous = state.get("turn:1")["selected_id"]
                expected_actor = alternative if mode == "recover" else favorite
                proposed = transfer(
                    expected_actor, obj if mode != "distract" else "мяч"
                )
                outcome, seconds = review(state, proposed)
                after = state.get("turn:1")
                actor = (
                    after["selected_meaning"]["event"]["actor"]
                    if after["selected_meaning"]
                    else None
                )
                record(
                    "structured_" + mode,
                    f"{obj}:{favorite}",
                    outcome["complete"] and actor == expected_actor,
                    previous_id=previous,
                    selected_id=after["selected_id"],
                    expected_actor=expected_actor,
                    actual_actor=actor,
                    seconds=seconds,
                    reason=outcome["reason"],
                    work=outcome["work"],
                    trace=outcome["summary"],
                )

    for actor in ("петя", "миша"):
        state = AttentionState(fingerprint)
        batch = add(state, 1, [], real=True)
        outcome, seconds = review(state, transfer(actor))
        after = state.get("turn:1")
        record(
            "language_candidates_structured_cue",
            actor,
            outcome["complete"]
            and after["selected_meaning"] is not None
            and after["selected_meaning"]["event"]["actor"] == actor,
            candidate_count=len(batch.candidates),
            seconds=seconds,
            trace=outcome["summary"],
        )

    for first_actor in ("петя", "миша"):
        second_actor = "миша" if first_actor == "петя" else "петя"
        state = AttentionState(fingerprint)
        add(state, 1, [transfer(first_actor), transfer(second_actor)])
        review(state, transfer(first_actor), number=10)
        outcome, seconds = review(state, transfer(second_actor), number=11)
        record(
            "conflicting_cues",
            first_actor,
            outcome["complete"] and state.get("turn:1")["selected_id"] is None,
            seconds=seconds,
            trace=outcome["summary"],
        )

    for mode in ("clarification", "ordinary_update"):
        session = LearnedSession(bundle)
        transcripts = []
        for text in [
            "Маша положила книгу в ящик",
            "Петя положил игрушку в коробку",
            "Она на столе",
            *(["Привет"] * 18),
        ]:
            response = session.respond(text)
            transcripts.append(
                {
                    "input": text,
                    "action": response["action"],
                    "reason": response["reason"],
                }
            )
        previous = session.attention.get("turn:3")["selected_id"]
        latest = "Нет, книга на столе" if mode == "clarification" else "Книга в ящике"
        response = session.respond(latest)
        after = session.attention.get("turn:3")
        passed = all(
            r["action"] not in {"limit", "error"} for r in transcripts
        ) and response["action"] not in {"limit", "error"}
        if mode == "clarification":
            passed &= (
                after["selected_meaning"] is not None
                and after["selected_meaning"]["event"]["object"] == "книга"
            )
        else:
            passed &= after["selected_id"] == previous
        LearnedSession.from_dict(session.to_dict(), bundle)
        record(
            "live_dialogue",
            mode,
            passed,
            previous_id=previous,
            selected_id=after["selected_id"],
            history=transcripts,
            final_input=latest,
            final_response=response,
            historical_original=session.attention.get("turn:3")["original_meaning"],
        )

    state = AttentionState(fingerprint, AttentionLimits(max_archives=2))
    add(state, 1, [transfer("петя"), transfer("миша")])
    for number in range(2, 62):
        if number % 3 == 0:
            meaning = Meaning(
                "inform", Event("locate", object="книга", place="стол", time="present")
            )
        elif number % 3 == 1:
            meaning = replace(
                transfer("петя"), event=replace(transfer("петя").event, time="future")
            )
        else:
            meaning = transfer("петя", "ключ")
        add(state, number, [meaning])
    work = Work(state.limits)
    retrieved = state.retrieve(transfer("миша"), work)
    ids = [r["observation"]["observation_id"] for r in retrieved]
    outcome, seconds = review(state, transfer("миша"))
    selected = state.get("turn:1")["selected_meaning"]
    record(
        "distant_retrieval",
        "60 mixed distractors",
        ids[0] == "turn:1"
        and outcome["complete"]
        and selected is not None
        and selected["event"]["actor"] == "миша",
        retrieved=ids,
        retrieval_work=work.counts,
        source_count=len(state.records),
        seconds=seconds,
        trace=outcome["summary"],
    )

    groups = sorted({r["group"] for r in data["cases"]})
    data["totals"] = {
        group: {
            "passed": sum(r["passed"] for r in data["cases"] if r["group"] == group),
            "total": sum(r["group"] == group for r in data["cases"]),
        }
        for group in groups
    }
    data["peak_self_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data["complete"] = True
    save()
    print(json.dumps(data["totals"], ensure_ascii=False))
    return 0 if all(r["passed"] for r in data["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
