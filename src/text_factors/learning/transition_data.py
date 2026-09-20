"""Public annotated training/development episodes; no independent test split.

Reference projection below is a teacher-data generator, never imported by the
runtime predictor. The small ontology and functional whereabouts convention are
declared scaffolds. Participants are varied independently of structural effects.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .dynamics import TransitionEpisode, checked_facts
from .schema import Event


def _fact(
    subject: str,
    relation: str,
    value: str,
    *,
    negated: bool = False,
    spatial: str = "in",
) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "value": value,
        "negated": negated,
        "spatial": spatial,
    }


def _teacher_after(before: list[dict[str, Any]], event: Event) -> list[dict[str, Any]]:
    if not event.actual or (event.negated and event.predicate in {"move", "give"}):
        return deepcopy(before)
    relation = "location" if event.predicate in {"locate", "move"} else "holder"
    value = (
        event.place
        if relation == "location"
        else event.recipient
        if event.predicate == "give"
        else event.actor
    )
    added = _fact(
        event.object,
        relation,
        value,
        negated=event.negated,
        spatial=event.spatial if relation == "location" else "in",
    )
    if not event.negated:
        retained = [
            fact
            for fact in before
            if not (
                fact["subject"] == event.object
                and (
                    not fact["negated"]
                    or all(
                        fact[k] == added[k] for k in ("relation", "value", "spatial")
                    )
                )
            )
        ]
    else:
        retained = [
            fact
            for fact in before
            if not all(
                fact[k] == added[k] for k in ("subject", "relation", "value", "spatial")
            )
        ]
    return checked_facts(retained + [added])


def _episodes(seed: int, *, development: bool) -> list[TransitionEpisode]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid transition-data seed")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 971, int(development)]))
    actor, recipient, thing, place = (
        ("данил", "алиса", "компас", "кабинет")
        if development
        else ("миша", "маша", "ключ", "ящик")
    )
    result = []
    # Same structural role situations, with independently renamed entities in
    # development. Future/modality examples collapse only along the disclosed
    # actual/nonactual feature axis, not by a runtime no-op rule.
    for predicate in ("locate", "move", "give", "have"):
        for negated in (False, True):
            for nonactual in (False, True):
                for spatial in ("in", "on"):
                    for prior in (
                        "empty",
                        "actor",
                        "third_party",
                        "other_place",
                        "other_surface",
                        "same_place",
                    ):
                        if prior == "same_place" and predicate not in {
                            "locate",
                            "move",
                        }:
                            continue
                        before = []
                        if prior == "actor":
                            before = [_fact(thing, "holder", actor)]
                        elif prior == "third_party":
                            before = [_fact(thing, "holder", "наблюдатель")]
                        elif prior == "other_place":
                            before = [_fact(thing, "location", "шкаф")]
                        elif prior == "other_surface":
                            before = [_fact(thing, "location", "стол", spatial="on")]
                        elif prior == "same_place":
                            before = [_fact(thing, "location", place, spatial=spatial)]
                        event = Event(
                            predicate,
                            actor=actor if predicate != "locate" else "",
                            object=thing,
                            recipient=recipient if predicate == "give" else "",
                            place=place if predicate in {"locate", "move"} else "",
                            spatial=spatial,
                            negated=negated,
                            modality="possible" if nonactual else "actual",
                            time="present" if development else "past",
                        )
                        result.append(
                            TransitionEpisode(
                                before, event, _teacher_after(before, event)
                            )
                        )
    # Teacher episodes establish wrappers as non-projecting scopes. The fixed
    # dynamics abstraction preserves outer roles/scope and intentionally does
    # not attempt to reason inside promised/reported/conditional content.
    content = Event("give", actor=actor, object=thing, recipient=recipient)
    for predicate in ("promise", "report", "conditional"):
        for negated in (False, True):
            for with_before in (False, True):
                for outer_actor in (actor, ""):
                    before = [_fact(thing, "holder", actor)] if with_before else []
                    event = Event(
                        predicate, actor=outer_actor, negated=negated, content=content
                    )
                    result.append(TransitionEpisode(before, event, deepcopy(before)))
    rng.shuffle(result)
    return result


def training_episodes(seed: int = 42) -> list[TransitionEpisode]:
    return _episodes(seed, development=False)


def development_episodes(seed: int = 42) -> list[TransitionEpisode]:
    """Public renamed-role checks, not an independent quality benchmark."""
    return _episodes(seed, development=True)
