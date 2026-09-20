"""Independent sealed v0.5 evaluation data: never import from production.

Authored against the public ontology and declared training-family inventory,
before inspecting held-out predictions. No trainer/model is imported here.
Ordinary regression tests must not execute held_out or challenge predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schema import DialogueContext, Entity, Event, Meaning, Query

DATA_VERSION = "v05-independent-cases-1"
SPLITS = ("development", "held_out", "challenge")
POLICY_ACTIONS = (
    "answer",
    "ack",
    "clarify",
    "unknown",
    "explain",
    "greet",
    "thanks",
    "help",
    "retracted",
    "corrected",
    "nonactual",
)
# subject, relation, value, negated, spatial.
Fact = tuple[str, str, str, bool, str]
# op, subject, relation, value, spatial.
Effect = tuple[str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class UnderstandingCase:
    case_id: str
    family: str
    text: str
    expected: Meaning | None
    context: DialogueContext = DialogueContext()


@dataclass(frozen=True, slots=True)
class DynamicsCase:
    case_id: str
    before: tuple[Fact, ...]
    event: Event
    expected_effects: tuple[Effect, ...]
    expected_supported: bool = True
    rival: Event | None = None


@dataclass(frozen=True, slots=True)
class PolicyCase:
    case_id: str
    features: tuple[tuple[str, Any], ...]
    expected_action: str
    allowed_actions: tuple[str, ...] = POLICY_ACTIONS


@dataclass(frozen=True, slots=True)
class GenerationCase:
    case_id: str
    action: str
    slots: tuple[tuple[str, str], ...]
    evidence: tuple[Fact, ...]
    expected_grounded: bool
    forbidden_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    text: str
    state: tuple[Fact, ...]
    actions: tuple[str, ...]
    assertions: tuple[Fact, ...] | None = None


@dataclass(frozen=True, slots=True)
class DialogueCase:
    case_id: str
    turns: tuple[DialogueTurn, ...]


def fact_dict(fact: Fact, *, evidence: bool = False) -> dict[str, Any]:
    subject, relation, value, negated, spatial = fact
    data: dict[str, Any] = {
        "subject": subject,
        "relation": relation,
        "value": value,
        "negated": negated,
        "spatial": spatial,
    }
    if evidence:
        data.update(source="user", event_id=1)
    return data


def effect_dict(effect: Effect) -> dict[str, str]:
    return dict(
        zip(("op", "subject", "relation", "value", "spatial"), effect, strict=True)
    )


def serialize_case(case: Any) -> dict[str, Any]:
    """Dataclass values are subsequently normalized by the report serializer."""
    return asdict(case)


def _loc(obj: str, place: str, spatial: str = "in", negated: bool = False) -> Fact:
    return (obj, "location", place, negated, spatial)


def _holder(obj: str, actor: str, negated: bool = False) -> Fact:
    return (obj, "holder", actor, negated, "in")


def _info(predicate: str, *, act: str = "inform", **kwargs: Any) -> Meaning:
    return Meaning(act, event=Event(predicate, **kwargs))


def _turn(
    text: str,
    state: tuple[Fact, ...],
    action: str | tuple[str, ...] = "ack",
    assertions: tuple[Fact, ...] | None = None,
) -> DialogueTurn:
    return DialogueTurn(
        text,
        tuple(sorted(state)),
        (action,) if isinstance(action, str) else action,
        assertions,
    )


def understanding_cases(split: str) -> tuple[UnderstandingCase, ...]:
    _validate_split(split)
    if split == "development":
        return (
            UnderstandingCase(
                "dev/u/move",
                "train_shape_move",
                "Миша положил ключ в сумку.",
                _info("move", actor="миша", object="ключ", place="сумка"),
            ),
            UnderstandingCase(
                "dev/u/have",
                "train_shape_have",
                "У Маши есть книга.",
                _info("have", actor="маша", object="книга", time="present"),
            ),
            UnderstandingCase(
                "dev/u/where",
                "train_shape_where",
                "Где паспорт?",
                Meaning("ask", query=Query("where", subject="паспорт")),
            ),
            UnderstandingCase(
                "dev/u/give",
                "train_shape_give",
                "Миша передал ключ Маше.",
                _info("give", actor="миша", object="ключ", recipient="маша"),
            ),
            UnderstandingCase(
                "dev/u/correct",
                "train_shape_correction",
                "Нет, ключ в ящике.",
                _info(
                    "locate", act="correct", object="ключ", place="ящик", time="present"
                ),
            ),
            UnderstandingCase(
                "dev/u/greet", "train_shape_social", "Привет.", Meaning("greet")
            ),
        )
    # Reserved set independently changes entities, phrase order and nesting.
    challenge = split == "challenge"
    actor, other, obj, place = (
        ("борис", "анна", "карандаш", "шкаф")
        if challenge
        else ("петя", "катя", "паспорт", "рюкзак")
    )
    dat = "анне" if challenge else "кате"
    prep = "шкафу" if challenge else "рюкзаке"
    person = Entity(actor, "person", "male")
    target = Entity(obj, "thing", "male")
    result: list[UnderstandingCase] = []

    def add(
        family: str,
        text: str,
        expected: Meaning | None,
        context: DialogueContext | None = None,
    ) -> None:
        result.append(
            UnderstandingCase(
                f"{split}/u/{family}",
                family,
                text,
                expected,
                DialogueContext() if context is None else context,
            )
        )

    add(
        "locative_fronting",
        f"В {prep} находится {obj}.",
        _info("locate", object=obj, place=place, time="present"),
    )
    add(
        "negative_locative_fronting",
        f"В {prep} не лежит {obj}.",
        _info("locate", object=obj, place=place, negated=True, time="present"),
    )
    add(
        "destination_fronting",
        f"В сумку {actor} положил {obj}.",
        _info("move", actor=actor, object=obj, place="сумка"),
    )
    add(
        "dative_fronting",
        f"{dat} {actor} передал {obj}.",
        _info("give", actor=actor, object=obj, recipient=other),
    )
    add(
        "verb_final_transfer",
        f"{actor} {dat} {obj} отдал.",
        _info("give", actor=actor, object=obj, recipient=other),
    )
    add(
        "negative_dative_fronting",
        f"{dat} {actor} не передал {obj}.",
        _info("give", actor=actor, object=obj, recipient=other, negated=True),
    )
    add(
        "object_fronted_have",
        f"{obj} есть у {actor[:-1] + 'и' if actor == 'петя' else 'бориса'}.",
        _info("have", actor=actor, object=obj, time="present"),
    )
    add(
        "question_subject_first",
        f"{obj} где находится?",
        Meaning("ask", query=Query("where", subject=obj)),
    )
    add(
        "holder_subject_first",
        f"{obj} у кого?",
        Meaning("ask", query=Query("who_has", subject=obj)),
    )
    add(
        "corrected_fronting",
        f"Нет, в {prep} находится {obj}.",
        _info("locate", act="correct", object=obj, place=place, time="present"),
    )
    promised = Event(
        "give",
        actor=actor,
        object=obj,
        recipient=other,
        modality="intended",
        time="future",
    )
    promise = Event("promise", actor=actor, content=promised, time="past")
    add(
        "promise_role_composition",
        f"Передать {obj} {dat} обещал {actor}.",
        Meaning("inform", event=promise),
    )
    reported = Event(
        "give",
        actor=actor,
        object=obj,
        recipient=other,
        modality="reported",
        time="past",
    )
    add(
        "report_fronted_content",
        f"Миша сообщил, что {dat} {actor} передал {obj}.",
        Meaning("inform", event=Event("report", actor="миша", content=reported)),
    )
    add(
        "possible_fronting",
        f"Возможно, в {prep} находится {obj}.",
        _info("locate", object=obj, place=place, modality="possible", time="present"),
    )
    conditional = Event(
        "locate", object=obj, place=place, modality="conditional", time="present"
    )
    add(
        "conditional_fronting",
        f"Если в {prep} находится {obj}.",
        Meaning(
            "inform",
            event=Event(
                "conditional",
                modality="conditional",
                time="unspecified",
                content=conditional,
            ),
        ),
    )
    context = DialogueContext(
        turns=(f"{actor} положил {obj} в ящик.",),
        entities=(person, target),
        focus=(obj,),
    )
    add(
        "resolved_object_reference",
        "Маша переложила его в сумку.",
        _info("move", actor="маша", object=obj, place="сумка"),
        context,
    )
    add("unresolved_reference", "Она передала его ему.", None)
    add("unsupported_multi_assertion", f"{obj} в {prep}; нарисуй галактику.", None)
    if challenge:
        nested = Event("promise", actor=actor, modality="reported", content=promised)
        add(
            "reported_fronted_promise",
            f"Маша сообщила, что передать {obj} {dat} обещал {actor}.",
            Meaning("inform", event=Event("report", actor="маша", content=nested)),
        )
        add(
            "ambiguous_two_objects",
            "Миша переложил его в ящик.",
            None,
            DialogueContext(
                entities=(target, Entity("ключ", "thing", "male")), focus=(obj, "ключ")
            ),
        )
    return tuple(result)


def dynamics_cases(split: str) -> tuple[DynamicsCase, ...]:
    _validate_split(split)
    actor, peer, obj, place = (
        ("миша", "маша", "ключ", "ящик")
        if split == "development"
        else ("исследователь", "курьер", "образец", "сейф")
        if split == "held_out"
        else ("архивариус", "хранитель", "свиток", "контейнер")
    )
    base = _holder(obj, actor)
    other = _holder(obj, peer)
    give = Event("give", actor=actor, object=obj, recipient=peer)
    cases: list[DynamicsCase] = []

    def add(
        name: str,
        before: tuple[Fact, ...],
        event: Event,
        effects: tuple[Effect, ...] = (),
        *,
        rival: Event | None = None,
    ) -> None:
        cases.append(
            DynamicsCase(f"{split}/d/{name}", before, event, effects, rival=rival)
        )

    add(
        "movement",
        (base,),
        Event("move", actor=actor, object=obj, place=place),
        (("set", obj, "location", place, "in"),),
    )
    add("transfer", (base,), give, (("set", obj, "holder", peer, "in"),))
    add(
        "negative_transfer",
        (base,),
        Event("give", actor=actor, object=obj, recipient=peer, negated=True),
    )
    add(
        "negative_location",
        (),
        Event("locate", object=obj, place=place, time="present", negated=True),
        (("exclude", obj, "location", place, "in"),),
    )
    add(
        "promise_scope",
        (base,),
        Event(
            "promise",
            actor=actor,
            content=Event(
                "give",
                actor=actor,
                object=obj,
                recipient=peer,
                modality="intended",
                time="future",
            ),
        ),
    )
    if split == "development":
        return tuple(cases)
    add(
        "role_reverse",
        (other,),
        Event("give", actor=peer, object=obj, recipient=actor),
        (("set", obj, "holder", actor, "in"),),
    )
    add(
        "spatial_on",
        (),
        Event("locate", object=obj, place=place, spatial="on", time="present"),
        (("set", obj, "location", place, "on"),),
    )
    add(
        "negative_possession",
        (base,),
        Event("have", actor=actor, object=obj, negated=True, time="present"),
        (("exclude", obj, "holder", actor, "in"),),
    )
    add(
        "future_move",
        (base,),
        Event(
            "move",
            actor=actor,
            object=obj,
            place=place,
            modality="intended",
            time="future",
        ),
    )
    add(
        "reported_transfer",
        (base,),
        Event(
            "report",
            actor=peer,
            content=Event(
                "give", actor=actor, object=obj, recipient=peer, modality="reported"
            ),
        ),
    )
    add(
        "possible_location",
        (base,),
        Event("locate", object=obj, place=place, modality="possible", time="present"),
    )
    add(
        "counterfactual_giver",
        (base,),
        give,
        (("set", obj, "holder", peer, "in"),),
        rival=Event("give", actor=peer, object=obj, recipient=actor),
    )
    if split == "challenge":
        add(
            "nested_report_promise",
            (base,),
            Event(
                "report",
                actor=peer,
                content=Event(
                    "promise",
                    actor=actor,
                    modality="reported",
                    content=Event(
                        "give",
                        actor=actor,
                        object=obj,
                        recipient=peer,
                        modality="intended",
                        time="future",
                    ),
                ),
            ),
        )
        add(
            "unrelated_fact_invariance",
            (base, _loc("лампа", "комната")),
            give,
            (("set", obj, "holder", peer, "in"),),
        )
    return tuple(cases)


def policy_cases(split: str) -> tuple[PolicyCase, ...]:
    _validate_split(split)
    cases: list[PolicyCase] = []

    def add(name: str, action: str, **features: Any) -> None:
        cases.append(
            PolicyCase(f"{split}/p/{name}", tuple(sorted(features.items())), action)
        )

    add(
        "known_answer",
        "answer",
        act="ask",
        has_answer=True,
        evidence_count=1,
        prev_action="ack",
        task="where",
    )
    add(
        "unknown_answer",
        "unknown",
        act="ask",
        has_answer=False,
        evidence_count=0,
        prev_action="answer",
        task="where",
    )
    add(
        "ambiguous_reference",
        "clarify",
        act="inform",
        ambiguous=True,
        has_answer=False,
        pending="reference",
        prev_action="answer",
    )
    add(
        "acknowledgement",
        "ack",
        act="inform",
        has_answer=False,
        evidence_count=1,
        prev_action="answer",
    )
    if split == "development":
        return tuple(cases)
    add(
        "correction_after_clarification",
        "corrected",
        act="correct",
        corrected=True,
        prev_action="clarify",
        pending="reference",
        evidence_count=1,
    )
    add(
        "retract_after_unknown",
        "retracted",
        act="retract",
        retracted=True,
        prev_action="unknown",
        evidence_count=1,
    )
    add(
        "scope_after_answer",
        "nonactual",
        act="inform",
        nonactual=True,
        prev_action="answer",
        evidence_count=1,
    )
    add(
        "unsupported_after_correction",
        "clarify",
        act="unknown",
        unsupported=True,
        prev_action="corrected",
        pending="task",
        evidence_count=0,
    )
    add(
        "explanation_after_correction",
        "explain",
        act="ask",
        has_answer=True,
        task="why",
        prev_action="corrected",
        evidence_count=1,
    )
    if split == "challenge":
        add(
            "ambiguous_overrides_answer",
            "clarify",
            act="ask",
            has_answer=True,
            ambiguous=True,
            evidence_count=2,
            prev_action="nonactual",
        )
        add(
            "failed_turn_overrides_ack",
            "clarify",
            act="inform",
            failed=True,
            evidence_count=1,
            prev_action="retracted",
        )
    return tuple(cases)


def generation_cases(split: str) -> tuple[GenerationCase, ...]:
    _validate_split(split)
    obj, place, person = (
        ("ключ", "ящик", "миша")
        if split == "development"
        else ("документ", "шкаф", "катя")
        if split == "held_out"
        else ("карандаш", "рюкзак", "борис")
    )
    location = _loc(obj, place)
    possession = _holder(obj, person)
    cases = [
        GenerationCase(
            f"{split}/g/location",
            "answer",
            (("object", obj), ("place", place), ("prep", "in")),
            (location,),
            True,
        ),
        GenerationCase(
            f"{split}/g/holder",
            "answer",
            (("object", obj), ("holder", person)),
            (possession,),
            True,
        ),
        GenerationCase(f"{split}/g/unknown", "unknown", (("object", obj),), (), True),
    ]
    if split == "development":
        return tuple(cases)
    cases.extend(
        (
            GenerationCase(
                f"{split}/g/unsupported_place",
                "answer",
                (("object", obj), ("place", "сейф"), ("prep", "in")),
                (location,),
                False,
                ("сейф",),
            ),
            GenerationCase(
                f"{split}/g/unsupported_holder",
                "answer",
                (("object", obj), ("holder", "курьер")),
                (possession,),
                False,
                ("курьер",),
            ),
            GenerationCase(
                f"{split}/g/negative_verification",
                "answer",
                (("object", obj), ("place", place), ("prep", "in"), ("truth", "no")),
                (_loc(obj, place, negated=True),),
                True,
            ),
        )
    )
    if split == "challenge":
        cases.append(
            GenerationCase(
                f"{split}/g/no_evidence_positive",
                "answer",
                (("object", obj), ("place", place), ("prep", "in")),
                (),
                False,
            )
        )
    return tuple(cases)


def dialogue_cases(split: str) -> tuple[DialogueCase, ...]:
    _validate_split(split)
    obj = (
        "ключ"
        if split == "development"
        else "паспорт"
        if split == "held_out"
        else "карандаш"
    )
    actor = (
        "миша" if split == "development" else "петя" if split == "held_out" else "борис"
    )
    peer = (
        "маша" if split == "development" else "катя" if split == "held_out" else "анна"
    )
    peer_dat = (
        "маше" if split == "development" else "кате" if split == "held_out" else "анне"
    )
    actor_dat = (
        "мише"
        if split == "development"
        else "пете"
        if split == "held_out"
        else "борису"
    )
    box, bag, table = _loc(obj, "ящик"), _loc(obj, "сумка"), _loc(obj, "стол", "on")
    recipient = _holder(obj, peer)
    author = _holder(obj, actor)
    turns = (
        _turn(f"{obj} в ящике.", (box,)),
        _turn(f"Где {obj}?", (box,), "answer", (box,)),
        _turn(f"{actor} переложил {obj} в сумку.", (bag,)),
        _turn(f"Где {obj}?", (bag,), "answer", (bag,)),
        _turn(f"{actor} не переложил {obj} в ящик.", (bag,)),
        _turn(f"Где {obj}?", (bag,), "answer", (bag,)),
        _turn(f"Нет, {obj} на столе.", (table,), "corrected"),
        _turn(f"Где {obj}?", (table,), "answer", (table,)),
        _turn("Отмени последнее утверждение.", (bag,), "retracted"),
        _turn(f"Где {obj}?", (bag,), "answer", (bag,)),
    )
    if split == "development":
        return (DialogueCase("development/chat/memory_updates", turns),)
    turns += (
        _turn(f"{actor} передал {obj} {peer_dat}.", (recipient,)),
        _turn(f"У кого {obj}?", (recipient,), "answer", (recipient,)),
        _turn(
            f"{peer} обещала передать {obj} {actor_dat}.", (recipient,), "nonactual", ()
        ),
        _turn(f"У кого {obj}?", (recipient,), "answer", (recipient,)),
        _turn(
            f"{peer} сообщила, что {actor} положил {obj} в ящик.",
            (recipient,),
            "nonactual",
            (),
        ),
        _turn(f"У кого {obj}?", (recipient,), "answer", (recipient,)),
        _turn(f"{peer} передала {obj} {actor_dat}.", (author,)),
        _turn(f"У кого {obj}?", (author,), "answer", (author,)),
        _turn("Где тессеракт?", (author,), ("unknown", "clarify"), ()),
        _turn(f"Где {obj}?", (author,), "answer", (author,)),
        _turn("Маша переложила его в сумку.", (bag,)),
        _turn(f"Где {obj}?", (bag,), "answer", (bag,)),
    )
    ambiguity = (
        _turn(f"{obj} в ящике.", (box,)),
        _turn("Ключ в сумке.", (box, _loc("ключ", "сумка"))),
        _turn(f"Где {obj}?", (box, _loc("ключ", "сумка")), "answer", (box,)),
        _turn("Миша переложил его на стол.", (table, _loc("ключ", "сумка"))),
        _turn(f"Где {obj}?", (table, _loc("ключ", "сумка")), "answer", (table,)),
        _turn(
            f"{obj} в ящике; открой портал.",
            (table, _loc("ключ", "сумка")),
            "clarify",
            (),
        ),
        _turn(f"Где {obj}?", (table, _loc("ключ", "сумка")), "answer", (table,)),
    )
    return (
        DialogueCase(f"{split}/chat/long_scope_updates", turns),
        DialogueCase(f"{split}/chat/query_focus_atomicity", ambiguity),
    )


def _validate_split(split: str) -> None:
    if type(split) is not str or split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")


def understanding_stratum(case: UnderstandingCase) -> str:
    """Declare phrasal, scope, reference and safe-rejection axes separately."""
    if case.case_id.startswith("dev/"):
        return "development_public"
    if case.family in {
        "promise_role_composition",
        "report_fronted_content",
        "possible_fronting",
        "conditional_fronting",
        "reported_fronted_promise",
    }:
        return "scoped_composition"
    if case.family == "resolved_object_reference":
        return "dialogue_reference"
    if case.expected is None:
        return "abstention_and_atomicity"
    return "independent_phrasal_family"


def all_cases(split: str = "development") -> dict[str, tuple[Any, ...]]:
    return {
        "understanding": understanding_cases(split),
        "dynamics": dynamics_cases(split),
        "policy": policy_cases(split),
        "generation": generation_cases(split),
        "dialogue": dialogue_cases(split),
    }
