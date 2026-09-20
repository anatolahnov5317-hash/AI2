"""Explicit annotated training/development data for the v0.5 language learner.

Templates here generate *training examples*, never inference-time parses. The
finite entity aliases and their grammatical annotations are disclosed input
scaffolding. This corpus does not teach arbitrary Russian morphology or claim
that supervised semantic labels were discovered by the model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .schema import (
    DialogueContext,
    Entity,
    Event,
    Meaning,
    Query,
    bounded_text,
    exact_fields,
)

TRAIN_DATA_VERSION = "v05-understanding-train-v2"
_EMPTY_CONTEXT = DialogueContext()
TRAIN_FAMILIES = (
    "location_bare",
    "location_svo",
    "movement_svo",
    "transfer_svo_dative",
    "transfer_received_ot",
    "possession_u",
    "possession_svo",
    "where_initial",
    "holder_initial",
    "inventory_initial",
    "provenance_explicit",
    "verification_trailing",
    "correction_prefix",
    "retraction_explicit",
    "social_explicit",
    "future_svo",
    "possibility_prefix",
    "promise_infinitive",
    "report_chto",
    "conditional_fragment",
    "report_promise",
    "context_pronoun_svo",
    "incomplete_roles",
)

_TOKENIZER = re.compile(r"[a-zа-я0-9]+(?:[-_][a-zа-я0-9]+)*|[^\s]", re.IGNORECASE)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKENIZER.findall(text.casefold().replace("ё", "е")))


@dataclass(frozen=True, slots=True)
class EntityForms:
    entity: Entity
    nominative: str
    accusative: str
    genitive: str
    dative: str
    locative: str


def _forms(
    name: str, kind: str, gender: str, acc: str, gen: str, dat: str, loc: str
) -> EntityForms:
    return EntityForms(Entity(name, kind, gender), name, acc, gen, dat, loc)


ENTITY_ROWS: tuple[EntityForms, ...] = (
    _forms("миша", "person", "male", "мишу", "миши", "мише", "мише"),
    _forms("петя", "person", "male", "петю", "пети", "пете", "пете"),
    _forms("маша", "person", "female", "машу", "маши", "маше", "маше"),
    _forms("катя", "person", "female", "катю", "кати", "кате", "кате"),
    _forms("саша", "person", "male", "сашу", "саши", "саше", "саше"),
    _forms("анна", "person", "female", "анну", "анны", "анне", "анне"),
    _forms("иван", "person", "male", "ивана", "ивана", "ивану", "иване"),
    _forms("оля", "person", "female", "олю", "оли", "оле", "оле"),
    _forms("я", "person", "unknown", "меня", "меня", "мне", "мне"),
    _forms("ключ", "thing", "male", "ключ", "ключа", "ключу", "ключе"),
    _forms("книга", "thing", "female", "книгу", "книги", "книге", "книге"),
    _forms("мяч", "thing", "male", "мяч", "мяча", "мячу", "мяче"),
    _forms("паспорт", "thing", "male", "паспорт", "паспорта", "паспорту", "паспорте"),
    _forms("телефон", "thing", "male", "телефон", "телефона", "телефону", "телефоне"),
    _forms("письмо", "thing", "neuter", "письмо", "письма", "письму", "письме"),
    _forms("игрушка", "thing", "female", "игрушку", "игрушки", "игрушке", "игрушке"),
    _forms("куб", "thing", "male", "куб", "куба", "кубу", "кубе"),
    _forms("ящик", "place", "male", "ящик", "ящика", "ящику", "ящике"),
    _forms("сумка", "place", "female", "сумку", "сумки", "сумке", "сумке"),
    _forms("стол", "place", "male", "стол", "стола", "столу", "столе"),
    _forms("комната", "place", "female", "комнату", "комнаты", "комнате", "комнате"),
    _forms("коробка", "place", "female", "коробку", "коробки", "коробке", "коробке"),
    _forms("шкаф", "place", "male", "шкаф", "шкафа", "шкафу", "шкафу"),
    _forms("рюкзак", "place", "male", "рюкзак", "рюкзака", "рюкзаку", "рюкзаке"),
    _forms("полка", "place", "female", "полку", "полки", "полке", "полке"),
)
ENTITY_BY_NAME = {row.entity.name: row for row in ENTITY_ROWS}
ENTITY_ALIASES: dict[str, tuple[Entity, str]] = {}
for _row in ENTITY_ROWS:
    for _case in ("nominative", "accusative", "genitive", "dative", "locative"):
        _word = getattr(_row, _case)
        _prior = ENTITY_ALIASES.get(_word)
        ENTITY_ALIASES[_word] = (
            _row.entity,
            f"{_prior[1]}|{_case}" if _prior else _case,
        )

# Token annotations, not a resolver. The learned contextual scorer chooses a
# referent (or its learned NULL candidate) from DialogueContext entities.
PRONOUN_FORMS: dict[str, tuple[str, str]] = {
    "он": ("male", "nominative"),
    "она": ("female", "nominative"),
    "оно": ("neuter", "nominative"),
    "его": ("male", "accusative"),
    "ее": ("female", "accusative"),
    "ему": ("male", "dative"),
    "ей": ("female", "dative"),
    "нем": ("male", "locative"),
    "ней": ("female", "locative"),
    "это": ("unknown", "nominative"),
}


@dataclass(frozen=True, slots=True)
class TrainingUtterance:
    text: str
    meaning: Meaning
    context: DialogueContext = DialogueContext()
    family: str = "annotated"
    links: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        bounded_text(self.text, "training text", cap=2048, empty=False)
        bounded_text(self.family, "training family", empty=False)
        if not isinstance(self.meaning, Meaning) or not isinstance(
            self.context, DialogueContext
        ):
            raise ValueError("invalid training annotation")
        if type(self.links) is not tuple or len(self.links) > 16:
            raise ValueError("invalid reference links")
        seen: set[int] = set()
        tokens = tokenize(self.text)
        referents = {entity.name for entity in self.context.entities}
        for pair in self.links:
            if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not int:
                raise ValueError("invalid reference link")
            if not 0 <= pair[0] < len(tokens) or pair[0] in seen:
                raise ValueError("invalid reference span")
            bounded_text(pair[1], "reference target", empty=False)
            if tokens[pair[0]] not in PRONOUN_FORMS or pair[1] not in referents:
                raise ValueError("reference link requires a contextual pronoun target")
            seen.add(pair[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "meaning": self.meaning.to_dict(),
            "context": self.context.to_dict(),
            "family": self.family,
            "links": [list(link) for link in self.links],
        }

    @classmethod
    def from_dict(cls, value: Any) -> TrainingUtterance:
        value = exact_fields(
            value,
            {"text", "meaning", "context", "family", "links"},
            "training utterance",
        )
        if type(value["links"]) is not list or len(value["links"]) > 16:
            raise ValueError("invalid reference links")
        if any(type(link) is not list or len(link) != 2 for link in value["links"]):
            raise ValueError("invalid reference link")
        return cls(
            value["text"],
            Meaning.from_dict(value["meaning"]),
            DialogueContext.from_dict(value["context"]),
            value["family"],
            tuple((link[0], link[1]) for link in value["links"]),
        )


@dataclass(frozen=True, slots=True)
class ReferenceExample:
    pronoun: str
    role: str
    context: DialogueContext
    target: str = ""


def data_fingerprint(examples: Iterable[TrainingUtterance]) -> str:
    payload = [
        {
            "text": ex.text,
            "meaning": ex.meaning.to_dict(),
            "context": ex.context.to_dict(),
            "family": ex.family,
            "links": list(ex.links),
        }
        for ex in examples
    ]
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _meaning(
    act: str, *, event: Event | None = None, query: Query | None = None
) -> Meaning:
    names: list[str] = []

    def visit(node: Event) -> None:
        for name in (node.actor, node.object, node.recipient, node.place):
            if name and name not in names:
                names.append(name)
        if node.content:
            visit(node.content)
        if node.condition:
            visit(node.condition)

    if event:
        visit(event)
    if query:
        names.extend(
            name for name in (query.subject, query.value) if name and name not in names
        )
    entities = tuple(ENTITY_BY_NAME[name].entity for name in names)
    return Meaning(act, event, query, entities)


def training_examples(seed: int = 42) -> list[TrainingUtterance]:
    """Annotated phrasal families; held-out phrase families live elsewhere."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid data seed")
    people = [row for row in ENTITY_ROWS if row.entity.kind == "person"]
    things = [row for row in ENTITY_ROWS if row.entity.kind == "thing"]
    places = [row for row in ENTITY_ROWS if row.entity.kind == "place"]
    result: list[TrainingUtterance] = []

    def add(
        text: str,
        family: str,
        *,
        event: Event | None = None,
        query: Query | None = None,
        act: str = "inform",
        context: DialogueContext = _EMPTY_CONTEXT,
        links: tuple[tuple[int, str], ...] = (),
    ) -> None:
        result.append(
            TrainingUtterance(
                text, _meaning(act, event=event, query=query), context, family, links
            )
        )

    combinations = []
    for index, person in enumerate(people):
        initial = things[(index * 3) % len(things)]
        contrast = things[0] if initial.entity.gender == "female" else things[1]
        combinations.extend(((index, person, initial), (index, person, contrast)))
    for index, person, thing in combinations:
        other = people[(index + 2) % 8]
        reporter = people[(index + 4) % 8]
        place = places[(index * 5) % len(places)]
        actor, recipient = person.entity.name, other.entity.name
        obj, where = thing.entity.name, place.entity.name
        female = person.entity.gender == "female"
        a, o, d = person.nominative, thing.accusative, other.dative
        for spatial, prep in (("in", "в"), ("on", "на")):
            for negative in (False, True):
                neg = "не " if negative else ""
                location = Event(
                    "locate",
                    object=obj,
                    place=where,
                    spatial=spatial,
                    negated=negative,
                    time="present",
                )
                add(
                    f"{thing.nominative} {neg}{prep} {place.locative}.",
                    "location_bare",
                    event=location,
                )
                for verb in ("лежит", "находится"):
                    add(
                        f"{thing.nominative} {neg}{verb} {prep} {place.locative}.",
                        "location_svo",
                        event=location,
                    )
                add(
                    f"{thing.nominative} {neg}{prep} {place.locative}?",
                    "verification_trailing",
                    act="ask",
                    query=Query(
                        "verify",
                        subject=obj,
                        value=where,
                        relation="location",
                        spatial=spatial,
                        negated=negative,
                    ),
                )
                for masculine, feminine in (
                    ("положил", "положила"),
                    ("переложил", "переложила"),
                    ("поместил", "поместила"),
                ):
                    verb = feminine if female else masculine
                    event = Event(
                        "move",
                        actor=actor,
                        object=obj,
                        place=where,
                        spatial=spatial,
                        negated=negative,
                    )
                    add(
                        f"{a} {neg}{verb} {o} {prep} {place.accusative}.",
                        "movement_svo",
                        event=event,
                    )
            event = Event(
                "move",
                actor=actor,
                object=obj,
                place=where,
                spatial=spatial,
                modality="intended",
                time="future",
            )
            future_verb = "положу" if actor == "я" else "положит"
            auxiliary = "буду" if actor == "я" else "будет"
            add(
                f"{a} {future_verb} {o} {prep} {place.accusative}.",
                "future_svo",
                event=event,
            )
            add(
                f"{a} {auxiliary} класть {o} {prep} {place.accusative}.",
                "future_svo",
                event=event,
            )
            loc = Event(
                "locate",
                object=obj,
                place=where,
                spatial=spatial,
                modality="possible",
                time="present",
            )
            add(
                f"возможно, {thing.nominative} {prep} {place.locative}.",
                "possibility_prefix",
                event=loc,
            )
            add(
                f"нет, {thing.nominative} {prep} {place.locative}.",
                "correction_prefix",
                act="correct",
                event=Event(
                    "locate", object=obj, place=where, spatial=spatial, time="present"
                ),
            )
            cond = Event(
                "locate",
                object=obj,
                place=where,
                spatial=spatial,
                modality="conditional",
                time="present",
            )
            add(
                f"если {thing.nominative} {prep} {place.locative}.",
                "conditional_fragment",
                event=Event(
                    "conditional",
                    content=cond,
                    modality="conditional",
                    time="unspecified",
                ),
            )
        for negative in (False, True):
            neg = "не " if negative else ""
            for masculine, feminine in (("передал", "передала"), ("отдал", "отдала")):
                verb = feminine if female else masculine
                give = Event(
                    "give",
                    actor=actor,
                    object=obj,
                    recipient=recipient,
                    negated=negative,
                )
                add(f"{a} {neg}{verb} {o} {d}.", "transfer_svo_dative", event=give)
            received = "получила" if other.entity.gender == "female" else "получил"
            add(
                f"{other.nominative} {neg}{received} {o} от {person.genitive}.",
                "transfer_received_ot",
                event=Event(
                    "give",
                    actor=actor,
                    object=obj,
                    recipient=recipient,
                    negated=negative,
                ),
            )
            have = Event(
                "have", actor=actor, object=obj, negated=negative, time="present"
            )
            phrase = (
                f"у {person.genitive} нет {thing.genitive}."
                if negative
                else f"у {person.genitive} есть {thing.nominative}."
            )
            add(phrase, "possession_u", event=have)
            possess_verb = "имею" if actor == "я" else "имеет"
            add(
                f"{a} {neg}{possess_verb} {thing.genitive if negative else o}.",
                "possession_svo",
                event=have,
            )
        add(
            f"{a} в {place.locative}.",
            "location_bare",
            event=Event("locate", object=actor, place=where, time="present"),
        )
        add(
            f"где {a}?", "where_initial", act="ask", query=Query("where", subject=actor)
        )
        for text in (f"где {thing.nominative}?", f"где находится {thing.nominative}?"):
            add(text, "where_initial", act="ask", query=Query("where", subject=obj))
        for text in (f"у кого {thing.nominative}?", f"у кого есть {thing.nominative}?"):
            add(text, "holder_initial", act="ask", query=Query("who_has", subject=obj))
        for text in (f"что у {person.genitive}?", f"что у {person.genitive} есть?"):
            add(
                text,
                "inventory_initial",
                act="ask",
                query=Query("what_has", subject=actor),
            )
        add(
            f"почему {thing.nominative}?",
            "provenance_explicit",
            act="ask",
            query=Query("why", subject=obj),
        )
        add(
            f"у {person.genitive} есть {thing.nominative}?",
            "verification_trailing",
            act="ask",
            query=Query("verify", subject=obj, value=actor, relation="holder"),
        )
        add(
            f"у {person.genitive} нет {thing.genitive}?",
            "verification_trailing",
            act="ask",
            query=Query(
                "verify", subject=obj, value=actor, relation="holder", negated=True
            ),
        )
        add(
            f"нет, у {person.genitive} есть {thing.nominative}.",
            "correction_prefix",
            act="correct",
            event=Event("have", actor=actor, object=obj, time="present"),
        )
        promise_verb = "обещала" if female else "обещал"
        promised = Event(
            "give",
            actor=actor,
            object=obj,
            recipient=recipient,
            modality="intended",
            time="future",
        )
        promise = Event("promise", actor=actor, content=promised)
        add(
            f"{a} {promise_verb} передать {o} {d}.", "promise_infinitive", event=promise
        )
        add(
            f"{a} не {promise_verb} передать {o} {d}.",
            "promise_infinitive",
            event=Event("promise", actor=actor, negated=True, content=promised),
        )
        promised_negative = Event(
            "give",
            actor=actor,
            object=obj,
            recipient=recipient,
            negated=True,
            modality="intended",
            time="future",
        )
        add(
            f"{a} {promise_verb} не передать {o} {d}.",
            "promise_infinitive",
            event=Event("promise", actor=actor, content=promised_negative),
        )
        report_verb = "сообщила" if reporter.entity.gender == "female" else "сообщил"
        give_verb = "передала" if female else "передал"
        reported = Event(
            "give", actor=actor, object=obj, recipient=recipient, modality="reported"
        )
        add(
            f"{reporter.nominative} {report_verb}, что {a} {give_verb} {o} {d}.",
            "report_chto",
            event=Event("report", actor=reporter.entity.name, content=reported),
        )
        nested = Event("promise", actor=actor, modality="reported", content=promised)
        add(
            f"{reporter.nominative} {report_verb}, что {a} "
            f"{promise_verb} передать {o} {d}.",
            "report_promise",
            event=Event("report", actor=reporter.entity.name, content=nested),
        )
        # Explicitly labelled abstentions teach the bounded ontology's missing
        # roles. No incomplete surface pattern is consulted during inference.
        move_verb = "положила" if female else "положил"
        for incomplete in (
            f"{a} {give_verb} {o}.",
            f"{a} {give_verb} {d}.",
            f"{other.nominative} {received} {o}.",
            f"{a} {promise_verb} передать {o}.",
            f"{a} {move_verb} {o}.",
            f"{a} {move_verb} в {place.accusative}.",
            f"{thing.nominative} находится.",
        ):
            add(incomplete, "incomplete_roles", act="unknown")

    for text in (
        "отмени последнее утверждение",
        "отмени последнее сообщение",
        "последнее сообщение было ошибкой",
        "отмена последнего утверждения",
    ):
        add(text, "retraction_explicit", act="retract")
    for act, texts in (
        ("greet", ("привет", "здравствуйте", "добрый день", "здравствуй")),
        ("thanks", ("спасибо", "благодарю", "большое спасибо")),
        ("help", ("помощь", "что ты умеешь", "как пользоваться")),
    ):
        for text in texts:
            add(text, "social_explicit", act=act)
    # Explicitly supervised contextual surface references. Gold links are used
    # only during fitting, never supplied to interpret().
    for person_name, thing_name, place_name in (
        ("миша", "ключ", "сумка"),
        ("петя", "паспорт", "коробка"),
        ("маша", "книга", "ящик"),
        ("катя", "игрушка", "шкаф"),
    ):
        p, t, location = (
            ENTITY_BY_NAME[name] for name in (person_name, thing_name, place_name)
        )
        context = DialogueContext(
            turns=(f"{p.nominative} положил {t.accusative} в {location.accusative}.",),
            entities=(p.entity, t.entity, location.entity),
            focus=(thing_name, person_name, place_name),
        )
        pronoun = "она" if p.entity.gender == "female" else "он"
        verb = "положила" if p.entity.gender == "female" else "положил"
        text = f"{pronoun} {verb} {t.accusative} в {location.accusative}."
        add(
            text,
            "context_pronoun_svo",
            context=context,
            links=((0, person_name),),
            event=Event("move", actor=person_name, object=thing_name, place=place_name),
        )
        object_pronoun = "ее" if t.entity.gender == "female" else "его"
        text = f"{p.nominative} {verb} {object_pronoun} в {location.accusative}."
        add(
            text,
            "context_pronoun_svo",
            context=context,
            links=((2, thing_name),),
            event=Event("move", actor=person_name, object=thing_name, place=place_name),
        )
        text = f"где {'она' if t.entity.gender == 'female' else 'он'}?"
        add(
            text,
            "context_pronoun_svo",
            context=context,
            links=((1, thing_name),),
            act="ask",
            query=Query("where", subject=thing_name),
        )
        add(
            "почему?",
            "provenance_explicit",
            context=context,
            act="ask",
            query=Query("why", subject=thing_name),
        )
        add(
            "почему ты так считаешь?",
            "provenance_explicit",
            context=context,
            act="ask",
            query=Query("why", subject=thing_name),
        )
        add(
            "откуда ты знаешь?",
            "provenance_explicit",
            context=context,
            act="ask",
            query=Query("why", subject=thing_name),
        )
    # Terminal full stops are presentation, not a new semantic phrasal family.
    # Keep query '?' when it is the only evidence distinguishing verification.
    result.extend(
        TrainingUtterance(ex.text[:-1], ex.meaning, ex.context, ex.family, ex.links)
        for index, ex in enumerate(tuple(result))
        if ex.text.endswith(".") and index % 2 == 0
    )
    random.Random(seed).shuffle(result)
    return result


def development_examples() -> list[TrainingUtterance]:
    """Different entity combinations of disclosed training phrasal families."""
    items = [
        (
            "Маша переложила паспорт в шкаф.",
            Event("move", actor="маша", object="паспорт", place="шкаф"),
        ),
        (
            "Петя передал книгу Анне.",
            Event("give", actor="петя", object="книга", recipient="анна"),
        ),
        (
            "Книга не на столе.",
            Event(
                "locate",
                object="книга",
                place="стол",
                spatial="on",
                negated=True,
                time="present",
            ),
        ),
        (
            "У Оли есть телефон.",
            Event("have", actor="оля", object="телефон", time="present"),
        ),
        (
            "Маша получила книгу от Пети",
            Event("give", actor="петя", object="книга", recipient="маша"),
        ),
        (
            "Петя передал книгу Маше",
            Event("give", actor="петя", object="книга", recipient="маша"),
        ),
        (
            "Оля получила паспорт от Ивана.",
            Event("give", actor="иван", object="паспорт", recipient="оля"),
        ),
        (
            "Иван передал паспорт Оле.",
            Event("give", actor="иван", object="паспорт", recipient="оля"),
        ),
        (
            "Катя передала игрушку Мише.",
            Event("give", actor="катя", object="игрушка", recipient="миша"),
        ),
        (
            "Миша получил игрушку от Кати.",
            Event("give", actor="катя", object="игрушка", recipient="миша"),
        ),
        (
            "Иван не передал книгу Оле.",
            Event("give", actor="иван", object="книга", recipient="оля", negated=True),
        ),
        (
            "Иван обещал передать книгу Оле.",
            Event(
                "promise",
                actor="иван",
                content=Event(
                    "give",
                    actor="иван",
                    object="книга",
                    recipient="оля",
                    modality="intended",
                    time="future",
                ),
            ),
        ),
    ]
    return [
        TrainingUtterance(
            text, _meaning("inform", event=event), family="dev_combinations"
        )
        for text, event in items
    ]


def reference_examples() -> list[ReferenceExample]:
    """Explicit positive, missing and ambiguous referent teaching episodes."""
    samples: list[ReferenceExample] = []
    by_name = {row.entity.name: row.entity for row in ENTITY_ROWS}
    settings = (
        ("миша", "ключ", "сумка"),
        ("петя", "паспорт", "коробка"),
        ("маша", "книга", "ящик"),
        ("катя", "игрушка", "шкаф"),
        ("петя", "книга", "стол"),
        ("маша", "ключ", "сумка"),
        ("миша", "игрушка", "коробка"),
        ("катя", "паспорт", "шкаф"),
    )
    for person, thing, place in settings:
        entities = tuple(by_name[name] for name in (person, thing, place))
        for focus in (
            (thing, person, place),
            (person, place, thing),
            (place, thing, person),
        ):
            context = DialogueContext(entities=entities, focus=focus)
            for role, name, masculine, feminine in (
                ("actor", person, "он", "она"),
                ("recipient", person, "ему", "ей"),
                ("object", thing, "его", "ее"),
                ("query_subject", thing, "он", "она"),
                ("place", place, "нем", "ней"),
            ):
                pronoun = feminine if by_name[name].gender == "female" else masculine
                samples.append(ReferenceExample(pronoun, role, context, name))
                samples.append(ReferenceExample(pronoun, role, DialogueContext(), ""))
                # Contrastive annotations include an otherwise plausible entity
                # with the wrong grammatical gender. Query subjects can also
                # name the person when that is the only matching referent.
                opposite = masculine if pronoun == feminine else feminine
                opposite_target = ""
                if role == "query_subject":
                    opposite_gender = PRONOUN_FORMS[opposite][0]
                    if by_name[person].gender == opposite_gender:
                        opposite_target = person
                samples.append(
                    ReferenceExample(opposite, role, context, opposite_target)
                )
    for first, second, pronoun, role in (
        ("миша", "петя", "он", "actor"),
        ("маша", "катя", "она", "actor"),
        ("ключ", "паспорт", "его", "object"),
        ("книга", "игрушка", "ее", "object"),
        ("ключ", "паспорт", "он", "query_subject"),
        ("сумка", "коробка", "ней", "place"),
    ):
        for focus in ((first, second), (second, first)):
            context = DialogueContext(
                entities=(by_name[first], by_name[second]), focus=focus
            )
            samples.append(ReferenceExample(pronoun, role, context, ""))
    for person, place, pronoun in (("миша", "сумка", "он"), ("маша", "ящик", "она")):
        for focus in ((person, place), (place, person)):
            context = DialogueContext(
                entities=(by_name[person], by_name[place]), focus=focus
            )
            samples.append(ReferenceExample(pronoun, "query_subject", context, person))
            opposite = "она" if pronoun == "он" else "он"
            samples.append(ReferenceExample(opposite, "query_subject", context, ""))
    # A provenance follow-up with no explicit noun links to the focused entity.
    for first, second in (
        ("ключ", "миша"),
        ("миша", "ключ"),
        ("книга", "маша"),
        ("маша", "книга"),
        ("паспорт", "петя"),
        ("петя", "паспорт"),
        ("сумка", "ключ"),
        ("ключ", "сумка"),
    ):
        context = DialogueContext(
            entities=(by_name[first], by_name[second]), focus=(first, second)
        )
        samples.append(ReferenceExample("<implicit>", "query_subject", context, first))
        samples.append(
            ReferenceExample(
                "<implicit>",
                "query_subject",
                DialogueContext(entities=context.entities),
                "",
            )
        )
    samples.append(
        ReferenceExample("<implicit>", "query_subject", DialogueContext(), "")
    )
    return samples
