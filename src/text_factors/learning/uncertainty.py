"""Local freshness guards, without inferring facts from unparsed language.

The deliberately conservative lexical gate is fixed scaffolding, not a trained
Russian parser. It accepts an uninterpreted declarative-looking utterance only
when it explicitly names a previously known subject and contains more than its
name. Question, request, social and overt nonactual cues veto the gate. Exact
names and the already disclosed finite entity aliases are its only bindings;
pronouns, arbitrary inflection, aliases of new names and hidden instances are
not resolved here. Consequently an unfamiliar description can over-trigger,
and an unfamiliar question or nonactual expression can evade the lexical veto.

A marker says that old evidence *may* be stale. It supplies no replacement
location, holder, event or truth value. There is no expiry or silent eviction.
Only a committed, understood positive actual assertion can restore the whole
subject's freshness. A negative constraint need not refresh surviving positive
facts. Original observations and resolution provenance stay in a bounded journal.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..conversation.persistence import encode_json
from ..conversation.schema import BudgetExceeded
from .hypotheses import Observation, digest
from .language_data import ENTITY_BY_NAME, tokenize
from .schema import (
    DialogueContext,
    Entity,
    Interpretation,
    Meaning,
    Query,
    bounded_text,
    exact_fields,
)
from .world import validate_assertion, validate_effects

SCHEMA = "ai2-local-uncertainty-v1"
REASON = "local_actuality_uncertain"
_MAX_TOKENS = 96
_MAX_FACTS = 4096
_MAX_BYTES = 4_000_000
_EMPTY_CONTEXT = DialogueContext()
# These are vetoes on a safety guard, never rules that produce semantic roles
# or assert an event. Their finite coverage is an explicit limitation above.
_QUESTION = frozenset(
    {
        "где",
        "куда",
        "откуда",
        "кто",
        "кого",
        "кому",
        "чей",
        "чья",
        "чье",
        "чьи",
        "что",
        "чем",
        "как",
        "почему",
        "зачем",
        "когда",
        "ли",
        "разве",
        "неужели",
        "which",
        "where",
        "who",
        "whose",
        "what",
        "why",
        "how",
        "when",
        "whether",
    }
)
_REQUEST = frozenset(
    {
        "пожалуйста",
        "скажи",
        "скажите",
        "расскажи",
        "расскажите",
        "покажи",
        "покажите",
        "объясни",
        "объясните",
        "уточни",
        "уточните",
        "помоги",
        "помогите",
        "please",
        "tell",
        "show",
        "explain",
        "help",
    }
)
_SOCIAL = frozenset(
    {
        "привет",
        "здравствуй",
        "здравствуйте",
        "спасибо",
        "благодарю",
        "пока",
        "hello",
        "hi",
        "thanks",
        "thank",
        "goodbye",
    }
)
_NONACTUAL = frozenset(
    {
        "если",
        "бы",
        "возможно",
        "может",
        "мог",
        "могла",
        "могли",
        "могло",
        "вдруг",
        "наверное",
        "завтра",
        "будет",
        "будут",
        "буду",
        "собираюсь",
        "обещал",
        "обещала",
        "обещали",
        "обещаю",
        "сказал",
        "сказала",
        "сказали",
        "говорит",
        "говорят",
        "якобы",
        "if",
        "would",
        "could",
        "might",
        "may",
        "will",
        "tomorrow",
        "promised",
        "said",
    }
)
_NON_CONTENT = frozenset(
    {
        "а",
        "и",
        "или",
        "но",
        "да",
        "ну",
        "вот",
        "это",
        "эта",
        "этот",
        "эти",
        "тот",
        "та",
        "те",
        "то",
        "о",
        "об",
        "про",
        "с",
        "со",
        "к",
        "ко",
        "у",
        "в",
        "во",
        "на",
        "из",
        "от",
        "до",
        "для",
        "по",
        "не",
        "нет",
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "with",
    }
)
_RECORD_FIELDS = {
    "record_id",
    "observation",
    "meaning",
    "accepted",
    "effects",
    "known_facts",
    "entities",
    "marked",
    "cleared",
}


def _turn(value: Any, *, zero: bool = False) -> int:
    if type(value) is not int or not (0 if zero else 1) <= value < 2**53:
        raise ValueError("invalid uncertainty turn")
    return value


def _facts(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) not in (list, tuple) or len(value) > _MAX_FACTS:
        raise ValueError("invalid uncertainty fact references")
    return tuple(validate_assertion(fact) for fact in value)


def _forms(name: str) -> tuple[tuple[str, ...], ...]:
    forms = {tokenize(name)}
    row = ENTITY_BY_NAME.get(name)
    if row is not None:
        forms.update(
            tokenize(getattr(row, case))
            for case in ("nominative", "accusative", "genitive", "dative", "locative")
        )
    return tuple(sorted(form for form in forms if form))


def _positions(tokens: tuple[str, ...], name: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                position
                for form in _forms(name)
                for start in range(len(tokens) - len(form) + 1)
                if tokens[start : start + len(form)] == form
                for position in range(start, start + len(form))
            }
        )
    )


def _eligible(tokens: tuple[str, ...], mentioned: set[int]) -> bool:
    if not mentioned or "?" in tokens:
        return False
    words = {token for token in tokens if any(char.isalnum() for char in token)}
    if words & (_QUESTION | _REQUEST | _SOCIAL | _NONACTUAL):
        return False
    # A salutation without a verb is common even when understanding abstains.
    if words & {"доброе", "добрый", "доброго"} and words & {
        "утро",
        "день",
        "вечер",
        "дня",
        "вечера",
    }:
        return False
    return any(
        i not in mentioned
        and token not in _NON_CONTENT
        and any(char.isalpha() for char in token)
        for i, token in enumerate(tokens)
    )


def _support(fact: dict[str, Any], meaning: Meaning) -> bool:
    event = meaning.event
    assert event is not None
    relation = "location" if event.predicate in {"locate", "move"} else "holder"
    value = (
        event.place
        if relation == "location"
        else event.recipient
        if event.predicate == "give"
        else event.actor
    )
    return (
        fact["subject"] == event.object
        and fact["relation"] == relation
        and fact["value"] == value
        and fact["negated"] == event.negated
        and fact["spatial"] == (event.spatial if relation == "location" else "in")
    )


class UncertaintyState:
    """Bounded transactional journal of unresolved subject-local guards.

    Call ``observe`` on a cloned state. ``accepted=True`` is for a mutation that
    survived policy and world validation; pass the world's facts *after* it.
    An accepted matching no-op is explicit new confirmation and can resolve a
    guard as well. ``query`` returns an outcome override, or ``None``.

    ``from_dict`` replays all guard decisions and references within the journal.
    External observation and world ledgers can additionally be supplied to bind
    provenance; a self-contained JSON archive is not an authenticated history.
    """

    def __init__(self, *, max_markers: int = 256, max_records: int = 512) -> None:
        for name, value, cap in (
            ("marker", max_markers, 1024),
            ("record", max_records, 4096),
        ):
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"invalid uncertainty {name} capacity")
        self.max_markers = max_markers
        self.max_records = max_records
        self._markers: dict[str, dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []

    @property
    def markers(self) -> list[dict[str, Any]]:
        return deepcopy([self._markers[name] for name in sorted(self._markers)])

    @property
    def records(self) -> list[dict[str, Any]]:
        return deepcopy(self._records)

    def clone(self) -> UncertaintyState:
        result = UncertaintyState(
            max_markers=self.max_markers, max_records=self.max_records
        )
        result._markers = deepcopy(self._markers)
        result._records = deepcopy(self._records)
        return result

    def observe(
        self,
        text: str,
        interpretation: Interpretation | Meaning | None,
        *,
        turn_id: int,
        known_facts: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        context: DialogueContext = _EMPTY_CONTEXT,
        effects: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        accepted: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Record a possible update or resolve exactly one confirmed subject.

        The returned markers are those introduced by this observation, not all
        unresolved markers. No changes, including a resolution, are committed
        when any journal, marker, token or encoded-size budget fails.
        """
        _turn(turn_id)
        bounded_text(text, "uncertainty observation", cap=2048)
        if not isinstance(context, DialogueContext):
            raise ValueError("expected uncertainty context")
        if type(accepted) is not bool:
            raise ValueError("invalid accepted uncertainty update")
        if isinstance(interpretation, Interpretation):
            meaning = interpretation.meaning
        elif interpretation is None or isinstance(interpretation, Meaning):
            meaning = interpretation
        else:
            raise ValueError("expected uncertainty interpretation")
        if type(effects) not in (tuple, list) or len(effects) > 1:
            raise ValueError("invalid uncertainty effects")
        if effects and not accepted:
            raise ValueError("uncommitted uncertainty effects")
        if accepted and (meaning is None or meaning.event is None):
            raise ValueError("accepted uncertainty update requires an event")
        facts = _facts(known_facts)
        observation = Observation(
            f"turn:{turn_id}", text, turn_id, f"сообщение {turn_id}"
        )
        marked: list[dict[str, Any]] = []
        cleared: list[str] = []
        anchors: list[dict[str, Any]] = []
        entities: list[Entity] = []
        if accepted:
            assert meaning is not None and meaning.event is not None
            event = meaning.event
            validate_effects(event, effects, before=facts)
            if event.actual and not event.negated and event.object in self._markers:
                matching = [fact for fact in facts if _support(fact, meaning)]
                if not matching:
                    raise ValueError("accepted uncertainty update lacks fact support")
                anchors = matching[:1]
                cleared = [self._markers[event.object]["marker_id"]]
        elif meaning is None or meaning.act == "unknown":
            tokens = tokenize(text)
            if len(tokens) > _MAX_TOKENS:
                raise BudgetExceeded("uncertainty_token_capacity")
            fact_by_subject = {fact["subject"]: fact for fact in reversed(facts)}
            entity_by_name = {
                entity.name: entity
                for entity in context.entities
                if entity.kind == "thing"
            }
            names = sorted(set(fact_by_subject) | set(entity_by_name))
            if len(names) > 1024:
                raise BudgetExceeded("uncertainty_subject_capacity")
            mentioned = {name: _positions(tokens, name) for name in names}
            mentioned = {
                name: positions for name, positions in mentioned.items() if positions
            }
            all_positions = {
                index for positions in mentioned.values() for index in positions
            }
            if not _eligible(tokens, all_positions):
                return ()
            for name, positions in mentioned.items():
                anchor = fact_by_subject.get(name)
                if anchor is not None:
                    anchors.append(anchor)
                elif name in entity_by_name:
                    entities.append(entity_by_name[name])
                basis = (
                    "known_fact_subject"
                    if anchor is not None
                    else "known_context_thing"
                )
                marker = {
                    "kind": "uncertain_actuality",
                    "subject": name,
                    "reason": REASON,
                    "observation": observation.to_dict(),
                    "source_positions": list(positions),
                    "basis": basis,
                    "known_event_ids": [anchor["event_id"]] if anchor else [],
                }
                marked.append({"marker_id": "u-" + digest(marker), **marker})
        if not marked and not cleared:
            return ()
        if self._records and turn_id <= self._records[-1]["observation"]["turn_id"]:
            raise ValueError("uncertainty record turns must increase")
        markers = deepcopy(self._markers)
        for marker in marked:
            markers[marker["subject"]] = marker
        if cleared:
            assert meaning is not None and meaning.event is not None
            del markers[meaning.event.object]
        if len(markers) > self.max_markers:
            raise BudgetExceeded("uncertainty_marker_capacity")
        if len(self._records) >= self.max_records:
            raise BudgetExceeded("uncertainty_record_capacity")
        record = {
            "observation": observation.to_dict(),
            "meaning": meaning.to_dict() if meaning else None,
            "accepted": accepted,
            "effects": deepcopy(list(effects)),
            "known_facts": deepcopy(anchors),
            "entities": [entity.to_dict() for entity in entities],
            "marked": marked,
            "cleared": cleared,
        }
        records = [*self._records, {"record_id": "ur-" + digest(record), **record}]
        candidate = self._snapshot(markers, records)
        try:
            encode_json(candidate, max_bytes=_MAX_BYTES)
        except ValueError as exc:
            if "budget" in str(exc):
                raise BudgetExceeded("uncertainty_state_capacity") from exc
            raise
        self._markers, self._records = markers, records
        return tuple(deepcopy(marked))

    def query(
        self,
        query: Query,
        *,
        known_facts: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        assertions: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    ) -> dict[str, Any] | None:
        """Return a clarification only for facts linked to a guarded subject."""
        if not isinstance(query, Query):
            raise ValueError("expected uncertainty query")
        facts, referenced = _facts(known_facts), _facts(assertions)
        if query.time not in {"present", "unspecified"}:
            return None
        if query.kind == "what_has":
            subjects = {
                fact["subject"]
                for fact in (*facts, *referenced)
                if fact["relation"] == "holder"
                and fact["value"] == query.subject
                and not fact["negated"]
            }
        elif query.kind == "why":
            subjects = {fact["subject"] for fact in referenced}
            if query.subject:
                subjects &= {query.subject}
        else:
            subjects = {query.subject}
        found = [
            self._markers[name] for name in sorted(subjects) if name in self._markers
        ]
        if not found:
            return None
        return {
            "action": "clarify",
            "reason": REASON,
            "assertions": [],
            "evidence": deepcopy(found),
        }

    def _snapshot(
        self, markers: dict[str, dict[str, Any]], records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "max_markers": self.max_markers,
            "max_records": self.max_records,
            "records": deepcopy(records),
            "markers": deepcopy([markers[name] for name in sorted(markers)]),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._snapshot(self._markers, self._records)
        encode_json(result, max_bytes=_MAX_BYTES)
        return result

    @staticmethod
    def _replay_record(state: UncertaintyState, raw: Any) -> None:
        record = exact_fields(raw, _RECORD_FIELDS, "uncertainty record")
        observation = Observation.from_dict(record["observation"])
        if type(record["entities"]) is not list or len(record["entities"]) > 64:
            raise ValueError("invalid uncertainty entity references")
        meaning = (
            Meaning.from_dict(record["meaning"])
            if record["meaning"] is not None
            else None
        )
        before = len(state._records)
        state.observe(
            observation.text,
            meaning,
            turn_id=observation.turn_id,
            known_facts=record["known_facts"],
            context=DialogueContext(
                entities=tuple(Entity.from_dict(e) for e in record["entities"])
            ),
            effects=record["effects"],
            accepted=record["accepted"],
        )
        if len(state._records) != before + 1 or encode_json(
            state._records[-1]
        ) != encode_json(record):
            raise ValueError("uncertainty record does not match deterministic replay")

    def at_turn(self, turn_id: int) -> UncertaintyState:
        """Reconstruct guards at the end of a historical turn, including clears."""
        _turn(turn_id, zero=True)
        state = UncertaintyState(
            max_markers=self.max_markers, max_records=self.max_records
        )
        # Internal records have already passed observe/from_dict. Reconstruct
        # heads in one pass; receipt replay must not repeatedly reparse history.
        for record in self._records:
            if record["observation"]["turn_id"] > turn_id:
                break
            state._records.append(deepcopy(record))
            for marker in record["marked"]:
                state._markers[marker["subject"]] = deepcopy(marker)
            cleared = set(record["cleared"])
            state._markers = {
                subject: marker
                for subject, marker in state._markers.items()
                if marker["marker_id"] not in cleared
            }
        return state

    def validate_references(
        self,
        *,
        observations: dict[str, Observation],
        events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        contexts: dict[int, DialogueContext] | None = None,
    ) -> None:
        """Cross-check provenance against independently loaded original ledgers.

        Supply original event records when a separate revision layer projects
        newer event versions. The journal can refer to formerly current facts.
        The enclosing session remains responsible for authenticating its ledgers.
        """
        if type(observations) is not dict or any(
            type(key) is not str
            or not isinstance(value, Observation)
            or value.observation_id != key
            for key, value in observations.items()
        ):
            raise ValueError("invalid uncertainty observation ledger")
        if type(events) not in (list, tuple) or len(events) > _MAX_FACTS:
            raise ValueError("invalid uncertainty event ledger")
        if contexts is not None and (
            type(contexts) is not dict
            or any(
                type(turn) is not int or not isinstance(context, DialogueContext)
                for turn, context in contexts.items()
            )
        ):
            raise ValueError("invalid uncertainty context ledger")
        by_id: dict[int, dict[str, Any]] = {}
        for event in events:
            if type(event) is not dict or type(event.get("id")) is not int:
                raise ValueError("invalid uncertainty event reference")
            if event["id"] in by_id:
                raise ValueError("duplicate uncertainty event reference")
            by_id[event["id"]] = event
        for record in self._records:
            observation = Observation.from_dict(record["observation"])
            if observations.get(observation.observation_id) != observation:
                raise ValueError("uncertainty observation is not original evidence")
            if contexts is not None and record["entities"]:
                context = contexts.get(observation.turn_id)
                if context is None or any(
                    Entity.from_dict(entity) not in context.entities
                    for entity in record["entities"]
                ):
                    raise ValueError(
                        "uncertainty entity is absent from original context"
                    )
            for fact in record["known_facts"]:
                event = by_id.get(fact["event_id"])
                if event is None or (
                    type(event.get("turn_id")) is not int
                    or event["turn_id"]
                    > observation.turn_id - int(not record["accepted"])
                    or event.get("source") != fact["source"]
                ):
                    raise ValueError("uncertainty fact has a forged event reference")
                effect = {
                    "op": "exclude" if fact["negated"] else "set",
                    "subject": fact["subject"],
                    "relation": fact["relation"],
                    "value": fact["value"],
                    "spatial": fact["spatial"],
                }
                if effect not in event.get("effects", ()):
                    raise ValueError(
                        "uncertainty fact is absent from its referenced event"
                    )

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        observations: dict[str, Observation] | None = None,
        events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        contexts: dict[int, DialogueContext] | None = None,
    ) -> UncertaintyState:
        value = exact_fields(
            value,
            {"schema", "max_markers", "max_records", "records", "markers"},
            "uncertainty state",
        )
        encode_json(value, max_bytes=_MAX_BYTES)
        if value["schema"] != SCHEMA:
            raise ValueError("invalid uncertainty schema")
        state = cls(max_markers=value["max_markers"], max_records=value["max_records"])
        if (
            type(value["records"]) is not list
            or len(value["records"]) > state.max_records
        ):
            raise ValueError("uncertainty record capacity exceeded")
        if (
            type(value["markers"]) is not list
            or len(value["markers"]) > state.max_markers
        ):
            raise ValueError("uncertainty marker capacity exceeded")
        try:
            for record in value["records"]:
                cls._replay_record(state, record)
        except BudgetExceeded as exc:
            raise ValueError("uncertainty archive exceeds capacity") from exc
        if encode_json(state.to_dict()) != encode_json(value):
            raise ValueError("uncertainty state does not match deterministic replay")
        if (observations is None) != (events is None):
            raise ValueError("uncertainty provenance requires both reference ledgers")
        if contexts is not None and observations is None:
            raise ValueError("uncertainty contexts require reference ledgers")
        if observations is not None and events is not None:
            state.validate_references(
                observations=observations, events=events, contexts=contexts
            )
        return state
