"""Conservative, bounded Russian *role grammar*, not a learned language model.

This explicit scaffold identifies roles and question/interpersonal acts. It does
not interpret lexical cues as canonical predicates: every returned frame has an
empty ``predicate``. ``DEFAULT_TEACHING_PAIRS`` is an explicit supervised corpus
for the separate learned bridge, never a runtime predicate-lookup fallback.

Entity morphology is a small declared case table. Unknown single-token entity
names are preserved literally; no stemming or open-ended Russian understanding
is claimed. Unconsumed, quoted, reported or ambiguous syntax rejects the whole
turn, including any otherwise valid prefix. These restrictions are deliberate.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .schema import ConversationLimits, ParseResult, SemanticFrame

# The labels below are teacher data, not parser rules or inference-time targets.
DEFAULT_TEACHING_PAIRS: list[tuple[str, str]] = [
    ("местонахождение", "locate"),
    ("лежит", "locate"),
    ("лежал", "locate"),
    ("лежала", "locate"),
    ("находится", "locate"),
    ("находился", "locate"),
    ("находилась", "locate"),
    ("остался", "locate"),
    ("осталась", "locate"),
    ("осталось", "locate"),
    ("стоит", "locate"),
    ("положил", "move"),
    ("положила", "move"),
    ("переложил", "move"),
    ("переложила", "move"),
    ("поместил", "move"),
    ("поместила", "move"),
    ("поставил", "move"),
    ("поставила", "move"),
    ("передал", "give"),
    ("передала", "give"),
    ("отдал", "give"),
    ("отдала", "give"),
    ("обладание", "have"),
    ("есть", "have"),
    ("нет", "have"),
    ("был", "have"),
    ("была", "have"),
    ("было", "have"),
    ("были", "have"),
    ("имеет", "have"),
    ("имел", "have"),
    ("имела", "have"),
]

# nominative, accusative, genitive, locative, dative; no guessed suffix rules.
_NOUN_ROWS = (
    ("ключ", "ключ", "ключа", "ключе", "ключу"),
    ("книга", "книгу", "книги", "книге", "книге"),
    ("паспорт", "паспорт", "паспорта", "паспорте", "паспорту"),
    ("мяч", "мяч", "мяча", "мяче", "мячу"),
    ("куб", "куб", "куба", "кубе", "кубу"),
    ("шар", "шар", "шара", "шаре", "шару"),
    ("телефон", "телефон", "телефона", "телефоне", "телефону"),
    ("письмо", "письмо", "письма", "письме", "письму"),
    ("яблоко", "яблоко", "яблока", "яблоке", "яблоку"),
    ("карандаш", "карандаш", "карандаша", "карандаше", "карандашу"),
    ("игрушка", "игрушку", "игрушки", "игрушке", "игрушке"),
    ("ящик", "ящик", "ящика", "ящике", "ящику"),
    ("сумка", "сумку", "сумки", "сумке", "сумке"),
    ("коробка", "коробку", "коробки", "коробке", "коробке"),
    ("стол", "стол", "стола", "столе", "столу"),
    ("шкаф", "шкаф", "шкафа", "шкафу", "шкафу"),
    ("комната", "комнату", "комнаты", "комнате", "комнате"),
    ("кухня", "кухню", "кухни", "кухне", "кухне"),
    ("рюкзак", "рюкзак", "рюкзака", "рюкзаке", "рюкзаку"),
    ("полка", "полку", "полки", "полке", "полке"),
    ("карман", "карман", "кармана", "кармане", "карману"),
    ("дом", "дом", "дома", "доме", "дому"),
    ("документ", "документ", "документа", "документе", "документу"),
    ("папка", "папку", "папки", "папке", "папке"),
)
_PERSON_ROWS = (
    ("миша", "мишу", "миши", "мише", "мише"),
    ("маша", "машу", "маши", "маше", "маше"),
    ("петя", "петю", "пети", "пете", "пете"),
    ("катя", "катю", "кати", "кате", "кате"),
    ("анна", "анну", "анны", "анне", "анне"),
    ("иван", "ивана", "ивана", "иване", "ивану"),
    ("оля", "олю", "оли", "оле", "оле"),
    ("саша", "сашу", "саши", "саше", "саше"),
    ("даша", "дашу", "даши", "даше", "даше"),
    ("борис", "бориса", "бориса", "борисе", "борису"),
)
_CASES: dict[str, list[tuple[str, int]]] = {}
_SURFACE_ROWS = {row[0]: row for row in (*_NOUN_ROWS, *_PERSON_ROWS)}
_PERSON_NAMES = frozenset(row[0] for row in _PERSON_ROWS)
_THING_NAMES = frozenset(row[0] for row in _NOUN_ROWS)
for _row in (*_NOUN_ROWS, *_PERSON_ROWS):
    for _case, _surface in enumerate(_row):
        _CASES.setdefault(_surface, []).append((_row[0], _case))

_WORD = re.compile(r"[a-zа-я0-9]+(?:[-_][a-zа-я0-9]+)*\Z")
_TOKEN = re.compile(r"[a-zа-я0-9]+(?:[-_][a-zа-я0-9]+)*|[^\s]")
_REPORTING = frozenset(
    [
        "сказал",
        "сказала",
        "сказали",
        "говорит",
        "говорила",
        "говорил",
        "сообщили",
        "сообщил",
        "сообщила",
        "утверждает",
        "утверждал",
        "утверждала",
        "думает",
        "думаю",
        "считаю",
        "кажется",
        "слышал",
        "слышала",
        "написал",
        "написала",
        "написали",
        "якобы",
        "по-моему",
        "думал",
        "думала",
        "подумал",
        "подумала",
        "полагал",
        "полагала",
        "предполагает",
        "предполагал",
        "предполагала",
        "предположил",
        "предположила",
        "предположили",
        "представил",
        "представила",
        "вообразил",
        "вообразила",
        "выдумал",
        "выдумала",
        "придумал",
        "придумала",
        "мечтал",
        "мечтала",
        "ожидал",
        "ожидала",
        "надеялся",
        "надеялась",
        "обещал",
        "обещала",
        "планировал",
        "планировала",
        "запланировал",
        "запланировала",
        "хотел",
        "хотела",
        "желал",
        "желала",
        "намеревался",
        "намеревалась",
        "попросил",
        "попросила",
        "приказал",
        "приказала",
        "велел",
        "велела",
        "отрицал",
        "отрицала",
        "сомневался",
        "сомневалась",
        "допустил",
        "допустила",
        "назвал",
        "назвала",
        "приснилось",
        "привиделось",
    ]
)
_RESERVED = frozenset(
    [
        "не",
        "ни",
        "нет",
        "да",
        "в",
        "на",
        "у",
        "к",
        "от",
        "из",
        "с",
        "со",
        "для",
        "до",
        "за",
        "под",
        "над",
        "и",
        "или",
        "но",
        "а",
        "что",
        "чтобы",
        "кто",
        "кого",
        "где",
        "куда",
        "откуда",
        "когда",
        "почему",
        "как",
        "если",
        "то",
        "ли",
        "бы",
        "пусть",
        "хотя",
        "пока",
        "после",
        "перед",
        "вместо",
        "только",
        "лишь",
        "никогда",
        "никто",
        "ничто",
        "ничей",
        "ничего",
        "нигде",
        "каждый",
        "любой",
        "все",
        "всем",
        "всегда",
        "иногда",
        "возможно",
        "наверное",
        "может",
        "нужно",
        "можно",
        "должен",
        "должна",
        "обязана",
        "обязан",
        "хочу",
        "хочет",
        "будет",
        "буду",
        "будешь",
        "будем",
        "будете",
        "будут",
        "был",
        "была",
        "было",
        "были",
        "есть",
        "имеет",
        "имел",
        "имела",
        "это",
        "этот",
        "эта",
        "эти",
        "того",
        "той",
        "тех",
        "один",
        "два",
        "три",
        "сегодня",
        "завтра",
        "вчера",
        "сначала",
        "потом",
        "затем",
        "сейчас",
    ]
)
_PRONOUNS = frozenset(
    [
        "я",
        "меня",
        "мне",
        "мной",
        "мы",
        "нас",
        "нам",
        "ты",
        "тебя",
        "тебе",
        "вы",
        "вас",
        "вам",
        "он",
        "она",
        "оно",
        "они",
        "его",
        "ее",
        "ему",
        "ей",
        "им",
        "их",
        "ними",
        "нем",
        "ней",
        "там",
        "туда",
        "тут",
        "сюда",
        "это",
        "этот",
        "эта",
        "тот",
        "та",
        "то",
        "себя",
        "себе",
        "свой",
        "свою",
        "мой",
        "моя",
        "твой",
        "твоя",
        "наш",
        "наша",
    ]
)
_FUTURE_AUX = frozenset(["буду", "будешь", "будет", "будем", "будете", "будут"])
_FUTURE_CUES = frozenset(
    [
        "положу",
        "положишь",
        "положит",
        "положим",
        "положите",
        "положат",
        "переложу",
        "переложишь",
        "переложит",
        "переложим",
        "переложите",
        "переложат",
        "помещу",
        "поместишь",
        "поместит",
        "передам",
        "передашь",
        "передаст",
        "передадим",
        "передадите",
        "передадут",
        "отдам",
        "отдашь",
        "отдаст",
        "отдадим",
        "отдадите",
        "отдадут",
        "окажется",
        "останется",
        "будет",
    ]
)
_CURRENT_CUES = frozenset({"лежит", "находится", "стоит", "имеет", "есть", "нет"})
_IRREGULAR_PAST = frozenset({"занес", "принес", "унес", "перенес", "внес"})
_PAST_ENDINGS = ("л", "ла", "ло", "ли", "лся", "лась", "лось", "лись")
_RETRACTIONS = frozenset(
    {
        "последнее сообщение было ошибкой",
        "последнее утверждение было ошибкой",
        "отмени последнее утверждение",
        "отмени последнее сообщение",
        "отменить последнее утверждение",
    }
)
_INTERPERSONAL = {
    "привет": "greet",
    "здравствуйте": "greet",
    "здравствуй": "greet",
    "добрый день": "greet",
    "доброе утро": "greet",
    "добрый вечер": "greet",
    "спасибо": "thanks",
    "благодарю": "thanks",
    "большое спасибо": "thanks",
    "помощь": "help",
    "помоги": "help",
    "что ты умеешь": "help",
    "как пользоваться": "help",
}


class _SyntaxFailure(ValueError):
    pass


def surface_entity(canonical: str, case: str = "nominative") -> str:
    """Explicit finite surface scaffold; unknown identities are quoted intact."""
    if type(canonical) is not str:
        raise TypeError("canonical entity must be a string")
    cases = {
        "nominative": 0,
        "accusative": 1,
        "genitive": 2,
        "locative": 3,
        "dative": 4,
    }
    if case not in cases:
        raise ValueError("unsupported noun case")
    row = _SURFACE_ROWS.get(canonical)
    if row is None:
        return f"«{canonical}»"
    surface = row[cases[case]]
    return surface.capitalize() if canonical in _PERSON_NAMES else surface


def _entity(token: str, role: str, *, negative: bool = False) -> str:
    """Resolve only declared cases and pronouns; preserve unknown tokens."""
    if not _WORD.fullmatch(token) or len(token) > 128:
        raise _SyntaxFailure("invalid_entity")
    markers = {
        "actor": {"я": "@speaker", "он": "@person", "она": "@person"},
        "subject": {
            "я": "@speaker",
            "он": "@ambiguous",
            "она": "@ambiguous",
            "оно": "@object",
        },
        "object": {
            "его": "@object",
            "ее": "@object",
            "это": "@ambiguous",
            "меня": "@speaker",
        },
        "absence": {"его": "@object", "ее": "@object", "меня": "@speaker"},
        "possessor": {"меня": "@speaker", "него": "@person", "нее": "@person"},
        "recipient": {"мне": "@speaker", "ему": "@person", "ей": "@person"},
        "place": {"нем": "@place", "ней": "@place"},
        "destination": {"него": "@place", "нее": "@place"},
    }
    if token in markers[role]:
        return markers[role][token]
    if token in {"это", "тот", "та"} and role in {"subject", "object"}:
        return "@ambiguous"
    if token in _PRONOUNS or token in _RESERVED or token in _REPORTING:
        raise _SyntaxFailure("unsupported_entity_reference")
    permitted = {
        "actor": {0},
        "subject": {0},
        "object": {1, 2} if negative else {1},
        "absence": {2},
        "possessor": {2},
        "recipient": {4},
        "place": {3},
        "destination": {1, 3},
    }[role]
    known = _CASES.get(token)
    if known is not None:
        candidates = {base for base, case in known if case in permitted}
        if len(candidates) != 1:
            raise _SyntaxFailure("unsupported_entity_case")
        return candidates.pop()
    return token


def _cue(token: str) -> str:
    if (
        not _WORD.fullmatch(token)
        or len(token) > 128
        or token in _CASES
        or token in _PRONOUNS
        or token in _RESERVED
        or token in _REPORTING
    ):
        raise _SyntaxFailure("unsupported_predicate_cue")
    return token


def _split_clauses(text: str, maximum: int) -> list[tuple[str, bool]]:
    clauses: list[tuple[str, bool]] = []
    buffer: list[str] = []
    for character in text:
        if character in ".!?;\n":
            value = "".join(buffer).strip()
            buffer.clear()
            if value:
                clauses.append((value, character == "?"))
                if len(clauses) > maximum:
                    raise _SyntaxFailure("clause_limit")
            elif character == "?" and clauses:
                clauses[-1] = (clauses[-1][0], True)
        else:
            buffer.append(character)
    value = "".join(buffer).strip()
    if value:
        clauses.append((value, False))
    if len(clauses) > maximum:
        raise _SyntaxFailure("clause_limit")
    return clauses


class RussianParser:
    """Bounded complete-clause parser with all-or-nothing turn semantics."""

    def __init__(self, limits: ConversationLimits | None = None) -> None:
        if limits is not None and not isinstance(limits, ConversationLimits):
            raise TypeError("limits must be ConversationLimits")
        self.limits = limits or ConversationLimits()

    def parse(self, text: str) -> ParseResult:
        if type(text) is not str:
            return ParseResult(complete=False, reason="invalid_text")
        if len(text) > self.limits.max_chars:
            return ParseResult(complete=False, reason="character_limit")
        normalized = text.casefold().replace("ё", "е").strip()
        if not normalized:
            return ParseResult(complete=False, reason="empty_input")
        if any(ord(char) < 32 and char not in "\n\t\r" for char in normalized):
            return ParseResult(complete=False, reason="control_character")
        if any(char in normalized for char in "\"'«»“”‘’`"):
            return ParseResult(complete=False, reason="quoted_speech_unsupported")
        tokens = _TOKEN.findall(normalized)
        if len(tokens) > self.limits.max_tokens:
            return ParseResult(complete=False, reason="token_limit")
        if any(token in _REPORTING for token in tokens):
            return ParseResult(complete=False, reason="reported_speech_unsupported")
        try:
            clauses = _split_clauses(normalized, self.limits.max_clauses)
            if not clauses:
                raise _SyntaxFailure("empty_input")
            frames = tuple(self._clause(raw, question) for raw, question in clauses)
        except _SyntaxFailure as error:
            return ParseResult(complete=False, reason=str(error))
        # Role type is declared morphology scaffolding, not a learned predicate
        # or an inferred ontology for open names and unresolved pronouns.
        frames = tuple(
            replace(
                frame,
                object_kind="person"
                if frame.object in _PERSON_NAMES
                else "thing"
                if frame.object in _THING_NAMES
                else "unknown",
            )
            for frame in frames
        )
        return ParseResult(frames=frames)

    def _clause(self, raw: str, question: bool) -> SemanticFrame:
        # A frame's schema cap is stricter than a configurable whole-turn cap.
        if len(raw) > 2048:
            raise _SyntaxFailure("clause_character_limit")
        body = " ".join(raw.split())
        act = "inform"
        modality = "asserted"
        tense = "current"
        if body.startswith("нет,"):
            act = "correct"
            body = body[4:].strip()
        if body in _RETRACTIONS:
            if question:
                raise _SyntaxFailure("question_is_not_retraction")
            return SemanticFrame(act="retract", raw=raw)
        numbered = re.fullmatch(r"отмени (?:утверждение|сообщение) ([0-9]{1,6})", body)
        if numbered:
            reference = int(numbered.group(1))
            if question or reference == 0:
                raise _SyntaxFailure("invalid_retraction_reference")
            return SemanticFrame(act="retract", reference=reference, raw=raw)
        if body.startswith("тема:") and act == "inform":
            topic = body[5:].strip()
            if (
                question
                or not topic
                or len(topic) > 128
                or not all(_WORD.fullmatch(part) for part in topic.split())
            ):
                raise _SyntaxFailure("invalid_topic")
            return SemanticFrame(act="topic", topic=topic, raw=raw)
        if body in _INTERPERSONAL and act == "inform":
            return SemanticFrame(act=_INTERPERSONAL[body], raw=raw)
        if body.startswith("правда ли, что "):
            question = True
            body = body[len("правда ли, что ") :]
        for prefix in (
            "если ",
            "возможно, ",
            "возможно ",
            "наверное, ",
            "наверное ",
            "предположим, ",
            "предположим ",
        ):
            if body.startswith(prefix):
                if act == "correct":
                    raise _SyntaxFailure("hypothetical_correction")
                act, modality = "hypothesis", "hypothetical"
                body = body[len(prefix) :]
                break
        words = body.split()
        if not words or not all(_WORD.fullmatch(word) for word in words):
            raise _SyntaxFailure("unconsumed_punctuation")
        if words[0] in {"сначала", "потом", "затем", "сейчас", "вчера", "завтра"}:
            temporal = words.pop(0)
            if temporal == "вчера":
                tense = "past"
            elif temporal == "завтра":
                tense, modality, act = "future", "hypothetical", "hypothesis"
        if "бы" in words:
            if words.count("бы") != 1 or words.index("бы") > 2:
                raise _SyntaxFailure("unsupported_conditional")
            words.remove("бы")
            modality, act = "hypothetical", "hypothesis"
        if not words:
            raise _SyntaxFailure("missing_clause")
        if any(word in _FUTURE_AUX or word in _FUTURE_CUES for word in words):
            tense, modality, act = "future", "hypothetical", "hypothesis"
        query = self._question(words, raw)
        if query is not None:
            if modality != "asserted" or act == "correct":
                raise _SyntaxFailure("unsupported_question_modality")
            return query
        frame = self._statement(words, raw)
        past = frame.cue.endswith(_PAST_ENDINGS) or frame.cue in _IRREGULAR_PAST
        if past and tense == "current":
            tense = "past"
        # Unknown future/present morphology is not reliably separable without a
        # morphological model. Accept raw unknown past cues, but abstain on
        # uncertain tense instead of silently asserting a future event.
        if not (
            past
            or frame.cue in _CURRENT_CUES
            or frame.cue in {"местонахождение", "обладание"}
            or frame.cue in _FUTURE_CUES
            or modality == "hypothetical"
            and frame.cue.endswith(("ть", "ти", "чь"))
        ):
            raise _SyntaxFailure("unsupported_cue_tense")
        if raw.startswith("нет,") and modality != "asserted":
            raise _SyntaxFailure("hypothetical_correction")
        if question:
            if modality != "asserted" or act == "correct":
                raise _SyntaxFailure("unsupported_question_modality")
            act = "ask"
        return replace(
            frame,
            act=act,
            modality=modality,
            tense=tense,
            query="verify" if question else "",
        )

    @staticmethod
    def _question(words: list[str], raw: str) -> SemanticFrame | None:
        # These exact phrases ask for recorded evidence, not causal reasoning.
        if words in (
            ["почему"],
            ["почему", "ты", "так", "считаешь"],
            ["откуда", "ты", "знаешь"],
        ):
            return SemanticFrame(act="ask", query="why", object="@object", raw=raw)
        if words[0] == "почему":
            if len(words) != 2:
                raise _SyntaxFailure("unsupported_provenance_question")
            return SemanticFrame(
                act="ask", query="why", object=_entity(words[1], "subject"), raw=raw
            )
        if words[0] == "где":
            rest = words[1:]
            if rest and rest[0] in {"сейчас", "находится", "лежит"}:
                rest = rest[1:]
            if len(rest) != 1:
                raise _SyntaxFailure("unsupported_where_question")
            return SemanticFrame(
                act="ask", query="where", object=_entity(rest[0], "subject"), raw=raw
            )
        if words[:2] == ["у", "кого"]:
            rest = words[2:]
            if rest and rest[0] == "есть":
                rest = rest[1:]
            if len(rest) != 1:
                raise _SyntaxFailure("unsupported_holder_question")
            return SemanticFrame(
                act="ask", query="who_has", object=_entity(rest[0], "subject"), raw=raw
            )
        if words[:2] == ["что", "у"]:
            rest = words[2:]
            if len(rest) == 2 and rest[1] == "есть":
                rest = rest[:1]
            if len(rest) != 1:
                raise _SyntaxFailure("unsupported_possession_question")
            return SemanticFrame(
                act="ask",
                query="what_has",
                actor=_entity(rest[0], "possessor"),
                raw=raw,
            )
        return None

    @staticmethod
    def _statement(words: list[str], raw: str) -> SemanticFrame:
        if words[0] == "у":
            if len(words) not in {3, 4, 5}:
                raise _SyntaxFailure("unsupported_possession_roles")
            actor = _entity(words[1], "possessor")
            rest = words[2:]
            negative = rest[0] == "нет" or rest[:1] == ["не"]
            if rest[:1] == ["не"]:
                rest = rest[1:]
                if not rest or rest[0] not in {"был", "была", "было", "были"}:
                    raise _SyntaxFailure("unsupported_possession_negation")
            if len(rest) == 1:
                cue, obj = "обладание", rest[0]
            elif len(rest) == 2 and rest[0] in {
                "есть",
                "нет",
                "был",
                "была",
                "было",
                "были",
            }:
                cue, obj = rest
            else:
                raise _SyntaxFailure("unsupported_possession_roles")
            return SemanticFrame(
                act="inform",
                cue=cue,
                actor=actor,
                object=_entity(obj, "absence" if negative else "subject"),
                negated=negative,
                raw=raw,
            )

        prepositions = [i for i, word in enumerate(words) if word in {"в", "на"}]
        if prepositions:
            if len(prepositions) != 1:
                raise _SyntaxFailure("multiple_spatial_roles")
            at = prepositions[0]
            if at != len(words) - 2:
                raise _SyntaxFailure("unconsumed_spatial_clause")
            left = words[:at]
            negative = "не" in left
            if negative:
                if left.count("не") != 1 or left.index("не") != 1:
                    raise _SyntaxFailure("misplaced_negation")
                left = left[:1] + left[2:]
            if len(left) > 1 and left[1] in _FUTURE_AUX:
                left = left[:1] + left[2:]
            if len(left) == 1:
                obj, cue, actor = _entity(left[0], "subject"), "местонахождение", ""
            elif len(left) == 2:
                obj, cue, actor = _entity(left[0], "subject"), _cue(left[1]), ""
            elif len(left) == 3:
                actor, cue = _entity(left[0], "actor"), _cue(left[1])
                obj = _entity(left[2], "object", negative=negative)
            else:
                raise _SyntaxFailure("unsupported_spatial_roles")
            place = _entity(words[-1], "destination" if actor else "place")
            return SemanticFrame(
                act="inform",
                cue=cue,
                actor=actor,
                object=obj,
                place=place,
                spatial="on" if words[at] == "на" else "in",
                negated=negative,
                raw=raw,
            )

        negative = "не" in words
        if negative:
            if words.count("не") != 1 or words.index("не") != 1:
                raise _SyntaxFailure("misplaced_negation")
            words = words[:1] + words[2:]
        if len(words) > 1 and words[1] in _FUTURE_AUX:
            words = words[:1] + words[2:]
        if len(words) == 4:
            actor, cue = _entity(words[0], "actor"), _cue(words[1])
            # Dative-before-accusative is accepted only when declared cases
            # identify that ordering, never by guessing an unknown name's stem.
            third = _CASES.get(words[2], [])
            if third and any(case == 4 for _, case in third):
                recipient = _entity(words[2], "recipient")
                obj = _entity(words[3], "object", negative=negative)
            else:
                obj = _entity(words[2], "object", negative=negative)
                recipient = _entity(words[3], "recipient")
            return SemanticFrame(
                act="inform",
                cue=cue,
                actor=actor,
                object=obj,
                recipient=recipient,
                negated=negative,
                raw=raw,
            )
        if len(words) == 3 and words[1] in {"имеет", "имел", "имела"}:
            return SemanticFrame(
                act="inform",
                cue=words[1],
                actor=_entity(words[0], "actor"),
                object=_entity(words[2], "object", negative=negative),
                negated=negative,
                raw=raw,
            )
        raise _SyntaxFailure("unsupported_complete_clause")
