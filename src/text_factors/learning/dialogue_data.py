"""Declared training demonstrations for the small learned dialogue components.

These are complete supervised turn sequences, not evaluation data. Response
strings are *training targets*: the runtime generator learns token transition
counts and never retrieves one of these complete responses. Case forms in copy
slots are annotated surface examples, not a general Russian morphology engine.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _fact(
    subject: str = "ключ",
    value: str = "ящик",
    *,
    relation: str = "location",
    negated: bool = False,
    spatial: str = "in",
    event_id: int = 1,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "value": value,
        "negated": negated,
        "spatial": spatial,
        "source": "user",
        "event_id": event_id,
    }


def _turn(
    user: str,
    action: str,
    response: str,
    *,
    act: str = "inform",
    task: str = "",
    evidence: dict[str, Any] | None = None,
    slots: dict[str, str] | None = None,
    **features: Any,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "act": act,
        "task": task,
        "has_answer": evidence is not None,
        "ambiguous": False,
        "unsupported": False,
        "nonactual": False,
        "corrected": action == "corrected",
        "retracted": action == "retracted",
        "pending": "",
        "evidence_count": int(evidence is not None),
        "failed": False,
    }
    values.update(features)
    return {
        "user": user,
        "features": values,
        "action": action,
        "response": response,
        "slots": slots or {},
        "evidence": [evidence] if evidence is not None else [],
    }


_TRAINING = [
    [
        _turn("Привет", "greet", "Привет!", act="greet"),
        _turn(
            "Ключ в ящике",
            "ack",
            "Запомнил: <object> <prep> <place>.",
            evidence=_fact(),
            slots={"object": "ключ", "place": "ящике"},
        ),
        _turn(
            "Где ключ?",
            "answer",
            "<object> <prep> <place>.",
            act="ask",
            task="where",
            evidence=_fact(),
            slots={"object": "ключ", "place": "ящике"},
        ),
        _turn(
            "Я переложил ключ в сумку",
            "ack",
            "Запомнил: <object> <prep> <place>.",
            evidence=_fact(value="сумка", event_id=4),
            slots={"place": "сумке"},
        ),
        _turn(
            "Где он?",
            "answer",
            "<object> <prep> <place>.",
            act="ask",
            task="where",
            evidence=_fact(value="сумка", event_id=4),
            slots={"place": "сумке"},
        ),
        _turn(
            "Нет, ключ на столе",
            "corrected",
            "Исправил: <object> <prep> <place>.",
            act="correct",
            evidence=_fact(value="стол", spatial="on", event_id=6),
            slots={"place": "столе"},
        ),
        _turn(
            "Почему?",
            "explain",
            "<object> <prep> <place>. Источник: <source>.",
            act="ask",
            task="why",
            evidence=_fact(value="стол", spatial="on", event_id=6),
            slots={"place": "столе", "source": "сообщение 6"},
        ),
        _turn("Отмени исправление", "retracted", "Отмена учтена.", act="retract"),
        _turn(
            "Где ключ теперь?",
            "answer",
            "<object> <prep> <place>.",
            act="ask",
            task="where",
            evidence=_fact(value="сумка", event_id=4),
            slots={"place": "сумке"},
        ),
        _turn("Спасибо", "thanks", "Пожалуйста.", act="thanks"),
    ],
    [
        _turn("Здравствуйте", "greet", "Здравствуйте!", act="greet"),
        _turn(
            "Миша передал книгу Маше",
            "ack",
            "Запомнил: <object> у <holder>.",
            evidence=_fact("книга", "маша", relation="holder"),
            slots={"holder": "Маши"},
        ),
        _turn(
            "У кого книга?",
            "answer",
            "<object> у <holder>.",
            act="ask",
            task="who_has",
            evidence=_fact("книга", "маша", relation="holder"),
            slots={"holder": "Маши"},
        ),
        _turn(
            "Что у Маши?",
            "answer",
            "<object> у <holder>.",
            act="ask",
            task="what_has",
            evidence=_fact("книга", "маша", relation="holder"),
            slots={"holder": "Маши"},
        ),
        _turn(
            "Нет, книга у Пети",
            "corrected",
            "Исправил: <object> у <holder>.",
            act="correct",
            evidence=_fact("книга", "петя", relation="holder", event_id=5),
            slots={"holder": "Пети"},
        ),
        _turn(
            "Откуда это известно?",
            "explain",
            "<object> у <holder>. Источник: <source>.",
            act="ask",
            task="why",
            evidence=_fact("книга", "петя", relation="holder", event_id=5),
            slots={"holder": "Пети", "source": "сообщение 5"},
        ),
        _turn("Отмени последнее", "retracted", "Отмена учтена.", act="retract"),
        _turn("Спасибо", "thanks", "Пожалуйста.", act="thanks"),
    ],
    [
        _turn(
            "Ключ не в коробке",
            "ack",
            "Запомнил: <object> не <prep> <place>.",
            evidence=_fact(value="коробка", negated=True),
            slots={"place": "коробке"},
        ),
        _turn(
            "Ключ в коробке?",
            "answer",
            "<truth>. <object> не <prep> <place>.",
            act="ask",
            task="verify",
            evidence=_fact(value="коробка", negated=True),
            slots={"place": "коробке", "truth": "no"},
        ),
        _turn(
            "Где же ключ?", "unknown", "Этого я пока не знаю.", act="ask", task="where"
        ),
        _turn(
            "Ключ на полке",
            "ack",
            "Запомнил: <object> <prep> <place>.",
            evidence=_fact(value="полка", spatial="on", event_id=4),
            slots={"place": "полке"},
        ),
        _turn(
            "Он на полке?",
            "answer",
            "<truth>. <object> <prep> <place>.",
            act="ask",
            task="verify",
            evidence=_fact(value="полка", spatial="on", event_id=4),
            slots={"place": "полке", "truth": "yes"},
        ),
        _turn(
            "Почему?",
            "explain",
            "<object> <prep> <place>. Источник: <source>.",
            act="ask",
            task="why",
            evidence=_fact(value="полка", spatial="on", event_id=4),
            slots={"place": "полке", "source": "сообщение 4"},
        ),
    ],
    [
        _turn(
            "У Маши нет паспорта",
            "ack",
            "Запомнил: <object> не у <holder>.",
            evidence=_fact("паспорт", "маша", relation="holder", negated=True),
            slots={"holder": "Маши"},
        ),
        _turn(
            "Паспорт у Маши?",
            "answer",
            "<truth>. <object> не у <holder>.",
            act="ask",
            task="verify",
            evidence=_fact("паспорт", "маша", relation="holder", negated=True),
            slots={"holder": "Маши", "truth": "no"},
        ),
        _turn(
            "У кого паспорт?",
            "unknown",
            "Этого я пока не знаю.",
            act="ask",
            task="who_has",
        ),
        _turn(
            "Нет, паспорт у Миши",
            "corrected",
            "Исправил: <object> у <holder>.",
            act="correct",
            evidence=_fact("паспорт", "миша", relation="holder", event_id=4),
            slots={"holder": "Миши"},
        ),
        _turn(
            "Он у Миши?",
            "answer",
            "<truth>. <object> у <holder>.",
            act="ask",
            task="verify",
            evidence=_fact("паспорт", "миша", relation="holder", event_id=4),
            slots={"holder": "Миши", "truth": "yes"},
        ),
    ],
    [
        _turn(
            "Где паспорт?", "unknown", "Этого я пока не знаю.", act="ask", task="where"
        ),
        _turn(
            "А он?",
            "clarify",
            "Уточните, какой предмет вы имеете в виду.",
            act="ask",
            ambiguous=True,
            task="where",
        ),
        _turn(
            "Я имею в виду книгу",
            "unknown",
            "Этого я пока не знаю.",
            act="ask",
            task="where",
            pending="clarify",
        ),
        _turn(
            "Книга в шкафу",
            "ack",
            "Запомнил: <object> <prep> <place>.",
            evidence=_fact("книга", "шкаф", event_id=4),
            slots={"place": "шкафу"},
        ),
        _turn(
            "Где она?",
            "answer",
            "<object> <prep> <place>.",
            act="ask",
            task="where",
            evidence=_fact("книга", "шкаф", event_id=4),
            slots={"place": "шкафу"},
        ),
        _turn(
            "А он?",
            "clarify",
            "Уточните, какой предмет вы имеете в виду.",
            act="ask",
            ambiguous=True,
            task="where",
        ),
        _turn(
            "Книга",
            "answer",
            "<object> <prep> <place>.",
            act="ask",
            task="where",
            pending="clarify",
            evidence=_fact("книга", "шкаф", event_id=4),
            slots={"place": "шкафу"},
        ),
    ],
    [
        _turn(
            "Миша обещал передать ключ",
            "nonactual",
            "Не отмечаю это действие как выполненное.",
            nonactual=True,
            task="promise",
        ),
        _turn(
            "У кого ключ?",
            "unknown",
            "Этого я пока не знаю.",
            act="ask",
            task="who_has",
        ),
        _turn(
            "Говорят, ключ в сумке",
            "nonactual",
            "Не отмечаю это действие как выполненное.",
            nonactual=True,
            task="report",
        ),
        _turn("Где ключ?", "unknown", "Этого я пока не знаю.", act="ask", task="where"),
        _turn(
            "Если он придет, ключ будет у Маши",
            "nonactual",
            "Не отмечаю это действие как выполненное.",
            nonactual=True,
            task="conditional",
        ),
        _turn(
            "Миша действительно передал ключ Маше",
            "ack",
            "Запомнил: <object> у <holder>.",
            evidence=_fact(value="маша", relation="holder", event_id=6),
            slots={"holder": "Маши"},
        ),
        _turn(
            "Теперь у кого ключ?",
            "answer",
            "<object> у <holder>.",
            act="ask",
            task="who_has",
            evidence=_fact(value="маша", relation="holder", event_id=6),
            slots={"holder": "Маши"},
        ),
    ],
    [
        _turn(
            "Расскажи о квантовой гравитации",
            "clarify",
            "Уточните запрос в поддерживаемом формате.",
            act="unknown",
            unsupported=True,
        ),
        _turn(
            "Помощь",
            "help",
            "Сообщите, где предмет, или задайте вопрос о нём.",
            act="help",
        ),
        _turn("Привет", "greet", "Привет!", act="greet"),
        _turn("Спасибо", "thanks", "Пожалуйста.", act="thanks"),
        _turn(
            "Отмени неизвестное утверждение",
            "clarify",
            "Уточните запрос в поддерживаемом формате.",
            act="retract",
            failed=True,
        ),
        _turn(
            "Что это значит?",
            "clarify",
            "Уточните запрос в поддерживаемом формате.",
            act="unknown",
            unsupported=True,
            pending="clarify",
        ),
    ],
    [
        _turn(
            "Телефон в кармане",
            "ack",
            "Учёл: <object> <prep> <place>.",
            evidence=_fact("телефон", "карман"),
            slots={"place": "кармане"},
        ),
        _turn(
            "Где телефон?",
            "answer",
            "<object> <prep> <place>.",
            act="ask",
            task="where",
            evidence=_fact("телефон", "карман"),
            slots={"place": "кармане"},
        ),
        _turn(
            "Нет, телефон не в кармане",
            "corrected",
            "Исправил: <object> не <prep> <place>.",
            act="correct",
            evidence=_fact("телефон", "карман", negated=True, event_id=3),
            slots={"place": "кармане"},
        ),
        _turn(
            "Почему?",
            "explain",
            "<object> не <prep> <place>. Источник: <source>.",
            act="ask",
            task="why",
            evidence=_fact("телефон", "карман", negated=True, event_id=3),
            slots={"place": "кармане", "source": "сообщение 3"},
        ),
        _turn(
            "Отмени последнее утверждение", "retracted", "Отмена учтена.", act="retract"
        ),
        _turn("Благодарю", "thanks", "Пожалуйста.", act="thanks"),
    ],
    [
        _turn(
            "У Пети нет мяча",
            "ack",
            "Запомнил: <object> не у <holder>.",
            evidence=_fact("мяч", "петя", relation="holder", negated=True),
            slots={"holder": "Пети"},
        ),
        _turn(
            "Нет, мяч не у Маши",
            "corrected",
            "Исправил: <object> не у <holder>.",
            act="correct",
            evidence=_fact("мяч", "маша", relation="holder", negated=True, event_id=2),
            slots={"holder": "Маши"},
        ),
        _turn(
            "Почему?",
            "explain",
            "<object> не у <holder>. Источник: <source>.",
            act="ask",
            task="why",
            evidence=_fact("мяч", "маша", relation="holder", negated=True, event_id=2),
            slots={"holder": "Маши", "source": "сообщение 2"},
        ),
        _turn(
            "Что у Маши?",
            "unknown",
            "Этого я пока не знаю.",
            act="ask",
            task="what_has",
        ),
        _turn(
            "Ключ не у Маши?",
            "answer",
            "<truth>. <object> не у <holder>.",
            act="ask",
            task="verify",
            evidence=_fact(value="маша", relation="holder", negated=True, event_id=5),
            slots={"holder": "Маши", "truth": "yes"},
        ),
    ],
]


def training_dialogues(seed: int = 42) -> list[list[dict[str, Any]]]:
    """A detached copy; callers may explicitly shuffle labels for controls."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid dialogue data seed")
    data = deepcopy(_TRAINING)
    data.append(
        [
            _turn("Я не перемещал ключ", "ack", "Сообщение учтено."),
            _turn(
                "Где ключ?", "unknown", "Этого я пока не знаю.", act="ask", task="where"
            ),
            _turn(
                "Исправляю: перемещение не произошло",
                "corrected",
                "Исправление учтено.",
                act="correct",
            ),
            _turn(
                "Покажи помощь",
                "help",
                "Сообщите, где предмет, или задайте вопрос о нём.",
                act="help",
            ),
        ]
    )
    return data


def development_dialogues() -> list[list[dict[str, Any]]]:
    """Declared development examples, disjoint from independent evaluation."""
    return [
        [
            _turn(
                "Письмо в папке",
                "ack",
                "Запомнил: <object> <prep> <place>.",
                evidence=_fact("письмо", "папка"),
                slots={"place": "папка"},
            ),
            _turn(
                "Где письмо?",
                "answer",
                "<object> <prep> <place>.",
                act="ask",
                task="where",
                evidence=_fact("письмо", "папка"),
                slots={"place": "папка"},
            ),
            _turn(
                "Может быть оно у Анны",
                "nonactual",
                "Не отмечаю это действие как выполненное.",
                nonactual=True,
                task="possible",
            ),
            _turn(
                "Откуда сведения?",
                "explain",
                "<object> <prep> <place>. Источник: <source>.",
                act="ask",
                task="why",
                evidence=_fact("письмо", "папка"),
                slots={"place": "папка", "source": "сообщение 1"},
            ),
        ]
    ]
