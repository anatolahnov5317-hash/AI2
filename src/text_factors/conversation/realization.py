"""Grounded Russian response templates, explicitly not learned generation."""

from __future__ import annotations

from .language import surface_entity
from .schema import Assertion, StateOutcome

HELP = (
    "Я веду небольшой мир предметов, мест и владельцев. "
    "Например: «Ключ в ящике», «Я переложил ключ в сумку», "
    "«Миша передал ключ Маше», «Где ключ?», «У кого ключ?». "
    "Исправление: «Нет, ключ на столе». Отмена: «Отмени последнее утверждение». "
    "Темы: «Тема: работа». Обучение: /teach спрятал move. "
    "Свободную речь вне этой грамматики я пока не понимаю."
)

_REASONS = {
    "ambiguous_reference": "Уточните, кого или какой предмет вы имеете в виду.",
    "unknown_reference": "Пока неясно, к кому или к чему относится местоимение.",
    "untaught_cue": "Я ещё не обучен значению этого слова действия.",
    "conflicting_teaching": "Для этого слова заданы противоречивые значения.",
    "no_retractable_assertion": "В этой теме нет утверждения, которое можно отменить.",
    "state_capacity": (
        "Достигнут предел памяти этой сессии. Сохраните её и создайте новую."
    ),
    "session_capacity": "Достигнут предел размера сессии. Ничего не изменено.",
    "turn_time_budget": "Время обработки истекло. Изменения этой реплики отменены.",
    "busy": "Предыдущая реплика ещё обрабатывается. Повторите запрос позднее.",
    "request_id_conflict": (
        "Этот идентификатор запроса уже использован для другой реплики."
    ),
    "teaching_capacity": "Достигнут предел обучающих примеров.",
    "invalid_teaching": "Формат обучения: /teach слово locate|move|give|have.",
    "internal_error": "Не удалось безопасно завершить обработку. Изменения отменены.",
}


def fact_text(fact: Assertion) -> str:
    subject = "вы" if fact.subject == "я" else surface_entity(fact.subject)
    negative = "не " if fact.negated else ""
    if fact.relation == "holder":
        owner = "вас" if fact.value == "я" else surface_entity(fact.value, "genitive")
        return f"{subject} {negative}у {owner}"
    prep = "на" if fact.qualifier == "on" else "в"
    return f"{subject} {negative}{prep} {surface_entity(fact.value, 'locative')}"


def realize(outcome: StateOutcome, *, max_facts: int = 8) -> str:
    """Only render returned facts; never fabricate a missing answer."""
    selected = outcome.assertions[:max_facts]
    facts = "; ".join(fact_text(fact) for fact in selected)
    if facts:
        facts = facts[0].upper() + facts[1:]
        if len(outcome.assertions) > len(selected):
            facts += (
                f". Показаны первые {len(selected)} из {len(outcome.assertions)} фактов"
            )
    if outcome.action == "answer":
        if outcome.reason in {"true", "false"}:
            return ("Да. " if outcome.reason == "true" else "Нет. ") + facts + "."
        if outcome.reason == "provenance":
            refs = ", ".join(
                f"{f.event_id} ("
                + ("ваше сообщение" if f.source == "user" else f"источник «{f.source}»")
                + ")"
                for f in selected
            )
            return f"{facts}. Основание: события {refs}."
        return facts + "."
    if outcome.action == "ack":
        if outcome.reason == "negated_action_not_state":
            return (
                "Учёл, что действие не произошло. "
                "Новое местонахождение из этого не следует."
            )
        prefix = "Исправил: " if outcome.reason == "corrected" else "Запомнил: "
        return prefix + facts + "." if facts else "Учёл сообщение."
    if outcome.action == "unknown":
        return "Этого я пока не знаю по сообщениям в текущей теме."
    if outcome.action == "clarify":
        message = _REASONS.get(
            outcome.reason,
            "Не смог однозначно разобрать реплику. "
            "Уточните её в поддерживаемом формате; "
            "память не изменена. Для примеров напишите «помощь».",
        )
        if outcome.alternatives:
            message += " Варианты: " + ", ".join(outcome.alternatives[:8]) + "."
        return message
    if outcome.action == "hypothetical":
        return (
            "Это предположение или будущее действие; "
            "как установленный факт не сохраняю."
        )
    if outcome.action == "retracted":
        return (
            "Отменил указанное утверждение; "
            "состояние восстановлено по оставшимся сообщениям."
        )
    if outcome.action == "topic":
        topic = outcome.resolved_frame.topic if outcome.resolved_frame else ""
        return f"Текущая тема: «{topic}». Факты других тем не подмешиваются."
    if outcome.action == "greet":
        return (
            "Привет! Расскажите, где лежит предмет, "
            "или спросите о том, что уже сообщили."
        )
    if outcome.action == "thanks":
        return "Пожалуйста."
    if outcome.action == "taught":
        return (
            "Обучающий пример сохранён. Значение проверяется факторной памятью; "
            "противоречивые пары приводят к уточнению."
        )
    return HELP
