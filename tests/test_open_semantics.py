"""Open train / synthetic development cases; no sealed corpus is read here."""

from __future__ import annotations

import copy
import json
import unittest

from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledEvent,
    LabeledLink,
    LabeledText,
    OpenSemanticModel,
    SemanticGraph,
    Span,
)


def _span(text: str, surface: str, occurrence: int = 1) -> Span:
    start = 0
    for _ in range(occurrence):
        start = text.index(surface, start)
        end = start + len(surface)
        start = end
    return Span(start - len(surface), start)


def _mention(
    text: str,
    mid: str,
    instance: str,
    surface: str,
    kind: str = "person",
    occurrence: int = 1,
    morphology: str | None = None,
) -> IdentifiedMention:
    return IdentifiedMention(
        mid, instance, _span(text, surface, occurrence), kind, morphology
    )


def _train() -> tuple[LabeledText, ...]:
    first = "Анна передала ключ Борису."
    second = "Борис вчера не передал ключ Анне."
    third = "Анна передала ключ Борису. Затем Борис исправил ключ."
    fourth = "Анна исправила ключ."
    return (
        LabeledText(
            first,
            (
                _mention(first, "a", "anna", "Анна", morphology="nom"),
                _mention(first, "k", "key-1", "ключ", "thing"),
                _mention(first, "b", "boris", "Борису", morphology="dat"),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(first, "передала"),
                    (("actor", "a"), ("object", "k"), ("recipient", "b")),
                ),
            ),
        ),
        LabeledText(
            second,
            (
                _mention(second, "b", "boris", "Борис", morphology="nom"),
                _mention(second, "k", "key-2", "ключ", "thing"),
                _mention(second, "a", "anna", "Анне", morphology="dat"),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(second, "передал"),
                    (("actor", "b"), ("object", "k"), ("recipient", "a")),
                    negation_cue=_span(second, "не"),
                    time_cue=_span(second, "вчера"),
                    time_label="past",
                ),
            ),
        ),
        LabeledText(
            third,
            (
                _mention(third, "a", "anna", "Анна", morphology="nom"),
                _mention(third, "k1", "key-1", "ключ", "thing", 1),
                _mention(third, "b1", "boris", "Борису", morphology="dat"),
                _mention(
                    third,
                    "b2",
                    "boris",
                    "Борис",
                    occurrence=2,
                    morphology="nom",
                ),
                _mention(third, "k2", "key-2", "ключ", "thing", 2),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(third, "передала"),
                    (("actor", "a"), ("object", "k1"), ("recipient", "b1")),
                ),
                LabeledEvent(
                    "repair",
                    _span(third, "исправил"),
                    (("actor", "b2"), ("object", "k2")),
                ),
            ),
            (
                LabeledLink(
                    "after",
                    _span(third, "передала"),
                    _span(third, "исправил"),
                    _span(third, "Затем"),
                ),
            ),
        ),
        LabeledText(
            fourth,
            (
                _mention(fourth, "a", "anna", "Анна", morphology="nom"),
                _mention(fourth, "k", "key-1", "ключ", "thing"),
            ),
            (
                LabeledEvent(
                    "repair",
                    _span(fourth, "исправила"),
                    (("actor", "a"), ("object", "k")),
                ),
            ),
        ),
    )


def _parse(
    model: OpenSemanticModel, text: str, mentions: tuple[IdentifiedMention, ...]
) -> SemanticGraph:
    return model.parse(
        text, mentions, source_id="synthetic-development", source_version=1
    )


class OpenSemanticTests(unittest.TestCase):
    def test_two_events_separate_same_surface_ids_negation_time_and_link(self) -> None:
        model = OpenSemanticModel().fit(_train())
        text = "Пётр вчера не передал ключ Анне. Затем Анна исправила непонятный ключ."
        graph = _parse(
            model,
            text,
            (
                _mention(text, "p", "peter", "Пётр", morphology="nom"),
                _mention(text, "k1", "key-new-1", "ключ", "thing", 1),
                _mention(text, "a1", "anna", "Анне", morphology="dat"),
                _mention(text, "a2", "anna", "Анна", morphology="nom"),
                _mention(text, "k2", "key-new-2", "ключ", "thing", 2),
            ),
        )
        self.assertEqual(
            [event.relation_id for event in graph.events],
            [
                "transfer",
                "repair",
            ],
        )
        self.assertEqual(
            [r.instance_id for r in graph.events[0].roles],
            ["peter", "key-new-1", "anna"],
        )
        self.assertEqual(
            [r.instance_id for r in graph.events[1].roles],
            ["anna", "key-new-2"],
        )
        self.assertTrue(graph.events[0].negated)
        self.assertEqual(graph.events[0].time_label, "past")
        self.assertFalse(graph.events[1].negated)
        self.assertEqual(len(graph.links), 1)
        self.assertEqual(graph.links[0].relation_id, "after")
        self.assertEqual(graph.links[0].source_event_id, graph.events[0].event_id)
        self.assertEqual(graph.links[0].target_event_id, graph.events[1].event_id)
        self.assertEqual(
            [text[span.start : span.end] for span in graph.unexplained],
            ["непонятный"],
        )
        self.assertFalse(graph.complete)
        self.assertEqual(
            SemanticGraph.from_dict(json.loads(json.dumps(graph.to_dict()))), graph
        )
        restored_model = OpenSemanticModel.from_dict(
            json.loads(json.dumps(model.to_dict()))
        )
        self.assertEqual(restored_model.model_fingerprint, model.model_fingerprint)
        self.assertEqual(
            _parse(
                restored_model,
                text,
                (
                    _mention(text, "p", "peter", "Пётр", morphology="nom"),
                    _mention(text, "k1", "key-new-1", "ключ", "thing", 1),
                    _mention(text, "a1", "anna", "Анне", morphology="dat"),
                    _mention(text, "a2", "anna", "Анна", morphology="nom"),
                    _mention(text, "k2", "key-new-2", "ключ", "thing", 2),
                ),
            ),
            graph,
        )

    def test_new_relation_requires_labeled_training(self) -> None:
        text = "Вера осветила зал."
        mentions = (
            _mention(text, "p", "vera", "Вера"),
            _mention(text, "r", "room-1", "зал", "thing"),
        )
        baseline = OpenSemanticModel().fit(_train())
        self.assertEqual(_parse(baseline, text, mentions).events, ())
        trained = OpenSemanticModel().fit(
            (
                *_train(),
                LabeledText(
                    "Анна осветила зал.",
                    (
                        _mention("Анна осветила зал.", "p", "anna", "Анна"),
                        _mention("Анна осветила зал.", "r", "room-0", "зал", "thing"),
                    ),
                    (
                        LabeledEvent(
                            "illuminate",
                            _span("Анна осветила зал.", "осветила"),
                            (("actor", "p"), ("target", "r")),
                        ),
                    ),
                ),
            )
        )
        graph = _parse(trained, text, mentions)
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(graph.events[0].relation_id, "illuminate")
        self.assertEqual(
            [r.instance_id for r in graph.events[0].roles],
            [
                "vera",
                "room-1",
            ],
        )
        self.assertTrue(graph.complete)

    def test_swapped_roles_missing_role_and_ambiguous_cues_abstain(self) -> None:
        model = OpenSemanticModel().fit(_train())
        text = "Ольга передала мяч Сергею."
        graph = _parse(
            model,
            text,
            (
                _mention(text, "o", "olga", "Ольга", morphology="nom"),
                _mention(text, "x", "ball", "мяч", "thing"),
                _mention(text, "s", "sergey", "Сергею", morphology="dat"),
            ),
        )
        self.assertEqual(
            [r.instance_id for r in graph.events[0].roles],
            [
                "olga",
                "ball",
                "sergey",
            ],
        )
        self.assertTrue(graph.complete)
        self.assertEqual(
            _parse(
                model,
                text,
                (
                    _mention(text, "o", "olga", "Ольга", morphology="nom"),
                    _mention(text, "x", "ball", "мяч", "thing"),
                ),
            ).events,
            (),
        )
        ambiguous = LabeledText(
            "Ольга передала мяч Сергею.",
            (
                _mention(text, "o", "olga", "Ольга", morphology="nom"),
                _mention(text, "x", "ball", "мяч", "thing"),
                _mention(text, "s", "sergey", "Сергею", morphology="dat"),
            ),
            (
                LabeledEvent(
                    "other_relation",
                    _span(text, "передала"),
                    (("actor", "o"), ("object", "x"), ("recipient", "s")),
                ),
            ),
        )
        model.fit((*_train(), ambiguous))
        self.assertEqual(_parse(model, text, ambiguous.mentions).events, ())

    def test_reversed_syntax_only_with_trained_layout_and_external_case(self) -> None:
        reversed_train = "Борису передала ключ Анна."
        reversed_example = LabeledText(
            reversed_train,
            (
                _mention(reversed_train, "b", "boris", "Борису", morphology="dat"),
                _mention(reversed_train, "k", "key", "ключ", "thing"),
                _mention(reversed_train, "a", "anna", "Анна", morphology="nom"),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(reversed_train, "передала"),
                    (("actor", "a"), ("object", "k"), ("recipient", "b")),
                ),
            ),
        )
        model = OpenSemanticModel().fit((*_train(), reversed_example))
        text = "Сергею передала мяч Ольга."
        with_cases = (
            _mention(text, "s", "sergey", "Сергею", morphology="dat"),
            _mention(text, "x", "ball", "мяч", "thing"),
            _mention(text, "o", "olga", "Ольга", morphology="nom"),
        )
        graph = _parse(model, text, with_cases)
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(
            {role.role: role.instance_id for role in graph.events[0].roles},
            {"actor": "olga", "object": "ball", "recipient": "sergey"},
        )
        self.assertTrue(graph.complete)
        self.assertEqual(
            _parse(
                OpenSemanticModel.from_dict(json.loads(json.dumps(model.to_dict()))),
                text,
                with_cases,
            ),
            graph,
        )
        without_cases = (
            _mention(text, "s", "sergey", "Сергею"),
            _mention(text, "x", "ball", "мяч", "thing"),
            _mention(text, "o", "olga", "Ольга"),
        )
        self.assertEqual(_parse(model, text, without_cases).events, ())
        self.assertFalse(_parse(model, text, without_cases).complete)

        # Even a single trained surface layout cannot certify two same-kind
        # roles when the caller supplies no grammatical distinction.
        canonical = "Ольга передала мяч Сергею."
        self.assertEqual(
            _parse(
                OpenSemanticModel().fit(_train()),
                canonical,
                (
                    _mention(canonical, "o", "olga", "Ольга"),
                    _mention(canonical, "x", "ball", "мяч", "thing"),
                    _mention(canonical, "s", "sergey", "Сергею"),
                ),
            ).events,
            (),
        )

    def test_graph_rejects_tampering_and_nested_spans(self) -> None:
        model = OpenSemanticModel().fit(_train())
        text = "Ольга передала мяч Сергею."
        mentions = (
            _mention(text, "o", "olga", "Ольга", morphology="nom"),
            _mention(text, "x", "ball", "мяч", "thing"),
            _mention(text, "s", "sergey", "Сергею", morphology="dat"),
        )
        result = _parse(model, text, mentions).to_dict()
        corrupt = copy.deepcopy(result)
        corrupt["complete"] = False
        with self.assertRaises(ValueError):
            SemanticGraph.from_dict(corrupt)
        corrupt = copy.deepcopy(result)
        corrupt["links"] = [
            {
                "relation_id": "after",
                "source_event_id": "unknown",
                "target_event_id": result["events"][0]["event_id"],
                "cue": {"start": 1, "end": 2},
            }
        ]
        with self.assertRaises(ValueError):
            SemanticGraph.from_dict(corrupt)
        nested = IdentifiedMention(
            "nested",
            "different",
            Span(
                mentions[0].span.start,
                mentions[0].span.end - 1,
            ),
        )
        with self.assertRaises(ValueError):
            _parse(model, text, (*mentions, nested))

    def test_unrecognized_symbol_and_intermediate_clause_preserve_residual(
        self,
    ) -> None:
        model = OpenSemanticModel().fit(_train())
        text = "Ольга передала $ мяч Сергею."
        graph = _parse(
            model,
            text,
            (
                _mention(text, "o", "olga", "Ольга", morphology="nom"),
                _mention(text, "x", "ball", "мяч", "thing"),
                _mention(text, "s", "sergey", "Сергею", morphology="dat"),
            ),
        )
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(
            [text[span.start : span.end] for span in graph.unexplained], ["$"]
        )
        self.assertFalse(graph.complete)

        text = "Ольга передала мяч Сергею. Туман. Затем Ольга исправила мяч."
        graph = _parse(
            model,
            text,
            (
                _mention(
                    text,
                    "o1",
                    "olga",
                    "Ольга",
                    occurrence=1,
                    morphology="nom",
                ),
                _mention(text, "x1", "ball", "мяч", "thing", 1),
                _mention(text, "s", "sergey", "Сергею", morphology="dat"),
                _mention(
                    text,
                    "o2",
                    "olga",
                    "Ольга",
                    occurrence=2,
                    morphology="nom",
                ),
                _mention(text, "x2", "ball", "мяч", "thing", 2),
            ),
        )
        self.assertEqual(len(graph.events), 2)
        self.assertEqual(graph.links, ())
        self.assertIn("Туман", [text[s.start : s.end] for s in graph.unexplained])


if __name__ == "__main__":
    unittest.main()
