"""Small open development tests; no sealed cases or oracle IDs at inference."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledEvent,
    LabeledText,
    SemanticGraph,
    Span,
)
from text_factors.real_data.raw_language import RawSemanticModel


def _span(text: str, word: str) -> Span:
    at = text.index(word)
    return Span(at, at + len(word))


def _train() -> tuple[LabeledText, ...]:
    ordinary = "Анна передала папку Борису."
    negated = "Борис не передал ключ Анне."
    reversed_order = "Борису передала папку Анна."
    return (
        LabeledText(
            ordinary,
            (
                IdentifiedMention(
                    "a", "anna", _span(ordinary, "Анна"), "person", "nom"
                ),
                IdentifiedMention("x", "folder", _span(ordinary, "папку"), "thing"),
                IdentifiedMention(
                    "b", "boris", _span(ordinary, "Борису"), "person", "dat"
                ),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(ordinary, "передала"),
                    (("actor", "a"), ("object", "x"), ("recipient", "b")),
                ),
            ),
        ),
        LabeledText(
            negated,
            (
                IdentifiedMention(
                    "b", "boris", _span(negated, "Борис"), "person", "nom"
                ),
                IdentifiedMention("x", "key", _span(negated, "ключ"), "thing"),
                IdentifiedMention("a", "anna", _span(negated, "Анне"), "person", "dat"),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(negated, "передал"),
                    (("actor", "b"), ("object", "x"), ("recipient", "a")),
                    negation_cue=_span(negated, "не"),
                ),
            ),
        ),
        LabeledText(
            reversed_order,
            (
                IdentifiedMention(
                    "b", "boris", _span(reversed_order, "Борису"), "person", "dat"
                ),
                IdentifiedMention(
                    "x", "folder", _span(reversed_order, "папку"), "thing"
                ),
                IdentifiedMention(
                    "a", "anna", _span(reversed_order, "Анна"), "person", "nom"
                ),
            ),
            (
                LabeledEvent(
                    "transfer",
                    _span(reversed_order, "передала"),
                    (("actor", "a"), ("object", "x"), ("recipient", "b")),
                ),
            ),
        ),
    )


def _parse(model: RawSemanticModel, text: str) -> SemanticGraph:
    return model.parse(text, source_id="heldout-source", source_version=2)


class RawLanguageTests(unittest.TestCase):
    def test_unseen_names_and_objects_roles_and_negation_without_oracle(self) -> None:
        model = RawSemanticModel().fit(_train())
        text = "Нина не передала книгу Дарье."
        graph = _parse(model, text)
        self.assertEqual(len(graph.events), 1)
        event = graph.events[0]
        self.assertEqual(event.relation_id, "transfer")
        self.assertTrue(event.negated)
        self.assertEqual(
            [
                text[mention.span.start : mention.span.end]
                for mention in model.infer_mentions(
                    text, source_id="heldout-source", source_version=2
                )
            ],
            ["Нина", "книгу", "Дарье"],
        )
        self.assertEqual(
            [
                text[
                    int(role.mention_id.rsplit(":", 2)[1]) : int(
                        role.mention_id.rsplit(":", 1)[1]
                    )
                ]
                for role in event.roles
            ],
            ["Нина", "книгу", "Дарье"],
        )
        self.assertEqual(
            [text[span.start : span.end] for span in graph.unexplained],
            ["Нина", "книгу", "Дарье"],
        )
        self.assertFalse(graph.complete)
        self.assertEqual(graph.text_sha256, hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(event.source_span, _span(text, "Нина не передала книгу Дарье"))

    def test_correction_prefix_stays_residual_without_losing_event(self) -> None:
        model = RawSemanticModel().fit(_train())
        text = "Исправление: Нина передала книгу Дарье."
        graph = _parse(model, text)
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(graph.events[0].relation_id, "transfer")
        self.assertEqual(
            [text[span.start : span.end] for span in graph.unexplained],
            ["Исправление:", "Нина", "книгу", "Дарье"],
        )
        self.assertFalse(graph.events[0].negated)

    def test_reversed_trained_layout_and_unknown_word_residual(self) -> None:
        model = RawSemanticModel().fit(_train())
        text = "Вере передала непонятный ключ Нина."
        graph = _parse(model, text)
        # An unknown adjective is kept as residue without discarding the event.
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(
            [text[span.start : span.end] for span in graph.unexplained],
            ["Вере", "непонятный", "Нина"],
        )
        known = _parse(model, "Вере передала ключ Нина.")
        self.assertEqual(len(known.events), 1)
        self.assertEqual(
            [role.role for role in known.events[0].roles],
            ["actor", "object", "recipient"],
        )
        self.assertEqual(
            [span for span in known.unexplained if span == known.events[0].source_span],
            [],
        )

    def test_unknown_relation_and_absent_argument_do_not_create_facts(self) -> None:
        model = RawSemanticModel().fit(_train())
        for text in ("Нина осмотрела книгу Дарье.", "Нина передала Дарье."):
            with self.subTest(text=text):
                graph = _parse(model, text)
                self.assertFalse(graph.events)
                self.assertFalse(graph.complete)

    def test_distinct_mentions_are_not_promoted_to_coreference(self) -> None:
        model = RawSemanticModel().fit(_train())
        text = "Нина передала книгу Дарье. Нина передала книгу Дарье."
        graph = _parse(model, text)
        self.assertEqual(len(graph.events), 2)
        self.assertNotEqual(
            graph.events[0].roles[0].instance_id,
            graph.events[1].roles[0].instance_id,
        )

    def test_model_and_graph_round_trip_and_invalid_input(self) -> None:
        model = RawSemanticModel().fit(_train())
        restored = RawSemanticModel.from_dict(json.loads(json.dumps(model.to_dict())))
        text = "Нина передала книгу Дарье."
        self.assertEqual(_parse(restored, text), _parse(model, text))
        graph = _parse(restored, text)
        self.assertEqual(SemanticGraph.from_dict(graph.to_dict()), graph)
        tampered = copy.deepcopy(model.to_dict())
        tampered["surfaces"].append(tampered["surfaces"][0])
        with self.assertRaises(ValueError):
            RawSemanticModel.from_dict(tampered)
        with self.assertRaises(ValueError):
            _parse(restored, "")
        with self.assertRaises(ValueError):
            restored.parse(text, source_id="heldout-source", source_version=0)


if __name__ == "__main__":
    unittest.main()
