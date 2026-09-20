import ast
import unittest
from argparse import Namespace
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from text_factors.conversation.persistence import decode_json, encode_json
from text_factors.learning import evaluation as evaluation
from text_factors.learning import evaluation_cases as cases
from text_factors.learning.schema import Event, Interpretation, Meaning


class _FixtureUnderstanding:
    """Authored test double only; never evidence of learned competence."""

    fingerprint = "fixture-training-fingerprint"

    def __init__(self, clock=None, fail_at=None):
        self.clock = clock
        self.fail_at = fail_at
        self.calls = 0

    def to_dict(self):
        return {"fixture": "understanding", "fingerprint": self.fingerprint}

    def interpret(self, text, context=None):
        if self.clock is not None:
            self.clock[0] += 0.1
        index = self.calls
        self.calls += 1
        if index == self.fail_at:
            raise RuntimeError("fixture understanding failure")
        for case in cases.understanding_cases("development"):
            if case.text == text:
                return Interpretation(case.expected, score=1.0)
        return Interpretation(None)


class _FixtureDynamics:
    def to_dict(self):
        return {"fixture": "dynamics"}

    def predict(self, before, event, seconds=1.0):
        for case in cases.dynamics_cases("development"):
            if event == case.event:
                return {
                    "effects": [
                        cases.effect_dict(effect) for effect in case.expected_effects
                    ],
                    "supported": case.expected_supported,
                }
        return {"effects": [], "supported": False}


class _FixturePolicy:
    def to_dict(self):
        return {"fixture": "policy"}

    def choose(self, features, allowed_actions):
        for case in cases.policy_cases("development"):
            if features == dict(case.features):
                return {"action": case.expected_action}
        return {"action": "unknown"}


class _FixtureGenerator:
    def to_dict(self):
        return {"fixture": "generator"}

    def generate(self, action, slots, evidence, max_tokens=48, seconds=0.5):
        return {
            "text": "Тестовая формулировка.",
            "tokens": ["тест"],
            "grounded": True,
            "reason": "fixture",
        }

    def verify(self, action, slots, evidence, reply):
        return True


class _FixtureBundle:
    fingerprint = "fixture-bundle-fingerprint"

    def __init__(self, clock=None, fail_at=None):
        self.understanding = _FixtureUnderstanding(clock=clock, fail_at=fail_at)
        self.dynamics = _FixtureDynamics()
        self.policy = _FixturePolicy()
        self.generator = _FixtureGenerator()
        self.metadata: dict[str, str] = {
            "dataset_kind": "bundled_synthetic_training_only"
        }
        self.version = 0

    def to_dict(self):
        return {
            "version": self.version,
            "models": [
                model.to_dict()
                for model in (
                    self.understanding,
                    self.dynamics,
                    self.policy,
                    self.generator,
                )
            ],
        }


class _FixtureWorld:
    def __init__(self):
        self._facts = []

    def facts(self):
        return tuple(self._facts)


class _FixtureSession:
    def __init__(self, bundle):
        self.world = _FixtureWorld()
        self.index = 0

    def respond(self, text, request_id=None):
        case = cases.dialogue_cases("development")[0]
        expected = case.turns[self.index]
        if text != expected.text or request_id != f"{case.case_id}:{self.index}":
            raise AssertionError("fixture input mismatch")
        self.index += 1
        self.world._facts = [
            cases.fact_dict(fact, evidence=True) for fact in expected.state
        ]
        return {
            "text": "Тестовый ответ.",
            "action": expected.actions[0],
            "assertions": [
                cases.fact_dict(fact, evidence=True)
                for fact in expected.assertions or ()
            ],
            "complete": True,
            "reason": "",
            "meaning": None,
            "evidence": [],
            "diagnostics": {},
            "turn_id": self.index,
        }


class LearnedCorpusTests(unittest.TestCase):
    def test_pinned_prospective_hashes_without_predictions(self):
        pinned = {
            "development": (
                "14468d13d23bc86594fed2c742bb9151da38d867335624a0dc033eae01944573"
            ),
            "held_out": (
                "56d1ff43603f2402abb5bfc7dad2a223a7f07d66c2da71ee5da769ee538f455f"
            ),
            "challenge": (
                "f1be18b8ab429fb23a24c0e4c0b7a1768dcf9330f58258584dd286c35835fede"
            ),
        }
        for split, digest in pinned.items():
            self.assertEqual(evaluation.corpus_manifest(split)["sha256"], digest)

    def test_case_builder_does_not_import_training_or_models(self):
        source = Path(cases.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        self.assertLessEqual(
            set(imports), {"__future__", "dataclasses", "typing", "schema"}
        )
        self.assertFalse(any(isinstance(node, ast.Import) for node in ast.walk(tree)))
        for forbidden in (
            "language_data",
            "training_examples",
            "understanding.py",
            ".interpret(",
        ):
            self.assertNotIn(forbidden, source)

    def test_whole_phrase_and_dialogue_splits_are_distinct(self):
        texts = []
        conversations = []
        for split in cases.SPLITS:
            items = cases.understanding_cases(split)
            self.assertEqual(len({item.case_id for item in items}), len(items))
            texts.append({item.text for item in items})
            conversations.append(
                {
                    tuple(turn.text for turn in case.turns)
                    for case in cases.dialogue_cases(split)
                }
            )
        self.assertFalse(texts[0] & texts[1])
        self.assertFalse(texts[0] & texts[2])
        self.assertFalse(conversations[0] & conversations[1])
        self.assertFalse(conversations[0] & conversations[2])
        self.assertFalse(conversations[1] & conversations[2])

    def test_structures_are_frozen_and_json_compatible(self):
        item = cases.understanding_cases("development")[0]
        with self.assertRaises(FrozenInstanceError):
            item.text = "changed"  # type: ignore[misc]
        for split in cases.SPLITS:
            manifest = evaluation.corpus_manifest(split, include_cases=True)
            self.assertEqual(decode_json(encode_json(manifest)), manifest)
            self.assertFalse(manifest["new_entity_morphology_claim"])

    def test_phrasal_and_compositional_axes_are_separate(self):
        from collections import Counter

        strata = Counter(
            cases.understanding_stratum(case)
            for case in cases.understanding_cases("held_out")
        )
        self.assertEqual(
            strata,
            {
                "independent_phrasal_family": 10,
                "scoped_composition": 4,
                "dialogue_reference": 1,
                "abstention_and_atomicity": 2,
            },
        )


class LearnedSemanticScoringTests(unittest.TestCase):
    def test_abstention_does_not_get_credit_for_absent_null_scope_paths(self):
        example = cases.understanding_cases("development")[0]
        score = evaluation._score_understanding(example, Interpretation(None))
        self.assertFalse(score["correct"])
        self.assertGreater(score["scope_paths_total"], 0)
        self.assertEqual(score["scope_paths_correct"], 0)
        self.assertEqual(score["role_paths_correct"], 0)

    def test_roles_negation_and_scope_are_not_bag_of_words(self):
        expected = Meaning(
            "inform", event=Event("give", actor="миша", object="ключ", recipient="маша")
        )
        example = cases.UnderstandingCase("test", "synthetic", "synthetic", expected)
        swapped = Meaning(
            "inform", event=Event("give", actor="маша", object="ключ", recipient="миша")
        )
        score = evaluation._score_understanding(example, Interpretation(swapped))
        self.assertFalse(score["correct"])
        self.assertLess(score["role_paths_correct"], score["role_paths_total"])
        negative = Meaning(
            "inform",
            event=Event(
                "give", actor="миша", object="ключ", recipient="маша", negated=True
            ),
        )
        score = evaluation._score_understanding(example, Interpretation(negative))
        self.assertFalse(score["correct"])
        self.assertLess(score["scope_paths_correct"], score["scope_paths_total"])

    def test_empty_effect_abstention_differs_from_supported_noop(self):
        example = cases.DynamicsCase(
            "test",
            (),
            Event("give", actor="а", object="о", recipient="б", negated=True),
            (),
        )
        abstained = evaluation._score_dynamics(
            example, {"effects": [], "supported": False}
        )
        noop = evaluation._score_dynamics(example, {"effects": [], "supported": True})
        self.assertFalse(abstained["correct"])
        self.assertTrue(noop["correct"])

    def test_generator_self_report_does_not_override_copy_verifier(self):
        example = cases.generation_cases("development")[0]
        result = {"text": "Непроверенное утверждение.", "tokens": [], "grounded": True}
        score = evaluation._score_generation(example, result, lambda *args: False)
        self.assertFalse(score["correct"])
        self.assertTrue(score["independent_slot_support"])
        invalid = cases.GenerationCase(
            "synthetic",
            "answer",
            (("object", "ключ"), ("place", "сейф")),
            (("ключ", "location", "ящик", False, "in"),),
            False,
            ("сейф",),
        )
        score = evaluation._score_generation(
            invalid, {"text": "Сейф.", "grounded": False}, lambda *args: True
        )
        self.assertFalse(score["correct"])

    def test_wrong_state_cannot_justify_unsupported_response(self):
        expected = cases.DialogueTurn(
            "где",
            (("ключ", "location", "ящик", False, "in"),),
            ("answer",),
            (("ключ", "location", "ящик", False, "in"),),
        )
        wrong = cases.fact_dict(("ключ", "location", "сумка", False, "in"))
        world = _FixtureWorld()
        world._facts = [wrong]
        score = evaluation._score_dialogue_turn(
            expected, {"action": "answer", "assertions": [wrong]}, world
        )
        self.assertFalse(score["correct"])
        self.assertEqual(len(score["unsupported_assertions"]), 1)

    def test_memorizer_uses_complete_training_phrases_only(self):
        meaning = Meaning(
            "inform", event=Event("locate", object="ключ", place="ящик", time="present")
        )
        baseline = evaluation._TrainingPhraseMemorizer(
            [SimpleNamespace(text="Ключ в ящике.", meaning=meaning)]
        )
        self.assertEqual(baseline.interpret("ключ в ящике").meaning, meaning)
        self.assertIsNone(baseline.interpret("В ящике ключ.").meaning)
        self.assertIsNone(baseline.interpret("Паспорт в ящике.").meaning)

    def test_nested_numeric_model_training_fingerprint_is_recorded(self):
        model = SimpleNamespace(
            to_dict=lambda: {"training": {"fingerprint": "training-only-digest"}}
        )
        manifest = evaluation._component_fingerprints(SimpleNamespace(dynamics=model))
        self.assertEqual(
            manifest["dynamics"]["training_fingerprint"], "training-only-digest"
        )
        self.assertEqual(
            evaluation._control_provenance(model)["training_fingerprint"],
            "training-only-digest",
        )

    def test_missing_custom_training_data_is_not_replaced_by_bundled_data(self):
        bundle = _FixtureBundle()
        bundle.metadata = {"dataset_kind": "explicit_annotated_dataset"}
        with patch.object(evaluation, "_training_language") as training:
            for control in ("shuffled", "memorization"):
                with (
                    self.subTest(control=control),
                    self.assertRaises(NotImplementedError),
                ):
                    evaluation._make_control(bundle, "understanding", control, 1.0)
        training.assert_not_called()


class LearnedEvaluationTests(unittest.TestCase):
    def _run(self, *, clock=None, fail_at=None, seconds: float = 60.0, progress=None):
        bundle = _FixtureBundle(clock=clock, fail_at=fail_at)
        with (
            patch.object(evaluation, "_new_session", side_effect=_FixtureSession),
            patch.object(
                evaluation, "_source_manifest", return_value={"test-source": "sha"}
            ),
        ):
            if clock is None:
                return evaluation.evaluate_learned_dialogue(
                    bundle, controls=("trained",), seconds=seconds, progress=progress
                )
            with patch.object(evaluation, "perf_counter", side_effect=lambda: clock[0]):
                return evaluation.evaluate_learned_dialogue(
                    bundle, controls=("trained",), seconds=seconds, progress=progress
                )

    def test_development_fixture_report_is_complete_and_strict_json(self):
        snapshots = []
        report = self._run(progress=snapshots.append)
        self.assertTrue(report["complete"], report["reason"])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["runs"]), 5)
        for run in report["runs"]:
            self.assertEqual(run["metrics"]["correct"], run["planned"])
            self.assertEqual(run["metrics"]["accuracy_all_requested"], 1.0)
        self.assertEqual(sum(run["planned"] for run in report["runs"]), 28)
        self.assertEqual(decode_json(encode_json(report)), report)
        self.assertTrue(all(not run["rows"] for run in snapshots[0]["runs"]))
        self.assertTrue(snapshots[-1]["complete"])

    def test_missing_and_stale_freeze_block_sealed_predictions(self):
        bundle = _FixtureBundle()
        for split in ("held_out", "challenge"):
            with (
                self.subTest(split=split),
                self.assertRaisesRegex(ValueError, "requires"),
            ):
                evaluation.evaluate_learned_dialogue(bundle, split=split)
        self.assertEqual(bundle.understanding.calls, 0)
        freeze = evaluation.capture_source_freeze(bundle)
        bundle.version = 1
        with self.assertRaisesRegex(ValueError, "stale"):
            evaluation.evaluate_learned_dialogue(
                bundle, split="held_out", source_freeze=freeze
            )
        self.assertEqual(bundle.understanding.calls, 0)

    def test_source_change_invalidates_freeze_without_model_prediction(self):
        bundle = _FixtureBundle()
        with patch.object(evaluation, "_source_manifest", return_value={"a": "before"}):
            freeze = evaluation.capture_source_freeze(bundle)
        with (
            patch.object(evaluation, "_source_manifest", return_value={"a": "after"}),
            self.assertRaisesRegex(ValueError, "stale"),
        ):
            evaluation.evaluate_learned_dialogue(
                bundle, split="challenge", source_freeze=freeze
            )
        self.assertEqual(bundle.understanding.calls, 0)

    def test_failed_example_is_not_dropped_from_denominator(self):
        report = self._run(fail_at=1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "completed_with_errors")
        understanding = report["runs"][0]["metrics"]
        self.assertEqual(understanding["requested"], 6)
        self.assertEqual(understanding["attempted"], 6)
        self.assertEqual(understanding["completed"], 5)
        self.assertEqual(understanding["correct"], 5)
        self.assertEqual(understanding["failed"], 1)
        self.assertEqual(understanding["execution_failed"], 1)
        self.assertEqual(understanding["incorrect"], 0)
        self.assertAlmostEqual(understanding["accuracy_all_requested"], 5 / 6)

    def test_returned_timeout_cannot_receive_correct_abstention_credit(self):
        with patch.object(
            evaluation,
            "_call_component",
            return_value={
                "correct": True,
                "abstained": True,
                "result": {"reason": "generation_deadline"},
            },
        ):
            report = self._run()
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "completed_with_errors")
        for run in report["runs"][:-1]:
            self.assertEqual(run["status"], "completed_with_errors")
            metrics = run["metrics"]
            self.assertEqual(metrics["timed_out"], metrics["requested"])
            self.assertEqual(metrics["correct"], 0)
            self.assertEqual(metrics["completed"], 0)

    def test_emergency_session_response_is_a_runtime_failure(self):
        with patch.object(
            _FixtureSession,
            "respond",
            return_value={"action": "error", "reason": "generation_grounding_failed"},
        ):
            report = self._run()
        self.assertFalse(report["complete"])
        metrics = report["runs"][-1]["metrics"]
        self.assertEqual(metrics["requested"], 10)
        self.assertEqual(metrics["execution_failed"], 10)
        self.assertEqual(metrics["correct"], 0)

    def test_semantic_errors_are_counted_separately_from_execution_errors(self):
        with patch.object(
            evaluation, "_call_component", return_value={"correct": False, "result": {}}
        ):
            report = self._run()
        for run in report["runs"]:
            metrics = run["metrics"]
            self.assertEqual(metrics["execution_failed"], 0)
            self.assertEqual(metrics["failed"], metrics["incorrect"])
            self.assertEqual(
                metrics["requested"],
                metrics["correct"]
                + metrics["failed"]
                + metrics["timed_out"]
                + metrics["skipped"],
            )

    def test_deadline_preserves_all_planned_denominators_and_prefix(self):
        report = self._run(clock=[0.0], seconds=0.15)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "timed_out")
        self.assertEqual(len(report["runs"]), 5)
        self.assertEqual(sum(run["planned"] for run in report["runs"]), 28)
        understanding = report["runs"][0]["metrics"]
        self.assertEqual(understanding["completed"], 2)
        self.assertEqual(understanding["requested"], 6)
        self.assertEqual(understanding["not_attempted"], 4)
        self.assertEqual(understanding["skipped"], 4)
        self.assertAlmostEqual(understanding["accuracy_all_requested"], 2 / 6)
        self.assertEqual(understanding["accuracy_observed_prefix"], 1.0)
        self.assertGreater(
            understanding["role_paths_requested"], understanding["role_paths_correct"]
        )
        stratum = understanding["strata"]["development_public"]
        self.assertEqual(stratum["requested"], 6)
        self.assertEqual(stratum["correct"], 2)
        self.assertEqual(decode_json(encode_json(report)), report)

    def test_unavailable_control_is_explicit_not_success(self):
        bundle = _FixtureBundle()
        with (
            patch.object(
                evaluation,
                "_make_control",
                side_effect=NotImplementedError("fixture unavailable"),
            ),
            patch.object(
                evaluation, "_source_manifest", return_value={"test-source": "sha"}
            ),
        ):
            report = evaluation.evaluate_learned_dialogue(
                bundle, controls=("shuffled",)
            )
        self.assertEqual(len(report["unavailable_controls"]), 4)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "unavailable")
        self.assertTrue(all(run["status"] == "unavailable" for run in report["runs"]))
        self.assertTrue(
            all(
                run["metrics"]["accuracy_all_requested"] is None
                for run in report["runs"]
            )
        )

    def test_worker_termination_recovers_full_denominators_and_completed_prefix(self):
        from text_factors.learning import commands, worker

        class HardStop(BaseException):
            pass

        write_checkpoint = worker.atomic_write_json

        def stop_after_two(path, report, **kwargs):
            write_checkpoint(path, report, **kwargs)
            if sum(run["metrics"]["attempted"] for run in report["runs"]) == 2:
                raise HardStop

        def supervisor(command, payload, **kwargs):
            del command, kwargs
            try:
                worker.execute(payload)
            except HardStop:
                return {
                    "status": "timed_out",
                    "elapsed_seconds": 1.0,
                    "error": "fixture supervisor termination",
                }
            raise AssertionError("worker must stop after a saved prefix")

        args = Namespace(
            model="fixture-model.json",
            seconds=1.0,
            split="development",
            freeze=None,
            output=None,
            overwrite=False,
        )
        with (
            patch.object(commands, "read_artifact", return_value={}),
            patch.object(commands, "run_json_worker", side_effect=supervisor),
            patch.object(commands, "_print") as output,
            patch.object(
                worker.ModelBundle, "from_dict", return_value=_FixtureBundle()
            ),
            patch.object(worker, "atomic_write_json", side_effect=stop_after_two),
            patch.object(evaluation, "_source_manifest", return_value={"test": "sha"}),
        ):
            self.assertEqual(commands.learned_evaluate(args), 1)
        report = output.call_args.args[0]
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["termination_reason"], "hard_worker_timed_out")
        self.assertEqual(sum(run["planned"] for run in report["runs"]), 82)
        self.assertEqual(sum(run["metrics"]["correct"] for run in report["runs"]), 2)
        metrics = report["runs"][0]["metrics"]
        self.assertEqual(metrics["accuracy_all_requested"], 2 / 6)
        self.assertEqual(metrics["accuracy_observed_prefix"], 1.0)
        self.assertEqual(decode_json(encode_json(report)), report)

    def test_invalid_inputs_rejected_without_prediction(self):
        bundle = _FixtureBundle()
        invalid: tuple[dict[str, Any], ...] = (
            {"seconds": 0},
            {"seconds": True},
            {"seconds": float("nan")},
            {"seconds": 601},
            {"controls": ()},
            {"controls": []},
            {"controls": ("trained", "trained")},
            {"controls": ([],)},
            {"controls": ("oracle",)},
            {"split": "test"},
            {"progress": 4},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluation.evaluate_learned_dialogue(bundle, **kwargs)
        self.assertEqual(bundle.understanding.calls, 0)


if __name__ == "__main__":
    unittest.main()
