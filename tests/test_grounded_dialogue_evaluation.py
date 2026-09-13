import json
import unittest
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

from text_factors.conversation.schema import Assertion, TurnResponse
from text_factors.evaluation import grounded_dialogue as evaluation


def _assertion(key, event_id=1):
    topic, subject, relation, value, negated, qualifier = key
    return Assertion(
        subject,
        relation,
        value,
        event_id,
        negated=negated,
        topic=topic,
        qualifier=qualifier,
    )


class _FakeState:
    def __init__(self):
        self._facts: tuple[Assertion, ...] = ()

    def facts(self):
        return self._facts


class _FixtureSession:
    """Evaluator tests only: deterministic authored outputs, not AI2 evidence."""

    def __init__(self, case, clock=None, fail_after=None):
        self.case = case
        self.index = 0
        self.state = _FakeState()
        self.clock = clock
        self.fail_after = fail_after

    def respond(self, text, *, request_id=None):
        if self.index == self.fail_after:
            raise RuntimeError("deliberate fixture failure")
        expected = self.case.turns[self.index]
        if text != expected.text or request_id != f"{self.case.case_id}:{self.index}":
            raise AssertionError("wrong fixture turn")
        self.state._facts = tuple(map(_assertion, expected.state))
        assertions = tuple(map(_assertion, expected.assertions or ()))
        self.index += 1
        if self.clock is not None:
            self.clock[0] += 0.1
        return TurnResponse(
            self.index,
            "Ответ тестовой заглушки.",
            expected.actions[0],
            assertions=assertions,
            reason=expected.reason,
        )


class GroundedDialogueCorpusTests(unittest.TestCase):
    def test_prospective_corpus_hashes_are_pinned(self):
        # Frozen before any engine predictions. Changing one requires an
        # explicit dataset-version/research-protocol decision, not test tuning.
        pinned = {
            "development": (
                "9ee2dd82e861ff459d51176cb1ba9c04b6736e5adf269bd1ddd280d041a624be"
            ),
            "held_out": (
                "d05b5e07f6c0cae7316bd1fd1c112b4def391e2afb2fc5a5ce3fb546e828892f"
            ),
            "challenge": (
                "298ad1cbec0f7892a6d08fcf9f2bb2e33adba8ca1b2344ada0aa099731ea2b5e"
            ),
        }
        for split, expected in pinned.items():
            self.assertEqual(evaluation.corpus_manifest(split)["sha256"], expected)

    def test_deterministic_frozen_seed_independent_corpus(self):
        first = evaluation.make_dialogue_cases()
        second = evaluation.make_dialogue_cases()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 14)
        self.assertEqual(len({case.case_id for case in first}), len(first))
        manifest = evaluation.corpus_manifest()
        self.assertEqual(manifest, evaluation.corpus_manifest())
        self.assertRegex(manifest["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["turn_count"], sum(len(case.turns) for case in first))
        self.assertTrue(manifest["seed_independent"])
        self.assertFalse(manifest["new_language_generalization"])
        for target, field in ((first[0], "case_id"), (first[0].turns[0], "text")):
            with self.assertRaises(FrozenInstanceError):
                setattr(target, field, "changed")

    def test_whole_dialogue_splits_are_disjoint(self):
        dialogues = []
        hashes = []
        for split in evaluation.SPLITS:
            cases = evaluation.make_dialogue_cases(split)
            dialogues.append(
                {tuple(turn.text for turn in case.turns) for case in cases}
            )
            hashes.append(evaluation.corpus_manifest(split)["sha256"])
            self.assertTrue(all(case.case_id.startswith(split + "/") for case in cases))
        self.assertEqual(len(set(hashes)), len(evaluation.SPLITS))
        self.assertFalse(dialogues[0] & dialogues[1])
        self.assertFalse(dialogues[0] & dialogues[2])
        self.assertFalse(dialogues[1] & dialogues[2])

    def test_required_families_and_explicit_semantic_expectations(self):
        cases = {case.category: case for case in evaluation.make_dialogue_cases()}
        self.assertLessEqual(
            {
                "move_pronoun",
                "transfer",
                "role_reversal",
                "negated_location",
                "negated_transfer",
                "negated_move",
                "correction_retraction",
                "retraction",
                "topic_isolation",
                "unknown_object",
                "hypothesis",
                "untaught_predicate",
                "atomic_partial_parse",
                "ambiguous_pronoun",
            },
            set(cases),
        )
        for case in cases.values():
            self.assertGreater(len(case.turns), 1)
            for turn in case.turns:
                self.assertIsInstance(turn.text, str)
                self.assertTrue(turn.text)
                self.assertTrue(turn.actions)
                for key in turn.state:
                    self.assertEqual(len(key), 6)
                    self.assertIs(type(key[4]), bool)
                    self.assertIn(key[2], {"holder", "location"})
        for category in (
            "atomic_partial_parse",
            "ambiguous_pronoun",
            "untaught_predicate",
        ):
            case = cases[category]
            self.assertEqual(case.turns[0].state, case.turns[1].state)
            self.assertEqual(case.turns[1].actions, ("clarify",))
            self.assertEqual(case.turns[1].assertions, ())
        forward = cases["transfer"].turns[-1].assertions
        reverse = cases["role_reversal"].turns[-1].assertions
        assert forward is not None and reverse is not None
        self.assertNotEqual(forward[0][3], reverse[0][3])

    def test_teacher_pairs_are_not_dialogues_or_test_answers(self):
        pairs = evaluation._teaching_pairs()
        cues = {cue for cue, _ in pairs}
        labels = {label for _, label in pairs}
        self.assertEqual(labels, {"locate", "move", "give", "have"})
        for split in evaluation.SPLITS:
            for case in evaluation.make_dialogue_cases(split):
                for turn in case.turns:
                    self.assertNotIn(turn.text, cues)
                    for key in turn.state:
                        self.assertNotIn(key[1], cues)
                        self.assertNotIn(key[3], cues)
        self.assertNotIn("телепортировал", cues)

    def test_oracle_only_maps_the_teacher_cue_not_state_or_answers(self):
        oracle = evaluation._EvaluatorOnlyOracle(7, [("передал", "give")])
        self.assertEqual(oracle.classify(" ПЕРЕДАЛ ").label, "give")
        self.assertIsNone(oracle.classify("телепортировал").label)
        self.assertIsNone(oracle.classify("миша").label)
        self.assertEqual(
            oracle.to_dict()["schema"], "evaluator-only-oracle-do-not-restore"
        )
        conflict = evaluation._EvaluatorOnlyOracle(
            7, [("cue", "give"), ("cue", "move")]
        )
        self.assertIsNone(conflict.classify("cue").label)
        self.assertEqual(set(conflict.classify("cue").candidates), {"give", "move"})

    def test_invalid_split_rejected(self):
        invalid: tuple[Any, ...] = ("", "test", None, 42)
        for split in invalid:
            with self.assertRaises(ValueError):
                evaluation.make_dialogue_cases(split)


class GroundedDialogueScoringTests(unittest.TestCase):
    def test_wrong_claim_is_unsupported_even_if_wrong_state_matches_it(self):
        expected = evaluation.make_dialogue_cases()[0].turns[-1]
        wrong = _assertion(("default", "ключ", "location", "ящик", False, "in"))
        response = TurnResponse(1, "Ключ в ящике.", "answer", assertions=(wrong,))
        scored = evaluation._score_turn(
            expected, response.to_dict(), [wrong.to_dict()], 0.1
        )
        self.assertFalse(scored["state_correct"])
        self.assertFalse(scored["semantic_answer_correct"])
        self.assertFalse(scored["success"])
        self.assertEqual(len(scored["unsupported_assertions"]), 1)
        self.assertEqual(scored["assertions_not_in_actual_state"], [])

    def test_ids_and_wording_do_not_change_semantic_answer(self):
        expected = evaluation.make_dialogue_cases()[0].turns[-1]
        fact = _assertion(expected.state[0], event_id=99)
        response = TurnResponse(
            7, "Иная корректная формулировка.", "answer", assertions=(fact,)
        )
        scored = evaluation._score_turn(
            expected, response.to_dict(), [fact.to_dict()], 0.2
        )
        self.assertTrue(scored["success"])
        self.assertTrue(scored["semantic_answer_correct"])

    def test_abstention_is_not_counted_as_correct_known_answer(self):
        expected = evaluation.make_dialogue_cases()[0].turns[-1]
        fact = _assertion(expected.state[0])
        response = TurnResponse(1, "Не знаю.", "unknown")
        scored = evaluation._score_turn(
            expected, response.to_dict(), [fact.to_dict()], 0.1
        )
        metrics = evaluation._metrics([scored])
        self.assertEqual(metrics["semantic_answers_correct"], 0)
        self.assertEqual(metrics["unnecessary_abstentions"], 1)
        self.assertEqual(metrics["unsupported_assertions"], 0)

    def test_verify_reason_is_part_of_semantics(self):
        case = next(
            case
            for case in evaluation.make_dialogue_cases()
            if case.category == "negated_location"
        )
        expected = case.turns[1]
        fact = _assertion(expected.state[0])
        response = TurnResponse(1, "Да.", "answer", assertions=(fact,), reason="true")
        scored = evaluation._score_turn(
            expected, response.to_dict(), [fact.to_dict()], 0.1
        )
        self.assertFalse(scored["semantic_answer_correct"])


class GroundedDialogueEvaluatorTests(unittest.TestCase):
    def _run_fixture(self, *, seconds=60.0, clock=None, fail_after=None, progress=None):
        cases = iter(evaluation.make_dialogue_cases("development"))

        def new_session(seed, bridge, seconds):
            return _FixtureSession(next(cases), clock=clock, fail_after=fail_after)

        with (
            patch.object(
                evaluation, "_make_bridge", return_value=(object(), {"complete": True})
            ),
            patch.object(evaluation, "_new_session", side_effect=new_session),
        ):
            if clock is None:
                return evaluation.evaluate_grounded_dialogue(
                    seeds=(7,),
                    modes=("oracle",),
                    seconds=seconds,
                    split="development",
                    progress=progress,
                )
            with patch.object(evaluation, "perf_counter", side_effect=lambda: clock[0]):
                return evaluation.evaluate_grounded_dialogue(
                    seeds=(7,),
                    modes=("oracle",),
                    seconds=seconds,
                    split="development",
                    progress=progress,
                )

    def test_completed_run_has_recomputable_traces_and_source_hashes(self):
        snapshots = []
        report = self._run_fixture(progress=snapshots.append)
        self.assertTrue(report["complete"])
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_runs"], report["planned_runs"])
        self.assertEqual(report["completed_dialogues"], report["planned_dialogues"])
        self.assertEqual(report["metrics"]["completed_turns"], report["planned_turns"])
        self.assertEqual(report["metrics"]["state_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["semantic_answer_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["unsupported_assertions"], 0)
        self.assertEqual(report["successful_dialogues"], report["planned_dialogues"])
        self.assertFalse(report["teaching"]["test_dialogues_used_for_fit"])
        self.assertIn("evaluation/grounded_dialogue.py", report["source_sha256"])
        self.assertTrue(
            all(len(value) == 64 for value in report["source_sha256"].values())
        )
        self.assertEqual(snapshots[0]["runs"], [])
        self.assertEqual(snapshots[0]["metrics"]["completed_turns"], 0)
        self.assertFalse(snapshots[0]["complete"])
        self.assertTrue(snapshots[-1]["complete"])
        json.dumps(report, allow_nan=False)

    def test_all_terminal_reports_use_strict_json_primitives(self):
        from text_factors.conversation.persistence import decode_json, encode_json

        reports = (
            self._run_fixture(),
            self._run_fixture(seconds=0.15, clock=[0.0]),
            self._run_fixture(fail_after=1),
        )
        self.assertEqual(
            {report["status"] for report in reports},
            {"completed", "timed_out", "failed"},
        )
        for report in reports:
            with self.subTest(status=report["status"]):
                encoded = encode_json(report)
                self.assertEqual(decode_json(encoded), report)

    def test_global_budget_returns_completed_turn_prefix_without_claiming_pass(self):
        clock = [0.0]
        snapshots = []
        report = self._run_fixture(seconds=0.15, clock=clock, progress=snapshots.append)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "timed_out")
        self.assertEqual(report["metrics"]["completed_turns"], 2)
        self.assertEqual(report["completed_dialogues"], 0)
        self.assertEqual(report["completed_runs"], 0)
        self.assertFalse(report["runs"][0]["cases"][0]["complete"])
        self.assertEqual(snapshots[-1]["status"], "timed_out")
        self.assertEqual(snapshots[-1]["metrics"]["completed_turns"], 2)

    def test_runtime_error_retains_prefix_and_error_location(self):
        report = self._run_fixture(fail_after=1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["metrics"]["completed_turns"], 1)
        self.assertEqual(report["failures"][0]["turn_index"], 1)
        self.assertEqual(report["failures"][0]["exception_type"], "RuntimeError")
        self.assertEqual(report["failures"][0]["case_id"], "development/move_pronoun")

    def test_training_timeout_never_starts_dialogues(self):
        with (
            patch.object(
                evaluation, "_make_bridge", side_effect=TimeoutError("training timeout")
            ),
            patch.object(evaluation, "_new_session") as new_session,
        ):
            report = evaluation.evaluate_grounded_dialogue(
                seeds=(7,), modes=("factor",)
            )
        self.assertEqual(report["status"], "timed_out")
        self.assertEqual(report["metrics"]["completed_turns"], 0)
        self.assertFalse(report["complete"])
        new_session.assert_not_called()

    def test_tiny_deadline_never_trains(self):
        with patch.object(evaluation, "_make_bridge") as make_bridge:
            report = evaluation.evaluate_grounded_dialogue(
                seeds=(7,),
                modes=("oracle",),
                seconds=1e-12,
            )
        self.assertEqual(report["status"], "timed_out")
        make_bridge.assert_not_called()

    def test_contained_engine_failures_are_not_ordinary_abstentions(self):
        for reason, status in (
            ("internal_error", "failed"),
            ("turn_time_budget", "timed_out"),
            ("invalid_input", "failed"),
            ("busy", "failed"),
        ):
            session = _FixtureSession(evaluation.make_dialogue_cases("development")[0])
            session.respond = lambda *args, _reason=reason, **kwargs: TurnResponse(
                1,
                "Не удалось выполнить ход.",
                "clarify",
                complete=False,
                reason=_reason,
            )
            with (
                self.subTest(reason=reason),
                patch.object(evaluation, "_new_session", return_value=session),
            ):
                report = evaluation.evaluate_grounded_dialogue(
                    seeds=(7,),
                    modes=("oracle",),
                    split="development",
                )
                self.assertFalse(report["complete"])
                self.assertEqual(report["status"], status)
                self.assertEqual(report["metrics"]["completed_turns"], 1)
                if status == "failed":
                    self.assertEqual(report["failures"][0]["turn_index"], 0)
                else:
                    self.assertEqual(report["interrupted_at"]["turn_index"], 0)

    def test_invalid_configuration_rejected(self):
        invalid: tuple[dict[str, Any], ...] = (
            {"seeds": ()},
            {"seeds": (True,)},
            {"seeds": (7, 7)},
            {"seeds": (-1,)},
            {"seeds": (2**32,)},
            {"seeds": [7]},
            {"modes": ()},
            {"modes": ("oracle", "oracle")},
            {"modes": ("fallback",)},
            {"modes": ["oracle"]},
            {"seconds": 0},
            {"seconds": float("nan")},
            {"seconds": float("inf")},
            {"seconds": True},
            {"seconds": 601},
            {"progress": 42},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluation.evaluate_grounded_dialogue(**kwargs)


class GroundedDialogueDevelopmentIntegrationTests(unittest.TestCase):
    def test_real_engine_oracle_on_public_development_only(self):
        from text_factors.conversation.persistence import decode_json, encode_json

        # Never exercise held_out/challenge in regression tests: doing so
        # would silently consume their intended prospective evaluation role.
        report = evaluation.evaluate_grounded_dialogue(
            seeds=(7,),
            modes=("oracle",),
            split="development",
            seconds=10.0,
        )
        self.assertTrue(report["complete"], report["reason"])
        self.assertEqual(report["metrics"]["completed_turns"], 10)
        self.assertEqual(report["metrics"]["state_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["semantic_answer_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["unsupported_assertions"], 0)
        self.assertEqual(report["successful_dialogues"], 3)
        self.assertEqual(decode_json(encode_json(report)), report)
        for run in report["runs"]:
            for case in run["cases"]:
                for turn in case["turns"]:
                    for evidence in turn["actual"]["evidence"]:
                        self.assertTrue(
                            evidence["evidence"]["oracle_confined_to_evaluation"]
                        )

    def test_production_restore_rejects_evaluator_oracle_recipe(self):
        from text_factors.conversation.bridge import FactorSemanticBridge

        recipe = evaluation._EvaluatorOnlyOracle(
            7, evaluation._teaching_pairs()
        ).to_dict()
        with self.assertRaises(ValueError):
            FactorSemanticBridge.from_dict(recipe)


if __name__ == "__main__":
    unittest.main()
