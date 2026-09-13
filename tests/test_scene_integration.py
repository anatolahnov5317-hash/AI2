import json
import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from text_factors.context_pipeline import LearnedContextPipeline
from text_factors.dialogue import GroundedDialogue
from text_factors.evaluation import scene_integration as experiment
from text_factors.evaluation.context_integration import (
    CONTEXTS,
    METHODS,
    ContextIntegrationConfig,
    common_memory_config,
)
from text_factors.evaluation.context_transfer import sdr_trace_metrics
from text_factors.evaluation.scene_integration import (
    SceneIntegrationConfig,
    evaluate_scene_query,
    run_scene_integration,
)
from text_factors.evaluation.transform_learning import (
    BIT_COUNT,
    PartEncoder,
    TransformLearningConfig,
    hidden_mappings,
    transform_memory_config,
)
from text_factors.memory import CombinatorialMemory
from text_factors.recognition import memory_encoding_id
from text_factors.scene_recognition import FactorSceneReader
from text_factors.transforms import LearnedSDRTransform


class SceneIntegrationTests(unittest.TestCase):
    def small_config(self) -> SceneIntegrationConfig:
        return SceneIntegrationConfig(seeds=(59,), points=16, epochs=1, seconds=30)

    def fixture(self):
        common = CombinatorialMemory(
            common_memory_config(ContextIntegrationConfig(seeds=(59,), points=16), 59)
        )
        namespace = memory_encoding_id(common)
        recipe = TransformLearningConfig(seeds=(59,), points=16, epochs=1)
        pipelines = {
            method: LearnedContextPipeline(
                common,
                {
                    context: LearnedSDRTransform(transform_memory_config(recipe, 59))
                    for context in CONTEXTS
                },
                input_encoding_id="source-59",
                memory_namespace=namespace,
            )
            for method in METHODS
        }
        reader = FactorSceneReader(common, encoding_id=namespace)
        dialogues = {
            name: GroundedDialogue(namespace, output_width=BIT_COUNT)
            for name in ("scene", "legacy")
        }
        source = PartEncoder(59, 0).encode((0, 1, 2))
        return source, pipelines, reader, dialogues

    def test_all_ordinary_reads_and_resolutions_precede_gold(self) -> None:
        source, pipelines, reader, dialogues = self.fixture()
        events: list[str] = []
        original_reader = reader.recognize_views
        original_summary = experiment._scene_summary
        original_legacy = experiment._legacy_summary

        def recognize(views, **kwargs):
            self.assertTrue(all(view.source_positions == () for view in views))
            self.assertTrue(all(view.bit_sources == () for view in views))
            self.assertEqual(set(kwargs), {"total_views", "limits"})
            events.append("read")
            return original_reader(views, **kwargs)

        def scene_summary(*args, **kwargs):
            events.append("scene_words")
            return original_summary(*args, **kwargs)

        def legacy_summary(*args, **kwargs):
            events.append("legacy_words")
            return original_legacy(*args, **kwargs)

        def gold():
            self.assertEqual(events, ["read", "scene_words", "legacy_words"] * 3)
            events.append("gold")
            return {
                context: {"bits": [0, 1, 2], "portraits": ["opaque"], "words": ["a"]}
                for context in CONTEXTS
            }

        with (
            patch.object(reader, "recognize_views", side_effect=recognize),
            patch.object(experiment, "_scene_summary", side_effect=scene_summary),
            patch.object(experiment, "_legacy_summary", side_effect=legacy_summary),
        ):
            row = evaluate_scene_query(
                source,
                pipelines,
                reader,
                dialogues,
                gold,
                observation_id="ordinary",
                diagnose=False,
            )
        self.assertTrue(row["complete"])
        self.assertEqual(set(row["methods"]), {*METHODS, "ideal"})
        self.assertEqual(events[-4:], ["gold", "read", "scene_words", "legacy_words"])
        for method in METHODS:
            self.assertTrue(
                all(not bits for bits in row["methods"][method]["predictions"].values())
            )

    def test_interrupted_query_retains_prediction_prefix_without_gold(self) -> None:
        source, pipelines, reader, dialogues = self.fixture()
        calls = [0]
        sink: dict[str, Any] = {}

        def check():
            calls[0] += 1
            if calls[0] == 2:
                raise TimeoutError("stop after prediction")

        with self.assertRaises(TimeoutError):
            evaluate_scene_query(
                source,
                pipelines,
                reader,
                dialogues,
                lambda: self.fail("gold must not be constructed"),
                observation_id="partial",
                check_budget=check,
                trace_sink=sink,
            )
        self.assertEqual(set(sink["methods"]), {"learned"})
        self.assertEqual(set(sink["methods"]["learned"]["predictions"]), set(CONTEXTS))
        self.assertNotIn("targets", sink)
        self.assertNotIn("complete", sink)

    def test_focal_gold_excludes_neighbors_only_at_evaluation(self) -> None:
        canonical = PartEncoder(59, 1)
        map0, map1 = hidden_mappings(59)
        mappings = {"primary": map0, "neighbor": map0, "alternative": map1}
        portraits = experiment._portrait_ids(canonical)
        for p in range(3):
            for v in range(3):
                case = tuple(v if slot == p else 1 for slot in range(3))
                whole = experiment._targets(
                    case, canonical, mappings, portraits, focal=(p, v)
                )
                isolated = experiment._targets(
                    case, canonical, mappings, portraits, focal=(p, v), isolated=True
                )
                for context, mapping in mappings.items():
                    self.assertEqual(len(whole[context]["bits"]), 18)
                    self.assertEqual(len(isolated[context]["bits"]), 6)
                    self.assertEqual(
                        whole[context]["focal"], isolated[context]["focal"]
                    )
                    slot = mapping.position_order.index(p)
                    self.assertEqual(whole[context]["focal"]["slot"], slot)
                    self.assertTrue(
                        set(isolated[context]["bits"]).issubset(whole[context]["bits"])
                    )

    def test_matched_point_control_filters_without_changing_original(self) -> None:
        def candidate(name, points):
            return {
                "candidate_id": name,
                "portrait_id": name,
                "context": "ctx",
                "active_points": points,
                "words": [name],
                "word_ambiguous": False,
                "ambiguous": False,
            }

        condition = {
            "scene": {"candidates": [candidate("a", 2), candidate("b", 4)]},
            "legacy": {"candidates": []},
            "predictions": {"ctx": [1, 2]},
        }
        targets = {
            "ctx": {"bits": [1, 2], "portraits": ["a", "b"], "words": ["a", "b"]}
        }
        experiment._score_condition(condition, targets, matched_min_points=4)
        self.assertEqual(condition["scores"]["ctx"]["correct_portraits"], ["a", "b"])
        self.assertEqual(
            condition["matched_points"]["scores"]["ctx"]["correct_portraits"], ["b"]
        )
        self.assertEqual(len(condition["scene"]["candidates"]), 2)

    def test_small_smoke_has_balanced_queries_frozen_state_and_recountable_scores(
        self,
    ) -> None:
        report = run_scene_integration(self.small_config())
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "complete")
        run = report["runs"][0]
        self.assertEqual(len(run["whole_traces"]), 9)
        self.assertEqual(len(run["focal_traces"]), 27)
        self.assertEqual(len(run["canonical_controls"]), 7)
        self.assertTrue(run["frozen_components_unchanged"])
        self.assertEqual(run["common_observation_calls"], 54)
        self.assertEqual(run["training_presentations"], 18 * 3 * 2)
        self.assertEqual(len(run["teaching"]), 9)
        data = run["training_payload"]["dataset"]
        self.assertFalse(set(data["train"]) & set(data["held_out"]))
        self.assertEqual(len(data["train"]), 18)
        for method in METHODS:
            traces = [
                {
                    "target_bits": target["bits"],
                    "predicted_bits": row["methods"][method]["predictions"][context],
                }
                for row in run["whole_traces"]
                if row["split"] == "test"
                for context, target in row["targets"].items()
            ]
            self.assertEqual(
                report["aggregate"]["whole"]["test"][method]["sdr"],
                sdr_trace_metrics(traces, BIT_COUNT),
            )
            self.assertEqual(
                report["aggregate"]["whole"]["test"][method]["context_queries"], 15
            )
            self.assertEqual(
                report["aggregate"]["whole"]["test"][method]["matched_points"][
                    "context_queries"
                ],
                15,
            )
            for condition in ("neighbors_0", "neighbors_1", "isolated"):
                result = report["aggregate"]["focal"][condition][method]
                self.assertEqual(result["unique_focal_context_queries"], 27)
                self.assertEqual(result["tp"] + result["fn"], 27 * 6)
        for row in run["focal_traces"]:
            expected_density = 6 if row["condition"] == "isolated" else 18
            self.assertEqual(len(row["source_bits"]), expected_density)
            for condition in row["methods"].values():
                self.assertEqual(
                    condition["scene"]["raw_count"],
                    condition["scene"]["selected_count"],
                )
                self.assertTrue(
                    all(
                        not c["source_positions"]
                        for c in condition["scene"]["candidates"]
                    )
                )
        query_ids = [
            row["query_id"]
            for key in ("whole_traces", "focal_traces")
            for row in run[key]
        ]
        self.assertEqual(len(query_ids), len(set(query_ids)))

    def test_timeout_has_no_aggregate_and_preserves_explicit_status(self) -> None:
        events: list[dict[str, Any]] = []
        with patch.object(experiment, "perf_counter", side_effect=range(1000)):
            report = run_scene_integration(
                replace(self.small_config(), seconds=0.5), progress=events.append
            )
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["aggregate"])
        self.assertEqual(report["runs"][0]["stop_reason"], "global_time_budget")
        self.assertTrue(events)

    def test_invalid_counts_and_budgets_are_rejected(self) -> None:
        for values in (
            {"seeds": ()},
            {"seeds": (59, 59)},
            {"seeds": (True,)},
            {"points": False},
            {"epochs": 0},
            {"seconds": np.inf},
            {"seconds": True},
        ):
            with self.assertRaises(ValueError):
                SceneIntegrationConfig(**cast(dict[str, Any], values))


if __name__ == "__main__":
    unittest.main()
