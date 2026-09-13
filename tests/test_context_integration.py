import json
import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

from text_factors.context_affinity import ContextAffinity
from text_factors.context_pipeline import LearnedContextPipeline
from text_factors.evaluation.context_integration import (
    CONTEXTS,
    METHODS,
    ContextIntegrationConfig,
    common_memory_config,
    evaluate_integration_query,
    run_context_integration,
)
from text_factors.evaluation.context_transfer import sdr_trace_metrics
from text_factors.evaluation.transform_learning import (
    BIT_COUNT,
    PartEncoder,
    TransformLearningConfig,
    transform_memory_config,
)
from text_factors.memory import CombinatorialMemory
from text_factors.recognition import memory_encoding_id, recognize_views
from text_factors.transforms import LearnedSDRTransform


class ContextIntegrationTests(unittest.TestCase):
    def small_config(self) -> ContextIntegrationConfig:
        return ContextIntegrationConfig(seeds=(11,), points=16, epochs=1)

    def fixture(self):
        config = self.small_config()
        common = CombinatorialMemory(common_memory_config(config, 11))
        namespace = memory_encoding_id(common)
        recipe = TransformLearningConfig(seeds=(11,), points=16, epochs=1)
        pipelines = {
            method: LearnedContextPipeline(
                common,
                {
                    context: LearnedSDRTransform(transform_memory_config(recipe, 11))
                    for context in CONTEXTS
                },
                input_encoding_id="source-11",
                memory_namespace=namespace,
            )
            for method in METHODS
        }
        affinities = {method: ContextAffinity(namespace) for method in METHODS}
        return PartEncoder(11, 0).codebook[0, 0], common, pipelines, affinities

    def test_ordinary_runtime_predictions_precede_gold_and_ideal(self) -> None:
        source, common, pipelines, affinities = self.fixture()
        events: list[str] = []
        originals = {
            method: pipeline.recognize for method, pipeline in pipelines.items()
        }

        def predictor(method):
            def recognize(bits, **kwargs):
                events.append(method)
                self.assertEqual(set(kwargs), {"observation_id", "source_positions"})
                return originals[method](bits, **kwargs)

            return recognize

        def gold():
            events.append("gold")
            return {context: (source.copy(), "known") for context in CONTEXTS}

        def ideal(*args, **kwargs):
            events.append("ideal")
            return recognize_views(*args, **kwargs)

        with (
            patch.object(
                pipelines["learned"], "recognize", side_effect=predictor("learned")
            ),
            patch.object(
                pipelines["shuffled"], "recognize", side_effect=predictor("shuffled")
            ),
            patch.object(
                pipelines["untrained"], "recognize", side_effect=predictor("untrained")
            ),
            patch(
                "text_factors.evaluation.context_integration.recognize_views",
                side_effect=ideal,
            ),
        ):
            result = evaluate_integration_query(
                source,
                pipelines,
                common,
                {},
                affinities,
                gold,
                observation_id="held-out",
                source_positions=(0,),
                semantic=True,
            )
        self.assertEqual(events, [*METHODS, "gold", "ideal"])
        self.assertEqual(set(result["methods"]), {*METHODS, "ideal"})
        self.assertTrue(
            all(
                not predicted
                for method in METHODS
                for predicted in result["methods"][method]["predictions"].values()
            )
        )

    def test_interrupted_query_retains_predictions_without_constructing_gold(
        self,
    ) -> None:
        source, common, pipelines, affinities = self.fixture()
        calls = [0]
        sink: dict[str, Any] = {}

        def check():
            calls[0] += 1
            if calls[0] == 2:
                raise TimeoutError("stop after first ordinary method")

        with self.assertRaises(TimeoutError):
            evaluate_integration_query(
                source,
                pipelines,
                common,
                {},
                affinities,
                lambda: self.fail("gold must not be constructed"),
                observation_id="partial",
                source_positions=(0,),
                semantic=False,
                check_budget=check,
                trace_sink=sink,
            )
        self.assertEqual(set(sink["methods"]), {"learned"})
        self.assertNotIn("targets", sink)

    def test_small_protocol_has_disjoint_splits_and_frozen_evaluation(self) -> None:
        report = run_context_integration(self.small_config())
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "complete")
        run = report["runs"][0]
        data = run["training_payload"]["dataset"]
        self.assertFalse(set(data["train"]) & set(data["held_out"]))
        self.assertEqual(len(data["train"]), 18)
        self.assertEqual(len(data["held_out"]), 9)
        self.assertEqual(len(run["whole_traces"]), 14)
        self.assertEqual(len(run["part_traces"]), 42)
        self.assertEqual(len(run["unknown_traces"]), 3)
        self.assertEqual(run["training_presentations"], 18 * 3 * 2)
        self.assertEqual(run["common_steps_before"], 54)
        self.assertEqual(run["common_steps_after"], 108)
        for phase in ("before", "after"):
            self.assertTrue(run[f"frozen_components_unchanged_{phase}"])
            self.assertTrue(run[f"common_unchanged_during_{phase}_queries"])
        for method in METHODS:
            self.assertEqual(len(run["affinity_state"][method]["events"]), 54)
            self.assertTrue(
                all(
                    count == (0 if method == "untrained" else 18)
                    for count in run["transform_observation_calls"][method].values()
                )
            )
            self.assertEqual(run["two_instances"][method]["lost_recognized_origins"], 0)
        self.assertEqual(len(run["teaching"]), 9)
        for key, phase, split in (
            ("before/dev", "before", "dev"),
            ("before/test", "before", "test"),
            ("after/test", "after", "test"),
        ):
            for method in METHODS:
                traces = [
                    {
                        "target_bits": target["bits"],
                        "predicted_bits": row["methods"][method]["predictions"][
                            context
                        ],
                    }
                    for row in run["whole_traces"]
                    if row["phase"] == phase and row["split"] == split
                    for context, target in row["targets"].items()
                ]
                self.assertEqual(
                    report["aggregate"][key]["whole_sdr"][method],
                    sdr_trace_metrics(traces, BIT_COUNT),
                )
        mapping = run["training_payload"]["mappings_evaluator_only"]
        self.assertEqual(mapping["primary"], mapping["neighbor"])
        self.assertNotEqual(mapping["primary"], mapping["alternative"])
        wiring = run["training_payload"]["transform_configs"]
        self.assertEqual([wiring[c]["seed"] for c in CONTEXTS], [11, 1020, 2028])

    def test_timeout_is_explicit_and_has_no_aggregate(self) -> None:
        events: list[dict[str, Any]] = []
        with patch(
            "text_factors.evaluation.context_integration.perf_counter",
            side_effect=range(1000),
        ):
            report = run_context_integration(
                replace(self.small_config(), seconds=0.5), progress=events.append
            )
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["aggregate"])
        self.assertEqual(report["completed_seeds"], [])
        self.assertEqual(report["runs"][0]["whole_traces"], [])
        self.assertTrue(events)

    def test_configuration_rejects_invalid_budget_and_counts(self) -> None:
        for values in (
            {"seeds": ()},
            {"seeds": (11, 11)},
            {"points": False},
            {"epochs": 0},
            {"seconds": float("inf")},
            {"seconds": True},
        ):
            with self.assertRaises(ValueError):
                ContextIntegrationConfig(**cast(dict[str, Any], values))


if __name__ == "__main__":
    unittest.main()
