"""P22 leakage, grouping, uncertainty and interrupted-run contracts."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from text_factors.real_data.independent_evaluation import (
    EvaluationCase,
    EvaluationPlan,
    EvidencePointer,
    FrozenPrediction,
    evaluate_open,
    wilson_95,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = EvaluationPlan(
    "candidate",
    "baseline",
    "ablation",
    frozenset({"train-history"}),
    frozenset({"old"}),
    frozenset({"new"}),
    bootstrap_repeats=200,
)


def _case(group: str, *, case_id: str = "query", wrong: bool = False) -> EvaluationCase:
    claim = "new|actor=alice|target=room"
    prediction = "invented" if wrong else claim
    return EvaluationCase(
        case_id,
        group,
        "new",
        "development",
        (claim,),
        True,
        {
            "candidate": FrozenPrediction(
                (prediction,),
                (EvidencePointer(prediction, "document", 1, "a" * 64),),
            ),
            "baseline": FrozenPrediction(()),
            "ablation": FrozenPrediction(()),
        },
    )


class IndependentP22EvaluationTests(unittest.TestCase):
    def test_independent_groups_not_queries_set_denominator(self) -> None:
        rows = (_case("history-a", case_id="1"), _case("history-a", case_id="2"))
        report = evaluate_open(rows, PLAN)
        candidate = report["per_method"]["candidate"]
        self.assertEqual(candidate["cases"], 2)
        self.assertEqual(candidate["independent_groups"], 1)
        self.assertEqual(candidate["accepted_groups"], 1)
        self.assertEqual(
            report["paired_candidate_minus_baseline"]["independent_groups"], 1
        )
        self.assertEqual(report["split"], "development")

    def test_bad_claim_and_absent_source_are_different_failures(self) -> None:
        wrong = _case("history-a", wrong=True)
        source = _case("history-b", case_id="other")
        bare = EvaluationCase(
            source.case_id,
            source.history_group_id,
            source.relation_id,
            source.split,
            source.gold_claims,
            True,
            {**source.predictions, "candidate": FrozenPrediction(source.gold_claims)},
        )
        report = evaluate_open((wrong, bare), PLAN)
        candidate = report["per_method"]["candidate"]
        self.assertEqual(candidate["accepted_cases"], 2)
        self.assertEqual(candidate["exact_cases"], 1)
        self.assertEqual(candidate["unverified_or_wrong_accepted_groups"], 2)
        self.assertEqual(candidate["useful_groups"], 0)
        self.assertEqual(candidate["accepted_group_risk_upper_95"], 1.0)

    def test_no_known_negative_from_missing_or_incomplete_gold(self) -> None:
        case = _case("history-a")
        with self.assertRaisesRegex(ValueError, "completely labeled"):
            evaluate_open(
                (
                    EvaluationCase(
                        case.case_id,
                        case.history_group_id,
                        case.relation_id,
                        case.split,
                        (),
                        False,
                        case.predictions,
                    ),
                ),
                PLAN,
            )

    def test_sealed_calibration_future_overlap_and_unpaired_are_rejected(self) -> None:
        base = _case("history-a")
        for split in ("sealed_test", "calibration", "future_stream", "train"):
            with (
                self.subTest(split=split),
                self.assertRaisesRegex(ValueError, "open development"),
            ):
                evaluate_open(
                    (
                        EvaluationCase(
                            base.case_id,
                            base.history_group_id,
                            base.relation_id,
                            split,
                            base.gold_claims,
                            True,
                            base.predictions,
                        ),
                    ),
                    PLAN,
                )
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            evaluate_open((_case("train-history"),), PLAN)
        with self.assertRaisesRegex(ValueError, "paired"):
            evaluate_open(
                (
                    EvaluationCase(
                        base.case_id,
                        base.history_group_id,
                        base.relation_id,
                        base.split,
                        base.gold_claims,
                        True,
                        {"candidate": base.predictions["candidate"]},
                    ),
                ),
                PLAN,
            )
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            evaluate_open((base, base), PLAN)

    def test_small_zero_error_sample_does_not_clear_one_percent_gate(self) -> None:
        result = evaluate_open((_case("a", case_id="1"), _case("b", case_id="2")), PLAN)
        upper = result["per_method"]["candidate"]["accepted_group_risk_upper_95"]
        self.assertGreater(upper, 0.01)
        self.assertGreater(
            result["paired_candidate_minus_baseline"][
                "win_fraction_wilson_95_non_ties"
            ][0],
            0,
        )
        self.assertIsNone(wilson_95(0, 0))

    def test_interrupted_attempt_is_included_alongside_subsequent_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "all-runs.sqlite"
            export = Path(temporary) / "all-runs.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_p22_open.py"),
                "--ledger",
                str(ledger),
                "--export",
                str(export),
            ]
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            first = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            report = json.loads(first.stdout)
            self.assertEqual(report["per_method"]["candidate"]["exact_cases"], 4)
            self.assertEqual(
                report["per_method"]["without_new_relation_ablation"]["exact_cases"], 2
            )
            self.assertEqual(
                report["per_method"]["exact_text_control"]["exact_cases"], 0
            )
            with sqlite3.connect(ledger) as connection:
                connection.execute(
                    "INSERT INTO attempts (id,started_at,status,source_revision) "
                    "VALUES (?,?,?,?)",
                    ("interrupted", "2026-09-23T00:00:00Z", "started", "{}"),
                )
            second = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            attempts = json.loads(second.stdout)["ledger_attempts"]
            self.assertEqual(len(attempts), 3)
            self.assertEqual([item["status"] for item in attempts].count("started"), 1)
            self.assertEqual([item["status"] for item in attempts].count("finished"), 2)
            exported = json.loads(export.read_text(encoding="utf-8"))
            self.assertEqual(len(exported["attempts"]), 3)
            self.assertEqual(
                {item["status"] for item in exported["attempts"]},
                {"started", "finished"},
            )
            self.assertTrue(
                all(
                    item["result"]["cost"]["wall_seconds"] >= 0
                    for item in exported["attempts"]
                    if item["status"] == "finished"
                )
            )

    def test_concurrent_attempts_export_every_committed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.sqlite"
            exported = Path(temporary) / "all-runs.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_p22_open.py"),
                "--ledger",
                str(ledger),
                "--export",
                str(exported),
            ]
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            processes = [
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            for process in processes:
                _, error = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, error)
            attempts = json.loads(exported.read_text(encoding="utf-8"))["attempts"]
            with sqlite3.connect(ledger) as connection:
                expected = connection.execute(
                    "SELECT count(*) FROM attempts"
                ).fetchone()
            self.assertEqual(expected, (2,))
            self.assertEqual(len(attempts), 2)
            self.assertEqual({item["status"] for item in attempts}, {"finished"})


if __name__ == "__main__":
    unittest.main()
