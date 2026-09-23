"""Open development checks for bounded dependent-history revision."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from text_factors.real_data.budget import BudgetTracker, ResourceBudget
from text_factors.real_data.dependencies import DependencyGraph
from text_factors.real_data.revision_workflow import (
    ReevaluationDecision,
    RevisionJob,
)
from text_factors.real_data.storage import OperationalStore, StalePublication


def budget(steps: int = 10_000) -> BudgetTracker:
    return BudgetTracker(ResourceBudget(max_steps=steps))


class OpenRevisionWorkflowTests(unittest.TestCase):
    def test_real_sql_barrier_survives_restart_and_rejects_revocation_epoch(
        self,
    ) -> None:
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        original = {"a": 1, "b": 2}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            barrier = OperationalStore(root / "access.sqlite")
            path = root / "revision.json"
            job = RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
                barrier=barrier,
                revision_id="rev-1",
            )
            self.assertEqual(barrier.revision_epoch(), 1)
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))
            job.run_batch(
                lambda node, view: ReevaluationDecision(value=view[node] + 1),
                batch_size=2,
                budget=budget(),
            )
            job.commit(current_state_version="v1", current_original=original)
            recovered = RevisionJob.resume(
                path=path,
                graph=graph,
                original=original,
                state_version="v1",
                barrier=barrier,
            )
            self.assertEqual(recovered.phase, "committed")
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))
            barrier.revoke_source("some-source", 1)
            with self.assertRaises(StalePublication):
                recovered.finish_barrier(
                    durable_state_version="v2",
                    durable_state=recovered.materialized_state(),
                )
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))

    def test_online_barrier_starts_before_checkpoint_and_clears_after_commit(
        self,
    ) -> None:
        class Barrier:
            epoch = 0
            pending: tuple[str, ...] = ()
            revision_id = ""

            def start_revision(
                self,
                revision_id: str,
                affected_claim_ids: tuple[str, ...],
                base_state_version: str,
            ) -> int:
                self.assert_before_write()
                assert base_state_version == "v1"
                self.revision_id = revision_id
                self.pending = affected_claim_ids
                self.epoch += 1
                return self.epoch

            def assert_before_write(self) -> None:
                assert not path.exists()

            def finish_revision(
                self,
                revision_id: str,
                new_state_version: str,
                *,
                expected_epoch: int,
            ) -> int:
                assert revision_id == self.revision_id
                assert expected_epoch == self.epoch
                assert new_state_version == "v2"
                self.pending = ()
                self.epoch += 1
                return self.epoch

            def is_revision_pending(self, revision_id: str) -> bool:
                return self.revision_id == revision_id and bool(self.pending)

            def pending_claim_ids(self) -> tuple[str, ...]:
                return self.pending

        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        original = {"a": 1, "b": 2}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "online.json"
            barrier = Barrier()
            job = RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
                barrier=barrier,
                revision_id="rev-1",
            )
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))
            job.run_batch(
                lambda node, values: ReevaluationDecision(value=values[node] + 1),
                batch_size=1,
                budget=budget(),
            )
            job = RevisionJob.resume(
                path=path,
                graph=graph,
                original=original,
                state_version="v1",
                barrier=barrier,
            )
            job.run_batch(
                lambda node, values: ReevaluationDecision(value=values[node] + 1),
                batch_size=1,
                budget=budget(),
            )
            job.commit(current_state_version="v1", current_original=original)
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))
            with self.assertRaisesRegex(ValueError, "durable state differs"):
                job.finish_barrier(durable_state_version="v2", durable_state=original)
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))
            self.assertEqual(
                job.finish_barrier(
                    durable_state_version="v2", durable_state=job.materialized_state()
                ),
                2,
            )
            self.assertEqual(barrier.pending_claim_ids(), ())

    def test_online_unresolved_revision_remains_blocked(self) -> None:
        class Barrier:
            def start_revision(
                self,
                revision_id: str,
                affected_claim_ids: tuple[str, ...],
                base_state_version: str,
            ) -> int:
                self.pending = affected_claim_ids
                return 1

            def is_revision_pending(self, revision_id: str) -> bool:
                return True

            def pending_claim_ids(self) -> tuple[str, ...]:
                return self.pending

            def finish_revision(
                self,
                revision_id: str,
                new_state_version: str,
                *,
                expected_epoch: int,
            ) -> int:
                raise AssertionError("unresolved claims must never be unblocked")

        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        original = {"a": 1, "b": 2}
        with TemporaryDirectory() as temp:
            barrier = Barrier()
            job = RevisionJob.start(
                path=Path(temp) / "online.json",
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
                barrier=barrier,
                revision_id="rev-unresolved",
            )
            job.run_batch(
                lambda _node, _values: ReevaluationDecision(reason="ambiguous"),
                batch_size=2,
                budget=budget(),
            )
            with self.assertRaisesRegex(ValueError, "unresolved"):
                job.commit(current_state_version="v1", current_original=original)
            self.assertEqual(barrier.pending_claim_ids(), ("a", "b"))
            self.assertEqual(job.phase, "ready")

    def test_two_topics_and_indirect_correction_respect_all_parents(self) -> None:
        graph = DependencyGraph()
        graph.add_dependency("person", "receipt")
        graph.add_dependency("person", "location")
        graph.add_dependency("location", "receipt")
        graph.add_dependency("receipt", "answer")
        graph.add_dependency("other_person", "other_answer")
        original = {
            "person": {"name": "Ира", "version": 1},
            "location": {"owner": "Ира", "place": "полка"},
            "receipt": {"owner": "Ира", "place": "полка"},
            "answer": {"text": "Ира положила документ на полку"},
            "other_person": {"name": "Анна"},
            "other_answer": {"text": "Анна работает"},
        }
        self.assertEqual(
            graph.ordered_affected(("person",)),
            ("person", "location", "receipt", "answer"),
        )
        with TemporaryDirectory() as temp:
            job = RevisionJob.start(
                path=Path(temp) / "job.json",
                graph=graph,
                original=original,
                changed_ids=("person",),
                state_version="real-data-state-5",
            )
            self.assertEqual(
                job.pending_claim_ids,
                ("answer", "location", "person", "receipt"),
            )
            self.assertNotIn("other_answer", job.pending_claim_ids)

            def recompute(node: str, values: Mapping[str, Any]) -> ReevaluationDecision:
                if node == "person":
                    return ReevaluationDecision(value={"name": "Оля", "version": 2})
                if node == "location":
                    self.assertEqual(values["person"]["name"], "Оля")
                    return ReevaluationDecision(
                        value={"owner": values["person"]["name"], "place": "стол"}
                    )
                if node == "receipt":
                    self.assertEqual(values["location"]["owner"], "Оля")
                    return ReevaluationDecision(value=dict(values["location"]))
                return ReevaluationDecision(value={"text": str(values["receipt"])})

            self.assertEqual(
                job.run_batch(recompute, batch_size=2, budget=budget()).processed, 2
            )
            with self.assertRaisesRegex(ValueError, "committed"):
                job.materialized_state()
            recovered = RevisionJob.resume(
                path=job.path,
                graph=graph,
                original=original,
                state_version="real-data-state-5",
            )
            self.assertEqual(
                recovered.run_batch(recompute, batch_size=4, budget=budget()).phase,
                "ready",
            )
            self.assertEqual(recovered.pending_claim_ids, job.pending_claim_ids)
            recovered.commit(
                current_state_version="real-data-state-5", current_original=original
            )
            result = recovered.materialized_state()
            self.assertEqual(result["receipt"], {"owner": "Оля", "place": "стол"})
            self.assertEqual(result["other_answer"], original["other_answer"])
            self.assertEqual(original["person"]["name"], "Ира")
            self.assertEqual(recovered.pending_claim_ids, ())
            self.assertEqual(
                recovered.commit(
                    current_state_version="real-data-state-5",
                    current_original=original,
                ),
                recovered.progress,
            )

    def test_unresolved_parent_preserves_scoped_uncertainty_and_independent_state(
        self,
    ) -> None:
        graph = DependencyGraph()
        graph.add_dependency("report", "decision")
        graph.add_dependency("decision", "historical_answer")
        graph.add_node("independent")
        source = {
            item: {"value": item}
            for item in ("report", "decision", "historical_answer", "independent")
        }
        with TemporaryDirectory() as temp:
            job = RevisionJob.start(
                path=Path(temp) / "pending.json",
                graph=graph,
                original=source,
                changed_ids=("report",),
                state_version="v5",
            )
            evaluated: list[str] = []

            def unknown(node: str, _values: Mapping[str, Any]) -> ReevaluationDecision:
                evaluated.append(node)
                return ReevaluationDecision(reason="unknown_reference")

            job.run_batch(unknown, batch_size=12, budget=budget())
            self.assertEqual(evaluated, ["report"])
            job.commit(current_state_version="v5", current_original=source)
            self.assertEqual(
                job.pending_claim_ids, ("decision", "historical_answer", "report")
            )
            self.assertEqual(job.materialized_state(), source)

    def test_many_uncertain_parents_keep_checkpoint_reason_bounded(self) -> None:
        graph = DependencyGraph()
        parents = tuple(f"parent-{i:02d}-" + "x" * 270 for i in range(20))
        for parent in parents:
            graph.add_dependency(parent, "combined")
        original = {parent: 0 for parent in parents}
        original["combined"] = 0
        with TemporaryDirectory() as temp:
            job = RevisionJob.start(
                path=Path(temp) / "many.json",
                graph=graph,
                original=original,
                changed_ids=parents,
                state_version="v1",
            )
            job.run_batch(
                lambda _node, _values: ReevaluationDecision(reason="unknown"),
                batch_size=21,
                budget=budget(),
            )
            self.assertEqual(job.phase, "ready")
            self.assertEqual(
                RevisionJob.resume(
                    path=job.path,
                    graph=graph,
                    original=original,
                    state_version="v1",
                ).progress.processed,
                21,
            )

    def test_long_history_budget_and_resume_with_small_batches(self) -> None:
        graph = DependencyGraph()
        original = {f"step-{i:04d}": {"index": i, "value": i} for i in range(600)}
        original["different_topic"] = {"value": -99}
        for i in range(1, 600):
            graph.add_dependency(f"step-{i - 1:04d}", f"step-{i:04d}")
        graph.add_node("different_topic")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "history.json"
            job = RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("step-0000",),
                state_version="v1",
            )

            def evaluate(node: str, context: Mapping[str, Any]) -> ReevaluationDecision:
                number = context[node]["index"]
                if number == 0:
                    return ReevaluationDecision(value={"index": 0, "value": 8})
                return ReevaluationDecision(
                    value={
                        "index": number,
                        "value": context[f"step-{number - 1:04d}"]["value"] + 1,
                    }
                )

            stopped = job.run_batch(evaluate, batch_size=50, budget=budget(17))
            self.assertEqual(stopped.stop_reason, "max_steps")
            self.assertEqual(stopped.processed, 17)
            while job.phase == "pending":
                job = RevisionJob.resume(
                    path=path, graph=graph, original=original, state_version="v1"
                )
                job.run_batch(evaluate, batch_size=41, budget=budget())
            self.assertEqual(job.progress.processed, 600)
            job.commit(current_state_version="v1", current_original=original)
            state = job.materialized_state()
            self.assertEqual(state["step-0599"]["value"], 607)
            self.assertEqual(state["different_topic"], {"value": -99})

    def test_refuse_changed_graph_state_and_tampered_checkpoint(self) -> None:
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        original = {"a": 1, "b": 1}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
            )
            with self.assertRaisesRegex(FileExistsError, ""):
                RevisionJob.start(
                    path=path,
                    graph=graph,
                    original=original,
                    changed_ids=("a",),
                    state_version="v1",
                )
            graph.add_dependency("a", "c")
            with self.assertRaisesRegex(ValueError, "another state or graph"):
                RevisionJob.resume(
                    path=path, graph=graph, original=original, state_version="v1"
                )
            fresh = DependencyGraph()
            fresh.add_dependency("a", "b")
            for changed_original, version in (
                ({"a": 2, "b": 1}, "v1"),
                (original, "v2"),
            ):
                with self.assertRaisesRegex(ValueError, "another state or graph"):
                    RevisionJob.resume(
                        path=path,
                        graph=fresh,
                        original=changed_original,
                        state_version=version,
                    )
            raw = json.loads(path.read_text())
            raw["payload"]["cursor"] = 1
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                RevisionJob.resume(
                    path=path, graph=fresh, original=original, state_version="v1"
                )

    def test_competing_workers_cannot_replace_newer_checkpoint(self) -> None:
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        original = {"a": 0, "b": 0}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "shared.json"
            RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
            )
            first = RevisionJob.resume(
                path=path, graph=graph, original=original, state_version="v1"
            )
            second = RevisionJob.resume(
                path=path, graph=graph, original=original, state_version="v1"
            )

            def change(_node: str, _context: Mapping[str, Any]) -> ReevaluationDecision:
                return ReevaluationDecision(value=1)

            first.run_batch(change, batch_size=1, budget=budget())
            with self.assertRaisesRegex(ValueError, "advanced in another worker"):
                second.run_batch(change, batch_size=2, budget=budget())
            self.assertEqual(
                RevisionJob.resume(
                    path=path, graph=graph, original=original, state_version="v1"
                ).progress.processed,
                1,
            )

    def test_cycle_rejected_without_checkpoint(self) -> None:
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("b", "a")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "job.json"
            with self.assertRaisesRegex(ValueError, "cycle"):
                RevisionJob.start(
                    path=path,
                    graph=graph,
                    original={"a": 1, "b": 2},
                    changed_ids=("a",),
                    state_version="v1",
                )
            self.assertFalse(path.exists())

    def test_changed_state_or_graph_blocks_commit_without_exposing_output(self) -> None:
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        original = {"a": 1, "b": 2}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "job.json"
            job = RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
            )
            job.run_batch(
                lambda node, view: ReevaluationDecision(value=view[node] + 10),
                batch_size=2,
                budget=budget(),
            )
            with self.assertRaisesRegex(ValueError, "source changed"):
                job.commit(current_state_version="v2", current_original=original)
            with self.assertRaisesRegex(ValueError, "source changed"):
                job.commit(
                    current_state_version="v1", current_original={"a": 3, "b": 2}
                )
            self.assertEqual(job.phase, "ready")
            with self.assertRaisesRegex(ValueError, "committed"):
                job.materialized_state()
            graph.add_dependency("a", "unrelated")
            with self.assertRaisesRegex(ValueError, "graph changed"):
                job.commit(current_state_version="v1", current_original=original)
            self.assertEqual(job.phase, "ready")

    def test_abrupt_process_death_resumes_before_unpublished_batch(self) -> None:
        graph = DependencyGraph()
        for first, second in (("a", "b"), ("b", "c")):
            graph.add_dependency(first, second)
        original = {"a": 1, "b": 2, "c": 3}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "job.json"
            job = RevisionJob.start(
                path=path,
                graph=graph,
                original=original,
                changed_ids=("a",),
                state_version="v1",
            )
            job.run_batch(
                lambda _node, _values: ReevaluationDecision(value=11),
                batch_size=1,
                budget=budget(),
            )
            program = "\n".join(
                (
                    "import os",
                    "from text_factors.real_data.dependencies import DependencyGraph",
                    "from text_factors.real_data.revision_workflow import RevisionJob",
                    "from text_factors.real_data.budget import BudgetTracker",
                    "from text_factors.real_data.budget import ResourceBudget",
                    "g = DependencyGraph()",
                    "g.add_dependency('a', 'b')",
                    "g.add_dependency('b', 'c')",
                    f"p = {str(path)!r}",
                    "j = RevisionJob.resume(path=p, graph=g,",
                    "                       original={'a': 1, 'b': 2, 'c': 3},",
                    "                       state_version='v1')",
                    "def kill(node, view): os._exit(37)",
                    "tracker = BudgetTracker(ResourceBudget())",
                    "j.run_batch(kill, batch_size=2, budget=tracker)",
                )
            )
            result = subprocess.run(
                [sys.executable, "-c", program],
                env=dict(os.environ),
                capture_output=True,
                check=False,
                timeout=20,
            )
            self.assertEqual(result.returncode, 37)
            resumed = RevisionJob.resume(
                path=path, graph=graph, original=original, state_version="v1"
            )
            self.assertEqual(resumed.progress.processed, 1)
            self.assertEqual(resumed.pending_claim_ids, ("a", "b", "c"))
            resumed.run_batch(
                lambda node, view: ReevaluationDecision(value=view[node] + 10),
                batch_size=2,
                budget=budget(),
            )
            resumed.commit(current_state_version="v1", current_original=original)
            self.assertEqual(resumed.materialized_state(), {"a": 11, "b": 12, "c": 13})
