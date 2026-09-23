"""P23 integration with P16: task crash, races and up-to-date authority copy."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from text_factors.real_data import (
    Claim,
    ClaimStatus,
    EvidenceRoot,
    RealDataEngine,
    RoleValue,
    SourceSlice,
)
from text_factors.real_data.bundle import load_bundle, save_bundle
from text_factors.real_data.composition import CompositionalEncoder
from text_factors.real_data.contexts import ContextRegistry
from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledEvent,
    LabeledText,
    OpenSemanticModel,
    Span,
)
from text_factors.real_data.reliability import publish_run, read_run
from text_factors.real_data.storage import OperationalStore, StalePublication

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_RESULT = "P23_PRIVATE_SOURCE_BYTES_MUST_NOT_PERSIST"
_BEGIN_THEN_EXIT = """
import os
import sys
from pathlib import Path
from text_factors.real_data.storage import OperationalStore

path, marker = map(Path, sys.argv[1:])
store = OperationalStore(path)
assert store.begin_task('resume-me', 'request-sha', 'user').status == 'pending'
marker.write_text('pending committed', encoding='utf-8')
os._exit(31)
"""


def _semantic_model() -> OpenSemanticModel:
    text = "Анна дала ключ."
    return OpenSemanticModel().fit(
        (
            LabeledText(
                text,
                (
                    IdentifiedMention("person", "anna", Span(0, 4)),
                    IdentifiedMention("thing", "key", Span(10, 14)),
                ),
                (
                    LabeledEvent(
                        "transfer",
                        Span(5, 9),
                        (("actor", "person"), ("object", "thing")),
                    ),
                ),
            ),
        )
    )


class OperationalRecoveryTests(unittest.TestCase):
    def test_crashed_pending_task_restarted_then_competing_identical_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "authority.db"
            marker = root / "child.txt"
            interrupted = subprocess.run(
                [sys.executable, "-c", _BEGIN_THEN_EXIT, str(db), str(marker)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                interrupted.returncode, 31, (interrupted.stdout, interrupted.stderr)
            )
            self.assertEqual(marker.read_text(), "pending committed")
            store = OperationalStore(db)
            pending = store.task_status("resume-me")
            assert pending is not None
            self.assertEqual(pending.status, "pending")

            def resume(_: int) -> str:
                current = OperationalStore(db)
                current.begin_task("resume-me", "request-sha", "user")
                return current.complete_task(
                    "resume-me",
                    "request-sha",
                    "user",
                    {"result": 1, "source_excerpt": PRIVATE_RESULT},
                ).status

            with ThreadPoolExecutor(max_workers=12) as pool:
                statuses = list(pool.map(resume, range(24)))
            self.assertEqual(set(statuses), {"complete"})
            completed = store.task_status("resume-me")
            assert completed is not None
            self.assertEqual(completed.status, "complete")
            with sqlite3.connect(db) as conn:
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM tasks WHERE request_id='resume-me'"
                    ).fetchone()[0],
                    1,
                )
            self.assertNotIn(PRIVATE_RESULT.encode(), db.read_bytes())
            with self.assertRaises(StalePublication):
                store.complete_task("resume-me", "request-sha", "user", {"result": 2})

            def independent(index: int) -> str:
                current = OperationalStore(db)
                request_id = f"independent-{index}"
                current.begin_task(request_id, "same-shape", "user")
                return current.complete_task(
                    request_id, "same-shape", "user", {"result": index}
                ).status

            with ThreadPoolExecutor(max_workers=12) as pool:
                independent_statuses = list(pool.map(independent, range(16)))
            self.assertEqual(independent_statuses, ["complete"] * 16)
            with sqlite3.connect(db) as conn:
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status='complete'"
                    ).fetchone()[0],
                    17,
                )

    def test_recovery_from_copy_replays_current_revocation_over_older_bundle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = SourceSlice("source", 1, 0, 4, "0" * 64, "private")
            engine = RealDataEngine()
            engine.register_evidence(
                EvidenceRoot("root", "family", "source", 1, "private", source)
            )
            engine.add_claim(
                Claim(
                    "fact",
                    "location",
                    (RoleValue("item", "key"), RoleValue("place", "shelf")),
                    ClaimStatus.OBSERVED,
                    source=source,
                    evidence_roots=("root",),
                    model_version="model-v1",
                )
            )
            bundle = root / "old-bundle.json"
            save_bundle(
                bundle,
                engine,
                contexts=ContextRegistry(width=128),
                semantics=_semantic_model(),
                encoder=CompositionalEncoder(width=128),
                model_version="model-v1",
                corpus_version="corpus-v1",
                policy_version="policy-v1",
            )
            authority_path = root / "authority.db"
            authority = OperationalStore(authority_path)
            authority.grant_scope("user", "private")
            authority.revoke_source("source", 1)
            authoritative_backup = root / "authority-copy.db"
            with (
                sqlite3.connect(authority_path) as active,
                sqlite3.connect(authoritative_backup) as backup,
            ):
                active.backup(backup)

            publish_run(
                root / "copies",
                "recovery-copy",
                {
                    "bundle.json": bundle.read_bytes(),
                    "authority.db": authoritative_backup.read_bytes(),
                },
            )
            # Simulate loss of active files after a completed, verified backup.
            bundle.unlink()
            authority_path.unlink()
            recovered_bytes = read_run(root / "copies", "recovery-copy")
            recovered_root = root / "recovered"
            recovered_root.mkdir()
            recovered_bundle = recovered_root / "bundle.json"
            recovered_db = recovered_root / "authority.db"
            recovered_bundle.write_bytes(recovered_bytes["bundle.json"])
            recovered_db.write_bytes(recovered_bytes["authority.db"])
            restored_authority = OperationalStore(recovered_db)
            self.assertEqual(restored_authority.revoked_sources(), (("source", 1),))
            loaded = load_bundle(
                recovered_bundle,
                expected_model_version="model-v1",
                expected_corpus_version="corpus-v1",
                expected_policy_version="policy-v1",
                authority=restored_authority,
            )
            self.assertFalse(
                loaded.engine.receipt(
                    question_id="where",
                    answer_text="Ключ на полке",
                    claim_ids=("fact",),
                    model_version="model-v1",
                    allowed_scopes=("private",),
                ).complete
            )


if __name__ == "__main__":
    unittest.main()
