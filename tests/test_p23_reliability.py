"""Open development checks for crash visibility, backup and competing readers."""

from __future__ import annotations

import os
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
    load_state,
    save_state,
)
from text_factors.real_data.reliability import publish_run, read_run

ROOT = Path(__file__).resolve().parents[1]


def sample_engine(*, extra: bool = False) -> RealDataEngine:
    engine = RealDataEngine()
    source = SourceSlice("source-1", 1, 0, 4, "0" * 64)
    engine.register_evidence(
        EvidenceRoot("root-1", "family-1", "source-1", 1, "default", source)
    )
    for claim_id in ("fact-1", "fact-2") if extra else ("fact-1",):
        engine.add_claim(
            Claim(
                claim_id,
                "located",
                (RoleValue("item", claim_id), RoleValue("place", "shelf")),
                ClaimStatus.ASSERTED,
                source=source,
                evidence_roots=("root-1",),
                model_version="p23-dev",
            )
        )
    return engine


_CRASH_BEFORE_RENAME = """
import os
import sys
from pathlib import Path
from text_factors.real_data import reliability

root, marker = map(Path, sys.argv[1:])
def fail_before_commit(*args):
    marker.write_text('fully-staged', encoding='utf-8')
    os._exit(23)
reliability.os.rename = fail_before_commit
reliability.publish_run(
    root, 'pending', {'state.json': b'complete'}, max_total_bytes=100
)
"""

_CRASH_AFTER_RENAME = """
import os
import sys
from pathlib import Path
from text_factors.real_data import reliability

root, marker = map(Path, sys.argv[1:])
original = os.rename
def exit_after_commit(*args):
    original(*args)
    marker.write_text('published', encoding='utf-8')
    os._exit(25)
reliability.os.rename = exit_after_commit
reliability.publish_run(root, 'committed', {'state.json': b'complete'})
"""

_CRASH_STATE_BEFORE_REPLACE = """
import os
import sys
from pathlib import Path
from text_factors.real_data import persistence
from text_factors.real_data import Claim, ClaimStatus, RoleValue

path, marker = map(Path, sys.argv[1:])
engine, _, model = persistence.load_state(path)
engine.add_claim(Claim('new-claim', 'new-relation',
    (RoleValue('subject', 'new'),), ClaimStatus.ASSERTED))
def fail_before_commit(*args):
    marker.write_text('fully-staged', encoding='utf-8')
    os._exit(24)
persistence.os.replace = fail_before_commit
persistence.save_state(path, engine, model_version=model)
"""


class OperationalPublicationTests(unittest.TestCase):
    def child(self, program: str, *args: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", program, *(str(arg) for arg in args)],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            timeout=30,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_immutable_publication_and_bounded_validated_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = publish_run(
                root, "first", {"state.json": b"ok", "metrics.json": b"{}"}
            )
            self.assertEqual(
                read_run(root, "first"), {"state.json": b"ok", "metrics.json": b"{}"}
            )
            self.assertEqual(
                publish_run(
                    root, "first", {"metrics.json": b"{}", "state.json": b"ok"}
                ),
                run,
            )
            with self.assertRaisesRegex(ValueError, "different artifacts"):
                publish_run(root, "first", {"state.json": b"changed"})
            with self.assertRaisesRegex(ValueError, "max_total_bytes"):
                publish_run(
                    root, "second", {"state.json": b"too-long"}, max_total_bytes=1
                )
            with self.assertRaisesRegex(ValueError, "artifact name"):
                publish_run(root, "second", {"../outside": b"x"})
            with self.assertRaisesRegex(ValueError, "run ID"):
                publish_run(root, "../outside", {"good.json": b"x"})
            (run / "metrics.json").write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_run(root, "first")

    def test_real_process_exit_before_directory_commit_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish_run(root, "previous", {"state.json": b"old"})
            marker = root / "reached.txt"
            stopped = self.child(_CRASH_BEFORE_RENAME, root, marker)
            self.assertEqual(stopped.returncode, 23, (stopped.stdout, stopped.stderr))
            self.assertEqual(marker.read_text(), "fully-staged")
            self.assertEqual(read_run(root, "previous"), {"state.json": b"old"})
            with self.assertRaisesRegex(ValueError, "absent"):
                read_run(root, "pending")
            self.assertTrue(
                any(path.name.startswith(".pending.") for path in root.iterdir())
            )
            publish_run(root, "pending", {"state.json": b"complete"})
            self.assertEqual(read_run(root, "pending"), {"state.json": b"complete"})

    def test_real_process_exit_after_directory_commit_exposes_complete_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "published.txt"
            stopped = self.child(_CRASH_AFTER_RENAME, root, marker)
            self.assertEqual(stopped.returncode, 25, (stopped.stdout, stopped.stderr))
            self.assertEqual(marker.read_text(), "published")
            self.assertEqual(read_run(root, "committed"), {"state.json": b"complete"})

    def test_same_id_concurrent_retry_has_one_complete_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                completed = list(
                    pool.map(
                        lambda _: publish_run(root, "shared", {"result": b"ok"}),
                        range(16),
                    )
                )
            self.assertEqual(set(completed), {root / "shared"})
            self.assertEqual(read_run(root, "shared"), {"result": b"ok"})
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_state_exit_before_replace_and_restore_from_verified_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.json"
            save_state(current, sample_engine(), model_version="p23-dev")
            original = current.read_bytes()
            publish_run(root / "backups", "backup-1", {"state.json": original})
            marker = root / "reached.txt"
            stopped = self.child(_CRASH_STATE_BEFORE_REPLACE, current, marker)
            self.assertEqual(stopped.returncode, 24, (stopped.stdout, stopped.stderr))
            self.assertEqual(marker.read_text(), "fully-staged")
            self.assertEqual(current.read_bytes(), original)
            self.assertEqual(
                load_state(current)[0].to_dict(), sample_engine().to_dict()
            )

            current.write_bytes(b"corrupted main copy")
            with self.assertRaises(ValueError):
                load_state(current)

            # Recovery uses a separate staging destination and validates both
            # the run manifest and the real-data state before publishing.
            backup_bytes = read_run(root / "backups", "backup-1")["state.json"]
            staging = root / "staging.json"
            staging.write_bytes(backup_bytes)
            recovered, contexts, model = load_state(staging)
            self.assertIsNone(contexts)
            self.assertEqual(model, "p23-dev")
            save_state(current, recovered, model_version=model)
            self.assertEqual(current.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
