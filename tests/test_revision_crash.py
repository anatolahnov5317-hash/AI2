"""Real process death before revision commit, followed by a fresh process restart.

The scene is public development material. This checks the checkpoint boundary;
it is not a language benchmark or a claim about power-loss durability.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from text_factors.learning.hypotheses import digest
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact, save_artifact
from text_factors.learning.session import LearnedSession

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "docs/results/v05_model_42.json"
CUE = "Нет, книга на столе"

# Executable code is a fixed test literal. Paths are passed as argv, never
# interpolated into code or shell commands. The restart uses the same program
# without the interruption hook and saves through the real persistence API.
_CHILD = """
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from text_factors.learning import session as session_module
from text_factors.learning.hypotheses import digest
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact, save_artifact
from text_factors.learning.session import LearnedSession

mode, model_name, checkpoint_name, marker_name, destination_name = sys.argv[1:]
assert mode in {"crash", "restart"}
checkpoint = Path(checkpoint_name)
marker = Path(marker_name)
destination = Path(destination_name)
bundle = ModelBundle.from_dict(read_artifact(Path(model_name), kind="model"))
session = LearnedSession.from_dict(read_artifact(checkpoint, kind="session"), bundle)
original_prepare = session_module.prepare_archive_revision

def interrupt_after_preparation(*args, **kwargs):
    prepared = original_prepare(*args, **kwargs)
    if prepared.applicable:
        assert prepared.complete and prepared.world is not None
        marker.write_text(json.dumps({
            "applicable": prepared.applicable,
            "complete": prepared.complete,
            "live_snapshot_digest": digest(session._snapshot()),
            "prepared_world_snapshot": prepared.world.snapshot_id,
            "live_world_snapshot": session.world.snapshot_id,
            "plan": prepared.to_dict(),
        }, ensure_ascii=False), encoding="utf-8")
        print("revision prepared; terminating before commit", flush=True)
        os._exit(23)
    return prepared

try:
    if mode == "crash":
        with patch.object(session_module, "prepare_archive_revision",
                          side_effect=interrupt_after_preparation):
            response = session.respond("Нет, книга на столе", request_id="revision")
    else:
        response = session.respond("Нет, книга на столе", request_id="revision")
    assert response["action"] == "corrected", response
    save_artifact(session.to_dict(), destination, kind="session",
                  overwrite=destination == checkpoint)
    print("revision committed and saved", flush=True)
finally:
    if mode == "crash":
        marker.with_suffix(".finally").write_text("finally ran", encoding="utf-8")
"""


def without_elapsed(value: Any) -> Any:
    if type(value) is dict:
        return {
            key: without_elapsed(item)
            for key, item in value.items()
            if key != "elapsed_seconds"
        }
    if type(value) is list:
        return [without_elapsed(item) for item in value]
    return value


class RevisionCrashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))
        cls.model_checkpoint = cls.bundle.to_dict()

    def tearDown(self) -> None:
        self.assertEqual(self.bundle.to_dict(), self.model_checkpoint)

    def child(
        self, mode: str, checkpoint: Path, marker: Path, destination: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _CHILD,
                mode,
                str(MODEL),
                str(checkpoint),
                str(marker),
                str(destination),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_hard_exit_before_revision_commit_then_fresh_restart(self) -> None:
        base = LearnedSession(self.bundle)
        scene = (
            ("Маша положила книгу в ящик", "ack"),
            ("Петя положил игрушку в коробку", "ack"),
            ("Она на столе", "clarify"),
            ("Где книга?", "answer"),
        )
        for text, action in scene:
            response = base.respond(text)
            self.assertEqual(response["action"], action, response)
        base_snapshot = base.to_dict()
        self.assertEqual(base.world.revisions, [])
        self.assertEqual(
            base_snapshot["history"][-1]["response"]["assertions"][0]["value"], "ящик"
        )

        with tempfile.TemporaryDirectory(prefix="ai2-revision-crash-") as directory:
            checkpoint = Path(directory) / "session.json"
            marker = Path(directory) / "prepared.json"
            final = Path(directory) / "restarted.json"
            roundtrip = Path(directory) / "roundtrip.json"
            save_artifact(base_snapshot, checkpoint, kind="session")
            saved_bytes = checkpoint.read_bytes()

            crashed = self.child("crash", checkpoint, marker, checkpoint)
            self.assertEqual(crashed.returncode, 23, (crashed.stdout, crashed.stderr))
            self.assertIn(
                "revision prepared; terminating before commit", crashed.stdout
            )
            self.assertTrue(marker.is_file())
            self.assertFalse(marker.with_suffix(".finally").exists())
            reached = json.loads(marker.read_text(encoding="utf-8"))
            self.assertTrue(reached["complete"])
            self.assertTrue(reached["applicable"])
            self.assertEqual(reached["plan"]["target_observation_ids"], ["turn:3"])
            self.assertEqual(reached["plan"]["turn_id"], 5)
            self.assertEqual(reached["live_snapshot_digest"], digest(base_snapshot))
            self.assertEqual(reached["live_world_snapshot"], base.world.snapshot_id)
            self.assertNotEqual(
                reached["prepared_world_snapshot"], base.world.snapshot_id
            )

            self.assertEqual(checkpoint.read_bytes(), saved_bytes)
            recovered_payload = read_artifact(checkpoint, kind="session")
            self.assertEqual(recovered_payload, base_snapshot)
            recovered = LearnedSession.from_dict(recovered_payload, self.bundle)
            self.assertEqual(recovered.to_dict(), base_snapshot)

            expected = base.respond(CUE, request_id="revision")
            self.assertEqual(expected["action"], "corrected", expected)
            self.assertEqual(
                expected["diagnostics"]["revision"]["plan"], reached["plan"]
            )
            uninterrupted = base.to_dict()
            self.assertEqual(uninterrupted["history"][:-1], base_snapshot["history"])

            restarted = self.child("restart", checkpoint, marker, final)
            self.assertEqual(
                restarted.returncode, 0, (restarted.stdout, restarted.stderr)
            )
            self.assertIn("revision committed and saved", restarted.stdout)
            self.assertEqual(checkpoint.read_bytes(), saved_bytes)
            final_payload = read_artifact(final, kind="session")
            self.assertEqual(
                without_elapsed(final_payload), without_elapsed(uninterrupted)
            )
            restored = LearnedSession.from_dict(final_payload, self.bundle)
            self.assertEqual(restored.to_dict(), final_payload)
            self.assertEqual(restored.world.facts(), base.world.facts())
            self.assertEqual(len(restored.world.revisions), 1)
            self.assertEqual(restored.world.events, recovered.world.events)

            save_artifact(restored.to_dict(), roundtrip, kind="session")
            self.assertEqual(roundtrip.read_bytes(), final.read_bytes())
            reloaded = LearnedSession.from_dict(
                read_artifact(roundtrip, kind="session"), self.bundle
            )
            self.assertEqual(reloaded.to_dict(), final_payload)


if __name__ == "__main__":
    unittest.main()
