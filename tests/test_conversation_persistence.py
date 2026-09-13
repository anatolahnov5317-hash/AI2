from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from text_factors.conversation import persistence
from text_factors.conversation.persistence import (
    atomic_write_json,
    decode_json,
    encode_json,
    load_session,
    read_json,
    save_session,
    save_session_state,
    session_envelope,
    state_from_envelope,
)
from text_factors.conversation.schema import ConversationLimits


class _TestSession:
    """A small session double isolates the on-disk contract from the engine."""

    def __init__(self, state: dict[str, Any], *, max_bytes: int = 4_000_000) -> None:
        self.state = state
        self.limits = ConversationLimits(max_state_bytes=max_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai2-grounded-conversation-v1",
            "limits": self.limits.to_dict(),
            "bridge": {"text": self.state["text"]},
            "world": {},
            "turn_count": self.state.get("turn", 0),
            "history": [],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> _TestSession:
        if state.get("schema") != "ai2-grounded-conversation-v1":
            raise ValueError("invalid test session")
        return cls({"turn": state["turn_count"], "text": state["bridge"]["text"]})


class ConversationPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="ai2-persistence-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "session.json"

    def save(self, state: dict[str, Any], *, overwrite: bool = False) -> Path:
        return save_session(
            cast(Any, _TestSession(state)), self.path, overwrite=overwrite
        )

    def test_json_round_trip_is_canonical_and_private(self) -> None:
        value = {"z": [True, None, 1, 1.25], "a": "ключ в ящике"}
        result = atomic_write_json(self.path, value)
        self.assertEqual(result, self.path.absolute())
        self.assertEqual(read_json(self.path), value)
        self.assertTrue(self.path.read_bytes().startswith(b'{"a":'))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_write_requires_explicit_overwrite(self) -> None:
        atomic_write_json(self.path, {"a": 1})
        with self.assertRaises(FileExistsError):
            atomic_write_json(self.path, {"a": 2})
        self.assertEqual(read_json(self.path), {"a": 1})
        atomic_write_json(self.path, {"a": 2}, overwrite=True)
        self.assertEqual(read_json(self.path), {"a": 2})

    def test_session_envelope_checksum_and_lazy_round_trip(self) -> None:
        state = {"turn": 1, "text": "Ключ в ящике."}
        self.save(state)
        envelope = read_json(self.path)
        self.assertEqual(set(envelope), {"kind", "version", "sha256", "state"})
        self.assertEqual(envelope["kind"], "ai2.conversation")
        self.assertEqual(envelope["version"], 1)
        canonical = json.dumps(
            _TestSession(state).to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(envelope["sha256"], hashlib.sha256(canonical).hexdigest())
        engine = types.ModuleType("text_factors.conversation.engine")
        engine.ConversationSession = _TestSession  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {engine.__name__: engine}):
            restored = load_session(self.path)
        self.assertEqual(restored.to_dict(), _TestSession(state).to_dict())
        self.save({"turn": 2, "text": "Теперь в сумке."}, overwrite=True)
        self.assertEqual(read_json(self.path)["state"]["turn_count"], 2)

    def test_unrelated_and_corrupt_snapshots_cannot_be_overwritten(self) -> None:
        for unrelated in ({"other": "valuable"}, {"kind": "ai2.conversation"}):
            atomic_write_json(self.path, unrelated, overwrite=self.path.exists())
            original = self.path.read_bytes()
            with self.assertRaises(ValueError):
                self.save({"turn": 1, "text": "new"}, overwrite=True)
            self.assertEqual(self.path.read_bytes(), original)
        self.path.unlink()
        self.save({"turn": 1, "text": "old"})
        envelope = read_json(self.path)
        envelope["state"]["bridge"]["text"] = "corrupted"
        atomic_write_json(self.path, envelope, overwrite=True)
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.save({"turn": 2, "text": "new"}, overwrite=True)
        self.assertEqual(self.path.read_bytes(), original)

    def test_public_in_memory_json_and_state_apis_do_not_replay_engine(self) -> None:
        state = _TestSession({"turn": 1, "text": "Ключ в сумке."}).to_dict()
        engine = types.ModuleType("text_factors.conversation.engine")
        # No ConversationSession attribute: any accidental numerical load fails.
        with patch.dict(sys.modules, {engine.__name__: engine}):
            envelope = session_envelope(state)
            self.assertEqual(state_from_envelope(envelope), state)
            self.assertEqual(decode_json(encode_json(envelope)), envelope)
            save_session_state(state, self.path)
            save_session_state(state, self.path, overwrite=True)
        self.assertEqual(state_from_envelope(read_json(self.path)), state)
        unrelated = self.path.parent / "valuable.json"
        atomic_write_json(unrelated, {"unrelated": True})
        with self.assertRaises(ValueError):
            save_session_state(state, unrelated, overwrite=True)
        self.assertEqual(read_json(unrelated), {"unrelated": True})

    def test_actual_engine_session_round_trip(self) -> None:
        from text_factors.conversation.engine import ConversationSession

        session = ConversationSession(train_defaults=False)
        session.respond("Привет", request_id="persistence-greeting")
        save_session(session, self.path)
        restored = load_session(self.path)
        self.assertEqual(restored.to_dict(), session.to_dict())

    def test_state_envelope_checks_shape_and_declared_limits_without_replay(
        self,
    ) -> None:
        state = _TestSession({"turn": 1, "text": "old"}).to_dict()
        bad_states = [
            {**state, "extra": 1},
            {**state, "schema": "unknown"},
            {**state, "limits": {**state["limits"], "max_state_bytes": True}},
            {**state, "limits": {**state["limits"], "max_state_bytes": 1}},
            {**state, "bridge": []},
            {**state, "world": []},
            {**state, "history": {}},
            {**state, "history": [{}] * 129},
            {**state, "turn_count": -1},
            {**state, "turn_count": True},
            {**state, "turn_count": 2**53 - 1},
        ]
        for bad in bad_states:
            with self.subTest(state=bad):
                with self.assertRaises(ValueError):
                    session_envelope(bad)
                envelope = {
                    "kind": "ai2.conversation",
                    "version": 1,
                    "sha256": hashlib.sha256(encode_json(bad)).hexdigest(),
                    "state": bad,
                }
                with self.assertRaises(ValueError):
                    state_from_envelope(envelope)
        with self.assertRaises(ValueError):
            session_envelope(state, max_bytes=16_000_001)

    def test_snapshot_rejects_missing_extra_and_invalid_fields(self) -> None:
        self.save({"turn": 1, "text": "old"})
        envelope = read_json(self.path)
        invalid = [
            {key: value for key, value in envelope.items() if key != "sha256"},
            {**envelope, "unexpected": 1},
            {**envelope, "version": True},
            {**envelope, "version": 2},
            {**envelope, "kind": "unrelated"},
            {**envelope, "sha256": "0" * 64},
            {**envelope, "sha256": "G" * 64},
            {**envelope, "state": []},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                persistence._unpack_session(value)

    def test_reader_rejects_duplicates_nonfinite_and_invalid_json(self) -> None:
        invalid = [
            b'{"a":1,"a":2}',
            b'{"a":{"b":1,"b":2}}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":-Infinity}',
            b'{"x":1e999}',
            b'{"x":' + b"9" * 129 + b"}",
            b'{"x":1.' + b"0" * 300 + b"}",
            b"[]",
            b'{"x":1} trailing',
            b'{"x":"\xff"}',
        ]
        for raw in invalid:
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_json(self.path)

    def test_nesting_is_bounded_but_quoted_brackets_are_not_nesting(self) -> None:
        self.path.write_bytes(b'{"a":' + b"[" * 80 + b"0" + b"]" * 80 + b"}")
        with self.assertRaisesRegex(ValueError, "nesting"):
            read_json(self.path)
        valid = {"quoted": ('[\\"}' * 100) + "]" * 100}
        atomic_write_json(self.path, valid, overwrite=True)
        self.assertEqual(read_json(self.path), valid)
        nested: Any = 0
        for _ in range(63):
            nested = [nested]
        atomic_write_json(self.path, {"a": nested}, overwrite=True)
        self.assertEqual(read_json(self.path), {"a": nested})
        with self.assertRaisesRegex(ValueError, "nesting"):
            atomic_write_json(self.path, {"a": [nested]}, overwrite=True)

    def test_invalid_serialization_does_not_touch_destination(self) -> None:
        atomic_write_json(self.path, {"keep": 1})
        cycle: dict[str, Any] = {}
        cycle["self"] = cycle
        for data in (
            {"x": float("nan")},
            {"x": float("inf")},
            {1: "not a string key"},
            {"x": (1, 2)},
            {"x": object()},
            {"x": 10**128},
            {"x": "\ud800"},
            cycle,
        ):
            with self.subTest(data=repr(data)), self.assertRaises(ValueError):
                atomic_write_json(self.path, cast(Any, data), overwrite=True)
            self.assertEqual(read_json(self.path), {"keep": 1})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_byte_budgets_are_enforced_on_read_and_save(self) -> None:
        atomic_write_json(self.path, {"text": "я" * 40})
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "budget"):
            read_json(self.path, max_bytes=20)
        session = _TestSession({"text": "я" * 40}, max_bytes=70)
        with self.assertRaisesRegex(ValueError, "budget"):
            save_session(cast(Any, session), self.path, overwrite=True)
        self.assertEqual(self.path.read_bytes(), original)
        for limit in (0, -1, True, 16_001_025):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                read_json(self.path, max_bytes=limit)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_symlinks_and_nonregular_paths_are_refused(self) -> None:
        other = self.path.parent / "other.json"
        other.write_text('{"keep":1}', encoding="utf-8")
        try:
            self.path.symlink_to(other)
        except OSError:
            self.skipTest("symlink creation not permitted")
        for action in (
            lambda: read_json(self.path),
            lambda: atomic_write_json(self.path, {"bad": 2}, overwrite=True),
            lambda: self.save({"turn": 1, "text": "bad"}, overwrite=True),
            lambda: read_json(self.path.parent),
        ):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(other.read_text(encoding="utf-8"), '{"keep":1}')

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO support required")
    def test_fifo_is_rejected_before_blocking_open(self) -> None:
        os.mkfifo(self.path)
        with self.assertRaises(ValueError):
            read_json(self.path)

    def test_failed_replace_preserves_original_and_cleans_own_temp(self) -> None:
        atomic_write_json(self.path, {"keep": 1})
        with (
            patch.object(
                persistence.os, "replace", side_effect=OSError("disk failure")
            ),
            self.assertRaisesRegex(OSError, "disk failure"),
        ):
            atomic_write_json(self.path, {"new": 2}, overwrite=True)
        self.assertEqual(read_json(self.path), {"keep": 1})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_exclusive_creation_does_not_clobber_racing_writer(self) -> None:
        original_link = os.link

        def competing_link(source: Any, target: Any) -> None:
            self.path.write_text('{"winner":"other"}', encoding="utf-8")
            original_link(source, target)

        with (
            patch.object(persistence.os, "link", side_effect=competing_link),
            self.assertRaises(FileExistsError),
        ):
            atomic_write_json(self.path, {"loser": "us"})
        self.assertEqual(read_json(self.path), {"winner": "other"})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_changed_overwrite_target_is_preserved(self) -> None:
        self.save({"turn": 1, "text": "old"})
        original_fsync = os.fsync

        def competing_write(descriptor: int) -> None:
            self.path.write_text('{"unrelated":"keep this"}', encoding="utf-8")
            original_fsync(descriptor)

        with (
            patch.object(persistence.os, "fsync", side_effect=competing_write),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            self.save({"turn": 2, "text": "new"}, overwrite=True)
        self.assertEqual(read_json(self.path), {"unrelated": "keep this"})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
