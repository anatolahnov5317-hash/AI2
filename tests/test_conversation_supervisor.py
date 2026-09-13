from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

from text_factors.conversation.persistence import atomic_write_json, read_json
from text_factors.conversation.supervisor import run_json_worker


class ConversationSupervisorTests(unittest.TestCase):
    def run_worker(
        self,
        code: str,
        payload: dict[str, Any] | None = None,
        *,
        seconds: float = 3.0,
        max_output_bytes: int = 4_000_000,
    ) -> dict[str, Any]:
        return run_json_worker(
            [sys.executable, "-c", code],
            payload or {},
            seconds=seconds,
            max_output_bytes=max_output_bytes,
        )

    def test_json_round_trip(self) -> None:
        result = self.run_worker(
            "import sys,json; p=json.load(sys.stdin); "
            "print(json.dumps({'echo':p},ensure_ascii=False))",
            {"text": "Где ключ?", "value": [1, True, None]},
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["echo"]["text"], "Где ключ?")
        self.assertIsNone(result["error"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)

    def test_hang_and_busy_loop_are_killed_by_deadline(self) -> None:
        for code in ("import time; time.sleep(60)", "while True: pass"):
            with self.subTest(code=code):
                started = time.perf_counter()
                result = self.run_worker(code, seconds=0.25)
                self.assertEqual(result["status"], "timeout")
                self.assertIsNone(result["result"])
                self.assertLess(time.perf_counter() - started, 2.0)

    def test_large_stdin_does_not_block_a_worker_that_never_reads(self) -> None:
        result = self.run_worker(
            "import time; time.sleep(60)", {"input": "x" * 800_000}, seconds=0.3
        )
        self.assertEqual(result["status"], "timeout")
        self.assertLess(result["elapsed_seconds"], 2.0)

    def test_wire_input_allows_metadata_beyond_default_four_mb_state(self) -> None:
        result = self.run_worker(
            "import sys,json; p=json.load(sys.stdin); "
            "print(json.dumps({'received_chars':len(p['state'])}))",
            {"state": "x" * 4_000_001, "text": "Где ключ?"},
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"], {"received_chars": 4_000_001})

    def test_stdout_and_stderr_floods_are_bounded(self) -> None:
        for descriptor in (1, 2):
            code = f"import os\nwhile True:\n os.write({descriptor}, b'x' * 65536)\n"
            with self.subTest(descriptor=descriptor):
                result = self.run_worker(code, seconds=3, max_output_bytes=10_000)
                self.assertEqual(result["status"], "output_limit")
                self.assertIsNone(result["result"])
                self.assertLess(result["elapsed_seconds"], 2.0)
                self.assertLessEqual(len(result["error"]), 2048)

    def test_stdout_and_stderr_share_budget(self) -> None:
        result = self.run_worker(
            "import os; os.write(1,b' ' * 700); os.write(2,b'w' * 700)",
            max_output_bytes=1000,
        )
        self.assertEqual(result["status"], "output_limit")

    def test_invalid_json_duplicates_and_nonfinite_values_are_rejected(self) -> None:
        invalid = (
            "print('not JSON')",
            "print('[]')",
            'print(\'{"a":1,"a":2}\')',
            "print('{\"x\":NaN}')",
            "print('{\"x\":1e999}')",
            "print('{\"x\":'+ '['*100 + '0' + ']'*100 +'}')",
            "import os; os.write(1,b'\\xff')",
        )
        for code in invalid:
            with self.subTest(code=code):
                result = self.run_worker(code)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["result"])
                self.assertIn("invalid JSON", result["error"])

    def test_nonzero_exit_is_not_success_even_with_valid_json(self) -> None:
        result = self.run_worker(
            "import sys; print('{}'); "
            "print('worker failed',file=sys.stderr); sys.exit(7)"
        )
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["result"])
        self.assertIn("code 7", result["error"])
        self.assertIn("worker failed", result["error"])

    def test_error_messages_are_bounded(self) -> None:
        result = self.run_worker(
            "import sys; print('x'*10000,file=sys.stderr); sys.exit(1)"
        )
        self.assertEqual(result["status"], "error")
        self.assertLessEqual(len(result["error"]), 2048)

    def test_missing_executable_is_a_structured_error(self) -> None:
        result = run_json_worker(
            ["/this-ai2-test-executable-does-not-exist"], {}, seconds=1.0
        )
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["result"])

    def test_invalid_arguments_are_rejected_without_launch(self) -> None:
        for seconds in (0, -1, float("nan"), float("inf"), True, 3601):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.run_worker("print('{}')", seconds=seconds)
        for cap in (0, -1, True, 16_000_001):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.run_worker("print('{}')", max_output_bytes=cap)
        for command in ([], "python", [""], ["python", "\x00"]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                run_json_worker(cast(Any, command), {}, seconds=1)
        with self.assertRaises(ValueError):
            self.run_worker("print('{}')", {"bad": float("nan")})
        with self.assertRaises(ValueError):
            self.run_worker("print('{}')", {"big": "x" * 16_000_001})

    def test_timeout_does_not_modify_parent_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai2-supervisor-state-") as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"turn": 1, "fact": "original"})
            original = path.read_bytes()
            result = self.run_worker(
                "import sys,json,time; data=json.load(sys.stdin); "
                "data['turn']=2; time.sleep(60); print(json.dumps(data))",
                read_json(path),
                seconds=0.25,
            )
            if result["status"] == "completed":
                atomic_write_json(path, result["result"], overwrite=True)
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "posix", "POSIX process group semantics")
    def test_descendants_are_killed_on_timeout_and_parent_exit(self) -> None:
        for parent_hangs in (False, True):
            with (
                self.subTest(parent_hangs=parent_hangs),
                tempfile.TemporaryDirectory(prefix="ai2-descendant-test-") as directory,
            ):
                marker = Path(directory) / "should-not-exist"
                child_code = (
                    "import time,pathlib; time.sleep(0.7); "
                    f"pathlib.Path({str(marker)!r}).write_text('survived')"
                )
                code = (
                    "import subprocess,sys,time; "
                    f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                    + ("time.sleep(60)" if parent_hangs else "print('{}',flush=True)")
                )
                result = self.run_worker(code, seconds=0.3)
                self.assertEqual(
                    result["status"], "timeout" if parent_hangs else "completed"
                )
                time.sleep(0.8)
                self.assertFalse(
                    marker.exists(), "a worker descendant survived cleanup"
                )

    def test_no_worker_reader_threads_remain_after_repeated_timeouts(self) -> None:
        for _ in range(3):
            result = self.run_worker("import time; time.sleep(60)", seconds=0.1)
            self.assertEqual(result["status"], "timeout")
        remaining = [
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("ai2-worker-")
        ]
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
