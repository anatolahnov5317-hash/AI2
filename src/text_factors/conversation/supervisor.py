"""Run pure JSON workers behind a hard process deadline and output budget.

The worker returns data only.  Callers must validate and persist a returned state
*after* a completed result, never from inside a timed worker.  This module is a
resource supervisor, not a security sandbox for hostile executable code.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import threading
from contextlib import suppress
from time import perf_counter
from typing import Any, BinaryIO

from .persistence import MAX_STATE_BYTES, _decode_json, _encode_json

_POLL_SECONDS = 0.02
_ERROR_CHARS = 2048


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        # Workers start in their own session/process group.  Kill the group even
        # if its original leader exited: descendants may still hold pipe FDs.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return
    if os.name == "nt":
        # taskkill is the portable Windows fallback; a Windows Job Object would
        # be required to guarantee containment of deliberately escaping children.
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=1.0,
            )
    if process.poll() is None:
        with suppress(ProcessLookupError):
            process.kill()


def run_json_worker(
    command: list[str],
    payload: dict[str, Any],
    *,
    seconds: float,
    max_output_bytes: int = 4_000_000,
) -> dict[str, Any]:
    """Return status/result/elapsed_seconds/error without blocking on workers.

    Input JSON may use up to 16 MB, independently of the session's state budget,
    leaving room for request metadata around a default 4 MB state snapshot.
    ``max_output_bytes`` bounds retained stdout and stderr *together*.  Invalid
    arguments raise ValueError; runtime faults return a structured failure.  The
    deadline covers startup and execution, with bounded process cleanup after
    the kill signal.  POSIX descendants in the worker process group are killed
    on success as well as failure.  Windows descendant cleanup is best effort.
    """
    started = perf_counter()
    if (
        type(command) is not list
        or not command
        or len(command) > 128
        or any(
            type(part) is not str or len(part) > 32768 or "\x00" in part
            for part in command
        )
        or not command[0]
    ):
        raise ValueError("command must be a nonempty bounded list of strings")
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 3600
    ):
        raise ValueError("seconds must be finite and in (0, 3600]")
    if (
        type(max_output_bytes) is not int
        or not 1 <= max_output_bytes <= MAX_STATE_BYTES
    ):
        raise ValueError(f"max_output_bytes must be in [1, {MAX_STATE_BYTES}]")
    encoded = _encode_json(payload, max_bytes=MAX_STATE_BYTES)
    deadline = started + seconds

    def outcome(
        status: str, result: dict[str, Any] | None = None, error: str | None = None
    ) -> dict[str, Any]:
        return {
            "status": status,
            "result": result,
            "elapsed_seconds": round(perf_counter() - started, 6),
            "error": error[:_ERROR_CHARS] if error else None,
        }

    if perf_counter() >= deadline:
        return outcome("timeout", error="worker deadline exceeded before launch")

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total_bytes = 0
    lock = threading.Lock()
    output_limit = threading.Event()
    reader_errors: list[str] = []
    readers: list[threading.Thread] = []
    process: subprocess.Popen[bytes] | None = None
    status = "error"
    return_code: int | None = None

    def drain(name: str, stream: BinaryIO) -> None:
        nonlocal total_bytes
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                with lock:
                    available = max_output_bytes - total_bytes
                    kept = chunk[:available]
                    buffers[name].extend(kept)
                    total_bytes += len(kept)
                    if len(chunk) > available:
                        output_limit.set()
                        return
        except (OSError, ValueError) as exc:
            with lock:
                reader_errors.append(str(exc)[:256])
        finally:
            stream.close()

    try:
        # A regular temporary input file avoids blocking the supervising thread
        # on a worker that never reads stdin.  Worker output remains in bounded
        # RAM buffers, not unbounded temporary files.
        with tempfile.TemporaryFile(mode="w+b") as stdin_file:
            stdin_file.write(encoded)
            stdin_file.seek(0)
            process = subprocess.Popen(
                command,
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                bufsize=0,
                text=False,
                start_new_session=os.name == "posix",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            assert process.stdout is not None and process.stderr is not None
            for name, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                reader = threading.Thread(
                    target=drain,
                    args=(name, stream),
                    name=f"ai2-worker-{name}-{process.pid}",
                    daemon=True,
                )
                reader.start()
                readers.append(reader)
            descendants_stopped = False
            while True:
                if output_limit.is_set():
                    status = "output_limit"
                    break
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    status = "timeout"
                    break
                return_code = process.poll()
                if return_code is not None:
                    if not descendants_stopped:
                        _terminate_tree(process)
                        descendants_stopped = True
                    if all(not reader.is_alive() for reader in readers):
                        status = "completed" if return_code == 0 else "error"
                        break
                output_limit.wait(min(_POLL_SECONDS, remaining))
    except OSError as exc:
        return outcome("error", error=f"worker could not run: {exc}")
    finally:
        if process is not None:
            _terminate_tree(process)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.5)
        for reader in readers:
            reader.join(timeout=0.2)

    if status == "timeout":
        return outcome("timeout", error="worker exceeded its wall-clock deadline")
    if status == "output_limit" or output_limit.is_set():
        return outcome("output_limit", error="worker exceeded its output byte budget")
    if any(reader.is_alive() for reader in readers):
        return outcome("error", error="worker output pipes did not close after cleanup")
    if reader_errors:
        return outcome("error", error=f"worker output read failed: {reader_errors[0]}")
    if status != "completed":
        stderr = bytes(buffers["stderr"]).decode("utf-8", errors="replace")
        return outcome(
            "error", error=f"worker exited with code {return_code}: {stderr}"
        )
    try:
        result = _decode_json(bytes(buffers["stdout"]), max_bytes=max_output_bytes)
    except (ValueError, TypeError) as exc:
        return outcome("error", error=f"worker returned invalid JSON: {exc}")
    if perf_counter() > deadline:
        return outcome(
            "timeout", error="worker result validation exceeded its deadline"
        )
    return outcome("completed", result=result)
