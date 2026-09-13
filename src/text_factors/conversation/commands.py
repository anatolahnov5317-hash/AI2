"""User-facing commands; computation is always supervised out of process."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .persistence import (
    atomic_write_json,
    read_json,
    save_session_state,
    state_from_envelope,
)
from .runtime import SessionFileLock, SupervisedConversation
from .schema import ConversationLimits
from .supervisor import run_json_worker

_DEMO = (
    "Привет",
    "Я положил ключ в ящик",
    "Где ключ?",
    "Потом я переложил его в сумку",
    "Где ключ?",
    "Нет, ключ на столе",
    "Где ключ?",
    "Отмени последнее утверждение",
    "Где ключ?",
    "Миша передал ключ Маше",
    "У кого ключ?",
    "Почему?",
)


def _json(value: Any) -> None:
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )


def converse(args: argparse.Namespace) -> int:
    limits = ConversationLimits(turn_seconds=args.turn_seconds)
    destination = Path(args.state).absolute() if args.state else None
    # Parent must be chosen deliberately; do not create arbitrary directory trees.
    if destination is not None and not destination.parent.is_dir():
        raise ValueError("state parent directory does not exist")
    guard = SessionFileLock(destination) if destination else nullcontext()
    with guard:
        state = None
        existed = bool(destination and destination.exists())
        if destination is not None and (existed or destination.is_symlink()):
            state = state_from_envelope(read_json(destination, max_bytes=16_001_024))
        runtime = SupervisedConversation(
            seed=args.seed,
            limits=limits,
            seconds=args.timeout,
            state=state,
        )
        messages = list(_DEMO) if args.demo else args.message
        interactive = messages is None
        if interactive:
            print(
                "AI2 v0.4: ограниченный русский диалог о предметах. "
                "«Помощь» — примеры, /quit — выход. "
                f"Предел вычисления реплики: {args.timeout:g} с.",
                flush=True,
            )
        status = 0
        index = 0
        while True:
            try:
                if interactive:
                    print("Вы: ", end="", flush=True)
                    message = sys.stdin.readline(runtime.limits.max_chars + 2)
                    if not message:
                        break
                    if len(message.rstrip("\r\n")) > runtime.limits.max_chars:
                        raise ValueError(
                            "input line exceeds the configured character limit"
                        )
                    message = message.rstrip("\r\n")
                else:
                    if index >= len(messages):
                        break
                    message = messages[index]
                    index += 1
            except (EOFError, KeyboardInterrupt):
                break
            if message.strip().casefold() in {"/quit", "/exit", "выход"}:
                break
            response = runtime.respond(message)
            if (
                destination is not None
                and runtime.state is not None
                and runtime.last_status == "completed"
            ):
                save_session_state(runtime.state, destination, overwrite=existed)
                existed = True
            if args.json:
                _json(
                    {
                        "input": message,
                        "worker_status": runtime.last_status,
                        "response": response.to_dict(),
                    }
                )
            else:
                if not interactive:
                    print(f"Вы: {message}", flush=True)
                print(f"AI2: {response.text}", flush=True)
            if runtime.last_status != "completed":
                status = 1
                if not interactive:
                    break
        return status


def dialogue_evaluate(args: argparse.Namespace) -> int:
    if (
        type(args.seconds) not in (int, float)
        or not math.isfinite(args.seconds)
        or not 0 < args.seconds <= 300
    ):
        raise ValueError("evaluation seconds must be finite and in (0, 300]")
    destination = Path(args.output).absolute() if args.output else None
    if destination is not None:
        if not destination.parent.is_dir():
            raise ValueError("report parent directory does not exist")
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise ValueError("report destination must be a regular file")
        if destination.exists() and not args.overwrite:
            raise ValueError("report already exists; use --overwrite to replace it")
    with tempfile.TemporaryDirectory(prefix="ai2-dialogue-eval-") as directory:
        checkpoint = Path(directory) / "prefix.json"
        result = run_json_worker(
            [sys.executable, "-m", "text_factors.conversation.worker"],
            {
                "operation": "evaluate",
                "seeds": args.seeds,
                "modes": args.modes,
                "seconds": args.seconds,
                "split": args.split,
                "checkpoint": str(checkpoint),
            },
            seconds=args.seconds + 5.0,
            max_output_bytes=16_000_000,
        )
        if result["status"] == "completed":
            report = result["result"]
        else:
            report = (
                read_json(checkpoint, max_bytes=16_000_000)
                if checkpoint.exists()
                else {
                    "schema_version": 1,
                    "completed_turns": 0,
                    "runs": [],
                }
            )
            report["status"] = "incomplete"
            report["complete"] = False
            report["termination_reason"] = "hard_worker_" + result["status"]
        report["supervisor"] = {
            key: result[key] for key in ("status", "elapsed_seconds", "error")
        }
    if destination is None:
        _json(report)
    else:
        atomic_write_json(
            destination, report, overwrite=args.overwrite, max_bytes=16_000_000
        )
        _json(
            {
                "saved_to": str(destination),
                "status": report["status"],
                "by_mode": report.get("by_mode", {}),
                "scope": (
                    "controlled Russian dialogue; not evidence of general intelligence"
                ),
            }
        )
    return 0 if report["status"] == "completed" else 1


def add_parsers(subparsers: Any) -> None:
    chat = subparsers.add_parser(
        "converse", help="grounded Russian dialogue with hard process deadlines"
    )
    source = chat.add_mutually_exclusive_group()
    source.add_argument(
        "--message", action="append", help="process one message; repeatable"
    )
    source.add_argument(
        "--demo", action="store_true", help="run a small Russian conversation"
    )
    chat.add_argument("--state", help="validated JSON state to create or resume")
    chat.add_argument("--seed", type=int, default=42)
    chat.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="hard seconds including model reconstruction",
    )
    chat.add_argument(
        "--turn-seconds",
        type=float,
        default=2.0,
        help="cooperative inference budget for a new session",
    )
    chat.add_argument(
        "--json",
        action="store_true",
        help="one response and evidence JSON object per line",
    )
    chat.set_defaults(handler=converse)
    evaluate = subparsers.add_parser(
        "dialogue-evaluate",
        help="frozen dialogue tests, ablations and recoverable checkpoints",
    )
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 42])
    evaluate.add_argument(
        "--modes",
        nargs="+",
        choices=("factor", "untrained", "shuffled", "nearest", "oracle"),
        default=["factor", "untrained", "shuffled", "nearest", "oracle"],
    )
    evaluate.add_argument(
        "--split", choices=("development", "held_out", "challenge"), default="held_out"
    )
    evaluate.add_argument("--seconds", type=float, default=120.0)
    evaluate.add_argument(
        "--output", help="destination JSON report; parent directory must exist"
    )
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=dialogue_evaluate)
