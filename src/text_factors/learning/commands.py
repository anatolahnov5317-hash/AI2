"""Explicit supervised training, inference and independent evaluation commands."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..conversation.persistence import atomic_write_json, read_json
from ..conversation.runtime import SessionFileLock
from ..conversation.schema import ConversationLimits
from ..conversation.supervisor import run_json_worker
from .persistence import read_artifact, save_artifact
from .runtime import MAX_WIRE_BYTES, WORKER, SupervisedLearnedConversation

_DEMO = (
    "Привет",
    "Маша положила книгу в ящик",
    "Где книга?",
    "Маша передала книгу Пете",
    "У кого книга?",
    "Почему?",
    "Петя обещал передать книгу Маше",
    "У кого книга?",
    "Нет, книга на столе",
    "Где книга?",
    "Отмени последнее утверждение",
    "У кого книга?",
)


def _print(value: Any) -> None:
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )


def _seconds(value: Any, *, cap: float = 300.0) -> float:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or not 0 < value <= cap
    ):
        raise ValueError(f"seconds must be finite and in (0, {cap:g}]")
    return float(value)


def _destination(value: str, overwrite: bool) -> Path:
    path = Path(value).absolute()
    if not path.parent.is_dir():
        raise ValueError("destination parent directory does not exist")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("destination must be a regular file")
    if path.exists() and not overwrite:
        raise ValueError("destination exists; use --overwrite explicitly")
    return path


def learned_train(args: argparse.Namespace) -> int:
    seconds = _seconds(args.seconds)
    path = _destination(args.output, args.overwrite)
    dataset = read_json(Path(args.data), max_bytes=12_000_000) if args.data else None
    result = run_json_worker(
        [sys.executable, "-m", WORKER],
        {
            "operation": "train",
            "seed": args.seed,
            "seconds": seconds,
            "dataset": dataset,
        },
        seconds=seconds + 10.0,
        max_output_bytes=MAX_WIRE_BYTES,
    )
    if result["status"] != "completed":
        _print(
            {
                "status": "incomplete",
                "saved": False,
                "worker_status": result["status"],
                "error": result["error"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        return 1
    model = result["result"]["model"]
    save_artifact(model, path, kind="model", overwrite=args.overwrite)
    _print(
        {
            "status": "completed",
            "saved_to": str(path),
            "fingerprint": result["result"]["fingerprint"],
            "metadata": model["metadata"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )
    return 0


def learned_chat(args: argparse.Namespace) -> int:
    model = read_artifact(Path(args.model), kind="model")
    path = Path(args.state).absolute() if args.state else None
    if path is not None and not path.parent.is_dir():
        raise ValueError("state parent directory does not exist")
    guard = SessionFileLock(path) if path is not None else nullcontext()
    with guard:
        exists = bool(path and path.exists())
        state = (
            read_artifact(path, kind="session")
            if path is not None and (exists or path.is_symlink())
            else None
        )
        runtime = SupervisedLearnedConversation(
            model,
            state=state,
            seconds=_seconds(args.timeout),
            limits=ConversationLimits(
                turn_seconds=_seconds(args.turn_seconds, cap=30.0),
                max_state_bytes=3_000_000,
            ),
        )
        messages = list(_DEMO) if args.demo else args.message
        interactive = messages is None
        if interactive:
            print(
                "AI2 v0.5: обучаемый диалог о ситуациях. /quit — выход. "
                "Неизвестные формулировки могут потребовать полного уточнения.",
                flush=True,
            )
        index = 0
        status = 0
        while True:
            try:
                if interactive:
                    print("Вы: ", end="", flush=True)
                    message = sys.stdin.readline(runtime.limits.max_chars + 2)
                    if not message:
                        break
                    message = message.rstrip("\r\n")
                    if len(message) > runtime.limits.max_chars:
                        raise ValueError("input line exceeds character capacity")
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
                path is not None
                and runtime.state is not None
                and runtime.last_status == "completed"
            ):
                save_artifact(runtime.state, path, kind="session", overwrite=exists)
                exists = True
            if args.json:
                _print(
                    {
                        "input": message,
                        "worker_status": runtime.last_status,
                        "response": response,
                    }
                )
            else:
                if not interactive:
                    print(f"Вы: {message}", flush=True)
                print(f"AI2: {response['text']}", flush=True)
            if runtime.last_status != "completed":
                status = 1
                if not interactive:
                    break
        return status


def learned_export_data(args: argparse.Namespace) -> int:
    from .dialogue_data import training_dialogues
    from .language_data import training_examples
    from .transition_data import training_episodes

    path = _destination(args.output, args.overwrite)
    data = {
        "schema": "ai2-learning-data-v1",
        "understanding": [e.to_dict() for e in training_examples(seed=args.seed)],
        "transitions": [e.to_dict() for e in training_episodes(seed=args.seed)],
        "dialogues": training_dialogues(seed=args.seed),
    }
    atomic_write_json(path, data, overwrite=args.overwrite, max_bytes=12_000_000)
    _print(
        {
            "saved_to": str(path),
            "scope": "annotated training data only; no held-out examples",
        }
    )
    return 0


def learned_freeze(args: argparse.Namespace) -> int:
    path = _destination(args.output, args.overwrite)
    model = read_artifact(Path(args.model), kind="model")
    result = run_json_worker(
        [sys.executable, "-m", WORKER],
        {"operation": "freeze", "model": model},
        seconds=30.0,
        max_output_bytes=MAX_WIRE_BYTES,
    )
    if result["status"] != "completed":
        _print(
            {"status": "incomplete", "worker_status": result["status"], "saved": False}
        )
        return 1
    atomic_write_json(
        path, result["result"], overwrite=args.overwrite, max_bytes=1_000_000
    )
    _print({"status": "completed", "saved_to": str(path)})
    return 0


def learned_evaluate(args: argparse.Namespace) -> int:
    seconds = _seconds(args.seconds)
    path = _destination(args.output, args.overwrite) if args.output else None
    model = read_artifact(Path(args.model), kind="model")
    freeze = read_json(Path(args.freeze), max_bytes=1_000_000) if args.freeze else None
    if args.split != "development" and freeze is None:
        raise ValueError(
            "held_out/challenge evaluation requires an explicit --freeze file"
        )
    report: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="ai2-learned-eval-") as directory:
        checkpoint = Path(directory) / "prefix.json"
        result = run_json_worker(
            [sys.executable, "-m", WORKER],
            {
                "operation": "evaluate",
                "model": model,
                "split": args.split,
                "seconds": seconds,
                "source_freeze": freeze,
                "checkpoint": str(checkpoint),
            },
            seconds=seconds + 5.0,
            max_output_bytes=MAX_WIRE_BYTES,
        )
        if result["status"] == "completed":
            report = result["result"]
        else:
            report = (
                read_json(checkpoint, max_bytes=MAX_WIRE_BYTES)
                if checkpoint.exists()
                else {"completed_cases": 0}
            )
            report.update(
                status="incomplete",
                complete=False,
                termination_reason="hard_worker_" + result["status"],
            )
    report["supervisor"] = {
        key: result[key] for key in ("status", "elapsed_seconds", "error")
    }
    if path is None:
        _print(report)
    else:
        atomic_write_json(
            path, report, overwrite=args.overwrite, max_bytes=MAX_WIRE_BYTES
        )
        _print(
            {
                "saved_to": str(path),
                "status": report.get("status"),
                "metrics": report.get("metrics", report.get("summary", {})),
            }
        )
    return 0 if report.get("status") == "completed" else 1


def add_parsers(subparsers: Any) -> None:
    train = subparsers.add_parser(
        "learned-train", help="fit all four learned components, with a hard deadline"
    )
    train.add_argument("--output", required=True)
    train.add_argument("--data", help="explicit annotated ai2-learning-data-v1 JSON")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--seconds", type=float, default=120.0)
    train.add_argument("--overwrite", action="store_true")
    train.set_defaults(handler=learned_train)
    chat = subparsers.add_parser(
        "learned-chat", help="chat using saved learned parameters; no implicit training"
    )
    chat.add_argument("--model", required=True)
    chat.add_argument("--state")
    source = chat.add_mutually_exclusive_group()
    source.add_argument("--message", action="append")
    source.add_argument("--demo", action="store_true")
    chat.add_argument("--timeout", type=float, default=15.0)
    chat.add_argument("--turn-seconds", type=float, default=2.0)
    chat.add_argument("--json", action="store_true")
    chat.set_defaults(handler=learned_chat)
    export = subparsers.add_parser(
        "learned-export-data", help="export only bundled annotated training examples"
    )
    export.add_argument("--output", required=True)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=learned_export_data)
    freeze = subparsers.add_parser(
        "learned-freeze", help="freeze model, code and evaluation manifest"
    )
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--overwrite", action="store_true")
    freeze.set_defaults(handler=learned_freeze)
    evaluate = subparsers.add_parser(
        "learned-evaluate",
        help="evaluate learned stages; holdout needs an explicit source freeze",
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument(
        "--split",
        choices=("development", "held_out", "challenge"),
        default="development",
    )
    evaluate.add_argument("--freeze")
    evaluate.add_argument("--seconds", type=float, default=120.0)
    evaluate.add_argument("--output")
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=learned_evaluate)
