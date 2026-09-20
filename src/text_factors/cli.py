"""Command-line interface for training and inspecting text-factor models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .encoder import DEFAULT_ALPHABET
from .model import TextFactorModel


def _print_json(value: Any) -> None:
    print(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    )


def _experiment_report(args: argparse.Namespace, report: dict[str, Any]) -> int:
    exit_code = 1 if report.get("status") == "incomplete" else 0
    if args.output is None:
        _print_json(report)
        return exit_code
    payload = (
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        stream.write(payload)
    _print_json(
        {
            "saved_to": str(destination),
            "status": report.get("status", "complete"),
            "summary": report.get("aggregate", report.get("metrics", {})),
            "scope": "controlled diagnostic experiment, not evidence of AGI",
        }
    )
    return exit_code


def _check_report_destination(args: argparse.Namespace) -> None:
    if args.output is not None:
        path = Path(args.output)
        if path.is_dir():
            raise ValueError("report destination must be a file, not a directory")
        if path.exists() and not args.overwrite:
            raise ValueError("report already exists; use --overwrite to replace it")


def _evaluate(args: argparse.Namespace) -> int:
    from .evaluation.runner import EvaluationConfig, run_evaluation

    _check_report_destination(args)
    config = EvaluationConfig(
        seeds=tuple(args.seeds),
        points=args.points,
        epochs=args.epochs,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        noise_size=args.noise_size,
        target_fpr=args.target_fpr,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    return _experiment_report(args, run_evaluation(config))


def _microworld(args: argparse.Namespace) -> int:
    from .evaluation.runner import source_manifest
    from .microworld import run_microworld

    _check_report_destination(args)
    report = run_microworld(
        seed=args.seed, episodes=args.episodes, action_budget=args.action_budget
    )
    report["manifest"]["source"] = source_manifest()
    report["schema_version"] = 1
    return _experiment_report(args, report)


def _read_training_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.input).read_text(encoding="utf-8")


def _context_transfer(args: argparse.Namespace) -> int:
    from .evaluation.context_transfer import ContextTransferConfig, run_context_transfer

    _check_report_destination(args)
    config = ContextTransferConfig(
        seeds=tuple(args.seeds),
        points=args.points,
        epochs=args.epochs,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        seconds_per_seed=args.seconds_per_seed,
    )
    return _experiment_report(
        args,
        run_context_transfer(
            config, progress=lambda message: print(message, file=sys.stderr, flush=True)
        ),
    )


def _factor_recovery(args: argparse.Namespace) -> int:
    from .evaluation.factor_recovery import run_factor_recovery

    _check_report_destination(args)
    return _experiment_report(
        args,
        run_factor_recovery(
            tuple(args.seeds),
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            point_count=args.points,
            seconds_per_seed=args.seconds_per_seed,
        ),
    )


def _experience_demo(args: argparse.Namespace) -> int:
    from .evaluation.runner import source_manifest
    from .experience import run_experience_demo

    _check_report_destination(args)
    report = run_experience_demo(seed=args.seed)
    report["source"] = source_manifest()
    report["metrics"] = {
        key: report[key]
        for key in (
            "held_out_exact_match",
            "held_out_bit_precision",
            "held_out_bit_recall",
            "training_observations",
            "held_out_count",
        )
    }
    return _experiment_report(args, report)


def _transform_learning(args: argparse.Namespace) -> int:
    from .evaluation.transform_learning import (
        TransformLearningConfig,
        run_transform_learning,
    )

    _check_report_destination(args)
    config = TransformLearningConfig(
        seeds=tuple(args.seeds),
        points=args.points,
        epochs=args.epochs,
        seconds=args.seconds,
    )
    report = run_transform_learning(
        config, progress=lambda message: print(message, file=sys.stderr, flush=True)
    )
    return _experiment_report(args, report)


def _dialogue_demo(args: argparse.Namespace) -> int:
    from .chat_lab import run_dialogue_demo
    from .evaluation.runner import source_manifest

    _check_report_destination(args)
    report = run_dialogue_demo(
        seed=args.seed,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    report["source"] = source_manifest()
    return _experiment_report(args, report)


def _context_integration(args: argparse.Namespace) -> int:
    from .evaluation.context_integration import (
        ContextIntegrationConfig,
        run_context_integration,
    )

    _check_report_destination(args)
    config = ContextIntegrationConfig(
        seeds=tuple(args.seeds),
        points=args.points,
        epochs=args.epochs,
        seconds=args.seconds,
    )
    report = run_context_integration(
        config, progress=lambda message: print(message, file=sys.stderr, flush=True)
    )
    return _experiment_report(args, report)


def _coactivation_structure(args: argparse.Namespace) -> int:
    from .evaluation.coactivation_structure import (
        CoactivationStructureConfig,
        run_coactivation_structure,
    )

    _check_report_destination(args)
    report = run_coactivation_structure(
        CoactivationStructureConfig(seeds=tuple(args.seeds), seconds=args.seconds),
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    return _experiment_report(args, report)


def _scene_integration(args: argparse.Namespace) -> int:
    from .evaluation.scene_integration import (
        SceneIntegrationConfig,
        run_scene_integration,
    )

    _check_report_destination(args)
    config = SceneIntegrationConfig(
        seeds=tuple(args.seeds),
        points=args.points,
        epochs=args.epochs,
        seconds=args.seconds,
    )
    report = run_scene_integration(
        config, progress=lambda message: print(message, file=sys.stderr, flush=True)
    )
    return _experiment_report(args, report)


def _recognize(args: argparse.Namespace) -> int:
    from .recognition import RecognitionLimits

    model = TextFactorModel.load(args.model)
    result = model.recognize_text(
        args.text,
        stride=args.stride,
        limits=RecognitionLimits(
            max_views=args.max_views,
            max_candidates=args.max_candidates,
            seconds=args.seconds,
        ),
        progress=(lambda message: print(message, file=sys.stderr, flush=True))
        if args.progress
        else None,
    )
    _print_json(result.to_dict())
    return 0 if result.complete else 1


def _chat(args: argparse.Namespace) -> int:
    from .chat_lab import ContextChatSession, make_chat_demo_model
    from .dialogue import GroundedDialogue, GroundingPolicy

    model = (
        make_chat_demo_model(args.seed)
        if args.demo
        else TextFactorModel.load(args.model)
    )
    state = Path(args.state) if args.state else None
    dialogue = (
        GroundedDialogue.load(state) if state is not None and state.exists() else None
    )
    if args.grounding is not None:
        if dialogue is None:
            dialogue = GroundedDialogue(
                model.recognition_encoding_id,
                output_width=model.config.output_bits,
                grounding_policy=GroundingPolicy(mode=args.grounding),
            )
        else:
            dialogue.grounding_policy = replace(
                dialogue.grounding_policy, mode=args.grounding
            )
    session = ContextChatSession(model, dialogue)
    if state is not None:
        state.parent.mkdir(parents=True, exist_ok=True)

    def respond(message: str) -> None:
        try:
            reply = session.handle(message)
            print(reply.text, flush=True)
            if state is not None:
                session.dialogue.save(state)
        except (ValueError, OSError) as error:
            if args.message is not None:
                raise
            print(f"Не удалось обработать: {error}", file=sys.stderr, flush=True)

    if args.message is not None:
        for message in args.message:
            respond(message)
        return 0
    print(
        "AI2 — учебный чат с собственной памятью. "
        "«помощь» — команды, «выход» — завершить.",
        flush=True,
    )
    if args.demo:
        print("Доступны наблюдения ab и cd. Например: покажи ab | cd", flush=True)
    try:
        for line in sys.stdin:
            if line.strip().casefold() in ("выход", "exit", "quit"):
                break
            if line.strip():
                respond(line.rstrip("\n"))
    except KeyboardInterrupt:
        print("Обработка остановлена.", flush=True)
    return 0


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        point_count=args.points,
        seed=args.seed,
        probation_after=args.probation_after,
        stable_after=args.stable_after,
        consolidation_method=args.consolidation,
        coactivation_history_size=args.history_size,
    )


def _add_model_size_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--points",
        type=int,
        default=20_000,
        help="number of random receptive points (default: 20000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--consolidation", choices=("frequency", "coactivation"), default="frequency"
    )
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument(
        "--probation-after",
        type=int,
        default=3,
        help="matching observations required for probation",
    )
    parser.add_argument(
        "--stable-after",
        type=int,
        default=6,
        help="matching observations required for a stable cluster",
    )
    parser.add_argument(
        "--alphabet",
        default=DEFAULT_ALPHABET,
        help="symbols retained by the encoder",
    )


def _train(args: argparse.Namespace) -> int:
    from .real_data.budget import BudgetExceeded, ResourceBudget

    text = _read_training_text(args)
    model = TextFactorModel(_config_from_args(args), alphabet=args.alphabet)
    budget = ResourceBudget(
        max_steps=args.max_windows,
        max_items=args.max_windows,
        max_bytes=args.max_training_bytes,
        max_wall_seconds=args.max_wall_seconds,
        max_artifact_bytes=args.max_artifact_bytes,
        max_clusters_total=args.max_clusters_total,
        checkpoint_every_steps=args.progress_every,
    )

    def progress(event: dict[str, Any]) -> None:
        if args.progress:
            print(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
                file=sys.stderr,
                flush=True,
            )

    try:
        model.fit_text(
            text,
            epochs=args.epochs,
            stride=args.stride,
            budget=budget,
            progress=progress,
        )
    except BudgetExceeded as error:
        report = model.summary(factor_limit=args.top)
        report.update(
            {
                "status": "incomplete",
                "saved_to": None,
                "stop_reason": error.reason,
                "budget": error.snapshot.to_dict(),
                "scope": (
                    "training stopped before the rejected unit; no partial model "
                    "artifact was published"
                ),
            }
        )
        _print_json(report)
        return 1

    destination = model.save(
        args.model,
        max_file_bytes=args.max_artifact_bytes,
    )
    report = model.summary(factor_limit=args.top)
    report.update(
        {
            "status": "complete",
            "saved_to": str(destination),
            "training_budget": {
                "max_windows": args.max_windows,
                "max_training_bytes": args.max_training_bytes,
                "max_wall_seconds": args.max_wall_seconds,
                "max_clusters_total": args.max_clusters_total,
                "max_artifact_bytes": args.max_artifact_bytes,
            },
        }
    )
    _print_json(report)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    model = TextFactorModel.load(args.model)
    results = [
        result.to_dict()
        for result in model.transform_text(args.text, stride=args.stride)
    ]
    factors = model.top_factors(args.top)
    _print_json(
        {
            "results": results,
            "top_factors": [
                {
                    **factor.to_dict(),
                    "evidence": [
                        item.to_dict()
                        for item in model.explain_factor(
                            factor.output_bit, limit=args.evidence
                        )
                    ],
                }
                for factor in factors
            ],
            "memory": model.memory.stats(),
        }
    )
    return 0


def _summary(args: argparse.Namespace) -> int:
    model = TextFactorModel.load(args.model)
    _print_json(model.summary(factor_limit=args.top))
    return 0


def _demo(args: argparse.Namespace) -> int:
    config = ModelConfig(
        point_count=args.points,
        seed=args.seed,
        probation_after=3,
        stable_after=6,
    )
    model = TextFactorModel(config)
    model.fit_text(args.text, epochs=args.epochs, stride=1)
    analysis = model.transform_text(args.text)
    _print_json(
        {
            "input": args.text,
            "last_windows": [item.to_dict() for item in analysis[-5:]],
            "summary": model.summary(factor_limit=args.top),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text-factors",
        description="Sparse online associative memory for recurring text factors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    from .conversation.commands import add_parsers as add_conversation_parsers
    from .learning.commands import add_parsers as add_learning_parsers
    from .observations.commands import add_parsers as add_observation_parsers
    from .observations.learning_commands import (
        add_parsers as add_mention_learning_parsers,
    )

    add_conversation_parsers(subparsers)
    add_learning_parsers(subparsers)
    add_observation_parsers(subparsers)
    add_mention_learning_parsers(subparsers)

    train = subparsers.add_parser("train", help="train a model from UTF-8 text")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="UTF-8 text file")
    source.add_argument("--text", help="literal training text")
    train.add_argument("--model", required=True, help="destination .npz model")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--stride", type=int, default=1)
    train.add_argument("--top", type=int, default=10)
    train.add_argument(
        "--max-windows",
        type=int,
        default=100_000,
        help="maximum successfully processed training windows",
    )
    train.add_argument(
        "--max-training-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="maximum cumulative UTF-8 window bytes processed",
    )
    train.add_argument(
        "--max-wall-seconds",
        type=float,
        default=300.0,
        help="maximum wall-clock seconds for training",
    )
    train.add_argument(
        "--max-clusters-total",
        type=int,
        default=3_000_000,
        help="global cluster budget across all memory points",
    )
    train.add_argument(
        "--max-artifact-bytes",
        type=int,
        default=2 * 1024**3,
        help="maximum persisted NPZ model size",
    )
    train.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="emit a progress checkpoint after this many accepted windows",
    )
    train.add_argument(
        "--progress",
        action="store_true",
        help="write JSON training checkpoints to stderr",
    )
    _add_model_size_options(train)
    train.set_defaults(handler=_train)

    analyze = subparsers.add_parser(
        "analyze", help="analyze text without modifying a saved model"
    )
    analyze.add_argument("--model", required=True, help="source .npz model")
    analyze.add_argument("--text", required=True)
    analyze.add_argument("--stride", type=int, default=1)
    analyze.add_argument("--top", type=int, default=10)
    analyze.add_argument("--evidence", type=int, default=8)
    analyze.set_defaults(handler=_analyze)

    recognize = subparsers.add_parser(
        "recognize", help="read multiple context results with located factor evidence"
    )
    recognize.add_argument("--model", required=True)
    recognize.add_argument("--text", required=True)
    recognize.add_argument("--stride", type=int, default=1)
    recognize.add_argument("--max-views", type=int, default=128)
    recognize.add_argument("--max-candidates", type=int, default=64)
    recognize.add_argument("--seconds", type=float, default=5.0)
    recognize.add_argument("--progress", action="store_true")
    recognize.set_defaults(handler=_recognize)

    chat = subparsers.add_parser(
        "chat", help="teach names to recognized contents in a bounded chat"
    )
    chat_source = chat.add_mutually_exclusive_group(required=True)
    chat_source.add_argument("--model", help="existing unsupervised .npz model")
    chat_source.add_argument(
        "--demo", action="store_true", help="use two unnamed demo sensor patterns"
    )
    chat.add_argument("--seed", type=int, default=7)
    chat.add_argument(
        "--grounding",
        choices=("exact", "factor"),
        help="word matching policy; defaults to saved policy, or exact for new state",
    )
    chat.add_argument(
        "--state", help="save/resume vocabulary and observation references"
    )
    chat.add_argument(
        "--message",
        action="append",
        help="process a command without entering interactive mode; repeatable",
    )
    chat.set_defaults(handler=_chat)

    summary = subparsers.add_parser("summary", help="inspect a saved model")
    summary.add_argument("--model", required=True, help="source .npz model")
    summary.add_argument("--top", type=int, default=10)
    summary.set_defaults(handler=_summary)

    demo = subparsers.add_parser("demo", help="run a small in-memory experiment")
    demo.add_argument(
        "--text",
        default="abracadabra abracadabra abracadabra",
        help="training and analysis text",
    )
    demo.add_argument("--epochs", type=int, default=8)
    demo.add_argument("--points", type=int, default=2_000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--top", type=int, default=10)
    demo.set_defaults(handler=_demo)

    evaluate = subparsers.add_parser(
        "evaluate", help="run frozen-rule text-memory experiments and matched baselines"
    )
    evaluate.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42])
    evaluate.add_argument("--points", type=int, default=128)
    evaluate.add_argument("--epochs", type=int, default=6)
    evaluate.add_argument("--train-size", type=int, default=24)
    evaluate.add_argument("--dev-size", type=int, default=32)
    evaluate.add_argument("--test-size", type=int, default=64)
    evaluate.add_argument("--noise-size", type=int, default=64)
    evaluate.add_argument("--target-fpr", type=float, default=0.05)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=1000)
    evaluate.set_defaults(handler=_evaluate)

    microworld = subparsers.add_parser(
        "microworld",
        help="run a two-hypothesis active-learning pilot (not SDR discovery)",
    )
    microworld.add_argument("--seed", type=int, default=42)
    microworld.add_argument("--episodes", type=int, default=32)
    microworld.add_argument("--action-budget", type=int, default=1)
    microworld.set_defaults(handler=_microworld)

    context_transfer = subparsers.add_parser(
        "context-transfer", help="measure learned SDR transforms on unseen combinations"
    )
    context_transfer.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42])
    context_transfer.add_argument("--points", type=int, default=1024)
    context_transfer.add_argument("--epochs", type=int, default=3)
    context_transfer.add_argument("--train-size", type=int, default=128)
    context_transfer.add_argument("--dev-size", type=int, default=32)
    context_transfer.add_argument("--test-size", type=int, default=64)
    context_transfer.add_argument("--seconds-per-seed", type=float, default=180)
    context_transfer.set_defaults(handler=_context_transfer)

    transform_learning = subparsers.add_parser(
        "transform-learning", help="test pair-learned SDR mappings and new compositions"
    )
    transform_learning.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42])
    transform_learning.add_argument("--points", type=int, default=512)
    transform_learning.add_argument("--epochs", type=int, default=3)
    transform_learning.add_argument("--seconds", type=float, default=45.0)
    transform_learning.set_defaults(handler=_transform_learning)

    dialogue_demo = subparsers.add_parser(
        "dialogue-demo", help="show recognition, taught names and a new arrangement"
    )
    dialogue_demo.add_argument("--seed", type=int, default=7)
    dialogue_demo.set_defaults(handler=_dialogue_demo)

    context_integration = subparsers.add_parser(
        "context-integration",
        help="test learned contexts, common recognition and persistent word grounding",
    )
    context_integration.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 23, 47]
    )
    context_integration.add_argument("--points", type=int, default=512)
    context_integration.add_argument("--epochs", type=int, default=3)
    context_integration.add_argument("--seconds", type=float, default=90.0)
    context_integration.set_defaults(handler=_context_integration)

    coactivation_structure = subparsers.add_parser(
        "coactivation-structure",
        help="diagnose joint structure at matched bit frequencies and fixed orders",
    )
    coactivation_structure.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 211, 307]
    )
    coactivation_structure.add_argument("--seconds", type=float, default=30.0)
    coactivation_structure.set_defaults(handler=_coactivation_structure)

    scene_integration = subparsers.add_parser(
        "scene-integration",
        help="test factor portraits in whole scenes and matched-density surroundings",
    )
    scene_integration.add_argument("--seeds", type=int, nargs="+", default=[59, 71, 89])
    scene_integration.add_argument("--points", type=int, default=512)
    scene_integration.add_argument("--epochs", type=int, default=3)
    scene_integration.add_argument("--seconds", type=float, default=120.0)
    scene_integration.set_defaults(handler=_scene_integration)

    factor_recovery = subparsers.add_parser(
        "factor-recovery", help="measure local factors and matched-marginal controls"
    )
    factor_recovery.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 42])
    factor_recovery.add_argument("--points", type=int, default=64)
    factor_recovery.add_argument("--train-samples", type=int, default=192)
    factor_recovery.add_argument("--test-samples", type=int, default=64)
    factor_recovery.add_argument("--seconds-per-seed", type=float, default=180)
    factor_recovery.set_defaults(handler=_factor_recovery)

    experience_demo = subparsers.add_parser(
        "experience-demo", help="exercise observed outcomes and a bounded context bank"
    )
    experience_demo.add_argument("--seed", type=int, default=42)
    experience_demo.set_defaults(handler=_experience_demo)

    for experiment in (
        evaluate,
        microworld,
        context_transfer,
        transform_learning,
        dialogue_demo,
        context_integration,
        coactivation_structure,
        scene_integration,
        factor_recovery,
        experience_demo,
    ):
        experiment.add_argument(
            "--output", help="save complete JSON traces to this file"
        )
        experiment.add_argument(
            "--overwrite", action="store_true", help="replace existing report"
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2
