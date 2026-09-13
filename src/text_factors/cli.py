"""Command-line interface for training and inspecting text-factor models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
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
    if args.output is None:
        _print_json(report)
        return 0
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
            "summary": report.get("aggregate", report.get("metrics", {})),
            "scope": "controlled diagnostic experiment, not evidence of AGI",
        }
    )
    return 0


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


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        point_count=args.points,
        seed=args.seed,
        probation_after=args.probation_after,
        stable_after=args.stable_after,
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
    text = _read_training_text(args)
    model = TextFactorModel(_config_from_args(args), alphabet=args.alphabet)
    model.fit_text(text, epochs=args.epochs, stride=args.stride)
    destination = model.save(args.model)
    report = model.summary(factor_limit=args.top)
    report["saved_to"] = str(destination)
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

    train = subparsers.add_parser("train", help="train a model from UTF-8 text")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="UTF-8 text file")
    source.add_argument("--text", help="literal training text")
    train.add_argument("--model", required=True, help="destination .npz model")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--stride", type=int, default=1)
    train.add_argument("--top", type=int, default=10)
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

    for experiment in (evaluate, microworld):
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
