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
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2
