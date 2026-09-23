"""Validate, optionally run, and score a public raw-dialogue diagnostic.

Without --runner, only an archive of existing predictions is scored. With a
factory, the runner streams one past raw event at a time to ConnectedDialogue,
archives all predictions, then scores them. `expected` is evaluator-only and
never passed to the model, reviewer or session factory. This is an OPEN
diagnostic and does not implement P22/P24 pilot gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/dialogue_research_v1.json"
SPLITS = {"train_open", "development_open", "evaluation_open_held_out"}
ACTIONS = {"answer", "ask_clarification", "abstain"}
PUBLIC_FIELDS = {
    "ingest": ("kind", "source_id", "version", "text"),
    "query": ("kind", "query_id", "question"),
    "revoke_source_family": ("kind", "source_id"),
    "restart": ("kind",),
}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def validate_fixture(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return frozen per-query gold and accessible documents before each query."""
    if fixture.get("schema") != "ai2-dialogue-research-open-v1":
        raise ValueError("wrong fixture schema")
    families = fixture.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("fixture requires nonempty families")
    gold: dict[str, dict[str, Any]] = {}
    seen_families: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("family must be an object")
        family_id = family.get("family_id")
        if (
            not isinstance(family_id, str)
            or not family_id
            or family_id in seen_families
        ):
            raise ValueError(f"duplicate or invalid family: {family_id!r}")
        seen_families.add(family_id)
        split = family.get("split")
        if split not in SPLITS:
            raise ValueError(f"invalid split for {family_id}")
        events = family.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"no events for {family_id}")
        documents: dict[tuple[str, int], str] = {}
        latest: dict[str, int] = {}
        revoked: set[str] = set()
        primary = 0
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"non-object event in {family_id}")
            kind = event.get("kind")
            if kind == "ingest":
                source_id, version, text = (
                    event.get("source_id"),
                    event.get("version"),
                    event.get("text"),
                )
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or type(version) is not int
                    or version != latest.get(source_id, 0) + 1
                    or not isinstance(text, str)
                    or not text
                    or source_id in revoked
                ):
                    raise ValueError(f"invalid source/version in {family_id}")
                documents[source_id, version] = text
                latest[source_id] = version
            elif kind == "revoke_source_family":
                source_id = event.get("source_id")
                if source_id not in latest or source_id in revoked:
                    raise ValueError(f"invalid revocation in {family_id}")
                revoked.add(source_id)
            elif kind == "restart":
                if set(event) != {"kind"}:
                    raise ValueError(f"restart has unexpected data in {family_id}")
            elif kind == "query":
                query_id = event.get("query_id")
                expected = event.get("expected")
                if (
                    not isinstance(query_id, str)
                    or not query_id
                    or query_id in gold
                    or not isinstance(event.get("question"), str)
                    or not isinstance(expected, dict)
                    or expected.get("action") not in ACTIONS
                ):
                    raise ValueError(f"invalid or duplicated query in {family_id}")
                if expected.get("primary", False):
                    primary += 1
                evidence = expected.get("evidence")
                if not isinstance(evidence, list):
                    raise ValueError(f"missing evidence array for {query_id}")
                if expected["action"] == "answer" and (
                    not isinstance(expected.get("fact"), dict) or not evidence
                ):
                    raise ValueError(f"answer requires fact and evidence: {query_id}")
                if expected["action"] != "answer" and (evidence or "fact" in expected):
                    raise ValueError(f"non-answer has gold fact: {query_id}")
                if expected["action"] == "ask_clarification" and not expected.get(
                    "required_slot"
                ):
                    raise ValueError(f"clarification needs slot: {query_id}")
                active = {
                    (source_id, version): text
                    for (source_id, version), text in documents.items()
                    if source_id not in revoked and latest[source_id] == version
                }
                for pointer in evidence:
                    if not isinstance(pointer, dict):
                        raise ValueError(f"non-object evidence for {query_id}")
                    source_id = pointer.get("source_id")
                    version = pointer.get("version")
                    if not isinstance(source_id, str) or type(version) is not int:
                        raise ValueError(f"invalid source pointer for {query_id}")
                    key = (source_id, version)
                    quote = pointer.get("quote")
                    if (
                        key not in active
                        or not isinstance(quote, str)
                        or not quote
                        or quote not in active[key]
                    ):
                        raise ValueError(f"nonexistent/stale quote for {query_id}")
                gold[query_id] = {
                    "family_id": family_id,
                    "split": split,
                    "question": event["question"],
                    "expected": expected,
                    "active": active,
                }
            else:
                raise ValueError(f"unknown event kind {kind!r} in {family_id}")
        if primary != 1:
            raise ValueError(f"expected one primary query in {family_id}: {primary}")
    return gold


def iter_public_events(fixture: dict[str, Any]):
    """Yield raw events in order, without gold or future events in each payload.

    Feed exactly one returned payload to inference and collect its reply before
    requesting the next item from this generator. Never send the fixture object
    or its family['events'] list to the model.
    """
    for family in fixture["families"]:
        for event in family["events"]:
            visible = {key: event[key] for key in PUBLIC_FIELDS[event["kind"]]}
            yield family["family_id"], family["split"], visible


def _evidence_matches(
    expected: dict[str, Any], prediction: dict[str, Any], active: dict
) -> bool:
    actual = prediction.get("evidence", [])
    if not isinstance(actual, list):
        return False
    expected_keys = {
        (item["source_id"], item["version"]) for item in expected["evidence"]
    }
    actual_keys: set[tuple[str, int]] = set()
    for item in actual:
        if not isinstance(item, dict):
            return False
        source_id = item.get("source_id")
        version = item.get("version")
        if not isinstance(source_id, str) or type(version) is not int:
            return False
        key = (source_id, version)
        quote = item.get("quote")
        if (
            key not in expected_keys
            or key not in active
            or not isinstance(quote, str)
            or not quote
            or quote not in active[key]
        ):
            return False
        actual_keys.add(key)
    return actual_keys == expected_keys


def _score_row(
    item: dict[str, Any], prediction: dict[str, Any] | None
) -> dict[str, Any]:
    expected = item["expected"]
    if prediction is None:
        return {
            "mechanical_match": False,
            "reviewed_success": False,
            "review": "missing",
            "known_unsafe_accepted": False,
        }
    actual_action = prediction.get("action")
    mechanical = actual_action == expected["action"] and _evidence_matches(
        expected, prediction, item["active"]
    )
    if expected["action"] == "answer":
        mechanical = mechanical and prediction.get("fact") == expected["fact"]
        # Answer text may contain extra assertions even when fact and quote match.
        mechanical = mechanical and bool(prediction.get("answer_text"))
        adjudication = prediction.get("human_supported")
    else:
        mechanical = mechanical and not prediction.get("fact")
        if expected["action"] == "ask_clarification":
            mechanical = mechanical and (
                prediction.get("required_slot") == expected["required_slot"]
            )
        adjudication = prediction.get("human_appropriate")
    if type(adjudication) is not bool:
        review = "pending"
    else:
        review = "passed" if adjudication else "failed"
    accepted = actual_action == "answer"
    known_unsafe = accepted and (
        not mechanical or adjudication is False or expected["action"] != "answer"
    )
    return {
        "mechanical_match": bool(mechanical),
        "reviewed_success": bool(mechanical and adjudication is True),
        "review": review,
        "known_unsafe_accepted": known_unsafe,
    }


def _summary(
    gold: dict[str, dict[str, Any]],
    scored: list[dict],
    selected_splits: frozenset[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in sorted(SPLITS):
        if split not in selected_splits:
            output[split] = {
                "status": "not_run",
                "available_primary_families": sum(
                    item["split"] == split and item["expected"].get("primary", False)
                    for item in gold.values()
                ),
                "primary_reviewed_success": None,
                "useful_supported_primary": None,
            }
            continue
        all_rows = [row for row in scored if row["split"] == split]
        primary = [row for row in all_rows if row["primary"]]
        answerable = [
            row
            for row in primary
            if gold[row["query_id"]]["expected"]["action"] == "answer"
        ]
        accepted = [row for row in primary if row["actual_action"] == "answer"]
        latencies = sorted(
            row["latency_ms"] for row in all_rows if row["latency_ms"] is not None
        )
        memories = [
            row["peak_rss_kib"] for row in all_rows if row["peak_rss_kib"] is not None
        ]
        output[split] = {
            "status": "recorded",
            "primary_families": len(primary),
            "primary_reviewed_success": sum(row["reviewed_success"] for row in primary),
            "primary_mechanical_match_only": sum(
                row["mechanical_match"] for row in primary
            ),
            "primary_pending_review": sum(
                row["review"] == "pending" for row in primary
            ),
            "answerable_primary_families": len(answerable),
            "useful_supported_primary": sum(
                row["reviewed_success"] for row in answerable
            ),
            "accepted_primary": len(accepted),
            "known_unsafe_accepted_primary": sum(
                row["known_unsafe_accepted"] for row in accepted
            ),
            "all_questions": len(all_rows),
            "all_reviewed_success": sum(row["reviewed_success"] for row in all_rows),
            "latency_reported": len(latencies),
            "latency_median_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": (
                latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
            ),
            "peak_rss_reported": len(memories),
            "peak_rss_kib_max": max(memories) if memories else None,
        }
    return output


def score_predictions(
    gold: dict[str, dict[str, Any]], predictions: dict[str, Any]
) -> dict[str, Any]:
    if predictions.get("schema") != "ai2-dialogue-open-predictions-v1":
        raise ValueError("wrong prediction schema")
    methods = predictions.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("predictions require one or more methods")
    scored_methods = []
    method_names: set[str] = set()
    for method in methods:
        if not isinstance(method, dict) or not isinstance(method.get("name"), str):
            raise ValueError("method requires name")
        name = method["name"]
        if name in method_names:
            raise ValueError(f"duplicate method {name}")
        method_names.add(name)
        costs = method.get("costs")
        if isinstance(costs, dict) and "selected_splits" in costs:
            raw_splits = costs["selected_splits"]
            if (
                not isinstance(raw_splits, list)
                or not raw_splits
                or any(split not in SPLITS for split in raw_splits)
            ):
                raise ValueError(f"invalid selected_splits for {name}")
            selected_splits = frozenset(raw_splits)
        else:
            selected_splits = frozenset(SPLITS)
        predictions_for_method = method.get("predictions")
        if not isinstance(predictions_for_method, list):
            raise ValueError(f"method {name} requires predictions array")
        rows: dict[str, dict[str, Any]] = {}
        for row in predictions_for_method:
            if not isinstance(row, dict):
                raise ValueError(f"invalid prediction for {name}")
            query_id = row.get("query_id")
            if query_id not in gold or query_id in rows:
                raise ValueError(f"unknown/duplicate query_id for {name}: {query_id}")
            if gold[query_id]["split"] not in selected_splits:
                raise ValueError(
                    f"query outside selected splits for {name}: {query_id}"
                )
            if row.get("family_id") != gold[query_id]["family_id"]:
                raise ValueError(f"wrong family for {name}/{query_id}")
            for metric in ("latency_ms", "peak_rss_kib"):
                value = row.get(metric)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"invalid {metric} for {name}/{query_id}")
            rows[query_id] = row
        scored = []
        for query_id, item in gold.items():
            if item["split"] not in selected_splits:
                continue
            row = rows.get(query_id)
            result = _score_row(item, row)
            scored.append(
                {
                    "query_id": query_id,
                    "family_id": item["family_id"],
                    "split": item["split"],
                    "primary": item["expected"].get("primary", False),
                    "expected_action": item["expected"]["action"],
                    "actual_action": row.get("action") if row else "missing",
                    "latency_ms": row.get("latency_ms") if row else None,
                    "peak_rss_kib": row.get("peak_rss_kib") if row else None,
                    **result,
                }
            )
        scored_methods.append(
            {
                "name": name,
                "declared_status": method.get("status", "unknown"),
                "recorded_predictions": len(rows),
                "recorded_costs": method.get("costs"),
                "summary": _summary(gold, scored, selected_splits),
                "rows": scored,
            }
        )
    return {
        "schema": "ai2-dialogue-open-score-v1",
        "attempt_id": predictions.get("attempt_id"),
        "methods": scored_methods,
        "qualification": (
            "Open synthetic diagnostics, no P22/P24 gate; unknown human review "
            "does not count as a supported answer."
        ),
    }


def _load_callback(name: str) -> Callable[..., Any]:
    module_name, separator, function_name = name.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("callbacks need module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise ValueError(f"callback is not callable: {name}")
    return function


def run_connected_open(
    fixture: dict[str, Any],
    *,
    factory: Callable[..., Any],
    reviewer: Callable[..., Any] | None,
    restarter: Callable[..., Any] | None,
    ledger_path: Path,
    output_path: Path,
    fixture_sha256: str,
    splits: frozenset[str],
) -> dict[str, Any]:
    """Stream public events into fresh connected sessions, journaling each event.

    The factory supplies a pre-trained model and a new ConnectedDialogue for
    each independent family. The optional reviewer must decide solely from the
    public event and candidate; it does not receive the fixture or any gold.
    """
    from text_factors.real_data.connected_dialogue import ConnectedDialogue
    from text_factors.real_data.hypothesis_bridge import ReviewDecision

    if ledger_path.absolute() == output_path.absolute():
        raise ValueError("ledger and output must have different paths")
    if ledger_path.exists() or output_path.exists():
        raise ValueError("run paths must be new to preserve earlier attempts")
    principal = "dialogue-open-reader"
    attempt_id = uuid.uuid4().hex
    rows = []
    diagnostics = []
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def record(ledger, entry: dict[str, Any]) -> None:
        ledger.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())

    # 'x' ensures a crash or retry cannot silently replace a previous attempt.
    with ledger_path.open("x", encoding="utf-8") as ledger:
        record(
            ledger,
            {
                "status": "started",
                "attempt_id": attempt_id,
                "fixture_sha256": fixture_sha256,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer_supplied": reviewer is not None,
                "restarter_supplied": restarter is not None,
            },
        )
        for family in fixture["families"]:
            family_id, split = family["family_id"], family["split"]
            if split not in splits:
                continue
            session = factory(family_id, split)
            if not isinstance(session, ConnectedDialogue):
                raise ValueError("factory must return a ConnectedDialogue")
            session.store.grant_scope(principal, "default")
            imported_source_ids: dict[str, str] = {}
            external_by_internal: dict[str, str] = {}
            revoked_external: set[str] = set()
            pending_review = False
            unsupported_restart = False
            for event in family["events"]:
                # Make a new allowlisted payload; never send `expected` or any
                # future event to a session, callback, or factory.
                visible = {key: event[key] for key in PUBLIC_FIELDS[event["kind"]]}
                kind = visible["kind"]
                if kind == "ingest":
                    imported = session.import_message(
                        visible["text"],
                        namespace="dialogue-open-v1",
                        external_key=visible["source_id"],
                        group_id=family_id,
                    )
                    if imported.source_version != visible["version"]:
                        raise ValueError(f"source version mismatch in {family_id}")
                    old_id = imported_source_ids.get(visible["source_id"])
                    if old_id is not None and old_id != imported.source_id:
                        raise ValueError(f"source identity changed in {family_id}")
                    imported_source_ids[visible["source_id"]] = imported.source_id
                    external_by_internal[imported.source_id] = visible["source_id"]
                    approved = 0
                    if reviewer is None or not imported.candidates:
                        pending_review = True
                    for candidate in imported.candidates:
                        decision = (
                            reviewer(family_id, visible, imported, candidate)
                            if reviewer is not None
                            else None
                        )
                        correction_target_id = None
                        if isinstance(decision, tuple) and len(decision) == 2:
                            decision, correction_target_id = decision
                        if decision is None:
                            pending_review = True
                            continue
                        if not isinstance(decision, ReviewDecision):
                            raise ValueError(
                                "reviewer must return ReviewDecision or None"
                            )
                        if not decision.approved:
                            pending_review = True
                            continue
                        session.review(
                            candidate,
                            decision,
                            correction_target_id=correction_target_id,
                        )
                        approved += 1
                    diagnostics.append(
                        {
                            "family_id": family_id,
                            "event": visible,
                            "status": "imported",
                            "proposals": len(imported.candidates),
                            "reviewed": approved,
                            "pending_review": pending_review,
                            "unexplained_spans": len(imported.graph.unexplained),
                        }
                    )
                elif kind == "revoke_source_family":
                    session.store.revoke_source_family(
                        imported_source_ids[visible["source_id"]]
                    )
                    revoked_external.add(visible["source_id"])
                    diagnostics.append(
                        {"family_id": family_id, "event": visible, "status": "revoked"}
                    )
                elif kind == "restart":
                    if restarter is None:
                        unsupported_restart = True
                        status = "unsupported_no_checkpoint_restarter"
                    else:
                        session = restarter(session, family_id)
                        if not isinstance(session, ConnectedDialogue):
                            raise ValueError(
                                "restarter must return a ConnectedDialogue"
                            )
                        session.store.grant_scope(principal, "default")
                        status = "external_restarter_called"
                    diagnostics.append(
                        {"family_id": family_id, "event": visible, "status": status}
                    )
                else:
                    started = time.perf_counter()
                    if unsupported_restart:
                        row = {
                            "family_id": family_id,
                            "query_id": visible["query_id"],
                            "action": "unsupported",
                            "answer_text": "",
                            "fact": None,
                            "evidence": [],
                            "human_supported": None,
                            "human_appropriate": None,
                            "pending_review": pending_review,
                            "error": "restart not implemented by this runner",
                        }
                    else:
                        answer = session.ask(
                            visible["query_id"],
                            visible["question"],
                            principal_id=principal,
                        )
                        status_action = {
                            "answered": "answer",
                            "clarify": "ask_clarification",
                        }
                        evidence = [
                            {
                                "source_id": external_by_internal.get(
                                    quote.source.source_id, quote.source.source_id
                                ),
                                "version": quote.source.source_version,
                                "quote": quote.text,
                            }
                            for quote in answer.quotes
                        ]
                        row = {
                            "family_id": family_id,
                            "query_id": visible["query_id"],
                            "action": status_action.get(answer.status, "abstain"),
                            "raw_status": answer.status,
                            "answer_text": "\n".join(
                                quote.text for quote in answer.quotes
                            ),
                            # No inference of a scored fact from the gold. A
                            # separate adjudicator must attach an inspected
                            # semantic prediction and full-text review.
                            "fact": None,
                            "evidence": evidence,
                            "human_supported": None,
                            "human_appropriate": None,
                            "pending_review": pending_review,
                            "clarification": answer.clarification,
                            "claim_ids": (
                                list(answer.receipt.claim_ids)
                                if answer.receipt is not None
                                else []
                            ),
                        }
                    row["latency_ms"] = (time.perf_counter() - started) * 1000
                    row["peak_rss_kib"] = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    rows.append(row)
                    record(ledger, {"status": "query_recorded", "prediction": row})
                    if any(
                        pointer["source_id"] in revoked_external
                        for pointer in row["evidence"]
                    ):
                        record(
                            ledger,
                            {
                                "status": "critical_stopped",
                                "reason": "revoked source in published answer",
                                "family_id": family_id,
                                "query_id": visible["query_id"],
                            },
                        )
                        raise RuntimeError("revoked source appeared in answer")
            record(
                ledger,
                {"status": "family_completed", "family_id": family_id, "split": split},
            )
        payload = {
            "schema": "ai2-dialogue-open-predictions-v1",
            "attempt_id": attempt_id,
            "methods": [
                {
                    "name": "connected_dialogue",
                    "status": (
                        "complete" if splits == SPLITS else "partial_selected_splits"
                    ),
                    "costs": {
                        "selected_splits": sorted(splits),
                        "reviewer_supplied": reviewer is not None,
                        "restarter_supplied": restarter is not None,
                        "training_cpu_seconds": None,
                        "annotation_minutes": None,
                    },
                    "predictions": rows,
                    "diagnostics": diagnostics,
                }
            ],
        }
        with output_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        record(ledger, {"status": "completed", "predictions_path": str(output_path)})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--runner", help="module:factory returning fresh ConnectedDialogue"
    )
    parser.add_argument(
        "--reviewer", help="module:function explicitly reviewing raw candidates"
    )
    parser.add_argument(
        "--restarter", help="module:function restoring a connected session"
    )
    parser.add_argument("--ledger", type=Path, help="new append-only JSONL event log")
    parser.add_argument("--run-output", type=Path, help="new raw prediction archive")
    parser.add_argument(
        "--run-splits",
        default="train_open,development_open",
        help="comma-separated open splits; held-out requires an explicit selection",
    )
    args = parser.parse_args()
    try:
        fixture = _read_json(args.fixture)
        gold = validate_fixture(fixture)
        output: dict[str, Any] = {
            "schema": "ai2-dialogue-open-fixture-check-v1",
            "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            "families": dict(Counter(item["split"] for item in fixture["families"])),
            "queries": len(gold),
            "primary_queries": sum(
                bool(item["expected"].get("primary")) for item in gold.values()
            ),
        }
        if args.runner:
            if args.predictions or args.ledger is None or args.run_output is None:
                raise ValueError("runner requires --ledger and --run-output only")
            splits = frozenset(args.run_splits.split(","))
            if not splits or not splits <= SPLITS:
                raise ValueError("invalid --run-splits")
            run_connected_open(
                fixture,
                factory=_load_callback(args.runner),
                reviewer=_load_callback(args.reviewer) if args.reviewer else None,
                restarter=_load_callback(args.restarter) if args.restarter else None,
                ledger_path=args.ledger,
                output_path=args.run_output,
                fixture_sha256=output["fixture_sha256"],
                splits=splits,
            )
            args.predictions = args.run_output
        elif args.reviewer or args.restarter or args.ledger or args.run_output:
            raise ValueError(
                "--reviewer/--restarter/--ledger/--run-output need --runner"
            )
        if args.predictions:
            output["predictions_sha256"] = hashlib.sha256(
                args.predictions.read_bytes()
            ).hexdigest()
            output["score"] = score_predictions(gold, _read_json(args.predictions))
        encoded = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        else:
            print(encoded)
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps({"status": "invalid", "error": str(error)}, ensure_ascii=False)
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
