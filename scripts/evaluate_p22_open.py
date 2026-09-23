"""Run and record *every* open P22 synthetic attempt; never read sealed data.

Usage: PYTHONPATH=src python scripts/evaluate_p22_open.py --ledger /tmp/p22.sqlite
The durable ledger records an attempt before predictions start. An interrupted
attempt remains visible as ``started``, and reruns create distinct records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from text_factors.real_data.independent_evaluation import (
    EvaluationCase,
    EvaluationPlan,
    EvidencePointer,
    FrozenPrediction,
    evaluate_open,
)
from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledEvent,
    LabeledText,
    OpenSemanticModel,
    Span,
)

ROOT = Path(__file__).resolve().parents[1]


def _span(text: str, surface: str) -> Span:
    start = text.index(surface)
    return Span(start, start + len(surface))


def _case(text: str, actor: str, item: str, cue: str, relation: str) -> LabeledText:
    return LabeledText(
        text,
        (
            IdentifiedMention("a", actor.lower(), _span(text, actor), "person", "nom"),
            IdentifiedMention("o", item.lower(), _span(text, item), "thing"),
        ),
        (
            LabeledEvent(
                relation,
                _span(text, cue),
                (
                    ("actor", "a"),
                    ("target" if relation == "illuminate" else "object", "o"),
                ),
            ),
        ),
    )


def _label(relation: str, actor: str, item: str) -> str:
    role = "target" if relation == "illuminate" else "object"
    return f"{relation}|actor={actor}|{role}={item}"


def _claims(
    model: OpenSemanticModel, item: LabeledText, group: str
) -> FrozenPrediction:
    graph = model.parse(item.text, item.mentions, source_id=group, source_version=1)
    labels = tuple(
        sorted(
            f"{event.relation_id}|"
            + "|".join(f"{role.role}={role.instance_id}" for role in event.roles)
            for event in graph.events
        )
    )
    pointers = tuple(
        EvidencePointer(label, group, 1, hashlib.sha256(item.text.encode()).hexdigest())
        for label in labels
    )
    return FrozenPrediction(labels, pointers)


def _open_fixture() -> tuple[
    tuple[EvaluationCase, ...], EvaluationPlan, dict[str, str]
]:
    """Construct deliberately small, disclosed Russian mechanism examples."""
    train = (
        (
            "train-a",
            _case("Анна осветила зал.", "Анна", "зал", "осветила", "illuminate"),
        ),
        (
            "train-b",
            _case("Борис осветил склад.", "Борис", "склад", "осветил", "illuminate"),
        ),
        (
            "train-c",
            _case("Анна исправила файл.", "Анна", "файл", "исправила", "repair"),
        ),
        (
            "train-d",
            _case("Борис исправил прибор.", "Борис", "прибор", "исправил", "repair"),
        ),
    )
    dev = (
        (
            "dev-x",
            _case("Вера осветила двор.", "Вера", "двор", "осветила", "illuminate"),
            "illuminate",
        ),
        (
            "dev-x",
            _case(
                "Вера исправила датчик.",
                "Вера",
                "датчик",
                "исправила",
                "repair",
            ),
            "repair",
        ),
        (
            "dev-y",
            _case("Олег осветил цех.", "Олег", "цех", "осветил", "illuminate"),
            "illuminate",
        ),
        (
            "dev-y",
            _case("Олег исправил станок.", "Олег", "станок", "исправил", "repair"),
            "repair",
        ),
    )
    candidate = OpenSemanticModel().fit(tuple(item for _, item in train))
    ablated = OpenSemanticModel().fit(tuple(item for _, item in train[2:]))
    memorized = {item.text: item for _, item in train}
    plan = EvaluationPlan(
        "candidate",
        "exact_text_control",
        "without_new_relation_ablation",
        frozenset(group for group, _ in train),
        frozenset({"repair"}),
        frozenset({"illuminate"}),
    )
    result: list[EvaluationCase] = []
    for index, (group, item, relation) in enumerate(dev):
        gold = _label(
            relation, item.mentions[0].instance_id, item.mentions[1].instance_id
        )
        exact = memorized.get(item.text)
        result.append(
            EvaluationCase(
                f"dev-{index}",
                group,
                relation,
                "development",
                (gold,),
                True,
                {
                    "candidate": _claims(candidate, item, group),
                    "exact_text_control": (
                        _claims(candidate, exact, group)
                        if exact is not None
                        else FrozenPrediction(())
                    ),
                    "without_new_relation_ablation": _claims(ablated, item, group),
                },
            )
        )
    return (
        tuple(result),
        plan,
        {
            "candidate_model_sha256": candidate.model_fingerprint or "",
            "ablation_model_sha256": ablated.model_fingerprint or "",
            "plan_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "methods": plan.methods,
                        "train_groups": sorted(plan.trained_history_groups),
                        "baseline_relations": sorted(plan.baseline_relation_ids),
                        "introduced_relations": sorted(plan.newly_trained_relation_ids),
                        "bootstrap_seed": plan.bootstrap_seed,
                        "bootstrap_repeats": plan.bootstrap_repeats,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "fixture_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "train": [(group, item.text) for group, item in train],
                        "development": [
                            (group, item.text, relation)
                            for group, item, relation in dev
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
    )


def _revision() -> dict[str, str | bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    return {
        "git_head": head,
        "git_tree": tree,
        "worktree_dirty": dirty,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(
            (ROOT / "src/text_factors/real_data/independent_evaluation.py").read_bytes()
        ).hexdigest(),
        "semantic_source_sha256": hashlib.sha256(
            (ROOT / "src/text_factors/real_data/open_semantics.py").read_bytes()
        ).hexdigest(),
    }


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS attempts ("
        "id TEXT PRIMARY KEY, started_at TEXT NOT NULL, "
        "status TEXT NOT NULL, source_revision TEXT NOT NULL, "
        "result_json TEXT, error TEXT)"
    )
    connection.commit()
    return connection


def _export_all(connection: sqlite3.Connection, path: Path) -> None:
    """Serialize concurrent exports behind the ledger's own write lock."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        rows = connection.execute(
            "SELECT id,started_at,status,source_revision,result_json,error "
            "FROM attempts ORDER BY started_at,id"
        )
        entries = [
            {
                "run_id": run_id,
                "started_at": started,
                "status": status,
                "source_revision": json.loads(revision),
                "result": json.loads(result) if result is not None else None,
                "error": error,
            }
            for run_id, started, status, revision, result, error in rows
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(
                    {"schema": "ai2-p22-all-attempts-v1", "attempts": entries},
                    output,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        connection.commit()


def run_logged(ledger: Path, export: Path | None = None) -> dict[str, object]:
    """Record attempt then result; a killed process leaves a visible attempt."""
    from datetime import datetime, timezone

    revision = _revision()
    run_id = uuid.uuid4().hex
    connection = _connect(ledger)
    connection.execute(
        "INSERT INTO attempts (id,started_at,status,source_revision) VALUES (?,?,?,?)",
        (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            "started",
            json.dumps(revision, sort_keys=True),
        ),
    )
    connection.commit()
    wall = time.perf_counter()
    cpu = time.process_time()
    try:
        cases, plan, fingerprints = _open_fixture()
        result = evaluate_open(cases, plan)
        result["run_id"] = run_id
        result["fingerprints"] = {**revision, **fingerprints}
        result["cost"] = {
            "wall_seconds": time.perf_counter() - wall,
            "process_cpu_seconds": time.process_time() - cpu,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python": platform.python_version(),
            "machine": platform.machine(),
        }
        connection.execute(
            "UPDATE attempts SET status=?,result_json=? WHERE id=?",
            (
                "finished",
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                run_id,
            ),
        )
        connection.commit()
        if export is not None:
            _export_all(connection, export)
        result["ledger_attempts"] = [
            {"run_id": item[0], "status": item[1]}
            for item in connection.execute(
                "SELECT id,status FROM attempts ORDER BY started_at,id"
            )
        ]
        return result
    except BaseException as exc:
        connection.execute(
            "UPDATE attempts SET status=?,error=? WHERE id=?",
            ("failed", repr(exc)[:1024], run_id),
        )
        connection.commit()
        if export is not None:
            _export_all(connection, export)
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument(
        "--export", type=Path, help="complete JSON export of all attempts"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_logged(args.ledger, args.export),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
