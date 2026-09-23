"""Score a frozen P24 manifest without starting a real-data deployment.

Example: python scripts/score_p24_pilot.py --input pilot-trials.json \
    --contract docs/real_data/pilot_contract.yaml

Exit status 2 means the machine-readable prerequisites are blocked. References
to selection/annotations are recorded, not independently verified by this CLI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from validate_pilot_contract import validate_contract

from text_factors.real_data.pilot_readiness import PilotTrial, score_p24_trials


def score_manifest(manifest: Any, contract: Any) -> dict[str, Any]:
    """Give a machine gate and clear external prerequisites, never deploy."""
    if not isinstance(manifest, dict) or manifest.get("schema") != "ai2-p24-trials-v1":
        raise ValueError("expected ai2-p24-trials-v1 manifest")
    rows = manifest.get("trials")
    if not isinstance(rows, list):
        raise ValueError("trials must be a list")
    for row in rows:
        if not isinstance(row, dict) or "preselected" not in row:
            raise ValueError("each trial must explicitly record preselection")
    trials = tuple(PilotTrial(**row) for row in rows)
    score = score_p24_trials(trials, phase=manifest.get("phase", ""))
    errors, decisions = validate_contract(contract)
    blockers = set(score.blockers)
    blockers.update(f"contract:{error}" for error in errors)
    blockers.update(f"p01:{decision}" for decision in decisions)
    finalized = (
        isinstance(contract, dict)
        and contract.get("status") == "finalized"
        and not errors
        and not decisions
    )
    if not finalized:
        blockers.add("p01_contract_not_finalized")
    for field in (
        "frozen_selection_reference",
        "independence_review_reference",
        "adjudication_reference",
        "code_and_thresholds_freeze_reference",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            blockers.add(f"missing:{field}")
    if score.phase == "limited":
        # An authenticated owner approval and a real shadow report require the
        # external P01 process. A free-form local JSON value cannot grant this.
        blockers.add("limited_requires_verified_shadow_and_owner_approval")
    return {
        "schema": "ai2-p24-gate-report-v1",
        "score": asdict(score),
        "contract_valid": not errors,
        "contract_finalized": finalized,
        "blockers": sorted(blockers),
        "status": "blocked" if blockers else "ready_for_owner_review",
        "pilot_activated": False,
        "note": (
            "Group independence, frozen selection and rights require "
            "external verification."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.input.read_text(encoding="utf-8"))
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        result = score_manifest(manifest, contract)
    except (OSError, ValueError, TypeError) as exc:
        print(
            json.dumps({"status": "invalid", "errors": [str(exc)]}, ensure_ascii=False)
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
