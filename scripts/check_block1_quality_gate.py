"""Check an evaluated mention-learning report against the Block 1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    value = json.loads(args.report.read_text(encoding="utf-8"))
    gate = value.get("quality_gate")
    if type(gate) is not dict or gate.get("schema") != "ai2-block1-quality-gate-v1":
        raise SystemExit("report does not contain a Block 1 quality gate")
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if gate.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
