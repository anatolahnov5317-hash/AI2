"""Reproducible synthetic P23 I/O probe; never a real pilot hardware claim.

Measures one state writer against parallel readers, immutable result commits,
and recovery from an independently validated copy. The subprocess crash test
is in tests/test_p23_reliability.py and should be run alongside this probe.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from text_factors.real_data import (
    Claim,
    ClaimStatus,
    EvidenceRoot,
    RealDataEngine,
    RoleValue,
    SourceSlice,
    load_state,
    save_state,
)
from text_factors.real_data.reliability import publish_run, read_run


def fixture(count: int) -> RealDataEngine:
    """A disclosed structured workload without actual user source text."""
    engine = RealDataEngine()
    source = SourceSlice("synthetic", 1, 0, 1, "0" * 64)
    engine.register_evidence(
        EvidenceRoot(
            "synthetic-root", "synthetic-family", "synthetic", 1, "default", source
        )
    )
    for index in range(count):
        engine.add_claim(
            Claim(
                f"claim-{index:06d}",
                "located",
                (RoleValue("item", f"item-{index:06d}"), RoleValue("place", "box")),
                ClaimStatus.OBSERVED,
                source=source,
                evidence_roots=("synthetic-root",),
                model_version="p23-synthetic",
            )
        )
    return engine


def distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": round(statistics.median(ordered) * 1_000, 3),
        "p95_ms": round(
            ordered[min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1)] * 1_000,
            3,
        ),
        "max_ms": round(ordered[-1] * 1_000, 3),
    }


def probe(iterations: int, readers: int, claims: int) -> dict[str, object]:
    if iterations < 2 or not 1 <= readers <= 16 or not 1 <= claims <= 2_000:
        raise ValueError("iterations >=2, readers 1..16, claims 1..2000 required")
    old = fixture(claims)
    new = fixture(claims + 1)
    with tempfile.TemporaryDirectory(prefix="ai2-p23-") as temp:
        root = Path(temp)
        state_path = root / "state.json"
        save_state(state_path, old, model_version="p23-synthetic")
        original = state_path.read_bytes()
        publish_run(root / "copies", "snapshot", {"state.json": original})
        # Recovery is into a fresh staging path, validated by load_state.
        recovered_bytes = read_run(root / "copies", "snapshot")["state.json"]
        recovery_stage = root / "recovered-staging.json"
        recovery_stage.write_bytes(recovered_bytes)
        recovered, contexts, version = load_state(recovery_stage)
        recovery_valid = (
            contexts is None
            and version == "p23-synthetic"
            and recovered.to_dict() == old.to_dict()
        )
        if not recovery_valid:
            raise AssertionError("verified backup did not restore the original state")

        read_times: list[float] = []
        write_times: list[float] = []
        publication_times: list[float] = []
        stop = threading.Event()

        def read_worker() -> tuple[int, list[float]]:
            samples = []
            valid = 0
            while not stop.is_set() or valid == 0:
                started = perf_counter()
                engine, _, got_version = load_state(state_path)
                samples.append(perf_counter() - started)
                if got_version != "p23-synthetic" or len(engine.claims) not in {
                    claims,
                    claims + 1,
                }:
                    raise AssertionError("reader observed a partial state")
                valid += 1
            return valid, samples

        total_started = perf_counter()
        with ThreadPoolExecutor(max_workers=readers) as pool:
            reading = [pool.submit(read_worker) for _ in range(readers)]
            try:
                for index in range(iterations):
                    started = perf_counter()
                    save_state(
                        state_path,
                        old if index % 2 == 0 else new,
                        model_version="p23-synthetic",
                    )
                    write_times.append(perf_counter() - started)
                    started = perf_counter()
                    publish_run(
                        root / "runs",
                        f"run-{index:04d}",
                        {"result.json": json.dumps({"iteration": index}).encode()},
                    )
                    publication_times.append(perf_counter() - started)
            finally:
                stop.set()
            completed = [future.result() for future in reading]
        elapsed = perf_counter() - total_started
        for _, samples in completed:
            read_times.extend(samples)
        if not all(
            read_run(root / "runs", f"run-{index:04d}")["result.json"]
            == json.dumps({"iteration": index}).encode()
            for index in range(iterations)
        ):
            raise AssertionError("committed result bundle invalid")

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "schema": "ai2-p23-synthetic-probe-v1",
        "workload": {
            "claims_per_state": claims,
            "iterations": iterations,
            "reader_threads": readers,
            "data": "synthetic claims; one writer, concurrent readers",
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count_visible": os.cpu_count(),
            "hardware_selected_by_p01": False,
        },
        "results": {
            "readers_valid": sum(count for count, _ in completed),
            "write_operations": len(write_times),
            "published_runs": iterations,
            "recovery_from_copy_valid": recovery_valid,
            "total_seconds": round(elapsed, 3),
            "read": distribution(read_times),
            "write": distribution(write_times),
            "publish": distribution(publication_times),
            "peak_process_rss_kib": rss
            if sys.platform != "darwin"
            else round(rss / 1024),
        },
        "scope": (
            "process-level regression only; no real workload or selected "
            "target hardware"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--claims", type=int, default=128)
    args = parser.parse_args()
    result = probe(args.iterations, args.readers, args.claims)
    run_id = "p23-" + uuid.uuid4().hex
    folder = publish_run(
        args.output_dir,
        run_id,
        {
            "report.json": (
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            ).encode()
        },
    )
    print(
        json.dumps(
            {"report": str(folder / "report.json"), "results": result["results"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
