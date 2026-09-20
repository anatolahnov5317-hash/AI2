"""Run unittest cases in bounded processes with incremental, resumable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def worker(path, names):
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # This research suite uses small arrays; bound address space independently
    # of the supervisor's wall-clock timeout. Worker children inherit the bound.
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    with path.open("w", buffering=1) as log:

        class Result(unittest.TextTestResult):
            def startTest(self, test):
                self.outcome = "passed"
                self.started = time.perf_counter()
                super().startTest(test)

            def addFailure(self, test, err):
                self.outcome = "failed"
                super().addFailure(test, err)

            def addError(self, test, err):
                self.outcome = "error"
                super().addError(test, err)
                if not isinstance(test, unittest.TestCase):
                    log.write(
                        json.dumps({"id": test.id(), "status": "fixture_error"}) + "\n"
                    )

            def addSkip(self, test, reason):
                self.outcome = "skipped"
                super().addSkip(test, reason)

            def addExpectedFailure(self, test, err):
                self.outcome = "expected_failure"
                super().addExpectedFailure(test, err)

            def addUnexpectedSuccess(self, test):
                self.outcome = "unexpected_success"
                super().addUnexpectedSuccess(test)

            def addSubTest(self, test, subtest, err):
                if err is not None:
                    self.outcome = "failed"
                super().addSubTest(test, subtest, err)

            def stopTest(self, test):
                log.write(
                    json.dumps(
                        {
                            "id": test.id(),
                            "status": self.outcome,
                            "seconds": time.perf_counter() - self.started,
                        }
                    )
                    + "\n"
                )
                super().stopTest(test)

        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(path.with_suffix(".usage.json"), {"peak_rss_kib": peak})
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("bounded process supervision requires Linux, macOS or WSL")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
    if args.worker:
        return worker(args.output, args.names)
    if not 1 <= args.batch_size <= 20 or not 0 < args.timeout <= 600:
        parser.error("batch size must be 1..20; timeout must be (0,600] seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    loader = unittest.TestLoader()
    suite = (
        loader.loadTestsFromNames(args.names)
        if args.names
        else loader.discover(str(ROOT / "tests"))
    )
    if loader.errors:
        raise RuntimeError("\n".join(loader.errors))
    groups = []
    for test in flatten(suite):
        name = test.id()
        if (
            not groups
            or len(groups[-1]) >= args.batch_size
            or groups[-1][-1].rsplit(".", 2)[0] != name.rsplit(".", 2)[0]
        ):
            groups.append([])
        groups[-1].append(name)
    files = sorted(
        [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("src/**/*.json"),
            *ROOT.glob("tests/**/*.py"),
        ]
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    config = {
        "source_sha256": digest.hexdigest(),
        "groups": groups,
        "timeout": args.timeout,
        "python": sys.version,
        "numpy": version("numpy"),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "address_space_bytes": 4 * 1024**3,
    }
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != config:
            raise ValueError(
                "existing run requires --resume and identical sources/config"
            )
    else:
        write_json(manifest, config)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "tests"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    totals = []
    for index, names in enumerate(groups, 1):
        prefix = args.output / f"{index:03d}"
        receipt = prefix.with_suffix(".json")
        if args.resume and receipt.exists():
            result = json.loads(receipt.read_text())
            if result["status"] == "passed":
                totals.append(result)
                continue
        started = time.perf_counter()
        timed_out = False
        with prefix.with_suffix(".log").open("w") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(prefix.with_suffix(".jsonl")),
                    *names,
                ],
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        events_path = prefix.with_suffix(".jsonl")
        events = (
            [json.loads(row) for row in events_path.read_text().splitlines()]
            if events_path.exists()
            else []
        )
        recorded = [event["id"] for event in events]
        missing = sorted(set(names) - set(recorded))
        complete = len(recorded) == len(names) and set(recorded) == set(names)
        result = {
            "batch": index,
            "requested": names,
            "tests": events,
            "status": "timeout"
            if timed_out
            else "failed"
            if code != 0
            else "passed"
            if complete
            else "incomplete",
            "not_completed": missing,
            "returncode": code,
            "seconds": time.perf_counter() - started,
        }
        usage_path = prefix.with_suffix(".usage.json")
        result["peak_worker_rss_kib"] = (
            json.loads(usage_path.read_text())["peak_rss_kib"]
            if usage_path.exists()
            else None
        )
        write_json(receipt, result)
        totals.append(result)
        write_json(
            args.output / "summary.json",
            {
                "total_tests": sum(map(len, groups)),
                "completed_batches": len(totals),
                "total_batches": len(groups),
                "batches": totals,
            },
        )
        print(
            f"{index}/{len(groups)} batches; "
            f"{sum(len(r['tests']) for r in totals)}/{sum(map(len, groups))} cases; "
            f"batch {result['status']}; {result['seconds']:.1f}s",
            flush=True,
        )
    return 0 if all(r["status"] == "passed" for r in totals) else 1


if __name__ == "__main__":
    raise SystemExit(main())
