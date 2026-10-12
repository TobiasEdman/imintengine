#!/usr/bin/env python3
"""Do the S1 tests catch the riskiest one-token bugs? (review finding, 2026-10-10)

Applies each mutation to the working tree, runs tests/test_ai2_vessels_s1.py,
restores the file byte for byte (verified by sha256) and reports which tests
failed. Exit 0 only if every mutation is caught and every file is restored.

    python scripts/marine/s1_port/mutation_check.py --out mutation_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

import _record as rec

MUTATIONS = [
    # The historical window must start 60 days BEFORE the target.
    ("imint/fetch.py",
     "    lo = t0 - timedelta(days=S1_VESSEL_HISTORICAL_OFFSET_DAYS)\n",
     "    lo = t0 + timedelta(days=S1_VESSEL_HISTORICAL_OFFSET_DAYS)\n"),
    # Patch boxes are shifted by the patch COLUMN origin in x.
    ("imint/analyzers/ai2_vessels_s1.py",
     "                    b[:, [0, 2]] += c\n",
     "                    b[:, [0, 2]] += r\n"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    report = []
    for rel, old, new in MUTATIONS:
        path = rec.REPO / rel
        original = path.read_bytes()
        text = original.decode()
        if text.count(old) != 1:
            raise SystemExit(f"{rel}: mutation site not found exactly once: {old.strip()!r}")
        try:
            path.write_text(text.replace(old, new))
            proc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ai2_vessels_s1.py",
                                   "-q", "-p", "no:cacheprovider", "-rf"],
                                  cwd=rec.REPO, capture_output=True, text=True)
        finally:
            path.write_bytes(original)
        failed = sorted({line.split(" ")[1].split("::")[1].split("[")[0]
                         for line in proc.stdout.splitlines() if line.startswith("FAILED ")})
        report.append({
            "file": rel, "mutation": f"{old.strip()}  ->  {new.strip()}",
            "pytest_summary": proc.stdout.strip().splitlines()[-1],
            "failing_tests": failed,
            "restored": hashlib.sha256(path.read_bytes()).digest()
            == hashlib.sha256(original).digest(),
        })
    rec.write_json(a.out, report)
    return 0 if all(r["failing_tests"] and r["restored"] for r in report) else 1


if __name__ == "__main__":
    sys.exit(main())
