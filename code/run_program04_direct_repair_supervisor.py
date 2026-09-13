#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TASKS = [
    (20260826, 160), (20260826, 260), (20260826, 360),
    (20260827, 160), (20260827, 260), (20260827, 360),
    (20260828, 160), (20260828, 260), (20260828, 360),
]
MORE_WORK = 75

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    worker = root / "run_program04_direct_repair_worker.py"
    final_worker = root / "run_program04_rolling_repair_chunk.py"

    if not worker.exists():
        print(f"ERROR: missing {worker}", file=sys.stderr)
        return 2
    if not final_worker.exists():
        print(f"ERROR: missing {final_worker}", file=sys.stderr)
        return 2

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    start = time.time()

    print("=" * 94)
    print("PROGRAM 04 FAST DIRECT REPAIR SUPERVISOR")
    print("repair pairs        : 9")
    print(f"chunk size          : {args.chunk}")
    print("strategy            : exactly ONE comparator per fresh Python process")
    print("repeated identities : eliminated during scoring")
    print("=" * 94)

    for i, (seed, target) in enumerate(TASKS, 1):
        print()
        print(f"######## PAIR {i}/9 : seed={seed}  0->{target} ########")
        while True:
            t0 = time.time()
            cp = subprocess.run(
                [
                    sys.executable, str(worker),
                    "--root", str(root),
                    "--seed", str(seed),
                    "--target-step", str(target),
                    "--device", "cuda",
                    "--max-new-shards", str(args.chunk),
                ],
                cwd=str(root),
                creationflags=flags,
            )
            dt = time.time() - t0

            if cp.returncode == MORE_WORK:
                print(f"[SUPERVISOR] 20-shard chunk finished in {dt/60:.2f} min; restarting in 3 sec.")
                time.sleep(3)
                continue

            if cp.returncode == 0:
                print(f"[SUPERVISOR] PAIR COMPLETE seed={seed} 0->{target}")
                break

            print(f"ERROR: direct worker returned {cp.returncode} for seed={seed}, target={target}.")
            return cp.returncode

    print()
    print("=" * 94)
    print("ALL 9 DIRECT COMPARATORS COMPLETE.")
    print("Running ONE full 45-pair pass to rebuild the global Program04 indexes...")
    print("=" * 94)

    cp = subprocess.run(
        [
            sys.executable, str(final_worker),
            "--root", str(root),
            "--device", "cuda",
            "--max-new-shards", "999999",
        ],
        cwd=str(root),
    )
    if cp.returncode != 0:
        return cp.returncode

    print()
    print("Running final 45-pair verify-only...")
    cp = subprocess.run(
        [
            sys.executable, str(final_worker),
            "--root", str(root),
            "--device", "cuda",
            "--verify-only",
        ],
        cwd=str(root),
    )
    if cp.returncode != 0:
        return cp.returncode

    elapsed = time.time() - start
    print()
    print("=" * 94)
    print("PROGRAM 04 FAST DIRECT STRUCTURAL REPAIR COMPLETED AND VERIFIED")
    print(f"elapsed : {elapsed/3600:.2f} hours")
    print("=" * 94)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
