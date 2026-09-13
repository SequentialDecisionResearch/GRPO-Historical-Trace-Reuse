#!/usr/bin/env python3
"""
Development-only benchmark for reducing the official Program-04 pair grid.

Uses only completed DEVELOPMENT tables (T02/T04) and the already-frozen gate.
It does NOT read official-test OPE outcomes and does NOT modify official outputs.

Candidate grids are geometry-based:
  - all 4 identity pairs per seed are always retained;
  - fixed targets are evenly spaced over 20..400;
  - rolling targets use the same selected lag(s) in each anchor block
    b in {100,200,300}, with lag in {20,40,60,80,100}.

Outputs:
  benchmarks/pair_grid_reduction/
      pair_grid_candidates.csv
      pair_grid_metrics.csv
      pair_grid_recommendation.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SEEDS_DEFAULT = (20260826, 20260827, 20260828)
IDENTITY_STEPS = (0, 100, 200, 300)
FIXED_ALL = tuple(range(20, 401, 20))
ROLLING_ANCHORS = (100, 200, 300)
ROLLING_LAGS_ALL = (20, 40, 60, 80, 100)

# Geometry-only lag patterns; no development outcome is used to create them.
LAG_PATTERNS = {
    "mid": (60,),
    "endpoints": (20, 100),
    "balanced2": (40, 80),
    "end_mid": (20, 60, 100),
}


def die(msg: str):
    raise RuntimeError(msg)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        die(f"Missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ffloat(x: Any) -> float:
    v = float(x)
    if not math.isfinite(v):
        die(f"Expected finite float, got {x!r}")
    return v


def fint(x: Any) -> int:
    return int(float(x))


def mean(xs: Sequence[float]) -> float:
    if not xs:
        return math.nan
    return math.fsum(float(x) for x in xs) / len(xs)


def evenly_spaced_targets(n: int) -> tuple[int, ...]:
    """Choose n fixed targets from 20..400 using only grid geometry."""
    if n <= 0 or n > len(FIXED_ALL):
        die(f"Invalid fixed target count: {n}")
    if n == 1:
        return (200,)
    idxs = []
    last = len(FIXED_ALL) - 1
    for i in range(n):
        idxs.append(int(round(i * last / (n - 1))))
    # Resolve any accidental duplicates deterministically.
    out = []
    used = set()
    for idx in idxs:
        if idx not in used:
            out.append(FIXED_ALL[idx]); used.add(idx)
    if len(out) != n:
        # Fallback: greedily fill nearest unused grid points.
        for e in FIXED_ALL:
            if e not in used:
                out.append(e); used.add(e)
                if len(out) == n:
                    break
    return tuple(sorted(out))


def load_gate(root: Path) -> tuple[dict[float, dict[str, Any]], set[int], set[int]]:
    p = root / "outputs" / "frozen_gate.json"
    if not p.exists():
        die(f"Missing frozen gate: {p}")
    gate = json.loads(p.read_text(encoding="utf-8"))
    th = {}
    for r in ((gate.get("primary_gate") or {}).get("thresholds") or []):
        th[float(r["tolerance"])] = {
            "threshold": float(r["threshold"]),
            "reject_all": bool(r.get("reject_all", False)),
            "direction": str(r.get("direction", "ge")),
        }
    if not th:
        die("Frozen gate has no primary thresholds.")
    cal = {int(x) for x in gate.get("calibration_seeds", [])}
    val = {int(x) for x in gate.get("validation_seeds", [])}
    if not cal or not val:
        die("Frozen gate lacks calibration/validation seed sets.")
    return th, cal, val


def gate_accept(ress: float, rec: Mapping[str, Any]) -> bool:
    if bool(rec.get("reject_all", False)):
        return False
    if rec.get("direction") != "ge":
        die("Primary gate direction is not >=.")
    return ress >= float(rec["threshold"])


def fixed_cases(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if str(r.get("split")) != "development":
            continue
        if str(r.get("estimator")) != "prompt_wis":
            continue
        out.append({
            "seed": fint(r["training_seed"]),
            "b": fint(r["behavior_step"]),
            "e": fint(r["target_step"]),
            "purpose": "fixed",
            "absolute_error": ffloat(r["absolute_error"]),
            "ress": ffloat(r["median_prompt_relative_ess"]),
            "kl": ffloat(r["tokenwise_kl_proxy"]),
        })
    return out


def rolling_cases(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if str(r.get("split")) != "development":
            continue
        if str(r.get("estimator")) != "prompt_wis":
            continue
        out.append({
            "seed": fint(r["training_seed"]),
            "b": fint(r["recent_behavior_step"]),
            "e": fint(r["target_step"]),
            "purpose": "rolling",
            "absolute_error": ffloat(r["recent_absolute_error"]),
            "ress": ffloat(r["recent_median_relative_ess"]),
            "kl": ffloat(r["recent_tokenwise_kl_proxy"]),
            "old_absolute_error": ffloat(r["old_absolute_error"]),
            "recent_absolute_error": ffloat(r["recent_absolute_error"]),
            "error_reduction": ffloat(r["delta_error_old_minus_recent"]),
        })
    return out


def scope_filter(case: Mapping[str, Any], scope: str, cal: set[int], val: set[int]) -> bool:
    s = int(case["seed"])
    if scope == "all":
        return True
    if scope == "calibration":
        return s in cal
    if scope == "validation":
        return s in val
    raise ValueError(scope)


def summarize_cases(
    fixed: Sequence[Mapping[str, Any]],
    rolling: Sequence[Mapping[str, Any]],
    thresholds: Mapping[float, Mapping[str, Any]],
    scope: str,
    cal: set[int],
    val: set[int],
) -> dict[str, float]:
    f = [x for x in fixed if scope_filter(x, scope, cal, val)]
    r = [x for x in rolling if scope_filter(x, scope, cal, val)]
    cases = f + r
    if not f or not r or not cases:
        die(f"Empty summary scope={scope}: fixed={len(f)} rolling={len(r)}")

    out = {
        "n_fixed": float(len(f)),
        "n_rolling": float(len(r)),
        "fixed_mean_mae": mean([x["absolute_error"] for x in f]),
        "fixed_mean_ress": mean([x["ress"] for x in f]),
        "rolling_mean_recent_mae": mean([x["recent_absolute_error"] for x in r]),
        "rolling_mean_old_mae": mean([x["old_absolute_error"] for x in r]),
        "rolling_mean_error_reduction": mean([x["error_reduction"] for x in r]),
        "rolling_fraction_recent_better": mean([
            1.0 if x["recent_absolute_error"] < x["old_absolute_error"] else 0.0 for x in r
        ]),
        "rolling_mean_ress": mean([x["ress"] for x in r]),
    }

    for eps, rec in sorted(thresholds.items()):
        accepts = [gate_accept(x["ress"], rec) for x in cases]
        false_accept = [
            bool(a and x["absolute_error"] > eps) for a, x in zip(accepts, cases)
        ]
        accepted_errors = [
            x["absolute_error"] for a, x in zip(accepts, cases) if a
        ]
        out[f"gate_accept_rate_{eps:.2f}"] = mean([1.0 if a else 0.0 for a in accepts])
        out[f"gate_false_accept_fraction_all_{eps:.2f}"] = mean([
            1.0 if z else 0.0 for z in false_accept
        ])
        out[f"gate_accepted_mae_{eps:.2f}"] = (
            mean(accepted_errors) if accepted_errors else math.nan
        )
    return out


def absdiff(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return math.inf
    return abs(a - b)


def candidate_rows(
    fixed_all: Sequence[Mapping[str, Any]],
    rolling_all: Sequence[Mapping[str, Any]],
    thresholds: Mapping[float, Mapping[str, Any]],
    cal: set[int],
    val: set[int],
    min_total_pairs: int,
    max_total_pairs: int,
):
    # Full references by scope.
    refs = {
        scope: summarize_cases(fixed_all, rolling_all, thresholds, scope, cal, val)
        for scope in ("all", "calibration", "validation")
    }

    candidates = []
    metrics = []
    cid = 0

    # Same reduced grid is applied to all three training seeds.
    n_seeds = len({int(x["seed"]) for x in fixed_all + rolling_all})

    for lag_name, lags in LAG_PATTERNS.items():
        for n_fixed in range(1, 13):
            fixed_targets = evenly_spaced_targets(n_fixed)
            pairs_per_seed = len(IDENTITY_STEPS) + len(fixed_targets) + len(ROLLING_ANCHORS) * len(lags)
            total_pairs = n_seeds * pairs_per_seed
            if total_pairs < min_total_pairs or total_pairs > max_total_pairs:
                continue

            fsel = [x for x in fixed_all if int(x["e"]) in set(fixed_targets)]
            rsel = [
                x for x in rolling_all
                if (int(x["e"]) - int(x["b"])) in set(lags)
            ]
            # Ensure every expected seed/grid case exists.
            expected_fixed = n_seeds * len(fixed_targets)
            expected_rolling = n_seeds * len(ROLLING_ANCHORS) * len(lags)
            if len(fsel) != expected_fixed or len(rsel) != expected_rolling:
                continue

            cid += 1
            candidate_id = f"C{cid:03d}"
            candidates.append({
                "candidate_id": candidate_id,
                "total_pairs_all_seeds": total_pairs,
                "pairs_per_seed": pairs_per_seed,
                "identity_pairs_per_seed": len(IDENTITY_STEPS),
                "fixed_pairs_per_seed": len(fixed_targets),
                "rolling_pairs_per_seed": len(ROLLING_ANCHORS) * len(lags),
                "fixed_targets": ",".join(str(x) for x in fixed_targets),
                "rolling_lag_pattern": lag_name,
                "rolling_lags": ",".join(str(x) for x in lags),
                "rolling_pairs": ";".join(
                    f"{b}->{b+d}" for b in ROLLING_ANCHORS for d in lags
                ),
            })

            for scope in ("all", "calibration", "validation"):
                s = summarize_cases(fsel, rsel, thresholds, scope, cal, val)
                ref = refs[scope]
                row = {
                    "candidate_id": candidate_id,
                    "scope": scope,
                    "total_pairs_all_seeds": total_pairs,
                    "fixed_targets": ",".join(str(x) for x in fixed_targets),
                    "rolling_lags": ",".join(str(x) for x in lags),
                }
                for k, v in s.items():
                    row[k] = v
                    if k in ref and k not in ("n_fixed", "n_rolling"):
                        row[f"absdiff_{k}"] = absdiff(v, ref[k])
                metrics.append(row)

    return candidates, metrics, refs


def evaluate(
    candidates: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    *,
    max_fixed_mae_diff: float,
    max_fixed_ress_diff: float,
    max_rolling_error_reduction_diff: float,
    max_rolling_fraction_better_diff: float,
    max_gate_accept_rate_diff: float,
    max_gate_false_accept_fraction_diff: float,
    max_gate_accepted_mae_diff: float,
):
    byid = {}
    for r in metrics:
        byid.setdefault(r["candidate_id"], []).append(r)

    checks = []
    for c in candidates:
        cid = c["candidate_id"]
        rows = [r for r in byid[cid] if r["scope"] in ("all", "validation")]
        passed = True
        reasons = []
        worst = {
            "fixed_mae": 0.0,
            "fixed_ress": 0.0,
            "rolling_error_reduction": 0.0,
            "rolling_fraction_better": 0.0,
            "gate_accept": 0.0,
            "gate_false_accept": 0.0,
            "gate_accepted_mae": 0.0,
        }

        for r in rows:
            worst["fixed_mae"] = max(worst["fixed_mae"], r["absdiff_fixed_mean_mae"])
            worst["fixed_ress"] = max(worst["fixed_ress"], r["absdiff_fixed_mean_ress"])
            worst["rolling_error_reduction"] = max(
                worst["rolling_error_reduction"], r["absdiff_rolling_mean_error_reduction"])
            worst["rolling_fraction_better"] = max(
                worst["rolling_fraction_better"], r["absdiff_rolling_fraction_recent_better"])

            for eps in (0.02, 0.05):
                k = f"{eps:.2f}"
                if f"absdiff_gate_accept_rate_{k}" in r:
                    worst["gate_accept"] = max(
                        worst["gate_accept"], r[f"absdiff_gate_accept_rate_{k}"])
                    worst["gate_false_accept"] = max(
                        worst["gate_false_accept"], r[f"absdiff_gate_false_accept_fraction_all_{k}"])
                    worst["gate_accepted_mae"] = max(
                        worst["gate_accepted_mae"], r[f"absdiff_gate_accepted_mae_{k}"])

        tests = [
            ("fixed_mae", max_fixed_mae_diff),
            ("fixed_ress", max_fixed_ress_diff),
            ("rolling_error_reduction", max_rolling_error_reduction_diff),
            ("rolling_fraction_better", max_rolling_fraction_better_diff),
            ("gate_accept", max_gate_accept_rate_diff),
            ("gate_false_accept", max_gate_false_accept_fraction_diff),
            ("gate_accepted_mae", max_gate_accepted_mae_diff),
        ]
        for name, limit in tests:
            if worst[name] > limit:
                passed = False
                reasons.append(f"{name}:{worst[name]:.6g}>{limit:.6g}")

        # Distortion score is used only to break ties among grids with the same pair count.
        score = (
            worst["fixed_mae"] / max_fixed_mae_diff +
            worst["fixed_ress"] / max_fixed_ress_diff +
            worst["rolling_error_reduction"] / max_rolling_error_reduction_diff +
            worst["rolling_fraction_better"] / max_rolling_fraction_better_diff +
            worst["gate_accept"] / max_gate_accept_rate_diff +
            worst["gate_false_accept"] / max_gate_false_accept_fraction_diff +
            worst["gate_accepted_mae"] / max_gate_accepted_mae_diff
        )
        checks.append({
            **dict(c),
            "passed": passed,
            "distortion_score": score,
            "reasons": reasons,
            **{f"worst_{k}": v for k, v in worst.items()},
        })

    passing = [x for x in checks if x["passed"]]
    recommended = None
    if passing:
        passing.sort(key=lambda x: (
            int(x["total_pairs_all_seeds"]),
            float(x["distortion_score"]),
            x["candidate_id"],
        ))
        recommended = passing[0]
    return checks, recommended


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if isinstance(rr.get("reasons"), list):
                rr["reasons"] = " | ".join(rr["reasons"])
            w.writerow(rr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output-dir", default="benchmarks/pair_grid_reduction")
    ap.add_argument("--min-total-pairs", type=int, default=35)
    ap.add_argument("--max-total-pairs", type=int, default=50)

    # Conservative but not exact-equality criteria.
    ap.add_argument("--max-fixed-mae-diff", type=float, default=0.0025)
    ap.add_argument("--max-fixed-ress-diff", type=float, default=0.05)
    ap.add_argument("--max-rolling-error-reduction-diff", type=float, default=0.003)
    ap.add_argument("--max-rolling-fraction-better-diff", type=float, default=0.10)
    ap.add_argument("--max-gate-accept-rate-diff", type=float, default=0.08)
    ap.add_argument("--max-gate-false-accept-fraction-diff", type=float, default=0.05)
    ap.add_argument("--max-gate-accepted-mae-diff", type=float, default=0.004)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    tdir = root / "outputs" / "tables"
    t02 = fixed_cases(read_csv_rows(tdir / "T02_fixed_reuse.csv"))
    t04 = rolling_cases(read_csv_rows(tdir / "T04_rolling_comparison.csv"))
    thresholds, cal, val = load_gate(root)

    if not t02 or not t04:
        die("No development prompt_wis rows found in T02/T04.")

    print("=" * 96)
    print("DEVELOPMENT-ONLY PAIR-GRID REDUCTION BENCHMARK")
    print(f"full fixed cases   : {len(t02)}")
    print(f"full rolling cases : {len(t04)}")
    print(f"calibration seeds  : {sorted(cal)}")
    print(f"validation seeds   : {sorted(val)}")
    print("TEST OPE OUTCOMES ARE NOT READ.")
    print("=" * 96)

    candidates, metrics, refs = candidate_rows(
        t02, t04, thresholds, cal, val,
        args.min_total_pairs, args.max_total_pairs
    )
    if not candidates:
        die("No candidate grids fall inside the requested pair-count range.")

    checks, rec = evaluate(
        candidates, metrics,
        max_fixed_mae_diff=args.max_fixed_mae_diff,
        max_fixed_ress_diff=args.max_fixed_ress_diff,
        max_rolling_error_reduction_diff=args.max_rolling_error_reduction_diff,
        max_rolling_fraction_better_diff=args.max_rolling_fraction_better_diff,
        max_gate_accept_rate_diff=args.max_gate_accept_rate_diff,
        max_gate_false_accept_fraction_diff=args.max_gate_false_accept_fraction_diff,
        max_gate_accepted_mae_diff=args.max_gate_accepted_mae_diff,
    )

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir = outdir.resolve()
    # Never write into official data/manifests/outputs.
    for bad in (root/"data", root/"manifests", root/"outputs"):
        try:
            outdir.relative_to(bad.resolve())
            die(f"Refusing to write benchmark under official tree: {outdir}")
        except ValueError:
            pass
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir/"pair_grid_candidates.csv", checks)
    write_csv(outdir/"pair_grid_metrics.csv", metrics)

    payload = {
        "benchmark": "development_only_pair_grid_reduction",
        "full_design": {
            "identity_per_seed": list(IDENTITY_STEPS),
            "fixed_targets": list(FIXED_ALL),
            "rolling_anchors": list(ROLLING_ANCHORS),
            "rolling_lags": list(ROLLING_LAGS_ALL),
            "full_pairs_all_seeds": 117,
        },
        "pair_budget": {
            "min_total_pairs": args.min_total_pairs,
            "max_total_pairs": args.max_total_pairs,
        },
        "criteria": {
            "max_fixed_mae_diff": args.max_fixed_mae_diff,
            "max_fixed_ress_diff": args.max_fixed_ress_diff,
            "max_rolling_error_reduction_diff": args.max_rolling_error_reduction_diff,
            "max_rolling_fraction_better_diff": args.max_rolling_fraction_better_diff,
            "max_gate_accept_rate_diff": args.max_gate_accept_rate_diff,
            "max_gate_false_accept_fraction_diff": args.max_gate_false_accept_fraction_diff,
            "max_gate_accepted_mae_diff": args.max_gate_accepted_mae_diff,
        },
        "reference_summaries": refs,
        "n_candidates": len(checks),
        "n_passing": sum(1 for x in checks if x["passed"]),
        "recommended": rec,
        "note": (
            "Development-only benchmark. Any official-test grid change remains a protocol amendment; "
            "this script does not modify the frozen pair registry or official artifacts."
        ),
    }
    (outdir/"pair_grid_recommendation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8"
    )

    print("\n" + "=" * 96)
    print("DECISION")
    by_count = {}
    for x in checks:
        by_count.setdefault(int(x["total_pairs_all_seeds"]), []).append(x)
    for n in sorted(by_count):
        passed = [x for x in by_count[n] if x["passed"]]
        if passed:
            best = min(passed, key=lambda x: x["distortion_score"])
            print(
                f"total_pairs={n:2d} -> PASS  best={best['candidate_id']} "
                f"fixed=[{best['fixed_targets']}] rolling_lags=[{best['rolling_lags']}]"
            )
        else:
            print(f"total_pairs={n:2d} -> FAIL")

    if rec is None:
        print("RECOMMENDED GRID: NONE")
    else:
        print(
            "RECOMMENDED GRID: "
            f"{rec['candidate_id']}  total_pairs={rec['total_pairs_all_seeds']}  "
            f"fixed=[{rec['fixed_targets']}]  rolling_lags=[{rec['rolling_lags']}]"
        )
        print(f"rolling pairs per seed: {rec['rolling_pairs']}")
        print("identity steps per seed: 0,100,200,300")

    print(f"recommendation: {outdir/'pair_grid_recommendation.json'}")
    print(f"candidates    : {outdir/'pair_grid_candidates.csv'}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
