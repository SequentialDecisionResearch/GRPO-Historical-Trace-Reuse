#!/usr/bin/env python3
"""
Development-only Program 05 Monte-Carlo reference sample-size benchmark.

Purpose
-------
Evaluate whether the official-test Program 05 online-reference sample counts
can be reduced without materially changing the reference values or the
downstream Program 06 error summaries.

STRICT RESEARCH BOUNDARY
------------------------
* Reads ONLY Program 05 `split=development` online-reference samples.
* Optionally reads ONLY `split=development` rows from Program 06 T02/T04 CSVs.
* Reads the reduced-grid selection manifest only to determine target steps.
* NEVER reads official-test online samples, official-test OPE rows, or test
  outcomes.
* NEVER modifies Program 04/05/06/07 data, manifests, tables, or gate files.
* Writes only under:
      benchmarks/program05_reference_sample_size/

The benchmark mimics reducing L by taking sample indices 0..L-1, which is the
same prefix that a smaller Program 05 target sample limit would retain.

Default candidate designs
-------------------------
  main L  in {4, 8}
  audit L in {8, 16, 32}

The frozen Program 05 test design is main=8, audit=32.  Audit checkpoints are
0,100,200,300,400.

PASS is intentionally based on DEVELOPMENT evidence only.  Default tolerances
are configurable from the CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

AUDIT_STEPS = {0, 100, 200, 300, 400}
DEFAULT_CALIBRATION_SEEDS = {20260826, 20260827}
DEFAULT_VALIDATION_SEEDS = {20260828}

class BenchmarkError(RuntimeError):
    pass

def fail(msg: str) -> None:
    raise BenchmarkError(msg)

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()

def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def safe_float(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None

def percentile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w

def mean(xs: Sequence[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None

def max_or_none(xs: Sequence[float]) -> float | None:
    return max(xs) if xs else None

def fmt(x: float | None) -> str:
    return "NA" if x is None else f"{x:.6f}"

def parse_int_list(values: Sequence[str]) -> list[int]:
    out = []
    for v in values:
        for piece in str(v).split(","):
            piece = piece.strip()
            if piece:
                out.append(int(piece))
    return sorted(set(out))

def load_reduced_steps(root: Path) -> tuple[list[int], list[int], Path, str]:
    p = root / "manifests" / "reduced_test_pair_selection.json"
    if not p.exists():
        fail(f"Missing reduced-grid selection manifest: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    rows = obj.get("selected_pairs")
    if not isinstance(rows, list) or not rows:
        fail("reduced_test_pair_selection.json lacks selected_pairs.")
    steps = sorted({int(r["target_step"]) for r in rows})
    seeds = sorted({int(r["training_seed"]) for r in rows})
    return seeds, steps, p, sha256_file(p)

@dataclass(frozen=True)
class TargetData:
    seed: int
    step: int
    prompt_samples: dict[str, tuple[float, ...]]
    full_l: int
    full_reference: float

def discover_target_dirs(dev_root: Path) -> dict[tuple[int, int], Path]:
    out: dict[tuple[int, int], Path] = {}
    if not dev_root.exists():
        fail(f"Development online-reference root not found: {dev_root}")
    for seed_dir in sorted(dev_root.glob("seed=*")):
        try:
            seed = int(seed_dir.name.split("=", 1)[1])
        except Exception:
            continue
        for step_dir in sorted(seed_dir.glob("target_step=*")):
            try:
                step = int(step_dir.name.split("=", 1)[1])
            except Exception:
                continue
            out[(seed, step)] = step_dir
    return out

def load_target_data(seed: int, step: int, target_dir: Path) -> TargetData:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        fail(f"pyarrow is required: {exc}")

    files = sorted(target_dir.glob("sample_block=*/shard=*/online_samples.parquet"))
    if not files:
        fail(f"No development online_samples.parquet files under {target_dir}")

    by_prompt: dict[str, dict[int, float]] = defaultdict(dict)
    for pp in files:
        tab = pq.read_table(pp, columns=["prompt_id", "sample_index", "correct"])
        for r in tab.to_pylist():
            pid = str(r["prompt_id"])
            idx = int(r["sample_index"])
            val = float(r["correct"])
            if val not in (0.0, 1.0):
                fail(f"Non-binary reward in {pp}: {val}")
            if idx in by_prompt[pid]:
                prev = by_prompt[pid][idx]
                if prev != val:
                    fail(f"Conflicting duplicate sample seed={seed} step={step} prompt={pid} idx={idx}")
            else:
                by_prompt[pid][idx] = val

    if not by_prompt:
        fail(f"No rows loaded from {target_dir}")

    full_l = min(len(v) for v in by_prompt.values())
    if full_l <= 0:
        fail(f"No complete sample prefix for seed={seed}, step={step}")

    packed: dict[str, tuple[float, ...]] = {}
    for pid, d in by_prompt.items():
        # Require a common contiguous prefix 0..full_l-1.
        missing = [i for i in range(full_l) if i not in d]
        if missing:
            fail(f"Non-contiguous development samples seed={seed} step={step} prompt={pid}; first missing={missing[0]}")
        packed[pid] = tuple(float(d[i]) for i in range(full_l))

    prompt_means = [math.fsum(xs) / full_l for xs in packed.values()]
    full_ref = math.fsum(prompt_means) / len(prompt_means)
    return TargetData(seed, step, packed, full_l, full_ref)

def candidate_reference(td: TargetData, L: int) -> tuple[float, dict[str, float]]:
    if L > td.full_l:
        fail(f"Candidate L={L} exceeds development full L={td.full_l} for seed={td.seed}, step={td.step}")
    pacc: dict[str, float] = {}
    for pid, xs in td.prompt_samples.items():
        pacc[pid] = math.fsum(xs[:L]) / L
    ref = math.fsum(pacc.values()) / len(pacc)
    return ref, pacc

def read_csv_dev(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("split", "")).strip().lower() == "development":
                rows.append(dict(r))
    return rows

def candidate_cost_fraction(main_l: int, audit_l: int, target_steps: Sequence[int]) -> float:
    audit_n = sum(int(s in AUDIT_STEPS) for s in target_steps)
    main_n = len(target_steps) - audit_n
    frozen = main_n * 8 + audit_n * 32
    cand = main_n * main_l + audit_n * audit_l
    return cand / frozen if frozen else float("nan")

def subset_name(seed: int) -> str:
    if seed in DEFAULT_CALIBRATION_SEEDS:
        return "calibration"
    if seed in DEFAULT_VALIDATION_SEEDS:
        return "validation"
    return "other"

def summarize_diffs(xs: Sequence[float]) -> dict[str, Any]:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    return {
        "n": len(ys),
        "mean": mean(ys),
        "median": percentile(ys, 0.50),
        "p95": percentile(ys, 0.95),
        "max": max_or_none(ys),
    }

def evaluate_candidate(
    *,
    main_l: int,
    audit_l: int,
    data: Mapping[tuple[int, int], TargetData],
    seeds: Sequence[int],
    steps: Sequence[int],
    t02_rows: Sequence[Mapping[str, str]],
    t04_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cand_ref: dict[tuple[int, int], float] = {}
    per_target: list[dict[str, Any]] = []

    for seed in seeds:
        for step in steps:
            td = data[(seed, step)]
            L = audit_l if step in AUDIT_STEPS else main_l
            if L > td.full_l:
                return ({
                    "main_l": main_l, "audit_l": audit_l,
                    "valid": False,
                    "reason": f"L={L} exceeds development full L={td.full_l} at seed={seed}, step={step}",
                }, [])
            cref, cp = candidate_reference(td, L)
            cand_ref[(seed, step)] = cref

            # Prompt-level difference from the full-L prompt means.
            prompt_diffs = []
            for pid, xs in td.prompt_samples.items():
                full_pa = math.fsum(xs) / td.full_l
                prompt_diffs.append(abs(cp[pid] - full_pa))
            per_target.append({
                "seed": seed,
                "subset": subset_name(seed),
                "target_step": step,
                "kind": "audit" if step in AUDIT_STEPS else "main",
                "candidate_l": L,
                "full_l": td.full_l,
                "full_reference": td.full_reference,
                "candidate_reference": cref,
                "reference_abs_diff": abs(cref - td.full_reference),
                "prompt_mae_vs_full": mean(prompt_diffs),
                "prompt_p95_abs_diff_vs_full": percentile(prompt_diffs, 0.95),
            })

    fixed_diffs_by_scope: dict[str, list[float]] = defaultdict(list)
    rolling_diffs_by_scope: dict[str, list[float]] = defaultdict(list)

    allowed = {(int(s), int(e)) for s in seeds for e in steps}

    for r in t02_rows:
        try:
            seed = int(r["training_seed"]); step = int(r["target_step"])
        except Exception:
            continue
        if (seed, step) not in allowed:
            continue
        est = safe_float(r.get("estimate"))
        if est is None:
            continue
        base = data[(seed, step)].full_reference
        cand = cand_ref[(seed, step)]
        delta = abs(abs(est - cand) - abs(est - base))
        fixed_diffs_by_scope["all"].append(delta)
        fixed_diffs_by_scope[subset_name(seed)].append(delta)

    for r in t04_rows:
        try:
            seed = int(r["training_seed"]); step = int(r["target_step"])
        except Exception:
            continue
        if (seed, step) not in allowed:
            continue
        old_est = safe_float(r.get("old_estimate"))
        recent_est = safe_float(r.get("recent_estimate"))
        if old_est is None or recent_est is None:
            continue
        base = data[(seed, step)].full_reference
        cand = cand_ref[(seed, step)]
        base_delta = abs(old_est - base) - abs(recent_est - base)
        cand_delta = abs(old_est - cand) - abs(recent_est - cand)
        d = abs(cand_delta - base_delta)
        rolling_diffs_by_scope["all"].append(d)
        rolling_diffs_by_scope[subset_name(seed)].append(d)

    summary: dict[str, Any] = {
        "main_l": main_l,
        "audit_l": audit_l,
        "valid": True,
        "estimated_generation_cost_fraction_of_frozen_test": candidate_cost_fraction(main_l, audit_l, steps),
        "estimated_generation_saving_fraction": 1.0 - candidate_cost_fraction(main_l, audit_l, steps),
        "scopes": {},
    }

    for scope in ("all", "calibration", "validation"):
        trs = [x for x in per_target if scope == "all" or x["subset"] == scope]
        refdiff = [float(x["reference_abs_diff"]) for x in trs]
        pmae = [float(x["prompt_mae_vs_full"]) for x in trs if x["prompt_mae_vs_full"] is not None]
        summary["scopes"][scope] = {
            "target_reference_abs_diff": summarize_diffs(refdiff),
            "target_prompt_mae_vs_full": summarize_diffs(pmae),
            "fixed_ope_abs_error_change": summarize_diffs(fixed_diffs_by_scope.get(scope, [])),
            "rolling_delta_error_change": summarize_diffs(rolling_diffs_by_scope.get(scope, [])),
        }
    return summary, per_target

def pass_scope(scope: Mapping[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    failures: list[str] = []
    ref = scope["target_reference_abs_diff"]
    if ref["n"] <= 0:
        failures.append("no target-reference comparisons")
    else:
        if ref["mean"] is None or ref["mean"] > args.ref_mae_tol:
            failures.append(f"reference mean {ref['mean']} > {args.ref_mae_tol}")
        if ref["p95"] is None or ref["p95"] > args.ref_p95_tol:
            failures.append(f"reference p95 {ref['p95']} > {args.ref_p95_tol}")

    fx = scope["fixed_ope_abs_error_change"]
    if fx["n"] > 0:
        if fx["mean"] is not None and fx["mean"] > args.fixed_mae_tol:
            failures.append(f"fixed-error mean {fx['mean']} > {args.fixed_mae_tol}")
        if fx["p95"] is not None and fx["p95"] > args.fixed_p95_tol:
            failures.append(f"fixed-error p95 {fx['p95']} > {args.fixed_p95_tol}")

    ro = scope["rolling_delta_error_change"]
    if ro["n"] > 0:
        if ro["mean"] is not None and ro["mean"] > args.rolling_mae_tol:
            failures.append(f"rolling-delta mean {ro['mean']} > {args.rolling_mae_tol}")
        if ro["p95"] is not None and ro["p95"] > args.rolling_p95_tol:
            failures.append(f"rolling-delta p95 {ro['p95']} > {args.rolling_p95_tol}")
    return not failures, failures

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

def flatten_summary_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for c in candidates:
        if not c.get("valid"):
            rows.append({
                "main_l": c.get("main_l"), "audit_l": c.get("audit_l"),
                "cost_fraction": None, "scope": "invalid", "pass": False,
                "ref_mean": None, "ref_p95": None,
                "fixed_mean": None, "fixed_p95": None,
                "rolling_mean": None, "rolling_p95": None,
                "reason": c.get("reason"),
            })
            continue
        for scope_name, s in c["scopes"].items():
            rows.append({
                "main_l": c["main_l"],
                "audit_l": c["audit_l"],
                "cost_fraction": c["estimated_generation_cost_fraction_of_frozen_test"],
                "scope": scope_name,
                "pass": c.get(f"pass_{scope_name}"),
                "ref_mean": s["target_reference_abs_diff"]["mean"],
                "ref_p95": s["target_reference_abs_diff"]["p95"],
                "fixed_mean": s["fixed_ope_abs_error_change"]["mean"],
                "fixed_p95": s["fixed_ope_abs_error_change"]["p95"],
                "rolling_mean": s["rolling_delta_error_change"]["mean"],
                "rolling_p95": s["rolling_delta_error_change"]["p95"],
                "reason": "; ".join(c.get(f"failures_{scope_name}", [])),
            })
    return rows

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Development-only Program 05 reference sample-size benchmark.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--main-l", nargs="+", default=["4", "8"],
                    help="Candidate non-audit L values, e.g. --main-l 4 8")
    ap.add_argument("--audit-l", nargs="+", default=["8", "16", "32"],
                    help="Candidate audit L values, e.g. --audit-l 8 16 32")
    ap.add_argument("--ref-mae-tol", type=float, default=0.005)
    ap.add_argument("--ref-p95-tol", type=float, default=0.0125)
    ap.add_argument("--fixed-mae-tol", type=float, default=0.005)
    ap.add_argument("--fixed-p95-tol", type=float, default=0.0125)
    ap.add_argument("--rolling-mae-tol", type=float, default=0.005)
    ap.add_argument("--rolling-p95-tol", type=float, default=0.0125)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    main_ls = parse_int_list(args.main_l)
    audit_ls = parse_int_list(args.audit_l)
    if not main_ls or not audit_ls or min(main_ls + audit_ls) <= 0:
        fail("Candidate L values must be positive.")

    # Hard boundary: every online sample path opened below is DEVELOPMENT only.
    dev_root = root / "data" / "online_reference" / "gsm8k" / "split=development"
    if "split=test" in str(dev_root).lower():
        fail("Internal safety guard: attempted test online-reference path.")

    frozen_seeds, reduced_steps, selection_path, selection_sha = load_reduced_steps(root)
    available = discover_target_dirs(dev_root)
    missing = [(s, e) for s in frozen_seeds for e in reduced_steps if (s, e) not in available]
    if missing:
        fail(f"Development Program 05 data missing required reduced-grid seed-targets; first missing: {missing[:5]}")

    print("=" * 100)
    print("DEVELOPMENT-ONLY PROGRAM 05 REFERENCE SAMPLE-SIZE BENCHMARK")
    print("TEST ONLINE SAMPLES AND TEST OPE OUTCOMES ARE NOT READ.")
    print(f"project root          : {root}")
    print(f"development root      : {dev_root}")
    print(f"training seeds        : {frozen_seeds}")
    print(f"reduced target steps  : {reduced_steps}")
    print(f"audit steps           : {sorted(AUDIT_STEPS & set(reduced_steps))}")
    print(f"candidate main L      : {main_ls}")
    print(f"candidate audit L     : {audit_ls}")
    print("=" * 100)

    data: dict[tuple[int, int], TargetData] = {}
    for i, (seed, step) in enumerate([(s, e) for s in frozen_seeds for e in reduced_steps], 1):
        td = load_target_data(seed, step, available[(seed, step)])
        data[(seed, step)] = td
        print(f"[{i:02d}/{len(frozen_seeds)*len(reduced_steps):02d}] seed={seed} e={step:03d} full_L={td.full_l} V_full={td.full_reference:.6f}")

    t02_path = root / "outputs" / "tables" / "T02_fixed_reuse.csv"
    t04_path = root / "outputs" / "tables" / "T04_rolling_comparison.csv"
    t02_rows = read_csv_dev(t02_path)
    t04_rows = read_csv_dev(t04_path)

    print()
    print(f"development T02 rows  : {len(t02_rows)}")
    print(f"development T04 rows  : {len(t04_rows)}")
    if not t02_rows:
        print("[NOTE] T02 development rows unavailable; downstream fixed-error distortion will be omitted.")
    if not t04_rows:
        print("[NOTE] T04 development rows unavailable; downstream rolling distortion will be omitted.")

    candidates = []
    target_rows_all = []
    for ml in main_ls:
        for al in audit_ls:
            c, target_rows = evaluate_candidate(
                main_l=ml, audit_l=al, data=data, seeds=frozen_seeds, steps=reduced_steps,
                t02_rows=t02_rows, t04_rows=t04_rows)
            if c.get("valid"):
                for scope in ("all", "calibration", "validation"):
                    ok, reasons = pass_scope(c["scopes"][scope], args)
                    c[f"pass_{scope}"] = ok
                    c[f"failures_{scope}"] = reasons
                # Require all-development and held-out training-seed validation to pass.
                c["decision"] = "PASS" if c["pass_all"] and c["pass_validation"] else "FAIL"
                for r in target_rows:
                    target_rows_all.append({"main_l": ml, "audit_l": al, **r})
            else:
                c["decision"] = "INVALID"
            candidates.append(c)

    # Select the cheapest passing design. Tie-break toward larger audit L, then larger main L.
    passing = [c for c in candidates if c.get("decision") == "PASS"]
    passing.sort(key=lambda c: (
        c["estimated_generation_cost_fraction_of_frozen_test"],
        -(c["audit_l"]),
        -(c["main_l"]),
    ))
    recommended = passing[0] if passing else None

    print("\nDECISION")
    for c in sorted(candidates, key=lambda x: (x.get("main_l", 0), x.get("audit_l", 0))):
        if not c.get("valid"):
            print(f"main={c['main_l']:>2} audit={c['audit_l']:>2} -> INVALID  {c.get('reason')}")
            continue
        v = c["scopes"]["validation"]
        print(
            f"main={c['main_l']:>2} audit={c['audit_l']:>2} "
            f"cost={100*c['estimated_generation_cost_fraction_of_frozen_test']:5.1f}% "
            f"val_ref_mean={fmt(v['target_reference_abs_diff']['mean'])} "
            f"val_ref_p95={fmt(v['target_reference_abs_diff']['p95'])} "
            f"val_fixed_mean={fmt(v['fixed_ope_abs_error_change']['mean'])} "
            f"val_roll_mean={fmt(v['rolling_delta_error_change']['mean'])} "
            f"-> {c['decision']}"
        )

    if recommended is None:
        print("\nRECOMMENDATION: NO REDUCED SAMPLE-SIZE DESIGN PASSED.")
        print("Keep frozen test Program 05 L_main=8 and L_audit=32 unless a new pre-analysis criterion is justified.")
    else:
        print(
            "\nRECOMMENDED DEVELOPMENT-ONLY DESIGN: "
            f"L_main={recommended['main_l']}, L_audit={recommended['audit_l']} "
            f"(estimated generation cost {100*recommended['estimated_generation_cost_fraction_of_frozen_test']:.1f}% "
            "of frozen reduced-grid Program 05)."
        )

    outdir = root / "benchmarks" / "program05_reference_sample_size"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "candidate_summary.csv"
    targets_csv = outdir / "per_target_diagnostics.csv"
    report_json = outdir / "benchmark_report.json"

    write_csv(summary_csv, flatten_summary_rows(candidates))
    write_csv(targets_csv, target_rows_all)

    input_hashes = {
        "reduced_test_pair_selection": {
            "path": str(selection_path.relative_to(root)).replace("\\", "/"),
            "sha256": selection_sha,
        },
        "T02_fixed_reuse_development_source": (
            {"path": str(t02_path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(t02_path)}
            if t02_path.exists() else None
        ),
        "T04_rolling_comparison_development_source": (
            {"path": str(t04_path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(t04_path)}
            if t04_path.exists() else None
        ),
    }
    report = {
        "schema_version": "1.0",
        "manifest_type": "development_only_program05_reference_sample_size_benchmark",
        "created_at_utc": now_utc(),
        "research_boundary": {
            "development_online_reference_only": True,
            "development_program06_rows_only": True,
            "test_online_reference_read": False,
            "test_ope_outcomes_read": False,
            "official_outputs_modified": False,
            "note": "This benchmark is intended to support an outcome-blind computational amendment before inspecting official-test OPE outcomes.",
        },
        "seeds": frozen_seeds,
        "calibration_seeds": sorted(DEFAULT_CALIBRATION_SEEDS & set(frozen_seeds)),
        "validation_seeds": sorted(DEFAULT_VALIDATION_SEEDS & set(frozen_seeds)),
        "target_steps": reduced_steps,
        "audit_steps": sorted(AUDIT_STEPS & set(reduced_steps)),
        "frozen_test_design": {"L_main": 8, "L_audit": 32},
        "candidate_main_l": main_ls,
        "candidate_audit_l": audit_ls,
        "tolerances": {
            "reference_mean_abs_diff": args.ref_mae_tol,
            "reference_p95_abs_diff": args.ref_p95_tol,
            "fixed_ope_abs_error_change_mean": args.fixed_mae_tol,
            "fixed_ope_abs_error_change_p95": args.fixed_p95_tol,
            "rolling_delta_error_change_mean": args.rolling_mae_tol,
            "rolling_delta_error_change_p95": args.rolling_p95_tol,
        },
        "development_full_l_by_seed_target": {
            f"{s}:{e}": data[(s, e)].full_l for s in frozen_seeds for e in reduced_steps
        },
        "inputs": input_hashes,
        "candidates": candidates,
        "recommended": (
            {
                "L_main": recommended["main_l"],
                "L_audit": recommended["audit_l"],
                "estimated_generation_cost_fraction_of_frozen_test": recommended["estimated_generation_cost_fraction_of_frozen_test"],
            }
            if recommended else None
        ),
        "outputs": {
            "candidate_summary_csv": str(summary_csv.relative_to(root)).replace("\\", "/"),
            "per_target_diagnostics_csv": str(targets_csv.relative_to(root)).replace("\\", "/"),
        },
    }
    report["content_sha256"] = sha256_obj({k: v for k, v in report.items() if k != "content_sha256"})
    atomic_write_json(report_json, report)

    print(f"\nCSV : {summary_csv}")
    print(f"CSV : {targets_csv}")
    print(f"JSON: {report_json}")
    print("No official Program 05/06/07 files were modified.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"\nBENCHMARK FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
