#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DEFAULT = r"C:\lsg\grpo_ope_reuse"
BENCH_REL = Path("benchmarks/program05_reference_sample_size/benchmark_report.json")
SELECTION_REL = Path("manifests/reduced_test_pair_selection.json")
GRID_AMEND_REL = Path("manifests/protocol_amendment_reduced_grid.json")
OUT_REL = Path("manifests/protocol_amendment_program05_sample_size.json")

EXPECTED_STEPS = [0, 20, 100, 120, 160, 200, 220, 260, 300, 360, 400]
EXPECTED_SEEDS = [20260826, 20260827, 20260828]
CHOSEN_MAIN_L = 8
CHOSEN_AUDIT_L = 8
ORIGINAL_MAIN_L = 8
ORIGINAL_AUDIT_L = 32

class AmendmentError(RuntimeError):
    pass

def fail(msg: str) -> None:
    raise AmendmentError(msg)

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def canonical_bytes(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def sha256_obj(x: Any) -> str:
    return hashlib.sha256(canonical_bytes(x)).hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing required file: {path}")
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict):
        fail(f"Expected JSON object: {path}")
    return x

def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def scan_test_rows_in_tables(root: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    tdir = root / "outputs" / "tables"
    if not tdir.exists():
        return hits
    for p in sorted(tdir.glob("*.csv")):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                rd = csv.DictReader(f)
                if not rd.fieldnames or "split" not in rd.fieldnames:
                    continue
                n = 0
                for r in rd:
                    if str(r.get("split", "")).strip().lower() == "test":
                        n += 1
                if n:
                    hits.append({"path": str(p.relative_to(root)).replace("\\", "/"), "test_rows": n})
        except Exception as exc:
            fail(f"Could not inspect {p}: {exc}")
    return hits

def count_preexisting_test_online(root: Path) -> tuple[int, dict[str, int], list[str]]:
    base = root / "data" / "online_reference" / "gsm8k" / "split=test"
    total = 0
    by_target: dict[str, int] = {}
    ranges: set[str] = set()
    if not base.exists():
        return 0, by_target, []
    for mp in sorted(base.rglob("manifest.json")):
        try:
            m = read_json(mp)
        except AmendmentError:
            continue
        if m.get("manifest_type") != "online_reference_shard":
            continue
        seed = int(m.get("training_seed"))
        step = int(m.get("target_step"))
        s0 = int(m.get("sample_start"))
        s1 = int(m.get("sample_end_exclusive"))
        key = f"{seed}:{step}"
        by_target[key] = by_target.get(key, 0) + 1
        ranges.add(f"{s0}:{s1}")
        total += 1
    return total, by_target, sorted(ranges)

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Lock Program 05 8/8 sample-size amendment.")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    bench_path = root / BENCH_REL
    sel_path = root / SELECTION_REL
    grid_amend_path = root / GRID_AMEND_REL
    out_path = root / OUT_REL
    p05_path = root / "05_generate_online_reference.py"
    cfg_path = root / "configs" / "protocol.yaml"

    bench = read_json(bench_path)
    selection = read_json(sel_path)
    grid_amend = read_json(grid_amend_path)

    if bench.get("manifest_type") != "development_only_program05_reference_sample_size_benchmark":
        fail("Unexpected benchmark manifest_type.")
    rb = bench.get("research_boundary") or {}
    required_boundary = {
        "development_online_reference_only": True,
        "development_program06_rows_only": True,
        "test_online_reference_read": False,
        "test_ope_outcomes_read": False,
        "official_outputs_modified": False,
    }
    for k, v in required_boundary.items():
        if rb.get(k) is not v:
            fail(f"Benchmark research boundary failed for {k}: {rb.get(k)!r}")

    frozen = bench.get("frozen_test_design") or {}
    if int(frozen.get("L_main", -1)) != ORIGINAL_MAIN_L or int(frozen.get("L_audit", -1)) != ORIGINAL_AUDIT_L:
        fail(f"Benchmark frozen design is not {ORIGINAL_MAIN_L}/{ORIGINAL_AUDIT_L}.")

    candidates = bench.get("candidates") or []
    chosen = None
    for c in candidates:
        if int(c.get("main_l", -1)) == CHOSEN_MAIN_L and int(c.get("audit_l", -1)) == CHOSEN_AUDIT_L:
            chosen = c
            break
    if chosen is None:
        fail("Benchmark has no L_main=8, L_audit=8 candidate.")
    if chosen.get("decision") != "PASS" or chosen.get("pass_all") is not True or chosen.get("pass_validation") is not True:
        fail("Benchmark 8/8 candidate did not PASS all + held-out validation.")

    rows = selection.get("selected_pairs")
    if not isinstance(rows, list) or len(rows) != 36:
        fail("Reduced pair selection is not exactly 36 TEST pairs.")
    steps = sorted({int(r["target_step"]) for r in rows})
    seeds = sorted({int(r["training_seed"]) for r in rows})
    if steps != EXPECTED_STEPS or seeds != EXPECTED_SEEDS:
        fail(f"Unexpected reduced-grid plan: seeds={seeds}, steps={steps}")

    if ((grid_amend.get("selection_manifest") or {}).get("sha256") != sha256_file(sel_path)):
        fail("Reduced-grid amendment no longer matches selected-pair manifest.")

    # Outcome-blinding guard: no official Program 06 test table rows or test manifest may exist.
    table_test_rows = scan_test_rows_in_tables(root)
    p06_test_manifest = root / "outputs" / "diagnostics" / "program06_test_manifest.json"
    if p06_test_manifest.exists():
        fail("Program 06 TEST manifest already exists; refusing to create a pre-analysis sample-size amendment.")
    if table_test_rows:
        fail(f"Official TEST rows already exist in Program 06 tables: {table_test_rows}")

    pre_n, pre_by_target, pre_ranges = count_preexisting_test_online(root)

    val = (chosen.get("scopes") or {}).get("validation") or {}
    all_scope = (chosen.get("scopes") or {}).get("all") or {}

    manifest = {
        "schema_version": "1.0",
        "manifest_type": "preanalysis_program05_reference_sample_size_amendment",
        "created_at_utc": now_utc(),
        "scope": {
            "program": "05_generate_online_reference.py",
            "mode": "paper",
            "split": "test",
            "reduced_pair_grid_pairs": 36,
            "training_seeds": seeds,
            "target_steps": steps,
        },
        "original_frozen_design": {
            "L_main": ORIGINAL_MAIN_L,
            "L_audit": ORIGINAL_AUDIT_L,
            "audit_steps": [0, 100, 200, 300, 400],
        },
        "amended_effective_design": {
            "L_main": CHOSEN_MAIN_L,
            "L_audit": CHOSEN_AUDIT_L,
            "effective_samples_per_prompt": 8,
            "generation_distribution": "unchanged",
            "sample_block_size": 8,
            "note": (
                "The frozen Program 05 generation policy/configuration remains unchanged. "
                "The amendment changes only how many Monte Carlo samples per prompt are required "
                "for official TEST reference summaries: all selected target checkpoints use the first 8 samples."
            ),
        },
        "selection_basis": {
            "benchmark_report": {
                "path": str(BENCH_REL).replace("\\", "/"),
                "file_sha256": sha256_file(bench_path),
                "embedded_content_sha256": bench.get("content_sha256"),
            },
            "candidate": {"L_main": 8, "L_audit": 8, "decision": chosen.get("decision")},
            "estimated_generation_cost_fraction_of_frozen_reduced_grid": chosen.get(
                "estimated_generation_cost_fraction_of_frozen_test"
            ),
            "held_out_validation_metrics": {
                "target_reference_abs_diff": val.get("target_reference_abs_diff"),
                "target_prompt_mae_vs_full": val.get("target_prompt_mae_vs_full"),
                "fixed_ope_abs_error_change": val.get("fixed_ope_abs_error_change"),
                "rolling_delta_error_change": val.get("rolling_delta_error_change"),
            },
            "all_development_metrics": {
                "target_reference_abs_diff": all_scope.get("target_reference_abs_diff"),
                "fixed_ope_abs_error_change": all_scope.get("fixed_ope_abs_error_change"),
                "rolling_delta_error_change": all_scope.get("rolling_delta_error_change"),
            },
            "rationale": [
                "The development-only benchmark shows L_main=8, L_audit=8 passes both all-development and held-out-seed criteria.",
                "This choice preserves the originally frozen non-audit L_main=8 and changes only audit oversampling from 32 to 8.",
                "It is more conservative than the benchmark's cheapest passing 4/8 design while still materially reducing computation.",
                "It reuses already-published 0:8 official TEST online shards without truncating them or changing their generation semantics.",
            ],
        },
        "outcome_blinding_and_timing": {
            "program06_test_manifest_present_at_amendment": False,
            "test_rows_detected_in_outputs_tables_at_amendment": 0,
            "benchmark_declares_test_online_reference_read": False,
            "benchmark_declares_test_ope_outcomes_read": False,
            "program05_test_generation_had_already_started": pre_n > 0,
            "preexisting_program05_test_online_shard_units": pre_n,
            "preexisting_units_by_seed_target": pre_by_target,
            "preexisting_sample_ranges": pre_ranges,
            "statement": (
                "This is a post-start computational amendment made after some Program 05 TEST "
                "online generation had begun, but before any Program 06 official-TEST OPE/ESS/gate "
                "outputs existed. The selection criterion is development-only and does not use "
                "official-test OPE outcomes."
            ),
        },
        "immutability_and_provenance": {
            "reduced_pair_selection": {
                "path": str(SELECTION_REL).replace("\\", "/"),
                "sha256": sha256_file(sel_path),
            },
            "reduced_grid_amendment": {
                "path": str(GRID_AMEND_REL).replace("\\", "/"),
                "sha256": sha256_file(grid_amend_path),
            },
            "program05_source": {
                "path": "05_generate_online_reference.py",
                "sha256": sha256_file(p05_path),
            },
            "protocol_config": {
                "path": "configs/protocol.yaml",
                "sha256": sha256_file(cfg_path),
            },
            "frozen_gate_changed": False,
            "program04_outputs_changed": False,
            "pair_registry_changed": False,
        },
        "paper_disclosure": (
            "Before inspecting official-test OPE outcomes, a development-only sensitivity analysis "
            "showed that using eight on-policy Monte Carlo samples per prompt at the audit checkpoints "
            "introduced negligible reference distortion on a held-out development seed. We therefore "
            "retained the frozen L_main=8 and reduced only the audit reference size from L_audit=32 to 8 "
            "for the reduced test grid."
        ),
    }
    manifest["content_sha256"] = sha256_obj({k: v for k, v in manifest.items() if k != "content_sha256"})
    atomic_write_json(out_path, manifest)

    print("=" * 94)
    print("PROGRAM 05 SAMPLE-SIZE AMENDMENT LOCKED")
    print(f"benchmark SHA-256 : {sha256_file(bench_path)}")
    print(f"chosen design      : L_main=8, L_audit=8")
    print(f"test OPE rows      : 0")
    print(f"preexisting P05 TEST shard units: {pre_n}")
    print(f"manifest           : {out_path}")
    print(f"manifest SHA-256   : {sha256_file(out_path)}")
    print("=" * 94)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AmendmentError as exc:
        print(f"\nAMENDMENT PREPARATION FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
