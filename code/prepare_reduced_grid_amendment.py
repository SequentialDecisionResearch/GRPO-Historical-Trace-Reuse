#!/usr/bin/env python3
"""
Prepare a transparent post-start reduced-grid amendment for the GRPO-OPE study.

This script:
  1) verifies the DEVELOPMENT-ONLY pair-grid benchmark recommendation;
  2) verifies that official Program06/07 TEST outcome analysis has not run;
  3) leaves the original frozen pair registry unchanged;
  4) creates an immutable TEST pair-selection manifest for the C001 reduced grid;
  5) creates an amendment manifest that hashes the protocol, gate, benchmark,
     original pair registry, and selected pair manifest.

It DOES NOT:
  - delete or modify existing Program04 test shards;
  - change the frozen gate;
  - modify pair_registry.parquet;
  - modify Program04 source code;
  - run any TEST OPE analysis.

A later reduced-grid runner/patch must explicitly consume
manifests/reduced_test_pair_selection.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "1.0"
AMENDMENT_TYPE = "poststart_reduced_test_pair_grid_amendment"
SELECTION_TYPE = "reduced_test_pair_selection"

EXPECTED_CANDIDATE = "C001"
EXPECTED_TOTAL_PAIRS = 36
EXPECTED_IDENTITY_STEPS = (0, 100, 200, 300)
EXPECTED_FIXED_TARGETS = (20, 120, 220, 300, 400)
EXPECTED_ROLLING = ((100, 160), (200, 260), (300, 360))
EXPECTED_SEEDS = (20260826, 20260827, 20260828)


class PrepError(RuntimeError):
    pass


def die(msg: str) -> None:
    raise PrepError(msg)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(x: Any) -> bytes:
    return json.dumps(
        x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        die(f"Missing required file: {path}")
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"Cannot read JSON {path}: {exc}")
    if not isinstance(x, dict):
        die(f"Expected JSON object: {path}")
    return x


def atomic_write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == data:
            print(f"[OK existing identical] {path}")
            return
        die(f"Refusing to overwrite existing non-identical manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def verify_recommendation(root: Path) -> tuple[Path, dict[str, Any]]:
    p = root / "benchmarks" / "pair_grid_reduction" / "pair_grid_recommendation.json"
    rec = read_json(p)
    r = rec.get("recommended")
    if not isinstance(r, Mapping):
        die("Benchmark has no recommended grid.")
    if str(r.get("candidate_id")) != EXPECTED_CANDIDATE:
        die(
            f"Expected benchmark recommendation {EXPECTED_CANDIDATE}, "
            f"observed {r.get('candidate_id')!r}"
        )
    if int(r.get("total_pairs_all_seeds", -1)) != EXPECTED_TOTAL_PAIRS:
        die("Recommended total pair count is not 36.")
    fixed = parse_csv_ints(str(r.get("fixed_targets", "")))
    if fixed != EXPECTED_FIXED_TARGETS:
        die(f"Unexpected fixed targets: {fixed}")
    lags = parse_csv_ints(str(r.get("rolling_lags", "")))
    if lags != (60,):
        die(f"Unexpected rolling lags: {lags}")
    return p, rec


def verify_no_test_outcome_analysis(root: Path) -> None:
    blockers = [
        root / "outputs" / "diagnostics" / "program06_test_manifest.json",
        root / "outputs" / "diagnostics" / "program07_test-only_manifest.json",
    ]
    existing = [p for p in blockers if p.exists()]
    if existing:
        die(
            "Official TEST outcome analysis already exists; refusing to certify a "
            "pre-outcome reduced-grid amendment:\n  "
            + "\n  ".join(str(p) for p in existing)
        )

    # Also fail if T02/T04 already contain split=test rows.
    for name in ("T02_fixed_reuse.csv", "T04_rolling_comparison.csv"):
        p = root / "outputs" / "tables" / name
        if not p.exists():
            continue
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if str(row.get("split", "")).strip().lower() == "test":
                    die(
                        f"{name} already contains split=test rows; refusing to certify "
                        "that reduction was chosen before official TEST OPE outcome analysis."
                    )


def load_pair_registry(root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    pq_path = root / "manifests" / "pair_registry.parquet"
    m_path = root / "manifests" / "pair_registry_manifest.json"
    if not pq_path.exists() or not m_path.exists():
        die("Missing frozen pair registry or its manifest.")
    m = read_json(m_path)
    expected = ((m.get("parquet") or {}).get("file_sha256"))
    observed = sha256_file(pq_path)
    if expected is not None and str(expected) != observed:
        die("pair_registry.parquet hash mismatch with pair_registry_manifest.json")
    try:
        import pyarrow.parquet as pq
        rows = pq.read_table(pq_path).to_pylist()
    except Exception as exc:
        die(f"Cannot read pair registry parquet: {exc}")
    return pq_path, m_path, [dict(r) for r in rows]


def keep_pair(r: Mapping[str, Any]) -> bool:
    if str(r.get("split")) != "test":
        return False
    seed = int(r["training_seed"])
    if seed not in EXPECTED_SEEDS:
        return False
    b = int(r["behavior_step"])
    e = int(r["target_step"])
    purpose = str(r["purpose"])
    if purpose == "identity":
        return b == e and e in EXPECTED_IDENTITY_STEPS
    if purpose == "fixed":
        return b == 0 and e in EXPECTED_FIXED_TARGETS
    if purpose == "rolling":
        return (b, e) in EXPECTED_ROLLING
    return False


def selection_rows(registry: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in registry:
        if keep_pair(r):
            rows.append(
                {
                    "pair_id": str(r["pair_id"]),
                    "training_seed": int(r["training_seed"]),
                    "dataset": str(r["dataset"]),
                    "split": str(r["split"]),
                    "behavior_step": int(r["behavior_step"]),
                    "target_step": int(r["target_step"]),
                    "purpose": str(r["purpose"]),
                    "protocol_version": str(r["protocol_version"]),
                }
            )
    rows.sort(
        key=lambda x: (
            x["training_seed"],
            0 if x["purpose"] == "identity" else 1 if x["purpose"] == "fixed" else 2,
            x["target_step"],
            x["behavior_step"],
            x["pair_id"],
        )
    )

    if len(rows) != EXPECTED_TOTAL_PAIRS:
        die(f"Expected 36 selected TEST pairs, observed {len(rows)}")

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(r["training_seed"], []).append(r)
    if set(by_seed) != set(EXPECTED_SEEDS):
        die(f"Unexpected selected seed set: {sorted(by_seed)}")
    for seed, rr in sorted(by_seed.items()):
        n_id = sum(r["purpose"] == "identity" for r in rr)
        n_fx = sum(r["purpose"] == "fixed" for r in rr)
        n_ro = sum(r["purpose"] == "rolling" for r in rr)
        if (n_id, n_fx, n_ro) != (4, 5, 3):
            die(
                f"Seed {seed}: expected identity/fixed/rolling=(4,5,3), "
                f"observed {(n_id,n_fx,n_ro)}"
            )
    return rows


def count_existing_test_rescore_manifests(root: Path) -> int:
    d = root / "data" / "target_rescores" / "gsm8k" / "split=test"
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("manifest.json") if p.parent.name.startswith("shard="))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    print("=" * 96)
    print("PREPARE REDUCED TEST PAIR-GRID AMENDMENT")
    print("=" * 96)

    # Core frozen artifacts.
    protocol_lock = root / "manifests" / "protocol_lock.json"
    gate = root / "outputs" / "frozen_gate.json"
    gate_manifest = root / "manifests" / "frozen_gate_manifest.json"
    config = root / "configs" / "protocol.yaml"
    p04 = root / "04_rescore_target_checkpoints.py"

    for p in (protocol_lock, gate, gate_manifest, config, p04):
        if not p.exists():
            die(f"Missing required artifact: {p}")

    verify_no_test_outcome_analysis(root)
    benchmark_path, benchmark = verify_recommendation(root)
    registry_path, registry_manifest_path, registry = load_pair_registry(root)
    selected = selection_rows(registry)

    benchmark_candidates = (
        root / "benchmarks" / "pair_grid_reduction" / "pair_grid_candidates.csv"
    )
    if not benchmark_candidates.exists():
        die(f"Missing benchmark candidates CSV: {benchmark_candidates}")

    manifests = root / "manifests"
    selection_path = manifests / "reduced_test_pair_selection.json"
    amendment_path = manifests / "protocol_amendment_reduced_grid.json"

    selection_payload = {
        "schema_version": SCHEMA,
        "manifest_type": SELECTION_TYPE,
        "created_at_utc": now_utc(),
        "dataset": "GSM8K",
        "split": "test",
        "selection_basis": "development_only_pair_grid_reduction_benchmark",
        "benchmark_candidate": EXPECTED_CANDIDATE,
        "all_training_seeds": list(EXPECTED_SEEDS),
        "design": {
            "identity_steps_per_seed": list(EXPECTED_IDENTITY_STEPS),
            "fixed_pairs_per_seed": [[0, e] for e in EXPECTED_FIXED_TARGETS],
            "rolling_pairs_per_seed": [list(x) for x in EXPECTED_ROLLING],
            "pairs_per_seed": 12,
            "total_pairs": EXPECTED_TOTAL_PAIRS,
        },
        "selected_pairs": selected,
        "selected_pair_ids_sha256": sha256_bytes(
            canonical_bytes([r["pair_id"] for r in selected])
        ),
        "original_pair_registry": {
            "path": "manifests/pair_registry.parquet",
            "file_sha256": sha256_file(registry_path),
            "manifest_path": "manifests/pair_registry_manifest.json",
            "manifest_sha256": sha256_file(registry_manifest_path),
            "unchanged": True,
        },
        "test_outcome_firewall": {
            "program06_test_manifest_absent_at_amendment": True,
            "program07_test_only_manifest_absent_at_amendment": True,
            "T02_T04_contained_no_test_rows_at_amendment": True,
        },
    }
    atomic_write_json_once(selection_path, selection_payload)
    selection_sha = sha256_file(selection_path)

    old_p04_amendment = manifests / "protocol_amendment_program04.json"
    old_p04_amendment_info = None
    if old_p04_amendment.exists():
        old_p04_amendment_info = {
            "path": "manifests/protocol_amendment_program04.json",
            "sha256": sha256_file(old_p04_amendment),
        }

    amendment_payload = {
        "schema_version": SCHEMA,
        "manifest_type": AMENDMENT_TYPE,
        "created_at_utc": now_utc(),
        "project": "grpo_ope_reuse",
        "reason": (
            "After official Program04 TEST rescoring began, measured wall-clock cost "
            "was materially higher than desired. Before any official TEST OPE outcome "
            "analysis, a DEVELOPMENT-only benchmark compared geometry-defined reduced "
            "pair grids. Candidate C001 was the smallest tested grid satisfying the "
            "predeclared benchmark distortion criteria and was therefore selected to "
            "bound remaining computation."
        ),
        "scientific_status": (
            "post-start computational/design amendment selected without official TEST "
            "OPE outcome analysis; not the originally frozen full 117-pair test design"
        ),
        "original_full_test_pairs": 117,
        "amended_test_pairs": EXPECTED_TOTAL_PAIRS,
        "reduction_fraction": EXPECTED_TOTAL_PAIRS / 117.0,
        "selection_manifest": {
            "path": "manifests/reduced_test_pair_selection.json",
            "sha256": selection_sha,
        },
        "development_benchmark": {
            "recommendation_path": str(benchmark_path.relative_to(root)).replace("\\", "/"),
            "recommendation_sha256": sha256_file(benchmark_path),
            "candidates_path": str(benchmark_candidates.relative_to(root)).replace("\\", "/"),
            "candidates_sha256": sha256_file(benchmark_candidates),
            "recommended_candidate": EXPECTED_CANDIDATE,
            "recommended_total_pairs": EXPECTED_TOTAL_PAIRS,
            "test_outcomes_read_by_benchmark": False,
        },
        "frozen_inputs": {
            "protocol_yaml_sha256": sha256_file(config),
            "protocol_lock_sha256": sha256_file(protocol_lock),
            "frozen_gate_sha256": sha256_file(gate),
            "frozen_gate_manifest_sha256": sha256_file(gate_manifest),
            "pair_registry_parquet_sha256": sha256_file(registry_path),
            "pair_registry_manifest_sha256": sha256_file(registry_manifest_path),
            "program04_source_sha256_at_amendment": sha256_file(p04),
            "prior_program04_amendment": old_p04_amendment_info,
        },
        "invariants": {
            "frozen_gate_unchanged": True,
            "pair_registry_unchanged": True,
            "program04_scoring_contract_unchanged": True,
            "program04_existing_test_shards_are_not_deleted": True,
            "trajectory_sample_size_K_unchanged": True,
            "training_seed_set_unchanged": True,
            "reduction_is_pair_grid_only": True,
        },
        "existing_test_state": {
            "program04_test_had_started": True,
            "existing_test_rescore_manifest_count_at_amendment": count_existing_test_rescore_manifests(root),
            "official_test_ope_analysis_had_not_started": True,
        },
        "required_next_step": (
            "Use a transparent reduced-grid Program04 runner/patch that consumes "
            "manifests/reduced_test_pair_selection.json. Do not resume the unmodified "
            "full-grid Program04 command."
        ),
    }
    atomic_write_json_once(amendment_path, amendment_payload)

    print(f"[PASS] benchmark candidate : {EXPECTED_CANDIDATE}")
    print(f"[PASS] selected TEST pairs : {len(selected)}")
    print(f"[PASS] pair registry        : unchanged")
    print(f"[PASS] frozen gate          : unchanged")
    print(f"[PASS] TEST OPE analysis    : not yet run")
    print(f"selection manifest          : {selection_path}")
    print(f"selection SHA-256           : {sha256_file(selection_path)}")
    print(f"amendment manifest          : {amendment_path}")
    print(f"amendment SHA-256           : {sha256_file(amendment_path)}")
    print("\nNEXT: do NOT resume the original full-grid Program04 yet.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(f"\nPREPARATION FAILED\n{exc}")
        raise SystemExit(2)
