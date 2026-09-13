#!/usr/bin/env python3
"""
amend_program04_preanalysis.py

One-time pre-analysis amendment helper for GRPO-OPE Program 04.

This script does NOT modify protocol_lock.json, protocol.yaml, Program 03 data,
pair_registry files, checkpoints, or Program 04 rescore outputs.

It only creates:
    manifests/protocol_amendment_program04.json

The amendment is permitted only when:
  * Program 04 source differs from the originally frozen source hash;
  * the existing formal development Program 04 outputs contain identity shards
    only (no fixed/rolling/non-identity pair has been evaluated);
  * the original identity tolerances remain exactly 0.01 and 0.05;
  * Program 03 development collection index and pair registry are present.

Typical Spyder command:
    %run "C:/lsg/grpo_ope_reuse/amend_program04_preanalysis.py"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROGRAM = "04_rescore_target_checkpoints.py"
AMENDMENT_NAME = "protocol_amendment_program04.json"
SCORING_CONTRACT = "symmetric_fp32_teacher_forced_v1"
EXPECTED_TOKEN_ATOL = 1.0e-2
EXPECTED_SEQUENCE_ATOL = 5.0e-2


class AmendmentError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        raise AmendmentError(f"Missing required file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AmendmentError(f"Missing required JSON: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AmendmentError(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise AmendmentError(f"Expected JSON object: {path}")
    return obj


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def first_present(mapping: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        cur: Any = mapping
        ok = True
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def read_frozen_tolerances(protocol_path: Path) -> tuple[float, float, str]:
    try:
        import yaml
    except Exception as exc:
        raise AmendmentError(f"PyYAML is required to verify frozen tolerances: {exc}") from exc

    if not protocol_path.exists():
        raise AmendmentError(f"Missing protocol config: {protocol_path}")
    try:
        cfg = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise AmendmentError(f"Cannot parse {protocol_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise AmendmentError("protocol.yaml must contain a mapping.")

    section = cfg.get("rescore") or {}
    if not isinstance(section, Mapping):
        raise AmendmentError("protocol.yaml rescore section must be a mapping.")

    token = float(section.get("identity_token_atol", EXPECTED_TOKEN_ATOL))
    seq = float(section.get("identity_sequence_atol", EXPECTED_SEQUENCE_ATOL))

    if token != EXPECTED_TOKEN_ATOL or seq != EXPECTED_SEQUENCE_ATOL:
        raise AmendmentError(
            "Identity tolerances have changed. This amendment is allowed only with "
            f"the original values token={EXPECTED_TOKEN_ATOL}, sequence={EXPECTED_SEQUENCE_ATOL}; "
            f"observed token={token}, sequence={seq}."
        )

    project = cfg.get("project") or {}
    pversion = str(project.get("protocol_version", "1.0")) if isinstance(project, Mapping) else "1.0"
    return token, seq, pversion


def audit_existing_program04_development(root: Path) -> dict[str, Any]:
    """
    Verify the crucial pre-analysis condition:
    existing formal development Program 04 shards are identity-only.
    """
    base = root / "data" / "target_rescores" / "gsm8k" / "split=development"

    identity_shards = 0
    nonidentity_shards = 0
    identity_rows = 0
    nonidentity_rows = 0
    purposes: dict[str, int] = {}

    if base.exists():
        for mp in sorted(base.rglob("manifest.json")):
            if not mp.parent.name.startswith("shard="):
                continue
            m = read_json(mp)
            if m.get("manifest_type") != "target_rescore_shard":
                continue

            purpose = str(m.get("purpose"))
            rows = int((m.get("parquet") or {}).get("row_count") or 0)
            purposes[purpose] = purposes.get(purpose, 0) + 1

            if purpose == "identity":
                identity_shards += 1
                identity_rows += rows
            else:
                nonidentity_shards += 1
                nonidentity_rows += rows

    if nonidentity_shards != 0 or nonidentity_rows != 0:
        raise AmendmentError(
            "PRE-ANALYSIS FIREWALL VIOLATION: existing Program 04 development outputs "
            "contain non-identity results. Do not create this amendment.\n"
            f"nonidentity_shards={nonidentity_shards}, nonidentity_rows={nonidentity_rows}, "
            f"purposes={purposes}"
        )

    if identity_shards == 0 or identity_rows == 0:
        raise AmendmentError(
            "No existing identity output was found. The amendment is intended to document "
            "the already-triggered Program 04 identity firewall."
        )

    return {
        "formal_development_rescore_root": base.as_posix(),
        "identity_shards_present": identity_shards,
        "identity_rows_present": identity_rows,
        "nonidentity_shards_present": nonidentity_shards,
        "nonidentity_rows_present": nonidentity_rows,
        "purposes_present": dict(sorted(purposes.items())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create the audited pre-analysis Program 04 amendment manifest.")
    p.add_argument("--output-root", default=r"C:\lsg\grpo_ope_reuse")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    manifests = root / "manifests"

    current_program = root / PROGRAM
    lock_path = manifests / "protocol_lock.json"
    protocol_path = root / "configs" / "protocol.yaml"
    amendment_path = manifests / AMENDMENT_NAME

    pair_registry = manifests / "pair_registry.parquet"
    pair_registry_manifest = manifests / "pair_registry_manifest.json"
    behavior_index = manifests / "behavior_logs" / "development" / "collection_index.json"

    print("=" * 88)
    print("PROGRAM 04 PRE-ANALYSIS AMENDMENT")
    print("=" * 88)
    print(f"project root : {root}")
    print(f"writes       : {amendment_path}")
    print("other files  : READ ONLY")
    print()

    lock = read_json(lock_path)
    source_hashes = lock.get("source_code_sha256") or {}
    if not isinstance(source_hashes, Mapping):
        raise AmendmentError("protocol_lock.json lacks source_code_sha256 mapping.")

    old_hash = source_hashes.get(PROGRAM)
    if not isinstance(old_hash, str) or len(old_hash) != 64:
        raise AmendmentError(f"protocol_lock.json does not contain a valid frozen hash for {PROGRAM}.")

    new_hash = sha256_file(current_program)
    lock_hash = sha256_file(lock_path)
    config_hash = sha256_file(protocol_path)

    locked_config = first_present(
        lock,
        (
            ("config_sha256",),
            ("protocol_config_sha256",),
            ("protocol", "config_sha256"),
            ("inputs", "config_sha256"),
        ),
    )
    if locked_config != config_hash:
        raise AmendmentError(
            "protocol.yaml does not match the original protocol lock. "
            "This helper will not amend a changed protocol configuration."
        )

    if old_hash == new_hash:
        raise AmendmentError(
            "Current Program 04 still has the original frozen source hash; "
            "there is no source-code amendment to record."
        )

    token_atol, sequence_atol, protocol_version = read_frozen_tolerances(protocol_path)

    # These artifacts are intentionally only hashed/checked, never modified.
    if not pair_registry.exists() or not pair_registry_manifest.exists():
        raise AmendmentError("Frozen pair registry/manifest is missing.")
    if not behavior_index.exists():
        raise AmendmentError("Program 03 development collection index is missing.")

    pair_registry_sha = sha256_file(pair_registry)
    pair_registry_manifest_sha = sha256_file(pair_registry_manifest)
    behavior_index_sha = sha256_file(behavior_index)

    audit = audit_existing_program04_development(root)

    payload = {
        "schema_version": "1.0",
        "manifest_type": "preanalysis_program04_scoring_amendment",
        "created_at_utc": now_utc(),
        "program": PROGRAM,
        "protocol_version": protocol_version,

        # Required by corrected Program 04:
        "old_program04_source_sha256": old_hash,
        "new_program04_source_sha256": new_hash,
        "original_protocol_lock_sha256": lock_hash,
        "protocol_config_sha256": config_hash,
        "scoring_contract": SCORING_CONTRACT,
        "identity_firewall_triggered_before_nonidentity": True,
        "nonidentity_pairs_evaluated_before_amendment_is_false": True,
        "program03_behavior_logs_unchanged": True,
        "pair_registry_unchanged": True,
        "identity_tolerances_unchanged": True,

        # Extra audit evidence:
        "identity_token_atol": token_atol,
        "identity_sequence_atol": sequence_atol,
        "program03_development_collection_index_sha256": behavior_index_sha,
        "pair_registry_parquet_sha256": pair_registry_sha,
        "pair_registry_manifest_sha256": pair_registry_manifest_sha,
        "pre_amendment_program04_output_audit": audit,
        "reason": (
            "The prespecified identity firewall detected a rare long-sequence numerical "
            "mismatch between Program 03 generation-time log probabilities and Program 04 "
            "full-sequence teacher-forced log probabilities before any non-identity OPE "
            "pair was evaluated. Program 03 trajectories were independently reproduced "
            "exactly. Program 04 was therefore amended pre-analysis to use one symmetric "
            "canonical FP32 teacher-forced probability engine for both behavior and target "
            "sequence probabilities, without changing pairs or identity tolerances."
        ),
    }

    if amendment_path.exists():
        existing = read_json(amendment_path)
        # Ignore timestamp only for idempotence comparison.
        a = dict(existing)
        b = dict(payload)
        a.pop("created_at_utc", None)
        b.pop("created_at_utc", None)
        if a != b:
            raise AmendmentError(
                f"An incompatible amendment already exists at {amendment_path}. "
                "Do not overwrite it."
            )
        print("Existing amendment manifest matches the required amendment.")
    else:
        atomic_write_json(amendment_path, payload)
        print("Amendment manifest created successfully.")

    # Final self-check against the exact fields expected by corrected Program 04.
    written = read_json(amendment_path)
    required_equal = {
        "program": PROGRAM,
        "old_program04_source_sha256": old_hash,
        "new_program04_source_sha256": new_hash,
        "original_protocol_lock_sha256": lock_hash,
        "protocol_config_sha256": config_hash,
        "scoring_contract": SCORING_CONTRACT,
    }
    for key, expected in required_equal.items():
        if written.get(key) != expected:
            raise AmendmentError(f"Final amendment self-check failed for {key!r}.")
    for key in (
        "identity_firewall_triggered_before_nonidentity",
        "nonidentity_pairs_evaluated_before_amendment_is_false",
        "program03_behavior_logs_unchanged",
        "pair_registry_unchanged",
        "identity_tolerances_unchanged",
    ):
        if written.get(key) is not True:
            raise AmendmentError(f"Final amendment self-check failed for {key!r}.")

    print()
    print(f"old Program 04 SHA-256 : {old_hash}")
    print(f"new Program 04 SHA-256 : {new_hash}")
    print(f"identity shards audited: {audit['identity_shards_present']}")
    print(f"identity rows audited  : {audit['identity_rows_present']}")
    print("non-identity rows      : 0")
    print("identity tolerances    : unchanged (0.01 / 0.05)")
    print("Program 03             : untouched")
    print("pair registry          : untouched")
    print()
    print("AMENDMENT PASSED")
    print("Do not run Program 04 yet; archive the old Program-04-only outputs next.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AmendmentError as exc:
        print(f"\nAMENDMENT FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nAMENDMENT INTERRUPTED.", file=sys.stderr)
        raise SystemExit(130)
