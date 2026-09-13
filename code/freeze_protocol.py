#!/usr/bin/env python3
"""Freeze the post-pilot GRPO-OPE paper protocol into an immutable lock.

This utility closes the deliberate boundary between development/pilot decisions
and the formal paper run.  It does not train or evaluate a model.  It reads the
already-pinned Program 00/01 manifests plus the resolved Program 02 training
configuration and writes manifests/protocol_lock.json exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROGRAM = "freeze_protocol.py"
PROGRAM_VERSION = "1.0"
PROJECT_NAME = "grpo_ope_reuse"

class FreezeProtocolError(RuntimeError):
    pass

def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FreezeProtocolError(f"Missing required upstream asset: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise FreezeProtocolError(f"Expected a JSON object: {path}")
    return obj

def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise FreezeProtocolError(f"PyYAML is required: {exc}") from exc
    if not path.exists():
        raise FreezeProtocolError(f"Missing protocol config: {path}")
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise FreezeProtocolError("protocol.yaml must contain a mapping.")
    return obj

def atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(str(path.parent), os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink()

def import_program02(root: Path):
    path = root / "02_train_all_seeds.py"
    spec = importlib.util.spec_from_file_location("grpo_ope_program02_for_lock", path)
    if spec is None or spec.loader is None:
        raise FreezeProtocolError(f"Cannot import Program 02 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        return out if len(out) == 40 else None
    except Exception:
        return None

def source_hashes(root: Path) -> dict[str, str]:
    names = [f"{i:02d}_{name}" for i, name in []]  # documentation-only placeholder; explicit scan below
    del names
    files = sorted(root.glob("[0-0][0-9]_*.py")) + [root / "freeze_protocol.py"]
    # glob above intentionally matches 00-09 only in this project; assert exact set.
    numbered = sorted(root.glob("??_*.py"))
    numbered = [p for p in numbered if p.name[:2].isdigit() and 0 <= int(p.name[:2]) <= 9]
    if len(numbered) != 10:
        raise FreezeProtocolError(f"Expected exactly Programs 00-09 before freezing; found {[p.name for p in numbered]}")
    files = numbered + [root / "freeze_protocol.py"]
    return {p.name: sha256_file(p) for p in files}

def validate_upstream(data: Mapping[str, Any], model: Mapping[str, Any], env: Mapping[str, Any], split: Mapping[str, Any]) -> tuple[str, str, str, str]:
    data_fp = data.get("content_fingerprint_sha256")
    model_primary = ((model.get("models") or {}).get("primary") or {}) if isinstance(model.get("models"), Mapping) else {}
    model_rev = model_primary.get("resolved_revision")
    env_fp = env.get("environment_fingerprint_sha256")
    split_fp = split.get("content_fingerprint_sha256")
    for label, value in (("data manifest fingerprint", data_fp), ("model revision", model_rev), ("environment fingerprint", env_fp), ("split fingerprint", split_fp)):
        if not isinstance(value, str) or not value:
            raise FreezeProtocolError(f"Missing {label}; rerun/verify Programs 00-01 before freezing.")
    return data_fp, model_rev, env_fp, split_fp

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze the post-pilot GRPO-OPE paper protocol exactly once.")
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--output-root", default=".")
    p.add_argument("--verify-only", action="store_true", help="Verify an existing lock against current frozen inputs; never write.")
    return p.parse_args(argv)

def build_lock(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    project = cfg.get("project") or {}
    if not isinstance(project, Mapping) or project.get("name") not in (None, PROJECT_NAME):
        raise FreezeProtocolError("Unexpected project.name in protocol.yaml.")
    pv = str(project.get("protocol_version", "1.0"))

    manifests = root / "manifests"
    data_path = manifests / "data_manifest.json"
    model_path = manifests / "model_manifest.json"
    env_path = manifests / "environment_manifest.json"
    split_path = manifests / "split_registry_manifest.json"
    data, model, env, split = map(read_json, (data_path, model_path, env_path, split_path))
    data_fp, model_rev, env_fp, split_fp = validate_upstream(data, model, env, split)

    p02 = import_program02(root)
    parsed_cfg, observed_cfg_sha = p02.load_protocol(config_path)
    if observed_cfg_sha != sha256_file(config_path):
        raise FreezeProtocolError("Program 02 and freeze_protocol disagree on protocol.yaml SHA-256.")
    training_spec = p02.parse_training_spec(parsed_cfg)
    p02.validate_paper_contract(training_spec)
    training_fp = p02.spec_fingerprint(training_spec)

    basis = {
        "protocol_version": pv,
        "config_sha256": observed_cfg_sha,
        "training_config_sha256": training_fp,
        "data_manifest_sha256": data_fp,
        "model_revision": model_rev,
        "environment_fingerprint_sha256": env_fp,
        "split_registry_sha256": split_fp,
        "manifest_file_sha256": {
            "data_manifest.json": sha256_file(data_path),
            "model_manifest.json": sha256_file(model_path),
            "environment_manifest.json": sha256_file(env_path),
            "split_registry_manifest.json": sha256_file(split_path),
        },
        "source_code_sha256": source_hashes(root),
        "git_commit": git_commit(root),
    }
    lock = {
        "schema_version": "1.0",
        "manifest_type": "grpo_ope_protocol_lock",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "project_name": PROJECT_NAME,
        **basis,
        "lock_fingerprint_sha256": sha256_bytes(canonical_bytes(basis)),
        "research_firewall": {
            "development_decisions_frozen": True,
            "official_test_tuning_forbidden": True,
            "gate_frozen_separately_after_program07_calibration": True,
        },
    }
    return lock

def comparable(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in lock.items() if k not in {"created_at_utc", "created_by"}}

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    config = Path(args.config)
    if not config.is_absolute(): config = root / config
    config = config.resolve()
    expected = build_lock(root, config)
    lock_path = root / "manifests" / "protocol_lock.json"
    if lock_path.exists():
        current = read_json(lock_path)
        if comparable(current) != comparable(expected):
            raise FreezeProtocolError("Existing protocol_lock.json does not match current frozen inputs. Do not overwrite it; create a new protocol version/project directory.")
        print(f"PROTOCOL LOCK VERIFIED: {lock_path}")
        return 0
    if args.verify_only:
        raise FreezeProtocolError("--verify-only requested but protocol_lock.json does not exist.")
    atomic_write_json(lock_path, expected)
    # Lock file itself becomes read-only best-effort; contents are never rewritten by this tool.
    try: lock_path.chmod(0o444)
    except OSError: pass
    print(f"PROTOCOL LOCK CREATED: {lock_path}")
    print(f"lock fingerprint: {expected['lock_fingerprint_sha256']}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreezeProtocolError as exc:
        print(f"PROGRAM FAILED: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
