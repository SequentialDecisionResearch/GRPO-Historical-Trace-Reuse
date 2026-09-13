#!/usr/bin/env python3
"""
Run original Program 04 on the frozen reduced TEST pair selection.

This wrapper does NOT modify 04_rescore_target_checkpoints.py or pair_registry.parquet.
It verifies the reduced-grid amendment, imports the original Program 04, lets the
original full pair registry verify normally, then filters the in-memory registry
to the 36 predeclared TEST pair_ids before Program 04 selects/executes pairs.

Existing valid Program-04 shards are reused by the original --resume logic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

SELECTION_REL = Path("manifests/reduced_test_pair_selection.json")
AMENDMENT_REL = Path("manifests/protocol_amendment_reduced_grid.json")
RUNNER_MANIFEST_REL = Path("manifests/reduced_grid_runner_manifest.json")
ORIGINAL_P04_REL = Path("04_rescore_target_checkpoints.py")

EXPECTED_SELECTION_TYPE = "reduced_test_pair_selection"
EXPECTED_AMENDMENT_TYPE = "poststart_reduced_test_pair_grid_amendment"
EXPECTED_PAIR_COUNT = 36


class ReducedRunnerError(RuntimeError):
    pass


def fail(msg: str):
    raise ReducedRunnerError(msg)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing required file: {path}")
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Cannot read JSON {path}: {exc}")
    if not isinstance(x, dict):
        fail(f"Expected JSON object: {path}")
    return x


def atomic_write_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            fail(f"Refusing to overwrite non-identical runner manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("program04_original_reduced_runner", path)
    if spec is None or spec.loader is None:
        fail(f"Cannot import original Program 04: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_wrapper_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Reduced-grid wrapper around original Program 04."
    )
    ap.add_argument("--root", default=".")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--verify-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_wrapper_args(argv)
    root = Path(args.root).resolve()

    selection_path = root / SELECTION_REL
    amendment_path = root / AMENDMENT_REL
    original_path = root / ORIGINAL_P04_REL
    wrapper_path = Path(__file__).resolve()

    selection = read_json(selection_path)
    amendment = read_json(amendment_path)

    if selection.get("manifest_type") != EXPECTED_SELECTION_TYPE:
        fail("Invalid reduced TEST selection manifest type.")
    if amendment.get("manifest_type") != EXPECTED_AMENDMENT_TYPE:
        fail("Invalid reduced-grid amendment manifest type.")

    selected_pairs = selection.get("selected_pairs")
    if not isinstance(selected_pairs, list) or len(selected_pairs) != EXPECTED_PAIR_COUNT:
        fail(f"Expected exactly {EXPECTED_PAIR_COUNT} selected TEST pairs.")

    ids = [str(r["pair_id"]) for r in selected_pairs]
    if len(ids) != len(set(ids)):
        fail("Duplicate pair_id in reduced selection manifest.")
    selected_ids = set(ids)

    recorded_selection_sha = ((amendment.get("selection_manifest") or {}).get("sha256"))
    observed_selection_sha = sha256_file(selection_path)
    if recorded_selection_sha != observed_selection_sha:
        fail("Reduced selection SHA does not match amendment.")

    recorded_p04_sha = (
        (amendment.get("frozen_inputs") or {}).get("program04_source_sha256_at_amendment")
    )
    observed_p04_sha = sha256_file(original_path)
    if recorded_p04_sha != observed_p04_sha:
        fail(
            "Original Program 04 source changed after reduced-grid amendment. "
            "Do not run until reconciled."
        )

    # Record the exact wrapper that implements the amendment.
    runner_manifest = {
        "schema_version": "1.0",
        "manifest_type": "reduced_test_pair_grid_runner",
        "runner_path": wrapper_path.name,
        "runner_sha256": sha256_file(wrapper_path),
        "original_program04_path": str(ORIGINAL_P04_REL).replace("\\", "/"),
        "original_program04_sha256": observed_p04_sha,
        "selection_path": str(SELECTION_REL).replace("\\", "/"),
        "selection_sha256": observed_selection_sha,
        "amendment_path": str(AMENDMENT_REL).replace("\\", "/"),
        "amendment_sha256": sha256_file(amendment_path),
        "selected_pair_count": EXPECTED_PAIR_COUNT,
        "implementation": (
            "Verify original full frozen pair registry via original ensure_pair_registry, "
            "then return only predeclared selected TEST pair_ids in memory."
        ),
        "pair_registry_on_disk_modified": False,
        "original_program04_source_modified": False,
    }
    atomic_write_once(root / RUNNER_MANIFEST_REL, runner_manifest)

    p04 = load_original(original_path)
    original_ensure = p04.ensure_pair_registry

    def reduced_ensure_pair_registry(*, root, mode, split, expected_records,
                                     config_sha256, dataset_revision, model_revision):
        registry, pair_manifest = original_ensure(
            root=root,
            mode=mode,
            split=split,
            expected_records=expected_records,
            config_sha256=config_sha256,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
        )
        if mode != "paper" or split != "test":
            fail("Reduced runner is legal only for --mode paper --split test.")

        filtered = [p for p in registry if p.split != "test" or p.pair_id in selected_ids]
        actual_test_ids = {p.pair_id for p in filtered if p.split == "test"}
        if actual_test_ids != selected_ids:
            missing = sorted(selected_ids - actual_test_ids)
            extra = sorted(actual_test_ids - selected_ids)
            fail(
                "Reduced selection does not match frozen pair registry. "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        return filtered, pair_manifest

    # Monkeypatch only the in-memory return of the already-verified pair registry.
    p04.ensure_pair_registry = reduced_ensure_pair_registry

    p04_argv = [
        "--config", "configs/protocol.yaml",
        "--mode", "paper",
        "--split", "test",
        "--resume",
        "--device", args.device,
        "--output-root", str(root),
    ]
    if args.verify_only:
        p04_argv.append("--verify-only")

    print("=" * 92)
    print("REDUCED-GRID PROGRAM 04 RUNNER")
    print(f"selected TEST pairs : {len(selected_ids)}")
    print(f"selection SHA-256   : {observed_selection_sha}")
    print(f"original P04 SHA    : {observed_p04_sha}")
    print(f"verify only         : {args.verify_only}")
    print("=" * 92)

    return int(p04.main(p04_argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReducedRunnerError as exc:
        print(f"\nREDUCED RUNNER FAILED\n{exc}")
        raise SystemExit(2)
