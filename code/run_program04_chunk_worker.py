#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SELECTION_REL = Path("manifests/reduced_test_pair_selection.json")
AMENDMENT_REL = Path("manifests/protocol_amendment_reduced_grid.json")
P04_REL = Path("04_rescore_target_checkpoints.py")
EXPECTED_PAIR_COUNT = 36
MORE_WORK = 75

class WorkerError(RuntimeError):
    pass

class ChunkStop(RuntimeError):
    pass

def fail(msg: str):
    raise WorkerError(msg)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing required file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")
    return obj

def pair_order(r: Mapping[str, Any]):
    purpose = str(r["purpose"])
    seed = int(r["training_seed"])
    b = int(r["behavior_step"])
    e = int(r["target_step"])
    return (0 if purpose == "identity" else 1, seed, e, b, purpose)

def count_source_units(root: Path, r: Mapping[str, Any]) -> int:
    base = (
        root / "data" / "behavior_logs" / "gsm8k" / "split=test"
        / f"seed={int(r['training_seed'])}"
        / f"behavior_step={int(r['behavior_step']):04d}"
    )
    if not base.exists():
        fail(f"Missing source directory: {base}")
    return sum(
        1 for p in base.rglob("manifest.json")
        if p.parent.name.startswith("shard=")
    )

def count_output_units(root: Path, r: Mapping[str, Any]) -> int:
    base = (
        root / "data" / "target_rescores" / "gsm8k" / "split=test"
        / f"seed={int(r['training_seed'])}"
        / f"behavior_step={int(r['behavior_step']):04d}"
        / f"target_step={int(r['target_step']):04d}"
    )
    if not base.exists():
        return 0
    return sum(
        1 for p in base.rglob("manifest.json")
        if p.parent.name.startswith("shard=")
    )

def choose_pair(root: Path, rows: list[dict[str, Any]]):
    for r in sorted(rows, key=pair_order):
        src = count_source_units(root, r)
        out = count_output_units(root, r)
        if out > src:
            fail(
                f"Output count exceeds source count for pair {r['pair_id']}: "
                f"{out}>{src}"
            )
        if out < src:
            return r, out, src
    return None, None, None

def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("program04_chunk_original", path)
    if spec is None or spec.loader is None:
        fail(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def cleanup():
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--max-new-shards", type=int, default=40)
    args = ap.parse_args()

    if args.max_new_shards <= 0:
        fail("--max-new-shards must be > 0")

    root = Path(args.root).resolve()
    selection_path = root / SELECTION_REL
    amendment_path = root / AMENDMENT_REL
    p04_path = root / P04_REL

    selection = read_json(selection_path)
    amendment = read_json(amendment_path)

    if selection.get("manifest_type") != "reduced_test_pair_selection":
        fail("Invalid reduced_test_pair_selection.json")
    if amendment.get("manifest_type") != "poststart_reduced_test_pair_grid_amendment":
        fail("Invalid protocol_amendment_reduced_grid.json")

    selected = selection.get("selected_pairs")
    if not isinstance(selected, list) or len(selected) != EXPECTED_PAIR_COUNT:
        fail("Expected exactly 36 selected TEST pairs")
    selected = [dict(x) for x in selected]

    if ((amendment.get("selection_manifest") or {}).get("sha256")
            != sha256_file(selection_path)):
        fail("Selection manifest hash does not match amendment")

    expected_p04_sha = (
        (amendment.get("frozen_inputs") or {})
        .get("program04_source_sha256_at_amendment")
    )
    if expected_p04_sha != sha256_file(p04_path):
        fail("Original Program04 source changed after amendment")

    pair, done, total = choose_pair(root, selected)
    if pair is None:
        print("ALL REDUCED-GRID SHARD COUNTS COMPLETE")
        return 0

    pair_id = str(pair["pair_id"])
    seed = int(pair["training_seed"])
    b = int(pair["behavior_step"])
    e = int(pair["target_step"])
    purpose = str(pair["purpose"])

    print("=" * 90)
    print("FRESH-PROCESS PROGRAM04 CHUNK")
    print(f"pair      : seed={seed} b={b} -> e={e} purpose={purpose}")
    print(f"progress  : {done}/{total} shard units already published")
    print(f"new limit : {args.max_new_shards}")
    print("=" * 90)

    p04 = load_original(p04_path)

    original_ensure = p04.ensure_pair_registry
    def filtered_ensure_pair_registry(
        *, root, mode, split, expected_records,
        config_sha256, dataset_revision, model_revision
    ):
        registry, pair_manifest = original_ensure(
            root=root,
            mode=mode,
            split=split,
            expected_records=expected_records,
            config_sha256=config_sha256,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
        )
        matches = [p for p in registry if p.pair_id == pair_id]
        if len(matches) != 1:
            fail(f"Pair {pair_id} not uniquely present in frozen registry")
        return matches, pair_manifest

    p04.ensure_pair_registry = filtered_ensure_pair_registry

    original_score = p04.score_behavior_rows
    state = {"new": 0}

    def limited_score_behavior_rows(*a, **kw):
        if state["new"] >= args.max_new_shards:
            raise ChunkStop
        result = original_score(*a, **kw)
        state["new"] += 1
        return result

    p04.score_behavior_rows = limited_score_behavior_rows

    argv = [
        "--config", "configs/protocol.yaml",
        "--mode", "paper",
        "--split", "test",
        "--resume",
        "--device", args.device,
        "--output-root", str(root),
        "--seed", str(seed),
    ]

    try:
        rc = int(p04.main(argv))
    except ChunkStop:
        print()
        print(f"CHUNK COMPLETE: {state['new']} new shard units scored.")
        print("Exiting Python now to fully release CUDA process memory.")
        return MORE_WORK
    finally:
        cleanup()

    if rc != 0:
        return rc

    pair2, _, _ = choose_pair(root, selected)
    if pair2 is not None:
        print()
        print("PAIR COMPLETE. More reduced-grid work remains.")
        print("Exiting Python now; supervisor will start a fresh process.")
        return MORE_WORK

    print()
    print("ALL REDUCED-GRID SHARD COUNTS COMPLETE")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkerError as exc:
        print(f"\nCHUNK WORKER FAILED\n{exc}")
        raise SystemExit(2)
