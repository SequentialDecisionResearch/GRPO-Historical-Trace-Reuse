#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, importlib.util, json, sys
from pathlib import Path

TARGETS = {160, 260, 360}
SEEDS = {20260826, 20260827, 20260828}
SELECTION = Path("manifests/reduced_test_pair_selection_program06_compatible.json")
REPAIR = Path("manifests/protocol_amendment_program06_rolling_comparator_repair.json")
P04 = Path("04_rescore_target_checkpoints.py")
MORE_WORK = 75

class DirectRepairError(RuntimeError): pass
class ChunkStop(RuntimeError): pass

def fail(msg): raise DirectRepairError(msg)

def read_json(path):
    path = Path(path)
    if not path.exists(): fail(f"Missing required file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict): fail(f"Expected JSON object: {path}")
    return obj

def load_module(path):
    spec = importlib.util.spec_from_file_location("p04_direct_repair_original", path)
    if spec is None or spec.loader is None: fail(f"Cannot import Program 04: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def cleanup():
    try: gc.collect()
    except Exception: pass
    try:
        import torch
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except Exception: pass
            torch.cuda.empty_cache()
    except Exception:
        pass
    try: gc.collect()
    except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target-step", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-shards", type=int, default=20)
    a = ap.parse_args()

    if a.seed not in SEEDS: fail(f"Unexpected seed: {a.seed}")
    if a.target_step not in TARGETS: fail(f"Unexpected target: {a.target_step}")
    if a.max_new_shards <= 0: fail("--max-new-shards must be positive")

    root = Path(a.root).resolve()
    sel = read_json(root / SELECTION)
    _ = read_json(root / REPAIR)
    rows = sel.get("selected_pairs")
    if not isinstance(rows, list) or len(rows) != 45:
        fail("Repaired selection must contain exactly 45 pairs.")

    wanted = [r for r in rows
              if str(r.get("split")) == "test"
              and int(r.get("training_seed")) == a.seed
              and int(r.get("behavior_step")) == 0
              and int(r.get("target_step")) == a.target_step
              and str(r.get("purpose")) == "fixed"]
    if len(wanted) != 1: fail(f"Expected exactly one comparator; found {len(wanted)}")
    wanted_id = str(wanted[0]["pair_id"])

    p04 = load_module(root / P04)

    # Full frozen registry is verified first, then expose only this one comparator.
    original_ensure = p04.ensure_pair_registry
    def direct_ensure(**kwargs):
        registry, manifest = original_ensure(**kwargs)
        chosen = [p for p in registry
                  if p.split == "test"
                  and p.training_seed == a.seed
                  and p.behavior_step == 0
                  and p.target_step == a.target_step
                  and p.purpose == "fixed"
                  and p.pair_id == wanted_id]
        if len(chosen) != 1:
            fail(f"Frozen registry did not resolve exactly one requested comparator; found {len(chosen)}")
        return chosen, manifest
    p04.ensure_pair_registry = direct_ensure

    # Do not let a one-pair worker overwrite global 45-pair indexes.
    original_rebuild = p04.rebuild_rescore_collection_index
    original_atomic = p04.atomic_write_json
    dummy_index = root / "benchmarks" / "program04_direct_repair" / "global_index_suppressed.json"
    def suppress_rebuild(*args, **kwargs):
        return dummy_index
    def guarded_atomic(path, obj):
        path = Path(path)
        if path.name == "pair_summaries.json":
            return None
        return original_atomic(path, obj)
    p04.rebuild_rescore_collection_index = suppress_rebuild
    p04.atomic_write_json = guarded_atomic

    state = {"new": 0}
    original_publish = p04.publish_rescore_unit
    def counted_publish(**kwargs):
        result = original_publish(**kwargs)
        state["new"] += 1
        if state["new"] >= a.max_new_shards:
            raise ChunkStop
        return result
    p04.publish_rescore_unit = counted_publish

    print("=" * 92)
    print("PROGRAM 04 DIRECT STRUCTURAL-COMPARATOR REPAIR")
    print(f"seed / comparator : {a.seed} / 0 -> {a.target_step}")
    print("selected pairs     : 1")
    print(f"fresh chunk limit  : {a.max_new_shards}")
    print("identity pairs     : 0")
    print("global index writes: SUPPRESSED until finalization")
    print("=" * 92)

    argv = [
        "--config", "configs/protocol.yaml",
        "--mode", "paper",
        "--split", "test",
        "--resume",
        "--device", a.device,
        "--output-root", str(root),
        "--seed", str(a.seed),
    ]

    try:
        try:
            return int(p04.run(argv))
        except ChunkStop:
            print(f"\nDIRECT CHUNK COMPLETE: {state['new']} new shard units published.")
            print("Exiting Python now to fully release CUDA memory.")
            return MORE_WORK
    finally:
        p04.ensure_pair_registry = original_ensure
        p04.rebuild_rescore_collection_index = original_rebuild
        p04.atomic_write_json = original_atomic
        p04.publish_rescore_unit = original_publish
        cleanup()

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DirectRepairError as exc:
        print(f"\nDIRECT REPAIR WORKER FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
