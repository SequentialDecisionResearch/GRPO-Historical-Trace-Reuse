#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

AMEND_REL = Path("manifests/protocol_amendment_program05_sample_size.json")
SELECTION_REL = Path("manifests/reduced_test_pair_selection.json")
P05_REL = Path("05_generate_online_reference.py")
EFFECTIVE_L = 8
EXPECTED_STEPS = [0,20,100,120,160,200,220,260,300,360,400]

class WorkerError(RuntimeError): pass
class ChunkStop(RuntimeError): pass

def fail(msg): raise WorkerError(msg)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists(): fail(f"Missing required file: {path}")
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict): fail(f"Expected JSON object: {path}")
    return x

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("program05_8x8_concurrent_original", path)
    if spec is None or spec.loader is None: fail(f"Cannot import Program 05: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def validate_plan(root: Path):
    amend_path = root/AMEND_REL
    amend = read_json(amend_path)
    if amend.get("manifest_type") != "preanalysis_program05_reference_sample_size_amendment":
        fail("Unexpected Program05 amendment type.")
    d = amend.get("amended_effective_design") or {}
    if int(d.get("L_main",-1)) != 8 or int(d.get("L_audit",-1)) != 8:
        fail("Program05 amendment is not 8/8.")
    sel_path = root/SELECTION_REL
    sel = read_json(sel_path)
    expected_sel = (((amend.get("immutability_and_provenance") or {})
                     .get("reduced_pair_selection") or {}).get("sha256"))
    if expected_sel != sha256_file(sel_path):
        fail("Reduced pair selection hash no longer matches amendment.")
    rows = sel.get("selected_pairs")
    if not isinstance(rows, list) or len(rows) != 36:
        fail("Reduced pair selection is not 36 pairs.")
    seeds = sorted({int(r["training_seed"]) for r in rows})
    steps = sorted({int(r["target_step"]) for r in rows})
    if len(seeds) != 3 or steps != EXPECTED_STEPS:
        fail(f"Unexpected reduced plan: seeds={seeds}, steps={steps}")
    return seeds, steps, sha256_file(amend_path)

def summary_path(root: Path, seed: int, step: int) -> Path:
    return (root/"data"/"online_reference"/"gsm8k"/"split=test"/
            f"seed={seed}"/f"target_step={step:04d}"/"reference_summary.json")

def is_complete(root: Path, seed: int, step: int) -> bool:
    rp = summary_path(root, seed, step)
    pp = rp.parent/"prompt_summary.parquet"
    if not rp.exists() or not pp.exists(): return False
    try:
        m = read_json(rp)
        return int(m.get("samples_per_prompt",-1)) == 8
    except Exception:
        return False

def cleanup():
    try: gc.collect()
    except Exception: pass
    try:
        import torch
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except Exception: pass
            torch.cuda.empty_cache()
    except Exception: pass
    try: gc.collect()
    except Exception: pass

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target-step", type=int, required=True)
    ap.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    ap.add_argument("--max-new-shards", type=int, default=10)
    args = ap.parse_args()
    if args.max_new_shards <= 0: fail("--max-new-shards must be positive.")

    root = Path(args.root).resolve()
    seeds, steps, amend_sha = validate_plan(root)
    if args.seed not in seeds or args.target_step not in steps:
        fail(f"Requested task not in amended plan: {args.seed}/{args.target_step}")
    if is_complete(root, args.seed, args.target_step):
        print(f"TARGET ALREADY COMPLETE seed={args.seed} target={args.target_step} L=8")
        return 0

    p05_path = root/P05_REL
    p05 = load_module(p05_path)

    # Effective 8/8 sample size, while leaving the frozen OnlineSpec itself intact.
    original_target_sample_limit = p05.target_sample_limit
    p05.target_sample_limit = lambda spec, target_step, target_l_main: EFFECTIVE_L

    # Important for concurrent safety:
    # per-target workers may rebuild their own target summary, but they must NOT
    # race on the global collection_index.json or reference_summaries.json.
    original_rebuild_collection_index = p05.rebuild_collection_index
    original_atomic_write_json = p05.atomic_write_json
    dummy_index = root/"benchmarks"/"program05_parallel3_8x8"/"global_index_write_suppressed.json"

    def suppress_collection_index(*a, **kw):
        return dummy_index

    def guarded_atomic_write_json(path, obj):
        path = Path(path)
        if path.name == "reference_summaries.json":
            return None
        return original_atomic_write_json(path, obj)

    p05.rebuild_collection_index = suppress_collection_index
    p05.atomic_write_json = guarded_atomic_write_json

    state = {"published": 0}
    original_publish = p05.publish_online_unit
    original_generate = p05.generate_prompt_block

    def counted_publish(*a, **kw):
        result = original_publish(*a, **kw)
        state["published"] += 1
        return result

    def limited_generate(*a, **kw):
        if state["published"] >= args.max_new_shards:
            raise ChunkStop
        return original_generate(*a, **kw)

    p05.publish_online_unit = counted_publish
    p05.generate_prompt_block = limited_generate

    print("="*90)
    print("PROGRAM 05 FORMAL 8/8 CONCURRENT-SAFE CHUNK")
    print(f"seed / target       : {args.seed} / {args.target_step}")
    print(f"effective L         : 8")
    print(f"new shard limit     : {args.max_new_shards}")
    print(f"sample amendment SHA: {amend_sha}")
    print("global index writes : SUPPRESSED until single-process finalization")
    print("="*90)

    argv = [
        "--config","configs/protocol.yaml","--mode","paper","--split","test","--resume",
        "--device",args.device,"--output-root",str(root),
        "--seed",str(args.seed),"--target-step",str(args.target_step),
    ]

    try:
        try:
            rc = int(p05.run(argv))
        except ChunkStop:
            print(f"\nCHUNK COMPLETE: {state['published']} new official online shard units published.")
            rc = 0
    finally:
        p05.target_sample_limit = original_target_sample_limit
        p05.rebuild_collection_index = original_rebuild_collection_index
        p05.atomic_write_json = original_atomic_write_json
        cleanup()

    if rc != 0:
        return rc
    print(f"target_complete_after_chunk: {is_complete(root,args.seed,args.target_step)}")
    print("Exiting fresh Python process.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkerError as exc:
        print(f"\nPROGRAM 05 FORMAL 8/8 WORKER FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
