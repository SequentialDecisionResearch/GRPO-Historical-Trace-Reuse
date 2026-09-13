#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

P08_NAME = "08_run_svamp_robustness.py"


def load_p08(root: Path):
    path = root / P08_NAME
    spec = importlib.util.spec_from_file_location("program08_behavior_fresh_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Windows-only I/O compatibility fix. Frozen Program 08 source is unchanged.
    def write_parquet_windows(path, rows):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise mod.Program08Error(f"pyarrow is required: {exc}") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([dict(r) for r in rows])
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        with path.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())

    mod.write_parquet = write_parquet_windows
    return mod


def setup(root: Path):
    p08 = load_p08(root)
    cfg, cfg_sha = p08.load_protocol(root / "configs" / "protocol.yaml")
    pv = p08.protocol_version(cfg)
    p08.verify_environment_manifest(root / "manifests" / "environment_manifest.json")
    _, svamp_record = p08.verify_data_and_svamp(root)
    _, model_record = p08.verify_model_manifest(root)
    p08.verify_protocol_lock(root, cfg_sha)
    gate, _ = p08.verify_frozen_gate(root, cfg_sha, pv)
    _, t02_sha, t04_sha = p08.verify_program06_development(root)
    spec = p08.resolve_spec(cfg, gate)
    p08.verify_main_official_test_complete(root, spec.seeds)

    gate_sha = p08.sha256_file(root / "outputs" / "frozen_gate.json")
    p06_sha = p08.sha256_file(root / "outputs" / "diagnostics" / "program06_development_manifest.json")
    pairs = p08.freeze_or_verify_svamp_pair_registry(
        root=root, spec=spec, pv=pv, cfg_sha=cfg_sha,
        program06_sha=p06_sha, t02_sha=t02_sha, t04_sha=t04_sha,
        gate_sha=gate_sha,
    )

    required_steps = sorted({x for pair in pairs for x in (pair.behavior_step, pair.target_step)})
    prompts = p08.load_svamp_prompts(root, svamp_record, limit=None)
    dataset_revision = str(svamp_record.get("resolved_revision"))
    model_revision = str(model_record.get("resolved_revision"))
    model_dir = root / "models" / "qwen25_05b"
    base = p08.robustness_root(root, "paper")
    p08.cleanup_staging(base, True)
    return p08, pv, spec, pairs, prompts, model_record, model_revision, dataset_revision, required_steps, model_dir, base


def block_complete(p08, base: Path, prompts, spec, seed: int, b: int, s0: int, s1: int) -> bool:
    n_shards = math.ceil(len(prompts) / spec.prompts_per_shard)
    for shard_idx in range(n_shards):
        u = p08.unit_dir(base, "behavior", seed, b, None, s0, s1, shard_idx)
        if not u.exists():
            return False
        p08.verify_unit(u, "svamp_behavior_shard")
    return True


def worker(root: Path, seed: int, b: int, s0: int, device: str):
    (p08, pv, spec, pairs, prompts, model_record, model_revision,
     dataset_revision, required_steps, model_dir, base) = setup(root)

    behavior_steps = {p.behavior_step for p in pairs}
    if seed not in spec.seeds or b not in behavior_steps:
        raise RuntimeError(f"Worker request is outside frozen design: seed={seed}, b={b}")
    valid_blocks = dict(p08.sample_blocks(spec.k, spec.sample_block_size))
    if s0 not in valid_blocks:
        raise RuntimeError(f"Invalid behavior sample block start {s0}")
    s1 = valid_blocks[s0]

    if block_complete(p08, base, prompts, spec, seed, b, s0, s1):
        print(f"[SKIP] seed={seed} b={b} block={s0}:{s1} already complete")
        return

    _, adapters = p08.verify_training_seed(root, seed, required_steps)
    tokenizer = p08.load_tokenizer(model_dir, model_record)
    model, _ = p08.load_adapter_model(model_dir, adapters[b], device)
    try:
        for shard_idx, pshard in enumerate(p08.prompt_shards(prompts, spec.prompts_per_shard)):
            outdir = p08.unit_dir(base, "behavior", seed, b, None, s0, s1, shard_idx)
            if outdir.exists():
                p08.verify_unit(outdir, "svamp_behavior_shard")
                continue
            rows = []
            for prompt in pshard:
                rows.extend(p08.generate_prompt_samples(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    sample_start=s0, sample_end=s1, spec=spec, pv=pv,
                    dataset_revision=dataset_revision, model_revision=model_revision,
                    training_seed=seed, step=b,
                    adapter_sha=adapters[b].payload_sha256,
                    device=device, behavior=True,
                ))
            p08.publish_unit(outdir, rows, "svamp_behavior_shard", "behavior.parquet", {
                "training_seed": seed, "behavior_step": b,
                "sample_start": s0, "sample_end_exclusive": s1,
                "shard_index": shard_idx,
                "adapter_sha256": adapters[b].payload_sha256,
            })
    finally:
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    if not block_complete(p08, base, prompts, spec, seed, b, s0, s1):
        raise RuntimeError("Fresh behavior block did not verify complete")
    print(f"[PASS] seed={seed} b={b} block={s0}:{s1}")


def supervisor(root: Path, device: str):
    (p08, pv, spec, pairs, prompts, model_record, model_revision,
     dataset_revision, required_steps, model_dir, base) = setup(root)
    behavior_steps = sorted({p.behavior_step for p in pairs})
    blocks = p08.sample_blocks(spec.k, spec.sample_block_size)

    print("=" * 86)
    print("PROGRAM 08 — FRESH-PROCESS BEHAVIOR RESUME")
    print(f"root            : {root}")
    print(f"seeds           : {list(spec.seeds)}")
    print(f"behavior steps  : {behavior_steps}")
    print(f"sample blocks   : {blocks}")
    print(f"prompts         : {len(prompts)}")
    print("scientific change: NONE — process scheduling only")
    print("Each fresh Python worker handles at most one K sample block (125 prompt shards).")
    print("=" * 86)

    for seed in spec.seeds:
        for b in behavior_steps:
            for s0, s1 in blocks:
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--root", str(root), "--device", device,
                    "--worker", "--seed", str(seed), "--b", str(b), "--s0", str(s0),
                ]
                print("\n[FRESH]", " ".join(cmd), flush=True)
                cp = subprocess.run(cmd, cwd=root)
                if cp.returncode != 0:
                    raise RuntimeError(f"Worker failed, exit code {cp.returncode}")

    # Final strict verification of all expected behavior blocks.
    for seed in spec.seeds:
        for b in behavior_steps:
            for s0, s1 in blocks:
                if not block_complete(p08, base, prompts, spec, seed, b, s0, s1):
                    raise RuntimeError(f"Final behavior verification failed: seed={seed} b={b} block={s0}:{s1}")

    print("\n" + "=" * 86)
    print("PROGRAM 08 FRESH BEHAVIOR STAGE COMPLETED AND VERIFIED")
    print("All existing valid shards were reused; no frozen scientific parameter was changed.")
    print("Next step: resume the normal Program 08 Windows wrapper.")
    print("=" * 86)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--b", type=int)
    ap.add_argument("--s0", type=int)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.worker:
        if args.seed is None or args.b is None or args.s0 is None:
            raise RuntimeError("Worker requires --seed --b --s0")
        worker(root, args.seed, args.b, args.s0, args.device)
    else:
        supervisor(root, args.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nINTERRUPTED. Published shards remain valid; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
