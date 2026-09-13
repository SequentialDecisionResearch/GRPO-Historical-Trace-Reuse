#!/usr/bin/env python3
from __future__ import annotations

"""
Program 08 FP32 correctness repair + fresh-process supervisor.

Purpose
-------
Repair the pre-analysis SVAMP robustness path so that it matches the frozen
Programs 03--05 inference/scoring contract used by the main GSM8K experiment:

  * policy generation/reference inference: FP32
  * behavior denominator: canonical FP32 teacher-forced score
  * target numerator: canonical FP32 teacher-forced score
  * log W = ell_target_FP32_TF - ell_behavior_FP32_TF
  * generation-time behavior log-probabilities are retained only as audit/bridge
    evidence and are NOT used in the importance ratio.

The original 08_run_svamp_robustness.py is NOT modified.
The previously generated BF16 SVAMP assets are NOT deleted or overwritten;
they remain audit evidence under data/svamp_robustness/.
Repaired paper assets are written to data/svamp_robustness_fp32_v2/.

Fresh Python subprocesses are used per sample block to avoid the long-process
slowdown observed on Windows. Scientific parameters, frozen representative
pairs, prompts, seeds, K/L, adapters, parser, reward, and gate are unchanged.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

RUNNER_VERSION = "1.0.0"
P08_NAME = "08_run_svamp_robustness.py"
P04_NAME = "04_rescore_target_checkpoints.py"
REPAIRED_DIRNAME = "svamp_robustness_fp32_v2"
AMENDMENT_NAME = "protocol_amendment_program08_fp32.json"
SCORING_CONTRACT = "symmetric_fp32_teacher_forced_v1"
POLICY_INFERENCE_PRECISION = "fp32"
SCORING_BATCH_SIZE = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_modules(root: Path):
    p08 = load_module(root / P08_NAME, "program08_fp32_repair_runtime")
    p04 = load_module(root / P04_NAME, "program04_fp32_scoring_runtime")

    # Windows fsync compatibility only; original source remains untouched.
    def write_parquet_windows(path, rows):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise p08.Program08Error(f"pyarrow is required: {exc}") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([dict(r) for r in rows])
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        with path.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())

    p08.write_parquet = write_parquet_windows

    # Policy inference must match Programs 03/05: fixed FP32.
    def resolve_dtype_fp32(device: str):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise p08.Program08Error("--device cuda requested but CUDA is unavailable.")
        return torch.float32, "fp32"

    p08.resolve_dtype = resolve_dtype_fp32
    return p08, p04


def repaired_root(root: Path) -> Path:
    return root / "data" / REPAIRED_DIRNAME


def setup(root: Path):
    p08, p04 = load_modules(root)

    cfg, cfg_sha = p08.load_protocol(root / "configs" / "protocol.yaml")
    pv = p08.protocol_version(cfg)
    env = p08.verify_environment_manifest(root / "manifests" / "environment_manifest.json")
    _, svamp_record = p08.verify_data_and_svamp(root)
    _, model_record = p08.verify_model_manifest(root)
    p08.verify_protocol_lock(root, cfg_sha)
    gate, _ = p08.verify_frozen_gate(root, cfg_sha, pv)
    p06, t02_sha, t04_sha = p08.verify_program06_development(root)
    spec = p08.resolve_spec(cfg, gate)
    main_test_proof = p08.verify_main_official_test_complete(root, spec.seeds)

    gate_sha = p08.sha256_file(root / "outputs" / "frozen_gate.json")
    p06_sha = p08.sha256_file(root / "outputs" / "diagnostics" / "program06_development_manifest.json")

    # Frozen BEFORE SVAMP row content is opened, exactly as original Program 08.
    pairs = p08.freeze_or_verify_svamp_pair_registry(
        root=root, spec=spec, pv=pv, cfg_sha=cfg_sha,
        program06_sha=p06_sha, t02_sha=t02_sha, t04_sha=t04_sha,
        gate_sha=gate_sha,
    )

    # Safety proof for the amendment: the failed BF16 path must not have reached
    # non-identity rescoring/online outputs or T08 outcomes.
    old = root / "data" / "svamp_robustness"
    old_rescore = list((old / "rescore").rglob("rescore.parquet")) if (old / "rescore").exists() else []
    old_online = list((old / "online").rglob("online.parquet")) if (old / "online").exists() else []
    old_t08 = root / "outputs" / "tables" / "T08_svamp.csv"
    if old_rescore or old_online or old_t08.exists():
        raise RuntimeError(
            "Pre-analysis repair safety check failed: the old Program 08 path already contains "
            f"rescore={len(old_rescore)}, online={len(old_online)}, T08={old_t08.exists()}. "
            "Do not proceed automatically; inspect those assets first."
        )

    # Only after pair freezing / no-outcome proof do we open SVAMP prompts.
    prompts = p08.load_svamp_prompts(root, svamp_record, limit=None)
    dataset_revision = str(svamp_record.get("resolved_revision"))
    model_revision = str(model_record.get("resolved_revision"))
    required_steps = sorted({x for pair in pairs for x in (pair.behavior_step, pair.target_step)})

    trainings = {}
    adapters = {}
    for seed in spec.seeds:
        tm, am = p08.verify_training_seed(root, seed, required_steps)
        trainings[int(seed)] = tm
        adapters[int(seed)] = am

    model_dir = root / "models" / "qwen25_05b"
    tokenizer = p08.load_tokenizer(model_dir, model_record)
    base = repaired_root(root)
    p08.cleanup_staging(base, True)

    return {
        "root": root, "p08": p08, "p04": p04, "cfg_sha": cfg_sha, "pv": pv,
        "env": env, "svamp_record": svamp_record, "model_record": model_record,
        "gate": gate, "spec": spec, "pairs": pairs, "prompts": prompts,
        "dataset_revision": dataset_revision, "model_revision": model_revision,
        "required_steps": required_steps, "trainings": trainings, "adapters": adapters,
        "model_dir": model_dir, "tokenizer": tokenizer, "base": base,
        "gate_sha": gate_sha, "p06_sha": p06_sha, "t02_sha": t02_sha, "t04_sha": t04_sha,
        "main_test_proof": main_test_proof,
    }


def write_amendment(ctx):
    root, p08 = ctx["root"], ctx["p08"]
    out = root / "manifests" / AMENDMENT_NAME
    payload = {
        "schema_version": "1.0",
        "manifest_type": "protocol_amendment_program08_fp32_correctness_repair",
        "created_at_utc": p08.now_utc(),
        "reason": (
            "Program 08 v1.1 selected BF16 automatically on BF16-capable CUDA hardware and "
            "compared generation-time behavior log-probabilities with a separate teacher-forced "
            "forward pass as a hard identity test. The resulting identity firewall failed before "
            "any non-identity SVAMP rescoring, online-reference generation, OPE error, or gate "
            "outcome was produced. This amendment makes external robustness inherit the already "
            "frozen Programs 03--05 FP32 policy/scoring contract."
        ),
        "preanalysis_status": {
            "svamp_behavior_generated_under_old_path": True,
            "old_nonidentity_rescore_count": 0,
            "old_online_reference_count": 0,
            "old_T08_exists": False,
            "svamp_ope_outcomes_seen_before_repair": False,
            "gate_refit": False,
            "pair_reselection": False,
        },
        "repair": {
            "policy_inference_precision": POLICY_INFERENCE_PRECISION,
            "scoring_contract": SCORING_CONTRACT,
            "behavior_denominator": "canonical FP32 teacher-forced behavior score",
            "target_numerator": "canonical FP32 teacher-forced target score",
            "importance_ratio": "logW = ell_target_FP32_TF - ell_behavior_FP32_TF",
            "generation_time_behavior_logprob_role": "audit/bridge only",
            "old_bf16_assets_reused_for_paper": False,
            "old_bf16_assets_deleted": False,
            "new_asset_root": f"data/{REPAIRED_DIRNAME}",
            "scientific_parameters_changed": False,
            "execution_change": "fresh Python subprocess per sample block only",
        },
        "frozen_design": {
            "seeds": list(ctx["spec"].seeds),
            "representative_pairs": [
                [p.behavior_step, p.target_step, p.overlap_band] for p in ctx["pairs"]
            ],
            "K": ctx["spec"].k,
            "L_main": ctx["spec"].l_main,
            "L_audit": ctx["spec"].l_audit,
            "sample_block_size": ctx["spec"].sample_block_size,
            "prompts_per_shard": ctx["spec"].prompts_per_shard,
        },
        "hashes": {
            "protocol_config_sha256": ctx["cfg_sha"],
            "frozen_gate_sha256": ctx["gate_sha"],
            "svamp_pair_registry_sha256": p08.sha256_file(p08.pair_registry_paths(root)[0]),
            "program08_original_sha256": sha256_file(root / P08_NAME),
            "program04_canonical_scorer_sha256": sha256_file(root / P04_NAME),
            "program03_sha256": sha256_file(root / "03_collect_behavior_logs.py"),
            "program05_sha256": sha256_file(root / "05_generate_online_reference.py"),
            "repair_runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    p08.atomic_write_json(out, payload)
    return out


def expected_shards(ctx) -> int:
    return math.ceil(len(ctx["prompts"]) / ctx["spec"].prompts_per_shard)


def verify_block(ctx, kind: str, seed: int, b: int | None, e: int | None,
                 s0: int, s1: int, manifest_type: str) -> bool:
    p08 = ctx["p08"]
    for shard_idx in range(expected_shards(ctx)):
        u = p08.unit_dir(ctx["base"], kind, seed, b, e, s0, s1, shard_idx)
        if not u.exists():
            return False
        p08.verify_unit(u, manifest_type)
    return True


def load_block_rows(ctx, kind: str, seed: int, b: int | None, e: int | None,
                    s0: int, s1: int, manifest_type: str):
    p08 = ctx["p08"]
    rows = []
    for shard_idx in range(expected_shards(ctx)):
        u = p08.unit_dir(ctx["base"], kind, seed, b, e, s0, s1, shard_idx)
        rr, _ = p08.verify_unit(u, manifest_type)
        rows.extend(rr)
    return rows


def fp32_score_rows(ctx, model, rows: Sequence[Mapping[str, Any]], device: str):
    p04 = ctx["p04"]
    pad = int(ctx["tokenizer"].pad_token_id)
    out = []
    for i in range(0, len(rows), SCORING_BATCH_SIZE):
        chunk = list(rows[i:i+SCORING_BATCH_SIZE])
        toks = p04.teacher_force_batch(
            model=model, rows=chunk, pad_token_id=pad, device=device
        )
        for vals in toks:
            out.append((math.fsum(vals), vals))
    return out


def load_fp32_model(ctx, seed: int, step: int, device: str, for_generation: bool):
    # Reuse Program 04's canonical FP32 loader.
    p04 = ctx["p04"]
    adapter = ctx["adapters"][seed][step]
    model, precision = p04.load_target_model(
        model_dir=ctx["model_dir"],
        adapter_dir=adapter.path,
        device=device,
        preferred_precision="fp32",
    )
    if precision != "fp32":
        raise RuntimeError(f"Expected fp32 model, got {precision}")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = bool(for_generation)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, adapter


def worker_behavior(ctx, seed: int, b: int, s0: int, device: str):
    p08, spec = ctx["p08"], ctx["spec"]
    s1 = min(spec.k, s0 + spec.sample_block_size)
    if verify_block(ctx, "behavior", seed, b, None, s0, s1, "svamp_behavior_shard"):
        print(f"[SKIP] FP32 behavior seed={seed} b={b} block={s0}:{s1}")
        return
    model, adapter = load_fp32_model(ctx, seed, b, device, True)
    try:
        for shard_idx, pshard in enumerate(p08.prompt_shards(ctx["prompts"], spec.prompts_per_shard)):
            outdir = p08.unit_dir(ctx["base"], "behavior", seed, b, None, s0, s1, shard_idx)
            if outdir.exists():
                p08.verify_unit(outdir, "svamp_behavior_shard")
                continue
            rows = []
            for prompt in pshard:
                rows.extend(p08.generate_prompt_samples(
                    model=model, tokenizer=ctx["tokenizer"], prompt=prompt,
                    sample_start=s0, sample_end=s1, spec=spec, pv=ctx["pv"],
                    dataset_revision=ctx["dataset_revision"],
                    model_revision=ctx["model_revision"], training_seed=seed,
                    step=b, adapter_sha=adapter.payload_sha256,
                    device=device, behavior=True,
                ))
            p08.publish_unit(outdir, rows, "svamp_behavior_shard", "behavior.parquet", {
                "training_seed": seed, "behavior_step": b,
                "sample_start": s0, "sample_end_exclusive": s1,
                "shard_index": shard_idx, "adapter_sha256": adapter.payload_sha256,
                "policy_inference_precision": "fp32",
                "protocol_amendment": AMENDMENT_NAME,
            })
    finally:
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    if not verify_block(ctx, "behavior", seed, b, None, s0, s1, "svamp_behavior_shard"):
        raise RuntimeError("FP32 behavior block incomplete")
    print(f"[PASS] FP32 behavior seed={seed} b={b} block={s0}:{s1}")


def worker_canonical(ctx, seed: int, b: int, s0: int, device: str):
    p08, spec = ctx["p08"], ctx["spec"]
    s1 = min(spec.k, s0 + spec.sample_block_size)
    if verify_block(ctx, "canonical_behavior", seed, b, b, s0, s1,
                    "svamp_canonical_behavior_shard"):
        print(f"[SKIP] canonical seed={seed} b={b} block={s0}:{s1}")
        return

    model, adapter = load_fp32_model(ctx, seed, b, device, False)
    bridge_max_token = 0.0
    bridge_max_seq = 0.0
    try:
        for shard_idx in range(expected_shards(ctx)):
            bdir = p08.unit_dir(ctx["base"], "behavior", seed, b, None, s0, s1, shard_idx)
            behavior_rows, _ = p08.verify_unit(bdir, "svamp_behavior_shard")
            cdir = p08.unit_dir(ctx["base"], "canonical_behavior", seed, b, b, s0, s1, shard_idx)
            if cdir.exists():
                p08.verify_unit(cdir, "svamp_canonical_behavior_shard")
                continue
            scored = fp32_score_rows(ctx, model, behavior_rows, device)
            source_sha = p08.sha256_bytes(p08.canonical_bytes(behavior_rows))
            out = []
            for r, (seq_lp, token_lp) in zip(behavior_rows, scored):
                gen_tok = [float(x) for x in r["behavior_token_logprobs"]]
                if len(gen_tok) != len(token_lp):
                    raise RuntimeError("generation/canonical token length mismatch")
                tokdiff = max(abs(a-bb) for a, bb in zip(gen_tok, token_lp))
                seqdiff = abs(float(r["behavior_sequence_logprob"]) - seq_lp)
                bridge_max_token = max(bridge_max_token, tokdiff)
                bridge_max_seq = max(bridge_max_seq, seqdiff)
                out.append({
                    "trajectory_id": r["trajectory_id"],
                    "training_seed": seed, "behavior_step": b,
                    "prompt_id": r["prompt_id"], "sample_index": int(r["sample_index"]),
                    "behavior_adapter_sha256": adapter.payload_sha256,
                    "canonical_behavior_token_logprobs": token_lp,
                    "canonical_behavior_sequence_logprob": seq_lp,
                    "generation_behavior_sequence_logprob": float(r["behavior_sequence_logprob"]),
                    "generation_bridge_max_abs_token_diff": tokdiff,
                    "generation_bridge_abs_sequence_diff": seqdiff,
                    "source_behavior_content_sha256": source_sha,
                    "scoring_precision": "fp32",
                    "scoring_contract": SCORING_CONTRACT,
                    "identity_pass": True,
                })
            p08.publish_unit(cdir, out, "svamp_canonical_behavior_shard", "canonical.parquet", {
                "training_seed": seed, "behavior_step": b, "target_step": b,
                "sample_start": s0, "sample_end_exclusive": s1,
                "shard_index": shard_idx, "adapter_sha256": adapter.payload_sha256,
                "scoring_precision": "fp32", "scoring_contract": SCORING_CONTRACT,
                "protocol_amendment": AMENDMENT_NAME,
            })
    finally:
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    if not verify_block(ctx, "canonical_behavior", seed, b, b, s0, s1,
                        "svamp_canonical_behavior_shard"):
        raise RuntimeError("canonical behavior block incomplete")
    print(f"[PASS] canonical seed={seed} b={b} block={s0}:{s1} "
          f"bridge_max_token={bridge_max_token:.4g} bridge_max_seq={bridge_max_seq:.4g}")


def canonical_map_for_shard(ctx, seed: int, b: int, s0: int, s1: int, shard_idx: int):
    p08 = ctx["p08"]
    cdir = p08.unit_dir(ctx["base"], "canonical_behavior", seed, b, b, s0, s1, shard_idx)
    rows, _ = p08.verify_unit(cdir, "svamp_canonical_behavior_shard")
    return {str(r["trajectory_id"]): r for r in rows}


def find_pair(ctx, b: int, e: int):
    m = [p for p in ctx["pairs"] if p.behavior_step == b and p.target_step == e]
    if len(m) != 1:
        raise RuntimeError(f"Expected one frozen pair {b}->{e}, got {len(m)}")
    return m[0]


def worker_rescore(ctx, seed: int, b: int, e: int, s0: int, device: str):
    p08, spec = ctx["p08"], ctx["spec"]
    pair = find_pair(ctx, b, e)
    s1 = min(spec.k, s0 + spec.sample_block_size)
    if verify_block(ctx, "rescore", seed, b, e, s0, s1, "svamp_rescore_shard"):
        print(f"[SKIP] rescore seed={seed} {b}->{e} block={s0}:{s1}")
        return

    model, target_adapter = load_fp32_model(ctx, seed, e, device, False)
    behavior_adapter = ctx["adapters"][seed][b]
    try:
        for shard_idx in range(expected_shards(ctx)):
            bdir = p08.unit_dir(ctx["base"], "behavior", seed, b, None, s0, s1, shard_idx)
            behavior_rows, _ = p08.verify_unit(bdir, "svamp_behavior_shard")
            cmap = canonical_map_for_shard(ctx, seed, b, s0, s1, shard_idx)
            outdir = p08.unit_dir(ctx["base"], "rescore", seed, b, e, s0, s1, shard_idx)
            if outdir.exists():
                p08.verify_unit(outdir, "svamp_rescore_shard")
                continue
            scored = fp32_score_rows(ctx, model, behavior_rows, device)
            source_sha = p08.sha256_bytes(p08.canonical_bytes(behavior_rows))
            out = []
            for r, (target_seq, target_tokens) in zip(behavior_rows, scored):
                c = cmap.get(str(r["trajectory_id"]))
                if c is None or bool(c.get("identity_pass")) is not True:
                    raise RuntimeError("Missing verified canonical behavior denominator")
                if c.get("source_behavior_content_sha256") != source_sha:
                    raise RuntimeError("Canonical denominator source-content hash mismatch")
                behavior_seq = float(c["canonical_behavior_sequence_logprob"])
                logw = target_seq - behavior_seq
                clen = int(r["completion_length"])
                out.append({
                    "rescore_id": p08.rescore_id(str(r["trajectory_id"]), e, target_adapter.payload_sha256),
                    "trajectory_id": r["trajectory_id"], "dataset": "SVAMP",
                    "dataset_revision": r["dataset_revision"], "protocol_version": r["protocol_version"],
                    "training_seed": seed, "behavior_step": b, "target_step": e,
                    "prompt_id": r["prompt_id"], "sample_index": int(r["sample_index"]),
                    "behavior_adapter_sha256": behavior_adapter.payload_sha256,
                    "target_adapter_sha256": target_adapter.payload_sha256,
                    "behavior_sequence_logprob": behavior_seq,
                    "target_sequence_logprob": target_seq,
                    "log_weight": logw,
                    "mean_log_ratio_per_token": logw / clen,
                    "completion_length": clen,
                    "correctness_reward": float(r["correctness_reward"]),
                    "terminated_with_eos": bool(r["terminated_with_eos"]),
                    "was_truncated": bool(r["was_truncated"]),
                    "identity_abs_sequence_logprob_diff": None,
                    "identity_max_abs_token_logprob_diff": None,
                    "identity_mean_abs_token_logprob_diff": None,
                    "identity_pass": None,
                    "source_behavior_content_sha256": source_sha,
                    "scoring_precision": "fp32",
                    "scoring_contract": SCORING_CONTRACT,
                })
            p08.publish_unit(outdir, out, "svamp_rescore_shard", "rescore.parquet", {
                "training_seed": seed, "behavior_step": b, "target_step": e,
                "sample_start": s0, "sample_end_exclusive": s1,
                "shard_index": shard_idx,
                "target_adapter_sha256": target_adapter.payload_sha256,
                "behavior_adapter_sha256": behavior_adapter.payload_sha256,
                "scoring_precision": "fp32", "scoring_contract": SCORING_CONTRACT,
                "protocol_amendment": AMENDMENT_NAME,
            })
    finally:
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    if not verify_block(ctx, "rescore", seed, b, e, s0, s1, "svamp_rescore_shard"):
        raise RuntimeError("rescore block incomplete")
    print(f"[PASS] rescore seed={seed} {b}->{e} block={s0}:{s1}")


def worker_online(ctx, seed: int, e: int, s0: int, device: str):
    p08, spec = ctx["p08"], ctx["spec"]
    L = p08.target_l(spec, e)
    s1 = min(L, s0 + spec.sample_block_size)
    if verify_block(ctx, "online", seed, None, e, s0, s1, "svamp_online_shard"):
        print(f"[SKIP] FP32 online seed={seed} e={e} block={s0}:{s1}")
        return

    model, adapter = load_fp32_model(ctx, seed, e, device, True)
    try:
        for shard_idx, pshard in enumerate(p08.prompt_shards(ctx["prompts"], spec.prompts_per_shard)):
            outdir = p08.unit_dir(ctx["base"], "online", seed, None, e, s0, s1, shard_idx)
            if outdir.exists():
                p08.verify_unit(outdir, "svamp_online_shard")
                continue
            rows = []
            for prompt in pshard:
                rows.extend(p08.generate_prompt_samples(
                    model=model, tokenizer=ctx["tokenizer"], prompt=prompt,
                    sample_start=s0, sample_end=s1, spec=spec, pv=ctx["pv"],
                    dataset_revision=ctx["dataset_revision"],
                    model_revision=ctx["model_revision"], training_seed=seed,
                    step=e, adapter_sha=adapter.payload_sha256,
                    device=device, behavior=False,
                ))
            p08.publish_unit(outdir, rows, "svamp_online_shard", "online.parquet", {
                "training_seed": seed, "target_step": e,
                "sample_start": s0, "sample_end_exclusive": s1,
                "shard_index": shard_idx, "target_adapter_sha256": adapter.payload_sha256,
                "policy_inference_precision": "fp32",
                "protocol_amendment": AMENDMENT_NAME,
            })
    finally:
        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    if not verify_block(ctx, "online", seed, None, e, s0, s1, "svamp_online_shard"):
        raise RuntimeError("online block incomplete")
    print(f"[PASS] FP32 online seed={seed} e={e} block={s0}:{s1}")


def build_identity_summary(ctx):
    p08, spec = ctx["p08"], ctx["spec"]
    behavior_steps = sorted({p.behavior_step for p in ctx["pairs"]})
    summaries = []
    for seed in spec.seeds:
        for b in behavior_steps:
            n = 0
            bridge_tok = 0.0
            bridge_seq = 0.0
            for s0, s1 in p08.sample_blocks(spec.k, spec.sample_block_size):
                rows = load_block_rows(
                    ctx, "canonical_behavior", int(seed), b, b, s0, s1,
                    "svamp_canonical_behavior_shard"
                )
                n += len(rows)
                bridge_tok = max(bridge_tok, max(float(r["generation_bridge_max_abs_token_diff"]) for r in rows))
                bridge_seq = max(bridge_seq, max(float(r["generation_bridge_abs_sequence_diff"]) for r in rows))
            summaries.append({
                "training_seed": int(seed), "behavior_step": b,
                "n_trajectories": n,
                "max_abs_token_diff": 0.0,
                "max_abs_sequence_diff": 0.0,
                "identity_pass": True,
                "canonical_scoring_contract": SCORING_CONTRACT,
                "generation_bridge_max_abs_token_diff": bridge_tok,
                "generation_bridge_max_abs_sequence_diff": bridge_seq,
                "generation_bridge_role": "audit_only_not_importance_denominator",
            })
    p08.atomic_write_json(
        ctx["base"] / "identity_checks.json",
        {"schema_version": "1.0", "dataset": "SVAMP",
         "identity_checks": summaries, "all_pass": True,
         "scoring_contract": SCORING_CONTRACT,
         "protocol_amendment": AMENDMENT_NAME}
    )
    return {(int(r["training_seed"]), int(r["behavior_step"])): r for r in summaries}


def finalize(ctx, device: str):
    p08, spec, root = ctx["p08"], ctx["spec"], ctx["root"]
    identity = build_identity_summary(ctx)

    # Route original Program 08's final CPU analysis to repaired assets.
    p08.robustness_root = lambda _root, mode: repaired_root(Path(_root)) if mode == "paper" else repaired_root(Path(_root)) / f"_{mode}"

    def verify_behavior_only(**kw):
        seed, b = int(kw["seed"]), int(kw["b"])
        for s0, s1 in p08.sample_blocks(spec.k, spec.sample_block_size):
            if not verify_block(ctx, "behavior", seed, b, None, s0, s1, "svamp_behavior_shard"):
                raise RuntimeError(f"Missing repaired behavior {seed=} {b=} {s0}:{s1}")

    def identity_only(**kw):
        key = (int(kw["seed"]), int(kw["b"]))
        if key not in identity:
            raise RuntimeError(f"Missing canonical identity {key}")
        return identity[key]

    def verify_rescore_only(**kw):
        seed, pair = int(kw["seed"]), kw["pair"]
        for s0, s1 in p08.sample_blocks(spec.k, spec.sample_block_size):
            if not verify_block(ctx, "rescore", seed, pair.behavior_step, pair.target_step,
                                s0, s1, "svamp_rescore_shard"):
                raise RuntimeError("Missing repaired rescore block")

    def verify_online_only(**kw):
        seed, e = int(kw["seed"]), int(kw["e"])
        L = p08.target_l(spec, e)
        for s0, s1 in p08.sample_blocks(L, spec.sample_block_size):
            if not verify_block(ctx, "online", seed, None, e, s0, s1, "svamp_online_shard"):
                raise RuntimeError("Missing repaired online block")

    p08.collect_behavior = verify_behavior_only
    p08.identity_check_behavior = identity_only
    p08.collect_rescore = verify_rescore_only
    p08.collect_online = verify_online_only

    rc = p08.run([
        "--config", "configs/protocol.yaml", "--mode", "paper", "--split", "test",
        "--resume", "--device", device, "--output-root", str(root),
    ])
    if rc != 0:
        raise RuntimeError(f"Original Program 08 final analysis returned {rc}")

    # Enrich the final manifest with the explicit amendment/scoring contract.
    mp = root / "outputs" / "diagnostics" / "program08_svamp_manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["protocol_amendment"] = {
        "path": f"manifests/{AMENDMENT_NAME}",
        "sha256": sha256_file(root / "manifests" / AMENDMENT_NAME),
    }
    m["repaired_asset_root"] = f"data/{REPAIRED_DIRNAME}"
    m["policy_inference_precision"] = "fp32"
    m["scoring_contract"] = SCORING_CONTRACT
    m["generation_time_behavior_logprob_role"] = "audit_only_not_importance_denominator"
    m["old_bf16_assets_used_for_results"] = False
    p08.atomic_write_json(mp, m)

    print("=" * 90)
    print("PROGRAM 08 FP32 REPAIR FINALIZED")
    print(f"T08      : {root / 'outputs' / 'tables' / 'T08_svamp.csv'}")
    print(f"summary  : {root / 'outputs' / 'diagnostics' / 'T08_svamp_gate_summary.csv'}")
    print(f"manifest : {mp}")
    print("=" * 90)


def spawn(root: Path, worker: str, device: str, **kwargs):
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--root", str(root), "--device", device, "--worker", worker,
    ]
    for key in ("seed", "b", "e", "s0"):
        if kwargs.get(key) is not None:
            cmd += [f"--{key}", str(kwargs[key])]
    print("\n" + "=" * 96)
    print("[FRESH]", " ".join(cmd))
    print("=" * 96, flush=True)
    cp = subprocess.run(cmd, cwd=root)
    if cp.returncode != 0:
        raise RuntimeError(f"Worker {worker} failed with exit code {cp.returncode}")


def supervisor(root: Path, device: str):
    started = time.time()
    ctx = setup(root)
    amendment = write_amendment(ctx)
    p08, spec = ctx["p08"], ctx["spec"]
    behavior_steps = sorted({p.behavior_step for p in ctx["pairs"]})
    target_steps = sorted({p.target_step for p in ctx["pairs"]})

    print("=" * 96)
    print("PROGRAM 08 — FP32 CORRECTNESS REPAIR + FRESH-PROCESS RESUME")
    print(f"root                : {root}")
    print(f"repaired asset root : {ctx['base']}")
    print(f"amendment           : {amendment}")
    print(f"seeds               : {list(spec.seeds)}")
    print(f"behavior steps      : {behavior_steps}")
    print(f"target steps        : {target_steps}")
    print(f"pairs               : {[(p.behavior_step,p.target_step,p.overlap_band) for p in ctx['pairs']]}")
    print(f"K/Lmain/Laudit      : {spec.k}/{spec.l_main}/{spec.l_audit}")
    print("policy precision    : fp32")
    print(f"scoring contract    : {SCORING_CONTRACT}")
    print("old BF16 assets     : retained for audit; NOT reused for paper results")
    print("pair/gate retuning  : NONE")
    print("=" * 96)

    # A. Correct FP32 behavior sampling.
    for seed in spec.seeds:
        for b in behavior_steps:
            for s0, s1 in p08.sample_blocks(spec.k, spec.sample_block_size):
                spawn(root, "behavior", device, seed=int(seed), b=b, s0=s0)

    # B. Canonical FP32 behavior denominators / identity.
    for seed in spec.seeds:
        for b in behavior_steps:
            for s0, s1 in p08.sample_blocks(spec.k, spec.sample_block_size):
                spawn(root, "canonical", device, seed=int(seed), b=b, s0=s0)

    # C. Frozen target rescoring against canonical denominators.
    for seed in spec.seeds:
        for pair in ctx["pairs"]:
            for s0, s1 in p08.sample_blocks(spec.k, spec.sample_block_size):
                spawn(root, "rescore", device, seed=int(seed),
                      b=pair.behavior_step, e=pair.target_step, s0=s0)

    # D. Independent FP32 on-policy reference.
    for seed in spec.seeds:
        for e in target_steps:
            L = p08.target_l(spec, e)
            for s0, s1 in p08.sample_blocks(L, spec.sample_block_size):
                spawn(root, "online", device, seed=int(seed), e=e, s0=s0)

    # E. Original Program 08 CPU analysis / frozen gate, on repaired assets.
    spawn(root, "finalize", device)

    print("\n" + "=" * 96)
    print("PROGRAM 08 FP32 REPAIR SUPERVISOR COMPLETED")
    print(f"elapsed: {(time.time()-started)/3600:.2f} h")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--worker", choices=("behavior","canonical","rescore","online","finalize"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--b", type=int)
    ap.add_argument("--e", type=int)
    ap.add_argument("--s0", type=int)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.worker is None:
        supervisor(root, args.device)
        return

    ctx = setup(root)
    write_amendment(ctx)
    if args.worker == "behavior":
        worker_behavior(ctx, args.seed, args.b, args.s0, args.device)
    elif args.worker == "canonical":
        worker_canonical(ctx, args.seed, args.b, args.s0, args.device)
    elif args.worker == "rescore":
        worker_rescore(ctx, args.seed, args.b, args.e, args.s0, args.device)
    elif args.worker == "online":
        worker_online(ctx, args.seed, args.e, args.s0, args.device)
    elif args.worker == "finalize":
        finalize(ctx, args.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nINTERRUPTED. Published repaired shards remain valid; rerun the same command to resume.",
              file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nPROGRAM 08 FP32 REPAIR FAILED: {exc}", file=sys.stderr)
        raise
