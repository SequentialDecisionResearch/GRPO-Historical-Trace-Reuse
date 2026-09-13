#!/usr/bin/env python3
"""
diagnose_cached_identity.py

READ-ONLY diagnostic for GRPO-OPE Program 04 identity-gate failures.

Purpose
-------
Program 03 stores generation-time selected-token log-probabilities from
model.generate(..., output_scores=True) +
compute_transition_scores(..., normalize_logits=True).

Program 04 rescored the exact saved token sequence using a full-sequence
teacher-forced forward pass.  Rare long sequences can accumulate tiny FP32
per-token differences and cross the frozen sequence tolerance.

This diagnostic DOES NOT modify protocol.yaml, Programs 00-09, Program 03
behavior logs, Program 04 rescoring outputs, manifests, or checkpoints.

For each formal paper/development identity row that numerically fails the
frozen identity tolerance, it compares four paths:

  1. existing Program 04 full-sequence teacher forcing;
  2. single-trajectory incremental cached replay;
  3. original-generation-call-batch incremental cached replay, reconstructed
     from all saved trajectories with the same generation_call_id;
  4. exact re-execution of the original generation call using the stored RNG
     seed/config, with transition scores recomputed exactly as Program 03 did.

The call-batch cached replay is important: Program 03 generated multiple return
sequences together, so matching the original call batch removes a possible
batch-shape numerical confound.

Typical Spyder command
----------------------
%run "C:/lsg/grpo_ope_reuse/diagnose_cached_identity.py"

Optional:
%run "C:/lsg/grpo_ope_reuse/diagnose_cached_identity.py" --output-root "C:/lsg/grpo_ope_reuse"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


class DiagnosticError(RuntimeError):
    pass


def load_program04(root: Path) -> Any:
    path = root / "04_rescore_target_checkpoints.py"
    if not path.exists():
        raise DiagnosticError(f"Missing Program 04: {path}")
    spec = importlib.util.spec_from_file_location("grpo_ope_program04_diagnostic", path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"Cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses and some import machinery expect the module to exist here.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiagnosticError(f"Missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise DiagnosticError(f"Expected JSON object: {path}")
    return obj


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise DiagnosticError(f"Cannot import pyarrow: {exc}") from exc
    if not path.exists():
        raise DiagnosticError(f"Missing Parquet: {path}")
    try:
        table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    except Exception as exc:
        raise DiagnosticError(f"Cannot read {path}: {exc}") from exc
    return [dict(r) for r in table.to_pylist()]


def numeric_identity_fail(row: Mapping[str, Any], token_atol: float, sequence_atol: float) -> bool:
    seq = row.get("identity_abs_sequence_logprob_diff")
    tok = row.get("identity_max_abs_token_logprob_diff")
    if seq is None or tok is None:
        return False
    return float(tok) > token_atol or float(seq) > sequence_atol


def discover_formal_failures(
    root: Path,
    *,
    split: str,
    token_atol: float,
    sequence_atol: float,
) -> list[dict[str, Any]]:
    base = root / "data" / "target_rescores" / "gsm8k" / f"split={split}"
    if not base.exists():
        raise DiagnosticError(f"Formal Program 04 output root does not exist: {base}")

    cols = [
        "trajectory_id",
        "training_seed",
        "split",
        "behavior_step",
        "target_step",
        "purpose",
        "identity_abs_sequence_logprob_diff",
        "identity_max_abs_token_logprob_diff",
        "identity_mean_abs_token_logprob_diff",
        "identity_pass",
        "target_adapter_sha256",
        "behavior_adapter_sha256",
    ]

    failures: list[dict[str, Any]] = []
    for pp in sorted(base.rglob("rescored.parquet")):
        rows = read_parquet(pp, cols)
        for r in rows:
            if r.get("purpose") != "identity":
                continue
            if int(r.get("behavior_step", -1)) != int(r.get("target_step", -2)):
                raise DiagnosticError(f"Malformed identity row in {pp}")
            failed_numeric = numeric_identity_fail(r, token_atol, sequence_atol)
            stored_pass = bool(r.get("identity_pass"))
            if stored_pass == failed_numeric:
                # pass=True with numeric fail OR pass=False with numeric pass
                raise DiagnosticError(
                    f"Stored identity_pass is inconsistent with frozen tolerances in {pp}, "
                    f"trajectory={r.get('trajectory_id')}"
                )
            if not failed_numeric:
                continue

            mp = pp.parent / "manifest.json"
            manifest = read_json(mp)
            src_rel = (manifest.get("source_behavior") or {}).get("parquet_path")
            if not isinstance(src_rel, str) or not src_rel:
                raise DiagnosticError(f"Program 04 manifest lacks source behavior parquet path: {mp}")
            source_pp = root / Path(src_rel)

            failures.append(
                {
                    "rescore_parquet": pp,
                    "rescore_manifest": mp,
                    "source_parquet": source_pp,
                    "rescore_row": r,
                }
            )

    failures.sort(
        key=lambda x: (
            int(x["rescore_row"]["training_seed"]),
            int(x["rescore_row"]["target_step"]),
            str(x["rescore_row"]["trajectory_id"]),
        )
    )
    return failures


def source_row_and_generation_group(
    source_pp: Path,
    trajectory_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_parquet(source_pp)
    by_tid = {str(r.get("trajectory_id")): r for r in rows}
    if trajectory_id not in by_tid:
        raise DiagnosticError(
            f"Failed trajectory {trajectory_id} is absent from immutable source {source_pp}"
        )
    src = dict(by_tid[trajectory_id])
    call_id = src.get("generation_call_id")
    prompt_id = src.get("prompt_id")
    if not isinstance(call_id, str) or not call_id:
        raise DiagnosticError("Source behavior row lacks generation_call_id.")

    group = [
        dict(r)
        for r in rows
        if r.get("generation_call_id") == call_id and r.get("prompt_id") == prompt_id
    ]
    group.sort(key=lambda r: int(r["sample_index"]))
    if not group:
        raise DiagnosticError("Could not reconstruct the original generation-call group.")

    prompt0 = group[0].get("prompt_token_ids")
    seed0 = group[0].get("generation_seed")
    config0 = group[0].get("generation_config_sha256")
    call0 = group[0].get("generation_call_id")
    for r in group:
        if r.get("prompt_token_ids") != prompt0:
            raise DiagnosticError("Generation-call group contains different prompt_token_ids.")
        if r.get("generation_seed") != seed0:
            raise DiagnosticError("Generation-call group contains different generation_seed values.")
        if r.get("generation_config_sha256") != config0:
            raise DiagnosticError("Generation-call group contains different generation config hashes.")
        if r.get("generation_call_id") != call0:
            raise DiagnosticError("Generation-call grouping invariant failed.")

    sample_indices = [int(r["sample_index"]) for r in group]
    if sample_indices != list(range(min(sample_indices), min(sample_indices) + len(sample_indices))):
        raise DiagnosticError(f"Generation-call sample indices are not contiguous: {sample_indices}")

    return src, group


def cached_replay_group(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    device: str,
) -> list[list[float]]:
    """
    Incrementally teacher-force a saved generation-call group with KV cache.

    The prompt is duplicated across the same number of rows that were produced
    by the original Program 03 generation call.  At each generation step, saved
    tokens are forced into the next model call.  Rows already terminated with
    EOS feed pad_token_id on later global steps, matching the standard generate()
    finished-sequence behavior while the other rows remain active.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise DiagnosticError(f"Cannot import torch: {exc}") from exc

    if not rows:
        return []

    prompt = [int(x) for x in rows[0]["prompt_token_ids"]]
    if not prompt:
        raise DiagnosticError("Empty prompt in cached replay.")
    for r in rows:
        if [int(x) for x in r["prompt_token_ids"]] != prompt:
            raise DiagnosticError("Cached replay group has non-identical prompts.")

    completions = [[int(x) for x in r["completion_token_ids"]] for r in rows]
    if any(not c for c in completions):
        raise DiagnosticError("Cached replay received an empty completion.")

    batch = len(rows)
    max_steps = max(len(c) for c in completions)
    input_ids = torch.tensor([prompt] * batch, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    collected: list[list[float]] = [[] for _ in range(batch)]

    with torch.inference_mode():
        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        except TypeError:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = getattr(out, "logits", None)
        past = getattr(out, "past_key_values", None)
        if logits is None or logits.ndim != 3 or past is None:
            raise DiagnosticError("Initial cached replay forward did not return logits + past_key_values.")
        next_logits = logits[:, -1, :]

        for step in range(max_steps):
            log_probs = F.log_softmax(next_logits.to(dtype=torch.float32), dim=-1)

            feed_tokens: list[int] = []
            for i, comp in enumerate(completions):
                if step < len(comp):
                    token = int(comp[step])
                    value = log_probs[i, token].detach().to(dtype=torch.float64, device="cpu").item()
                    collected[i].append(float(value))
                    feed_tokens.append(token)
                else:
                    # generate() keeps finished rows in the batch and feeds pad
                    # while unfinished rows continue.
                    feed_tokens.append(int(pad_token_id))

            if step == max_steps - 1:
                break

            step_ids = torch.tensor(feed_tokens, dtype=torch.long, device=device).unsqueeze(1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch, 1), dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )

            try:
                out = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            except TypeError:
                out = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )

            logits = getattr(out, "logits", None)
            past = getattr(out, "past_key_values", None)
            if logits is None or logits.ndim != 3 or past is None:
                raise DiagnosticError(f"Cached replay failed at generation step {step + 1}.")
            next_logits = logits[:, -1, :]

    for r, vals in zip(rows, collected):
        expected = len(r["completion_token_ids"])
        if len(vals) != expected:
            raise DiagnosticError(
                f"Cached replay returned {len(vals)} scores for a {expected}-token completion."
            )
        if not all(math.isfinite(x) for x in vals):
            raise DiagnosticError("Cached replay produced non-finite token log-probabilities.")

    return collected


def exact_regenerate_call(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Re-run the exact Program 03 generation-call design from saved metadata.
    This is read-only and is used only as a diagnostic reference.
    """
    try:
        import torch
        from transformers import GenerationConfig
    except Exception as exc:
        raise DiagnosticError(f"Cannot import generation dependencies: {exc}") from exc

    if not rows:
        return [], []

    row0 = rows[0]
    prompt = [int(x) for x in row0["prompt_token_ids"]]
    n = len(rows)
    generation_seed = int(row0["generation_seed"])

    eos = tokenizer.eos_token_id
    if eos is None:
        raise DiagnosticError("Tokenizer has no eos_token_id.")
    if isinstance(eos, int):
        eos_value: int | list[int] = int(eos)
    else:
        eos_list = [int(x) for x in eos]
        if not eos_list:
            raise DiagnosticError("Tokenizer eos_token_id list is empty.")
        eos_value = sorted(set(eos_list))
        if len(eos_value) == 1:
            eos_value = eos_value[0]

    kwargs = {
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": n,
        "max_new_tokens": int(row0["max_completion_length"]),
        "temperature": float(row0["temperature"]),
        "top_p": float(row0["top_p"]),
        "top_k": int(row0["top_k"]),
        "repetition_penalty": float(row0["repetition_penalty"]),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": eos_value,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    generation_config = GenerationConfig(**kwargs)

    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
    attention = torch.ones_like(prompt_tensor, dtype=torch.long, device=device)

    devices: list[int] = []
    if device == "cuda":
        devices = [torch.cuda.current_device()]

    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(generation_seed)
        if device == "cuda":
            torch.cuda.manual_seed(generation_seed)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=prompt_tensor,
                attention_mask=attention,
                generation_config=generation_config,
            )

    scores = getattr(outputs, "scores", None)
    sequences = getattr(outputs, "sequences", None)
    if scores is None or sequences is None:
        raise DiagnosticError("Exact regeneration did not return sequences + scores.")

    n_steps = len(scores)
    if n_steps <= 0:
        raise DiagnosticError("Exact regeneration returned zero score steps.")

    try:
        transition = model.compute_transition_scores(
            sequences,
            scores,
            normalize_logits=True,
        )
    except Exception as exc:
        raise DiagnosticError(f"compute_transition_scores failed during regeneration: {exc}") from exc

    generated = sequences[:, len(prompt) : len(prompt) + n_steps]
    if int(generated.shape[0]) != n or int(transition.shape[0]) != n:
        raise DiagnosticError("Exact regeneration returned the wrong group size.")

    generated_cpu = generated.detach().to("cpu")
    transition_cpu = transition.detach().to(dtype=torch.float64, device="cpu")

    token_lists: list[list[int]] = []
    logp_lists: list[list[float]] = []
    for i, r in enumerate(rows):
        cut = len(r["completion_token_ids"])
        token_lists.append([int(x) for x in generated_cpu[i, :cut].tolist()])
        logp_lists.append([float(x) for x in transition_cpu[i, :cut].tolist()])

    return token_lists, logp_lists


def compare_logps(
    label: str,
    candidate: Sequence[float],
    behavior: Sequence[float],
    *,
    token_atol: float,
    sequence_atol: float,
) -> dict[str, Any]:
    if len(candidate) != len(behavior):
        raise DiagnosticError(
            f"{label}: candidate length {len(candidate)} != behavior length {len(behavior)}"
        )
    diffs = [abs(float(a) - float(b)) for a, b in zip(candidate, behavior)]
    max_token = max(diffs) if diffs else 0.0
    mean_token = math.fsum(diffs) / len(diffs) if diffs else 0.0
    seq_diff = abs(math.fsum(float(x) for x in candidate) - math.fsum(float(x) for x in behavior))
    passed = max_token <= token_atol and seq_diff <= sequence_atol
    result = {
        "label": label,
        "max_token_diff": max_token,
        "mean_token_diff": mean_token,
        "sequence_diff": seq_diff,
        "pass": passed,
    }
    print(
        f"{label:<31} "
        f"max_token={max_token:.12g}  "
        f"mean_token={mean_token:.12g}  "
        f"seq_diff={seq_diff:.12g}  "
        f"PASS={passed}"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only cached-replay diagnostic for formal Program 04 identity failures."
    )
    p.add_argument("--output-root", default=r"C:\lsg\grpo_ope_reuse")
    p.add_argument("--split", choices=("development",), default="development")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="Safety cap on the number of failed identity trajectories to diagnose.",
    )
    p.add_argument(
        "--skip-regenerate",
        action="store_true",
        help="Skip exact re-execution of the original Program 03 generation call.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    if args.max_failures <= 0:
        raise DiagnosticError("--max-failures must be positive.")

    print("=" * 96)
    print("GRPO-OPE CACHED IDENTITY DIAGNOSTIC — READ ONLY")
    print("=" * 96)
    print(f"project root : {root}")
    print(f"split        : {args.split}")
    print(f"device       : {args.device}")
    print("writes       : NONE")
    print()

    p04 = load_program04(root)

    cfg, _ = p04.load_protocol(root / "configs" / "protocol.yaml")
    frozen_spec = p04.parse_rescore_spec(cfg, None)
    token_atol = float(frozen_spec.identity_token_atol)
    sequence_atol = float(frozen_spec.identity_sequence_atol)

    print(f"frozen token tolerance    : {token_atol}")
    print(f"frozen sequence tolerance : {sequence_atol}")

    failures = discover_formal_failures(
        root,
        split=args.split,
        token_atol=token_atol,
        sequence_atol=sequence_atol,
    )
    print(f"formal numeric identity failures found: {len(failures)}")

    if not failures:
        print("\nNo formal identity failures are currently present. Nothing to diagnose.")
        return 0
    if len(failures) > args.max_failures:
        raise DiagnosticError(
            f"Found {len(failures)} failures, above --max-failures={args.max_failures}. "
            "Stop and inspect before running a broader diagnostic."
        )

    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as exc:
        raise DiagnosticError(f"Cannot import torch/transformers: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise DiagnosticError("--device cuda requested but CUDA is unavailable.")

    model_dir = root / "models" / "qwen25_05b"
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise DiagnosticError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    # Load source rows/groups once.
    prepared: list[dict[str, Any]] = []
    for item in failures:
        rr = dict(item["rescore_row"])
        tid = str(rr["trajectory_id"])
        src, group = source_row_and_generation_group(item["source_parquet"], tid)
        prepared.append(
            {
                **item,
                "source_row": src,
                "generation_group": group,
            }
        )

    # Group failures by identity adapter so each FP32 model is loaded once.
    by_model: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        rr = item["rescore_row"]
        by_model[(int(rr["training_seed"]), int(rr["target_step"]))].append(item)

    final_results: list[dict[str, Any]] = []

    for (seed, step), items in sorted(by_model.items()):
        adapter_dir = root / "checkpoints" / f"seed_{seed}" / "adapters" / f"step_{step:04d}"
        if not adapter_dir.exists():
            raise DiagnosticError(f"Missing adapter: {adapter_dir}")

        observed_adapter_hash = p04.adapter_payload_hash(adapter_dir)
        expected_hashes = {str(x["rescore_row"]["target_adapter_sha256"]) for x in items}
        if len(expected_hashes) != 1 or observed_adapter_hash not in expected_hashes:
            raise DiagnosticError(
                f"Adapter hash mismatch for seed={seed}, step={step}. "
                f"observed={observed_adapter_hash}, expected={sorted(expected_hashes)}"
            )

        print("\n" + "=" * 96)
        print(f"LOAD IDENTITY MODEL  seed={seed}  step={step}  precision=fp32")
        print(f"adapter: {adapter_dir}")

        model, precision = p04.load_target_model(
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            device=args.device,
            preferred_precision="fp32",
        )
        if precision != "fp32":
            raise DiagnosticError(f"Expected fp32 model, got {precision!r}")

        try:
            for item in items:
                rr = item["rescore_row"]
                src = item["source_row"]
                group = item["generation_group"]
                tid = str(rr["trajectory_id"])
                saved = [float(x) for x in src["behavior_token_logprobs"]]

                print("\n" + "-" * 96)
                print(f"trajectory       : {tid}")
                print(f"prompt_id        : {src['prompt_id']}")
                print(f"sample_index     : {src['sample_index']}")
                print(f"length           : {len(saved)}")
                print(f"truncated        : {src['was_truncated']}")
                print(f"generation_call  : {src['generation_call_id']}")
                print(f"generation_seed  : {src['generation_seed']}")
                print(f"call group size  : {len(group)}")
                print(f"group samples     : {[int(r['sample_index']) for r in group]}")
                print(
                    "stored P04       : "
                    f"max_token={float(rr['identity_max_abs_token_logprob_diff']):.12g}  "
                    f"mean_token={float(rr['identity_mean_abs_token_logprob_diff']):.12g}  "
                    f"seq_diff={float(rr['identity_abs_sequence_logprob_diff']):.12g}"
                )
                print()

                # 1. Full-sequence Program 04 path, recomputed now.
                full = p04.teacher_force_batch(
                    model=model,
                    rows=[src],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=args.device,
                )[0]
                r_full = compare_logps(
                    "full-sequence TF (batch=1)",
                    full,
                    saved,
                    token_atol=token_atol,
                    sequence_atol=sequence_atol,
                )

                # 2. Incremental cache, trajectory alone.
                single_cached = cached_replay_group(
                    model,
                    [src],
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=args.device,
                )[0]
                r_single = compare_logps(
                    "cached replay (single)",
                    single_cached,
                    saved,
                    token_atol=token_atol,
                    sequence_atol=sequence_atol,
                )

                # 3. Incremental cache with reconstructed original generation-call batch.
                group_cached = cached_replay_group(
                    model,
                    group,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=args.device,
                )
                group_index = next(
                    (i for i, r in enumerate(group) if str(r["trajectory_id"]) == tid),
                    None,
                )
                if group_index is None:
                    raise DiagnosticError("Failed trajectory vanished from reconstructed call group.")
                r_group = compare_logps(
                    "cached replay (call batch)",
                    group_cached[group_index],
                    saved,
                    token_atol=token_atol,
                    sequence_atol=sequence_atol,
                )

                regen_exact_tokens: bool | None = None
                r_regen: dict[str, Any] | None = None

                # 4. Re-run original Program 03 generation call, same group size/seed/config.
                if not args.skip_regenerate:
                    regen_tokens, regen_logps = exact_regenerate_call(
                        model,
                        tokenizer,
                        group,
                        device=args.device,
                    )
                    token_matches = [
                        regen_tokens[i] == [int(x) for x in group[i]["completion_token_ids"]]
                        for i in range(len(group))
                    ]
                    regen_exact_tokens = all(token_matches)
                    target_token_match = bool(token_matches[group_index])
                    print(
                        f"{'exact regenerate tokens':<31} "
                        f"target_match={target_token_match}  all_group_match={regen_exact_tokens}"
                    )
                    r_regen = compare_logps(
                        "exact regenerate logprobs",
                        regen_logps[group_index],
                        saved,
                        token_atol=token_atol,
                        sequence_atol=sequence_atol,
                    )

                final_results.append(
                    {
                        "trajectory_id": tid,
                        "length": len(saved),
                        "truncated": bool(src["was_truncated"]),
                        "full_pass": bool(r_full["pass"]),
                        "cached_single_pass": bool(r_single["pass"]),
                        "cached_call_batch_pass": bool(r_group["pass"]),
                        "regenerated_group_tokens_exact": regen_exact_tokens,
                        "regenerated_logprob_pass": None if r_regen is None else bool(r_regen["pass"]),
                        "full_sequence_diff": float(r_full["sequence_diff"]),
                        "cached_single_sequence_diff": float(r_single["sequence_diff"]),
                        "cached_call_batch_sequence_diff": float(r_group["sequence_diff"]),
                        "regenerated_sequence_diff": None if r_regen is None else float(r_regen["sequence_diff"]),
                    }
                )

        finally:
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print("\n" + "=" * 96)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 96)
    for r in final_results:
        print(
            f"trajectory={r['trajectory_id'][:16]}... len={r['length']} "
            f"full={r['full_pass']} "
            f"cached_single={r['cached_single_pass']} "
            f"cached_call_batch={r['cached_call_batch_pass']} "
            f"regen_tokens_exact={r['regenerated_group_tokens_exact']} "
            f"regen_logp={r['regenerated_logprob_pass']}"
        )

    all_call_cached = all(r["cached_call_batch_pass"] for r in final_results)
    regen_rows = [r for r in final_results if r["regenerated_logprob_pass"] is not None]
    all_regen = bool(regen_rows) and all(
        bool(r["regenerated_group_tokens_exact"]) and bool(r["regenerated_logprob_pass"])
        for r in regen_rows
    )

    print()
    print(f"all failed rows pass cached call-batch replay : {all_call_cached}")
    if regen_rows:
        print(f"all original calls reproduce tokens+logprobs  : {all_regen}")
    print("outputs modified                              : NO")
    print()
    print("Interpretation:")
    if all_call_cached:
        print(
            "  Cached call-batch replay satisfies the ORIGINAL frozen identity tolerances "
            "for every diagnosed failure."
        )
        print(
            "  This supports a numerical-path explanation for the full-sequence teacher-forcing failures."
        )
    else:
        print(
            "  At least one failure persists under cached call-batch replay. "
            "Do NOT change Program 04 or relax tolerances yet."
        )
    if regen_rows and all_regen:
        print(
            "  Re-executing Program 03 also reproduced the original saved tokens/logprobs, "
            "supporting the integrity of the immutable behavior evidence."
        )
    elif regen_rows:
        print(
            "  Exact regeneration did not reproduce every saved call exactly; inspect the detailed output "
            "before drawing conclusions."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"\nDIAGNOSTIC FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nDIAGNOSTIC INTERRUPTED. No research outputs were modified.", file=sys.stderr)
        raise SystemExit(130)
