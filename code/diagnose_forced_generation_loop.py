#!/usr/bin/env python3
"""
diagnose_forced_generation_loop.py

READ-ONLY diagnostic for the GRPO-OPE Program 04 identity-gate issue.

What this tests
---------------
Program 03 defined behavior-policy probability evidence using Hugging Face
`model.generate(..., output_scores=True)` followed by
`compute_transition_scores(..., normalize_logits=True)`.

Program 04 currently uses a separate full-sequence teacher-forced model forward.
For two long formal development trajectories, token-level differences are tiny
but their sequence sums exceed the frozen 0.05 identity tolerance.

This diagnostic runs the *actual Hugging Face generate loop* but inserts a
custom LogitsProcessor that:

  1. observes/captures the generation-loop scores BEFORE forcing;
  2. records the log-probability of the already-saved Program 03 token;
  3. replaces the outgoing scores so that the saved token is forced.

Therefore there is NO resampling of the research trajectory: the exact saved
tokens are replayed while Hugging Face itself handles prepare_inputs_for_generation,
cache_position, KV cache, attention-mask updates, finished-sequence handling, etc.

For every formal identity failure, the script reconstructs the original
Program 03 generation_call_id group (normally 4 samples) and tests the whole
group, so the batch structure also matches Program 03.

It also optionally re-runs the original unforced Program 03 generation call as
a reference check.

SAFETY
------
This script is read-only. It does NOT modify:
- configs/protocol.yaml
- Programs 00-09
- checkpoints
- Program 03 behavior logs
- Program 04 rescore outputs
- manifests
- protocol lock

Typical Spyder command:
    %run "C:/lsg/grpo_ope_reuse/diagnose_forced_generation_loop.py"

Optional:
    %run "C:/lsg/grpo_ope_reuse/diagnose_forced_generation_loop.py" --skip-reference-regeneration
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiagnosticError(f"Missing JSON file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiagnosticError(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise DiagnosticError(f"Expected JSON object: {path}")
    return obj


def read_parquet(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise DiagnosticError(f"Cannot import pyarrow: {exc}") from exc
    if not path.exists():
        raise DiagnosticError(f"Missing Parquet file: {path}")
    try:
        table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    except Exception as exc:
        raise DiagnosticError(f"Cannot read Parquet {path}: {exc}") from exc
    return [dict(r) for r in table.to_pylist()]


def load_program04(root: Path) -> Any:
    path = root / "04_rescore_target_checkpoints.py"
    if not path.exists():
        raise DiagnosticError(f"Missing Program 04: {path}")
    spec = importlib.util.spec_from_file_location("p04_forced_loop_diag", path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"Cannot import Program 04 from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def numeric_identity_fail(
    row: Mapping[str, Any],
    *,
    token_atol: float,
    sequence_atol: float,
) -> bool:
    tok = row.get("identity_max_abs_token_logprob_diff")
    seq = row.get("identity_abs_sequence_logprob_diff")
    if tok is None or seq is None:
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
        raise DiagnosticError(f"Formal Program 04 output directory is missing: {base}")

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
        "behavior_adapter_sha256",
        "target_adapter_sha256",
    ]

    out: list[dict[str, Any]] = []
    for pp in sorted(base.rglob("rescored.parquet")):
        rows = read_parquet(pp, columns=cols)
        for row in rows:
            if row.get("purpose") != "identity":
                continue
            if int(row.get("behavior_step", -1)) != int(row.get("target_step", -2)):
                raise DiagnosticError(f"Malformed identity row in {pp}")

            failed = numeric_identity_fail(
                row,
                token_atol=token_atol,
                sequence_atol=sequence_atol,
            )
            stored = bool(row.get("identity_pass"))
            expected_stored = not failed
            if stored != expected_stored:
                raise DiagnosticError(
                    "Stored identity_pass is inconsistent with frozen tolerances: "
                    f"{pp}, trajectory={row.get('trajectory_id')}"
                )
            if not failed:
                continue

            manifest_path = pp.parent / "manifest.json"
            manifest = read_json(manifest_path)
            src_rel = (manifest.get("source_behavior") or {}).get("parquet_path")
            if not isinstance(src_rel, str) or not src_rel:
                raise DiagnosticError(f"Missing source_behavior.parquet_path in {manifest_path}")

            out.append(
                {
                    "rescore_parquet": pp,
                    "rescore_manifest": manifest_path,
                    "source_parquet": root / Path(src_rel),
                    "rescore_row": row,
                }
            )

    out.sort(
        key=lambda x: (
            int(x["rescore_row"]["training_seed"]),
            int(x["rescore_row"]["target_step"]),
            str(x["rescore_row"]["trajectory_id"]),
        )
    )
    return out


def reconstruct_generation_group(
    source_parquet: Path,
    trajectory_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_parquet(source_parquet)
    target = None
    for r in rows:
        if str(r.get("trajectory_id")) == trajectory_id:
            target = dict(r)
            break
    if target is None:
        raise DiagnosticError(
            f"Trajectory {trajectory_id} not found in immutable source {source_parquet}"
        )

    call_id = target.get("generation_call_id")
    prompt_id = target.get("prompt_id")
    if not isinstance(call_id, str) or not call_id:
        raise DiagnosticError("Behavior row lacks generation_call_id.")

    group = [
        dict(r)
        for r in rows
        if r.get("generation_call_id") == call_id and r.get("prompt_id") == prompt_id
    ]
    group.sort(key=lambda r: int(r["sample_index"]))
    if not group:
        raise DiagnosticError("Could not reconstruct generation-call group.")

    reference_fields = (
        "prompt_token_ids",
        "generation_seed",
        "generation_call_id",
        "generation_config_sha256",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_completion_length",
    )
    for field in reference_fields:
        ref = group[0].get(field)
        if any(r.get(field) != ref for r in group):
            raise DiagnosticError(
                f"Generation-call group is inconsistent in field {field!r}."
            )

    samples = [int(r["sample_index"]) for r in group]
    if samples != list(range(samples[0], samples[0] + len(samples))):
        raise DiagnosticError(f"Generation-call sample indices are not contiguous: {samples}")

    return target, group


def eos_value_from_tokenizer(tokenizer: Any) -> int | list[int]:
    raw = tokenizer.eos_token_id
    if raw is None:
        raise DiagnosticError("Tokenizer has no eos_token_id.")
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, (list, tuple, set)) and raw:
        vals = sorted({int(x) for x in raw})
        return vals[0] if len(vals) == 1 else vals
    raise DiagnosticError(f"Unsupported tokenizer eos_token_id={raw!r}")


def generation_config_kwargs(
    tokenizer: Any,
    row0: Mapping[str, Any],
    *,
    n_return: int,
) -> dict[str, Any]:
    return {
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": int(n_return),
        "max_new_tokens": int(row0["max_completion_length"]),
        "temperature": float(row0["temperature"]),
        "top_p": float(row0["top_p"]),
        "top_k": int(row0["top_k"]),
        "repetition_penalty": float(row0["repetition_penalty"]),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": eos_value_from_tokenizer(tokenizer),
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }


class CaptureAndForceProcessor:
    """
    Logits processor used inside Hugging Face generate().

    `scores` are the generation-loop scores received after the model forward and
    previously constructed built-in processors.  We capture the normalized
    selected-token probability from these scores, then force the already-saved
    Program 03 token for the actual next-token choice.
    """

    def __init__(
        self,
        *,
        forced_completions: Sequence[Sequence[int]],
        prompt_length: int,
        pad_token_id: int,
    ) -> None:
        self.forced = [[int(x) for x in seq] for seq in forced_completions]
        self.prompt_length = int(prompt_length)
        self.pad_token_id = int(pad_token_id)
        self.captured: list[list[float]] = [[] for _ in self.forced]
        self.steps_seen: list[int] = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        if input_ids.ndim != 2 or scores.ndim != 2:
            raise DiagnosticError("Forced logits processor received unexpected tensor rank.")
        if int(scores.shape[0]) != len(self.forced):
            raise DiagnosticError(
                "Generation-loop batch size does not match reconstructed Program 03 call: "
                f"scores_batch={int(scores.shape[0])}, expected={len(self.forced)}"
            )

        step = int(input_ids.shape[1]) - self.prompt_length
        if step < 0:
            raise DiagnosticError("Generation-loop sequence shorter than original prompt.")
        self.steps_seen.append(step)

        log_probs = torch.nn.functional.log_softmax(scores.to(dtype=torch.float32), dim=-1)

        next_tokens: list[int] = []
        for i, completion in enumerate(self.forced):
            if step < len(completion):
                token = int(completion[step])
                lp = log_probs[i, token].detach().to(dtype=torch.float64, device="cpu").item()
                self.captured[i].append(float(lp))
                next_tokens.append(token)
            else:
                # The original row already ended with EOS. Hugging Face's own
                # unfinished-sequence logic will pad it while other rows continue.
                next_tokens.append(self.pad_token_id)

        # Exactly one finite outgoing logit per row. With do_sample=True this
        # still enters the same sampling branch, but the saved token is certain.
        forced_scores = torch.full_like(scores, -float("inf"))
        row_idx = torch.arange(len(next_tokens), device=scores.device)
        tok_idx = torch.tensor(next_tokens, dtype=torch.long, device=scores.device)
        forced_scores[row_idx, tok_idx] = 0.0
        return forced_scores


def run_forced_generation_loop(
    model: Any,
    tokenizer: Any,
    group: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> tuple[list[list[int]], list[list[float]], list[int]]:
    try:
        import torch
        from transformers import GenerationConfig, LogitsProcessorList
    except Exception as exc:
        raise DiagnosticError(f"Cannot import generation dependencies: {exc}") from exc

    if not group:
        raise DiagnosticError("Empty generation group.")

    prompt = [int(x) for x in group[0]["prompt_token_ids"]]
    if not prompt:
        raise DiagnosticError("Empty saved prompt.")
    completions = [[int(x) for x in r["completion_token_ids"]] for r in group]
    if any(not c for c in completions):
        raise DiagnosticError("Saved generation group contains an empty completion.")

    processor = CaptureAndForceProcessor(
        forced_completions=completions,
        prompt_length=len(prompt),
        pad_token_id=int(tokenizer.pad_token_id),
    )

    cfg = GenerationConfig(
        **generation_config_kwargs(tokenizer, group[0], n_return=len(group))
    )
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    # The random seed is not needed to select tokens because the processor makes
    # each saved token certain, but matching the original seed keeps the sampling
    # branch as close as possible to Program 03.
    seed = int(group[0]["generation_seed"])
    devices: list[int] = [torch.cuda.current_device()] if device == "cuda" else []

    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=cfg,
                logits_processor=LogitsProcessorList([processor]),
            )

    sequences = getattr(outputs, "sequences", None)
    if sequences is None:
        raise DiagnosticError("Forced generate() returned no sequences.")
    if int(sequences.shape[0]) != len(group):
        raise DiagnosticError(
            f"Forced generate() returned {int(sequences.shape[0])} rows, expected {len(group)}."
        )

    generated = sequences[:, len(prompt):].detach().to("cpu")
    generated_prefixes: list[list[int]] = []
    for i, comp in enumerate(completions):
        generated_prefixes.append([int(x) for x in generated[i, : len(comp)].tolist()])

    for i, (cap, comp) in enumerate(zip(processor.captured, completions)):
        if len(cap) != len(comp):
            raise DiagnosticError(
                f"Captured {len(cap)} log-probabilities for row {i}, expected {len(comp)}."
            )
        if not all(math.isfinite(float(x)) for x in cap):
            raise DiagnosticError("Forced generation captured a non-finite log-probability.")

    return generated_prefixes, processor.captured, processor.steps_seen


def run_reference_regeneration(
    model: Any,
    tokenizer: Any,
    group: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> tuple[list[list[int]], list[list[float]]]:
    """Re-run the original Program 03 generate()+transition-score path."""
    try:
        import torch
        from transformers import GenerationConfig
    except Exception as exc:
        raise DiagnosticError(f"Cannot import generation dependencies: {exc}") from exc

    prompt = [int(x) for x in group[0]["prompt_token_ids"]]
    cfg = GenerationConfig(
        **generation_config_kwargs(tokenizer, group[0], n_return=len(group))
    )

    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    seed = int(group[0]["generation_seed"])
    devices: list[int] = [torch.cuda.current_device()] if device == "cuda" else []

    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=cfg,
            )

    scores = getattr(outputs, "scores", None)
    sequences = getattr(outputs, "sequences", None)
    if scores is None or sequences is None:
        raise DiagnosticError("Reference regeneration returned no sequences/scores.")

    try:
        transition = model.compute_transition_scores(
            sequences,
            scores,
            normalize_logits=True,
        )
    except Exception as exc:
        raise DiagnosticError(f"compute_transition_scores failed: {exc}") from exc

    n_steps = len(scores)
    generated = sequences[:, len(prompt): len(prompt) + n_steps].detach().to("cpu")
    transition = transition.detach().to(dtype=torch.float64, device="cpu")

    token_prefixes: list[list[int]] = []
    logp_prefixes: list[list[float]] = []
    for i, r in enumerate(group):
        n = len(r["completion_token_ids"])
        token_prefixes.append([int(x) for x in generated[i, :n].tolist()])
        logp_prefixes.append([float(x) for x in transition[i, :n].tolist()])
    return token_prefixes, logp_prefixes


def compare(
    candidate: Sequence[float],
    saved: Sequence[float],
    *,
    token_atol: float,
    sequence_atol: float,
) -> dict[str, Any]:
    if len(candidate) != len(saved):
        raise DiagnosticError(
            f"Comparison length mismatch: candidate={len(candidate)}, saved={len(saved)}"
        )
    diffs = [abs(float(a) - float(b)) for a, b in zip(candidate, saved)]
    max_token = max(diffs) if diffs else 0.0
    mean_token = math.fsum(diffs) / len(diffs) if diffs else 0.0
    seq_diff = abs(
        math.fsum(float(x) for x in candidate)
        - math.fsum(float(x) for x in saved)
    )
    return {
        "max_token_diff": max_token,
        "mean_token_diff": mean_token,
        "sequence_diff": seq_diff,
        "pass": max_token <= token_atol and seq_diff <= sequence_atol,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only Hugging Face forced-generation-loop identity diagnostic."
    )
    p.add_argument("--output-root", default=r"C:\lsg\grpo_ope_reuse")
    p.add_argument("--split", choices=("development",), default="development")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--max-failures", type=int, default=20)
    p.add_argument(
        "--skip-reference-regeneration",
        action="store_true",
        help="Skip the already-established unforced Program 03 regeneration check.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()

    print("=" * 100)
    print("GRPO-OPE FORCED GENERATION-LOOP DIAGNOSTIC — READ ONLY")
    print("=" * 100)
    print(f"project root : {root}")
    print(f"split        : {args.split}")
    print(f"device       : {args.device}")
    print("writes       : NONE")

    p04 = load_program04(root)
    cfg, _ = p04.load_protocol(root / "configs" / "protocol.yaml")
    rescore_spec = p04.parse_rescore_spec(cfg, None)
    token_atol = float(rescore_spec.identity_token_atol)
    sequence_atol = float(rescore_spec.identity_sequence_atol)

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
        print("Nothing to diagnose.")
        return 0
    if len(failures) > args.max_failures:
        raise DiagnosticError(
            f"Found {len(failures)} failures, above safety cap {args.max_failures}."
        )

    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as exc:
        raise DiagnosticError(f"Cannot import torch/transformers: {exc}") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise DiagnosticError("CUDA requested but torch.cuda.is_available() is False.")

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

    prepared: list[dict[str, Any]] = []
    for item in failures:
        rr = item["rescore_row"]
        tid = str(rr["trajectory_id"])
        target, group = reconstruct_generation_group(item["source_parquet"], tid)
        prepared.append({**item, "target_source_row": target, "group": group})

    # Avoid running the same generation_call_id twice if two failures happened
    # to occur in one call.
    unique_calls: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in prepared:
        rr = item["rescore_row"]
        group = item["group"]
        key = (
            int(rr["training_seed"]),
            int(rr["target_step"]),
            str(group[0]["generation_call_id"]),
        )
        unique_calls.setdefault(key, item)

    by_model: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for (seed, step, _), item in unique_calls.items():
        by_model[(seed, step)].append(item)

    diagnosed_failed: dict[str, dict[str, Any]] = {}
    group_control_results: list[dict[str, Any]] = []

    for (seed, step), items in sorted(by_model.items()):
        adapter_dir = root / "checkpoints" / f"seed_{seed}" / "adapters" / f"step_{step:04d}"
        if not adapter_dir.exists():
            raise DiagnosticError(f"Missing adapter: {adapter_dir}")

        expected_hashes = {
            str(x["rescore_row"]["target_adapter_sha256"])
            for x in prepared
            if int(x["rescore_row"]["training_seed"]) == seed
            and int(x["rescore_row"]["target_step"]) == step
        }
        observed_hash = p04.adapter_payload_hash(adapter_dir)
        if len(expected_hashes) != 1 or observed_hash not in expected_hashes:
            raise DiagnosticError(
                f"Adapter hash mismatch for seed={seed}, step={step}: "
                f"observed={observed_hash}, expected={sorted(expected_hashes)}"
            )

        print("\n" + "=" * 100)
        print(f"LOAD MODEL seed={seed} step={step} precision=fp32")
        print(f"adapter: {adapter_dir}")

        model, precision = p04.load_target_model(
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            device=args.device,
            preferred_precision="fp32",
        )
        if precision != "fp32":
            raise DiagnosticError(f"Expected fp32 inference, got {precision!r}")

        try:
            for item in items:
                group = item["group"]
                call_id = str(group[0]["generation_call_id"])
                failed_ids_in_call = {
                    str(x["rescore_row"]["trajectory_id"])
                    for x in prepared
                    if str(x["group"][0]["generation_call_id"]) == call_id
                    and int(x["rescore_row"]["training_seed"]) == seed
                    and int(x["rescore_row"]["target_step"]) == step
                }

                print("\n" + "-" * 100)
                print(f"generation_call_id : {call_id}")
                print(f"generation_seed    : {group[0]['generation_seed']}")
                print(f"group size         : {len(group)}")
                print(f"sample indices     : {[int(r['sample_index']) for r in group]}")
                print(f"completion lengths : {[len(r['completion_token_ids']) for r in group]}")
                print(f"failed trajectories: {len(failed_ids_in_call)}")

                forced_tokens, forced_logps, steps_seen = run_forced_generation_loop(
                    model,
                    tokenizer,
                    group,
                    device=args.device,
                )

                print(f"forced-loop score steps observed: {len(steps_seen)}")
                print()

                all_forced_tokens_exact = True
                for i, r in enumerate(group):
                    tid = str(r["trajectory_id"])
                    saved_tokens = [int(x) for x in r["completion_token_ids"]]
                    saved_logps = [float(x) for x in r["behavior_token_logprobs"]]

                    tokens_exact = forced_tokens[i] == saved_tokens
                    all_forced_tokens_exact = all_forced_tokens_exact and tokens_exact
                    cmp = compare(
                        forced_logps[i],
                        saved_logps,
                        token_atol=token_atol,
                        sequence_atol=sequence_atol,
                    )
                    is_failed_target = tid in failed_ids_in_call

                    print(
                        f"{'*FAIL*' if is_failed_target else 'control':>7} "
                        f"sample={int(r['sample_index']):2d} len={len(saved_tokens):3d} "
                        f"tokens_exact={str(tokens_exact):5s} "
                        f"max_token={cmp['max_token_diff']:.12g} "
                        f"mean_token={cmp['mean_token_diff']:.12g} "
                        f"seq_diff={cmp['sequence_diff']:.12g} "
                        f"PASS={cmp['pass']}"
                    )

                    rec = {
                        "trajectory_id": tid,
                        "is_original_failure": is_failed_target,
                        "forced_tokens_exact": tokens_exact,
                        **cmp,
                    }
                    group_control_results.append(rec)
                    if is_failed_target:
                        diagnosed_failed[tid] = rec

                print(f"all forced output prefixes exact: {all_forced_tokens_exact}")

                if not args.skip_reference_regeneration:
                    ref_tokens, ref_logps = run_reference_regeneration(
                        model,
                        tokenizer,
                        group,
                        device=args.device,
                    )
                    ref_tokens_exact = all(
                        ref_tokens[i] == [int(x) for x in group[i]["completion_token_ids"]]
                        for i in range(len(group))
                    )
                    ref_logps_exact = True
                    max_ref_seq = 0.0
                    max_ref_tok = 0.0
                    for i, r in enumerate(group):
                        cmp_ref = compare(
                            ref_logps[i],
                            [float(x) for x in r["behavior_token_logprobs"]],
                            token_atol=token_atol,
                            sequence_atol=sequence_atol,
                        )
                        max_ref_seq = max(max_ref_seq, float(cmp_ref["sequence_diff"]))
                        max_ref_tok = max(max_ref_tok, float(cmp_ref["max_token_diff"]))
                        if float(cmp_ref["sequence_diff"]) != 0.0 or float(cmp_ref["max_token_diff"]) != 0.0:
                            ref_logps_exact = False

                    print(
                        "reference original generate: "
                        f"tokens_exact={ref_tokens_exact}, "
                        f"logprobs_bitwise_numeric_exact={ref_logps_exact}, "
                        f"max_token_diff={max_ref_tok:.12g}, "
                        f"max_seq_diff={max_ref_seq:.12g}"
                    )
        finally:
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print("\n" + "=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)

    missing = [
        str(x["rescore_row"]["trajectory_id"])
        for x in prepared
        if str(x["rescore_row"]["trajectory_id"]) not in diagnosed_failed
    ]
    if missing:
        raise DiagnosticError(f"Some failed trajectories were not diagnosed: {missing}")

    for item in prepared:
        tid = str(item["rescore_row"]["trajectory_id"])
        r = diagnosed_failed[tid]
        print(
            f"trajectory={tid[:16]}... "
            f"forced_tokens_exact={r['forced_tokens_exact']} "
            f"max_token={r['max_token_diff']:.12g} "
            f"mean_token={r['mean_token_diff']:.12g} "
            f"seq_diff={r['sequence_diff']:.12g} "
            f"PASS={r['pass']}"
        )

    all_failed_pass = all(
        bool(r["forced_tokens_exact"]) and bool(r["pass"])
        for r in diagnosed_failed.values()
    )
    all_controls_pass = all(
        bool(r["forced_tokens_exact"]) and bool(r["pass"])
        for r in group_control_results
    )

    print()
    print(f"all original failed rows pass forced generate loop : {all_failed_pass}")
    print(f"all rows in reconstructed call groups pass         : {all_controls_pass}")
    print("outputs modified                                   : NO")
    print()

    if all_failed_pass and all_controls_pass:
        print("RESULT: FORCED GENERATION-LOOP DIAGNOSTIC PASSED")
        print(
            "The Hugging Face generate-loop probability path can score the fixed saved "
            "trajectories within the ORIGINAL frozen identity tolerances."
        )
        print(
            "This supports replacing Program 04's full-sequence scoring engine with a "
            "generation-loop-equivalent forced replay, subject to a formal protocol amendment."
        )
    else:
        print("RESULT: FORCED GENERATION-LOOP DIAGNOSTIC DID NOT PASS")
        print(
            "Do NOT modify Program 04 or relax tolerances yet. The remaining mismatch "
            "requires further diagnosis."
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
