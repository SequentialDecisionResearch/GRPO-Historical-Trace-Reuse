#!/usr/bin/env python3
"""
diagnose_identity.py

Read-only forensic diagnostic for the GRPO-OPE Program03 -> Program04
identity mismatch.

This script does NOT modify, delete, reset, or publish any research artifact.
It inspects one original Program03 generation call and compares:

  A. saved Program03 completion tokens
     vs exact re-generation with the saved generation_seed/config;

  B. saved Program03 generation-time token log-probabilities
     vs re-generated generation-time token log-probabilities;

  C. saved Program03 generation-time token log-probabilities
     vs a separate raw full-sequence teacher-forced forward pass;

  D. when A succeeds, re-generated generation-time log-probabilities
     vs raw teacher-forced log-probabilities.

The purpose is localization only. Do not use this script to create paper
results, tune tolerances, or replace Programs 03/04.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


class DiagnosticError(RuntimeError):
    pass


def load_numeric_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise DiagnosticError(f"Missing source file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and some typing machinery expect the module to be registered.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiagnosticError(f"Missing JSON file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiagnosticError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise DiagnosticError(f"Expected a JSON object: {path}")
    return obj


def behavior_root(root: Path, mode: str, split: str, seed: int, behavior_step: int) -> Path:
    if mode == "paper":
        base = root / "data" / "behavior_logs"
    else:
        base = root / "data" / "behavior_logs" / f"_{mode}"
    return (
        base
        / "gsm8k"
        / f"split={split}"
        / f"seed={seed}"
        / f"behavior_step={behavior_step:04d}"
    )


def training_manifest_path(root: Path, mode: str, seed: int) -> Path:
    if mode == "paper":
        return root / "manifests" / "training" / f"seed_{seed}.json"
    return root / "manifests" / f"_{mode}" / "training" / f"seed_{seed}.json"


def adapter_path(root: Path, mode: str, seed: int, behavior_step: int) -> Path:
    if mode == "paper":
        seed_root = root / "checkpoints" / f"seed_{seed}"
    else:
        seed_root = root / "checkpoints" / f"_{mode}" / "program02" / f"seed_{seed}"
    return seed_root / "adapters" / f"step_{behavior_step:04d}"


def load_behavior_rows(base: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise DiagnosticError(f"pyarrow is required: {exc}") from exc

    if not base.exists():
        raise DiagnosticError(f"Behavior directory does not exist: {base}")

    parquet_files = sorted(base.rglob("*.parquet"))
    if not parquet_files:
        raise DiagnosticError(f"No behavior Parquet files found under: {base}")

    rows: list[dict[str, Any]] = []
    for path in parquet_files:
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise DiagnosticError(f"Cannot read {path}: {exc}") from exc
        for row in table.to_pylist():
            if isinstance(row, Mapping):
                rows.append(dict(row))

    if not rows:
        raise DiagnosticError(f"No behavior rows found under: {base}")
    return rows


def select_generation_call(
    rows: Sequence[Mapping[str, Any]],
    call_id: str | None,
    call_index: int,
) -> tuple[str, list[dict[str, Any]]]:
    usable = [
        dict(r)
        for r in rows
        if isinstance(r.get("generation_call_id"), str)
        and isinstance(r.get("sample_index"), int)
        and isinstance(r.get("prompt_id"), str)
    ]
    if not usable:
        raise DiagnosticError("Behavior rows contain no usable generation_call_id.")

    ordered = sorted(
        usable,
        key=lambda r: (
            str(r["prompt_id"]),
            int(r["sample_index"]),
            str(r["generation_call_id"]),
        ),
    )
    unique_ids: list[str] = []
    for r in ordered:
        cid = str(r["generation_call_id"])
        if cid not in unique_ids:
            unique_ids.append(cid)

    if call_id is None:
        if call_index < 0 or call_index >= len(unique_ids):
            raise DiagnosticError(
                f"--call-index={call_index} is outside available range 0..{len(unique_ids)-1}"
            )
        call_id = unique_ids[call_index]
    elif call_id not in unique_ids:
        raise DiagnosticError(f"Requested generation_call_id not present: {call_id}")

    group = [dict(r) for r in ordered if str(r["generation_call_id"]) == call_id]
    group.sort(key=lambda r: int(r["sample_index"]))
    if not group:
        raise DiagnosticError("Selected generation call has no rows.")
    return call_id, group


def ensure_same(group: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [r.get(field) for r in group]
    first = values[0]
    if any(v != first for v in values[1:]):
        raise DiagnosticError(f"Selected generation call mixes field {field!r}: {values}")
    return first


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return math.inf
    if not a:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def sequence_sum(xs: Sequence[float]) -> float:
    return math.fsum(float(x) for x in xs)


def first_token_mismatch(a: Sequence[int], b: Sequence[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    if len(a) != len(b):
        return n
    return None


def teacher_force_one(
    model: Any,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    device: str,
) -> list[float]:
    """Raw full-sequence teacher-forced selected-token log probabilities."""
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise DiagnosticError(f"Cannot import torch: {exc}") from exc

    p = [int(x) for x in prompt_ids]
    c = [int(x) for x in completion_ids]
    if not p or not c:
        raise DiagnosticError("Teacher forcing requires non-empty prompt and completion.")

    full = torch.tensor([p + c], dtype=torch.long, device=device)
    attention = torch.ones_like(full)

    with torch.inference_mode():
        try:
            out = model(
                input_ids=full,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            out = model(
                input_ids=full,
                attention_mask=attention,
                use_cache=False,
            )

    logits = getattr(out, "logits", None)
    if logits is None or logits.ndim != 3:
        raise DiagnosticError("Teacher-forcing forward pass returned invalid logits.")

    p_len = len(p)
    c_len = len(c)
    pred = logits[0, p_len - 1 : p_len + c_len - 1, :]
    if int(pred.shape[0]) != c_len:
        raise DiagnosticError("Teacher-forcing causal shift length mismatch.")

    targets = full[0, p_len : p_len + c_len]
    # Report both paths using the current Transformers/PyTorch environment.
    # float32 normalization is the most useful comparison to generation scores.
    log_probs = torch.nn.functional.log_softmax(pred.to(dtype=torch.float32), dim=-1)
    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    vals = [float(x) for x in selected.detach().to(dtype=torch.float64, device="cpu").tolist()]
    if len(vals) != c_len or not all(math.isfinite(x) for x in vals):
        raise DiagnosticError("Teacher-forced log probabilities are invalid.")
    return vals


def print_row_comparison(
    *,
    sample_index: int,
    saved_tokens: Sequence[int],
    regenerated_tokens: Sequence[int],
    saved_logps: Sequence[float],
    regenerated_logps: Sequence[float],
    teacher_logps: Sequence[float],
) -> tuple[bool, float | None, float, float | None]:
    token_match = list(map(int, saved_tokens)) == list(map(int, regenerated_tokens))
    mismatch = first_token_mismatch(saved_tokens, regenerated_tokens)

    saved_teacher_token = max_abs_diff(saved_logps, teacher_logps)
    saved_teacher_seq = abs(sequence_sum(saved_logps) - sequence_sum(teacher_logps))

    print("-" * 78)
    print(f"sample_index                 : {sample_index}")
    print(f"saved completion length      : {len(saved_tokens)}")
    print(f"regenerated completion length: {len(regenerated_tokens)}")
    print(f"saved == regenerated tokens  : {token_match}")
    print(f"first token mismatch index    : {mismatch}")

    saved_regen_token: float | None = None
    saved_regen_seq: float | None = None
    regen_teacher_token: float | None = None
    regen_teacher_seq: float | None = None

    if token_match:
        saved_regen_token = max_abs_diff(saved_logps, regenerated_logps)
        saved_regen_seq = abs(sequence_sum(saved_logps) - sequence_sum(regenerated_logps))
        regen_teacher_token = max_abs_diff(regenerated_logps, teacher_logps)
        regen_teacher_seq = abs(sequence_sum(regenerated_logps) - sequence_sum(teacher_logps))

        print(f"saved vs regenerated max |token logp diff| : {saved_regen_token:.12g}")
        print(f"saved vs regenerated |sequence logp diff|  : {saved_regen_seq:.12g}")
        print(f"regen vs teacher max |token logp diff|      : {regen_teacher_token:.12g}")
        print(f"regen vs teacher |sequence logp diff|       : {regen_teacher_seq:.12g}")
    else:
        print("saved vs regenerated logp comparison        : NOT MEANINGFUL (tokens differ)")
        print("regen vs teacher logp comparison            : NOT MEANINGFUL (tokens differ)")

    print(f"saved vs teacher max |token logp diff|       : {saved_teacher_token:.12g}")
    print(f"saved vs teacher |sequence logp diff|        : {saved_teacher_seq:.12g}")

    return token_match, saved_regen_token, saved_teacher_token, regen_teacher_token


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only forensic diagnostic for Program03 -> Program04 identity mismatch."
    )
    p.add_argument("--output-root", default=".")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="smoke")
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--behavior-step", type=int, default=0)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--call-index", type=int, default=0)
    p.add_argument("--generation-call-id", default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()

    p03_path = root / "03_collect_behavior_logs.py"
    p03 = load_numeric_module(p03_path, "_grpo_ope_program03_forensic")

    model_manifest = read_json(root / "manifests" / "model_manifest.json")
    models = model_manifest.get("models")
    if not isinstance(models, Mapping) or not isinstance(models.get("primary"), Mapping):
        raise DiagnosticError("model_manifest.json lacks models.primary")
    model_record = dict(models["primary"])

    model_dir = root / "models" / "qwen25_05b"
    if not model_dir.exists():
        raise DiagnosticError(f"Missing pinned model directory: {model_dir}")

    tm_path = training_manifest_path(root, args.mode, args.seed)
    training_manifest = read_json(tm_path)
    preferred_precision = training_manifest.get("resolved_precision")
    if preferred_precision is not None:
        preferred_precision = str(preferred_precision)

    adir = adapter_path(root, args.mode, args.seed, args.behavior_step)
    if not adir.exists():
        raise DiagnosticError(f"Missing adapter directory: {adir}")

    base = behavior_root(root, args.mode, args.split, args.seed, args.behavior_step)
    all_rows = load_behavior_rows(base)
    call_id, group = select_generation_call(
        all_rows,
        args.generation_call_id,
        args.call_index,
    )

    # Validate the key Program03 call invariants before touching the model.
    ensure_same(group, "prompt_id")
    prompt_ids = ensure_same(group, "prompt_token_ids")
    generation_seed = int(ensure_same(group, "generation_seed"))
    ensure_same(group, "generation_call_id")
    temperature = float(ensure_same(group, "temperature"))
    top_p = float(ensure_same(group, "top_p"))
    top_k = int(ensure_same(group, "top_k"))
    repetition_penalty = float(ensure_same(group, "repetition_penalty"))
    max_new_tokens = int(ensure_same(group, "max_completion_length"))
    saved_adapter_sha = str(ensure_same(group, "behavior_adapter_sha256"))
    saved_gen_hash = str(ensure_same(group, "generation_config_sha256"))

    sample_indices = [int(r["sample_index"]) for r in group]
    if sample_indices != list(range(min(sample_indices), min(sample_indices) + len(group))):
        raise DiagnosticError(
            f"Selected generation call does not have contiguous sample indices: {sample_indices}"
        )

    observed_adapter_sha = p03.adapter_payload_hash(adir)
    if observed_adapter_sha != saved_adapter_sha:
        raise DiagnosticError(
            "Adapter hash mismatch before diagnostic.\n"
            f"saved   ={saved_adapter_sha}\n"
            f"observed={observed_adapter_sha}"
        )

    print("=" * 78)
    print("GRPO-OPE identity forensic diagnostic — READ ONLY")
    print("=" * 78)
    print(f"project root        : {root}")
    print(f"mode / split        : {args.mode} / {args.split}")
    print(f"seed / behavior step: {args.seed} / {args.behavior_step}")
    print(f"generation_call_id  : {call_id}")
    print(f"generation_seed     : {generation_seed}")
    print(f"sample indices      : {sample_indices}")
    print(f"rows in call        : {len(group)}")
    print(f"device              : {args.device}")
    print(f"adapter             : {adir}")
    print(f"saved adapter SHA   : {saved_adapter_sha}")
    print("=" * 78)

    tokenizer = p03.load_tokenizer(model_dir, model_record)
    eos_set = p03.eos_ids(tokenizer)

    # Reconstruct exactly Program03's generation-config fingerprint.
    gen_record = {
        "generation_config_source": "fresh_transformers.GenerationConfig_not_model_generation_config",
        "do_sample": True,
        "num_beams": 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": sorted(eos_set),
        "bos_token_id": None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id),
        "return_dict_in_generate": True,
        "output_scores": True,
        "transition_score_normalization": True,
        "logprob_definition": "selected-token log softmax of generation-time processed scores",
    }
    observed_gen_hash = p03.sha256_bytes(p03.canonical_bytes(gen_record))
    print(f"saved generation config SHA   : {saved_gen_hash}")
    print(f"rebuilt generation config SHA : {observed_gen_hash}")
    print(f"generation config hash match  : {saved_gen_hash == observed_gen_hash}")
    if saved_gen_hash != observed_gen_hash:
        raise DiagnosticError(
            "Generation-config fingerprint does not reproduce. Stop before model comparison."
        )

    model, resolved_precision = p03.load_behavior_model(
        model_dir=model_dir,
        adapter_dir=adir,
        device=args.device,
        preferred_precision=preferred_precision,
    )
    print(f"resolved inference precision  : {resolved_precision}")

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise DiagnosticError(f"Cannot import torch: {exc}") from exc

    prompt_tensor = torch.tensor([list(map(int, prompt_ids))], dtype=torch.long, device=args.device)
    attention = torch.ones_like(prompt_tensor)
    n = len(group)

    kwargs = {
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": n,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": sorted(eos_set) if len(eos_set) > 1 else next(iter(eos_set)),
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    print("\n[1/3] Re-running the original Program03 generation call...")
    outputs = p03._seeded_generate(
        model,
        prompt_tensor,
        attention,
        kwargs,
        generation_seed,
        args.device,
    )
    scores_tuple = getattr(outputs, "scores", None)
    sequences = getattr(outputs, "sequences", None)
    if scores_tuple is None or sequences is None:
        raise DiagnosticError("Re-generation returned no sequences/scores.")

    n_steps = len(scores_tuple)
    transition = model.compute_transition_scores(
        sequences,
        scores_tuple,
        normalize_logits=True,
    )
    if int(sequences.shape[0]) != n or int(transition.shape[0]) != n:
        raise DiagnosticError("Re-generation returned an unexpected number of sequences.")
    if int(transition.shape[1]) != n_steps:
        raise DiagnosticError("Re-generation transition-score length mismatch.")

    generated = sequences[:, len(prompt_ids) : len(prompt_ids) + n_steps]
    generated_cpu = generated.detach().to("cpu")
    transition_cpu = transition.detach().to(dtype=torch.float64, device="cpu")

    regenerated_tokens: list[list[int]] = []
    regenerated_logps: list[list[float]] = []

    for local_i in range(n):
        token_all = [int(x) for x in generated_cpu[local_i].tolist()]
        cut, _, _ = p03.completion_cut_length(
            token_all,
            eos_set,
            n_steps,
            max_new_tokens,
        )
        regenerated_tokens.append(token_all[:cut])
        regenerated_logps.append(
            [float(x) for x in transition_cpu[local_i, :cut].tolist()]
        )

    print("[2/3] Computing raw full-sequence teacher-forced probabilities...")
    teacher_logps: list[list[float]] = []
    for row in group:
        teacher_logps.append(
            teacher_force_one(
                model,
                row["prompt_token_ids"],
                row["completion_token_ids"],
                args.device,
            )
        )

    print("[3/3] Comparing the three probability paths...\n")

    all_tokens_match = True
    saved_regen_diffs: list[float] = []
    saved_teacher_diffs: list[float] = []
    regen_teacher_diffs: list[float] = []

    for j, row in enumerate(group):
        saved_tokens = [int(x) for x in row["completion_token_ids"]]
        saved_logps = [float(x) for x in row["behavior_token_logprobs"]]

        token_match, sr, st, rt = print_row_comparison(
            sample_index=int(row["sample_index"]),
            saved_tokens=saved_tokens,
            regenerated_tokens=regenerated_tokens[j],
            saved_logps=saved_logps,
            regenerated_logps=regenerated_logps[j],
            teacher_logps=teacher_logps[j],
        )
        all_tokens_match = all_tokens_match and token_match
        saved_teacher_diffs.append(st)
        if sr is not None:
            saved_regen_diffs.append(sr)
        if rt is not None:
            regen_teacher_diffs.append(rt)

    print("\n" + "=" * 78)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(f"all saved tokens reproduce exactly : {all_tokens_match}")
    if saved_regen_diffs:
        print(f"max saved-vs-regenerated token logp diff : {max(saved_regen_diffs):.12g}")
    else:
        print("max saved-vs-regenerated token logp diff : n/a (tokens did not reproduce)")
    print(f"max saved-vs-teacher token logp diff     : {max(saved_teacher_diffs):.12g}")
    if regen_teacher_diffs:
        print(f"max regenerated-vs-teacher token logp diff: {max(regen_teacher_diffs):.12g}")
    else:
        print("max regenerated-vs-teacher token logp diff: n/a (tokens did not reproduce)")

    # Conservative interpretation. Threshold is diagnostic only, never a paper tolerance.
    tiny = 1e-5
    print("\nINTERPRETATION")
    if not all_tokens_match:
        print(
            "A: FAIL — the saved generation call does not reproduce its tokens from the "
            "saved seed/config in the current environment."
        )
        print(
            "Next target: Program03 generation reproducibility / RNG / environment. "
            "Do not alter Program04 identity tolerances."
        )
    elif saved_regen_diffs and max(saved_regen_diffs) <= tiny:
        print("A: PASS — Program03 saved tokens and generation-time logprobs reproduce.")
        if max(saved_teacher_diffs) > tiny:
            print(
                "B: MISMATCH LOCALIZED — the separate teacher-forced forward path differs "
                "from the actual generation-time probability path."
            )
            print(
                "Next target: define one common probability-scoring path for both behavior "
                "and target policies before any OPE run."
            )
        else:
            print(
                "B: PASS — teacher-forced and generation-time probabilities also agree to "
                "diagnostic precision. Investigate Program04 artifact/pair handling instead."
            )
    else:
        print(
            "A: PARTIAL — tokens reproduce, but saved generation-time logprobs do not "
            "reproduce closely."
        )
        print(
            "Next target: Program03 score capture / compute_transition_scores / numerical "
            "determinism in the frozen software environment."
        )

    print("\nREAD-ONLY DIAGNOSTIC COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"\nDIAGNOSTIC FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
