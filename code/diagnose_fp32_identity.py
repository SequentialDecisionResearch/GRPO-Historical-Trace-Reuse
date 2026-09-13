#!/usr/bin/env python3
"""
diagnose_fp32_identity.py

Read-only test of whether FP32 removes the Program03/Program04 probability-path
mismatch.

It uses one existing Program03 smoke prompt only as a fixed diagnostic input.
It then:
  1) loads the same base model + LoRA adapter in FP32 on CUDA (or CPU),
  2) generates a fresh deterministic diagnostic batch,
  3) obtains generation-time selected-token log probabilities exactly as
     Program03 does,
  4) teacher-forces those *freshly generated same tokens* in FP32,
  5) reports token- and sequence-level differences.

No existing research artifact is modified.
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


def load_module(path: Path, name: str) -> Any:
    if not path.exists():
        raise DiagnosticError(f"Missing source file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiagnosticError(f"Missing JSON file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise DiagnosticError(f"Expected JSON object: {path}")
    return obj


def behavior_root(root: Path, mode: str, split: str, seed: int, behavior_step: int) -> Path:
    base = root / "data" / "behavior_logs"
    if mode != "paper":
        base = base / f"_{mode}"
    return base / "gsm8k" / f"split={split}" / f"seed={seed}" / f"behavior_step={behavior_step:04d}"


def adapter_path(root: Path, mode: str, seed: int, step: int) -> Path:
    if mode == "paper":
        seed_root = root / "checkpoints" / f"seed_{seed}"
    else:
        seed_root = root / "checkpoints" / f"_{mode}" / "program02" / f"seed_{seed}"
    return seed_root / "adapters" / f"step_{step:04d}"


def load_first_behavior_row(base: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise DiagnosticError(f"pyarrow is required: {exc}") from exc
    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise DiagnosticError(f"No behavior parquet found under {base}")
    rows = pq.read_table(files[0]).to_pylist()
    if not rows:
        raise DiagnosticError(f"No behavior rows in {files[0]}")
    return dict(rows[0])


def load_fp32_model(model_dir: Path, adapter_dir: Path, device: str) -> Any:
    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as exc:
        raise DiagnosticError(f"Cannot import model stack: {exc}") from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise DiagnosticError("CUDA requested but unavailable.")

    try:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.float32,
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
            )
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        model.to(device)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    except Exception as exc:
        raise DiagnosticError(f"Cannot load FP32 model + adapter: {exc}") from exc


def teacher_force_one(model: Any, prompt_ids: Sequence[int], completion_ids: Sequence[int], device: str) -> list[float]:
    import torch  # type: ignore

    p = [int(x) for x in prompt_ids]
    c = [int(x) for x in completion_ids]
    full = torch.tensor([p + c], dtype=torch.long, device=device)
    mask = torch.ones_like(full)

    with torch.inference_mode():
        out = model(input_ids=full, attention_mask=mask, use_cache=False, return_dict=True)

    logits = out.logits[0, len(p) - 1 : len(p) + len(c) - 1, :]
    if int(logits.shape[0]) != len(c):
        raise DiagnosticError("Teacher-forcing causal shift length mismatch.")

    targets = full[0, len(p) : len(p) + len(c)]
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    vals = [float(x) for x in selected.double().cpu().tolist()]
    if not all(math.isfinite(x) for x in vals):
        raise DiagnosticError("Non-finite teacher-forced log probabilities.")
    return vals


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only FP32 generation-vs-teacher-forcing identity diagnostic.")
    p.add_argument("--output-root", default=".")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="smoke")
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--behavior-step", type=int, default=0)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--num-sequences", type=int, default=2)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    p03 = load_module(root / "03_collect_behavior_logs.py", "_grpo_ope_p03_fp32_diag")

    model_manifest = read_json(root / "manifests" / "model_manifest.json")
    models = model_manifest.get("models")
    if not isinstance(models, Mapping) or not isinstance(models.get("primary"), Mapping):
        raise DiagnosticError("model_manifest.json lacks models.primary")
    model_record = dict(models["primary"])

    model_dir = root / "models" / "qwen25_05b"
    adir = adapter_path(root, args.mode, args.seed, args.behavior_step)
    brow = load_first_behavior_row(
        behavior_root(root, args.mode, args.split, args.seed, args.behavior_step)
    )

    prompt_ids = [int(x) for x in brow["prompt_token_ids"]]
    generation_seed = int(brow["generation_seed"])
    temperature = float(brow["temperature"])
    top_p = float(brow["top_p"])
    top_k = int(brow["top_k"])
    repetition_penalty = float(brow["repetition_penalty"])
    max_new_tokens = int(brow["max_completion_length"])

    tokenizer = p03.load_tokenizer(model_dir, model_record)
    eos_set = p03.eos_ids(tokenizer)
    model = load_fp32_model(model_dir, adir, args.device)

    import torch  # type: ignore

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    attention = torch.ones_like(prompt_tensor)

    kwargs = {
        "do_sample": True,
        "num_beams": 1,
        "num_return_sequences": int(args.num_sequences),
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

    print("=" * 78)
    print("FP32 generation-vs-teacher-forcing diagnostic — READ ONLY")
    print("=" * 78)
    print(f"root                 : {root}")
    print(f"mode/split           : {args.mode}/{args.split}")
    print(f"seed/behavior step   : {args.seed}/{args.behavior_step}")
    print(f"device               : {args.device}")
    print(f"diagnostic precision : fp32")
    print(f"num sequences        : {args.num_sequences}")
    print(f"generation seed      : {generation_seed}")
    print("=" * 78)

    outputs = p03._seeded_generate(
        model, prompt_tensor, attention, kwargs, generation_seed, args.device
    )
    scores = outputs.scores
    if scores is None:
        raise DiagnosticError("generate() returned no scores.")
    n_steps = len(scores)

    transition = model.compute_transition_scores(
        outputs.sequences, scores, normalize_logits=True
    ).double().cpu()

    generated = outputs.sequences[:, len(prompt_ids) : len(prompt_ids) + n_steps].cpu()

    token_diffs: list[float] = []
    seq_diffs: list[float] = []

    for i in range(int(args.num_sequences)):
        all_tokens = [int(x) for x in generated[i].tolist()]
        cut, ended_eos, truncated = p03.completion_cut_length(
            all_tokens, eos_set, n_steps, max_new_tokens
        )
        comp = all_tokens[:cut]
        gen_logps = [float(x) for x in transition[i, :cut].tolist()]
        tf_logps = teacher_force_one(model, prompt_ids, comp, args.device)

        tdiff = max_abs_diff(gen_logps, tf_logps)
        sdiff = abs(math.fsum(gen_logps) - math.fsum(tf_logps))
        token_diffs.append(tdiff)
        seq_diffs.append(sdiff)

        print("-" * 78)
        print(f"sequence {i}")
        print(f"completion length               : {len(comp)}")
        print(f"ended_eos / truncated           : {ended_eos} / {truncated}")
        print(f"max |generation-teacher token|  : {tdiff:.12g}")
        print(f"|generation-teacher sequence|   : {sdiff:.12g}")

    print("\n" + "=" * 78)
    print("FP32 DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(f"max token diff    : {max(token_diffs):.12g}")
    print(f"max sequence diff : {max(seq_diffs):.12g}")

    # Diagnostic thresholds only. They are not protocol/paper tolerances.
    if max(token_diffs) <= 1e-5 and max(seq_diffs) <= 1e-4:
        print("RESULT: FP32 CONSISTENCY PASS")
        print("The large BF16 mismatch is numerical/path dependent. A shared FP32")
        print("probability-evidence path is a viable fix to test in Programs 03/04.")
    elif max(token_diffs) <= 1e-4 and max(seq_diffs) <= 1e-3:
        print("RESULT: FP32 CONSISTENCY NEAR-PASS")
        print("The BF16 mismatch is largely removed; inspect tolerances and kernels before redesign.")
    else:
        print("RESULT: FP32 CONSISTENCY FAIL")
        print("The mismatch persists even in FP32. Precision alone is not the fix;")
        print("Programs 03/04 need one common cached autoregressive scoring implementation.")

    print("\nREAD-ONLY FP32 DIAGNOSTIC COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"\nDIAGNOSTIC FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
