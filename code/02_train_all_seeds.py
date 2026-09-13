#!/usr/bin/env python3
"""GRPO-OPE Program 02: multi-seed GRPO training with dense immutable adapters.

Research boundary
-----------------
This program implements only the GRPO policy-drift training stage of the paper.
Its purpose is not to maximize GSM8K benchmark performance.  Its purpose is to
produce multiple reproducible policy paths

    pi_0 -> pi_20 -> pi_40 -> ... -> pi_400

from the pinned Qwen2.5-0.5B-Instruct model and the 6000 immutable GSM8K
training prompts created by Program 01.

Program 02 does NOT collect the paper's behavior logs, rescore historical token
paths, generate on-policy OPE references, compute IS/pWIS/DR, calibrate a reuse
gate, or access official-test rewards.

Paper-mode invariants
---------------------
* Program 00 manifests and Program 01 split registry are re-verified.
* Only research_split='training' (6000 prompts) is dereferenced.
* manifests/protocol_lock.json must already exist and match protocol.yaml.
* At least three training seeds are required.
* max_steps=400 and save_steps=20 are required by the paper protocol unless the
  frozen config explicitly upgrades the protocol version in a new project.
* Sampling is explicitly fixed: temperature=1, top_p=1, top_k=0,
  repetition_penalty=1.
* loss_type is explicitly set; the default research setting is original
  GRPO (loss_type='grpo'), never TRL's changing default.
* Training reward is correctness + 0.05 * format by default.  Final OPE reward
  remains correctness-only in later programs.
* vLLM is disabled for the main training path to avoid introducing a separate
  sampler/training-engine mismatch into the policy path.
* Permanent LoRA adapters are published at step 0,20,...,400.
* A single rolling full Trainer checkpoint is maintained at resume_latest/.
* Resume never overwrites a permanent adapter with different content.

Engineering modes
-----------------
--mode smoke and --mode pilot are isolated engineering runs.  They never write
paper checkpoints/manifests.  smoke uses a tiny development subset and a few
steps; pilot uses 50 development prompts and at most 50 steps, matching the
pre-freeze engineering stage described in the code-design memo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


PROGRAM = "02_train_all_seeds.py"
PROGRAM_VERSION = "1.1.0"
MANIFEST_SCHEMA = "1.0"
TRAINING_MANIFEST_SCHEMA = "1.0"
CHECKPOINT_INDEX_SCHEMA = "1.0"

PROJECT_NAME = "grpo_ope_reuse"
GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

EXPECTED_UPSTREAM_ROWS = {"train": 7473, "test": 1319}
EXPECTED_RESEARCH_ROWS = {"training": 6000, "development": 1473, "test": 1319}
PAPER_REQUIRED_MAX_STEPS = 400
PAPER_REQUIRED_SAVE_STEPS = 20
PAPER_REQUIRED_MIN_SEEDS = 3

CORE_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "huggingface_hub",
)
EXTRA_PACKAGES = ("tokenizers", "safetensors", "pyarrow", "numpy", "PyYAML")
HASH_EXCLUDE_DIRS = {".git", ".cache", "__pycache__"}

PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning briefly.\n"
    "Put only the final numerical answer inside <answer>...</answer>.\n\n"
    "Problem:\n{question}"
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
GSM8K_GOLD_RE = re.compile(r"####\s*(.*?)\s*$", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*)?")

T01_COLUMNS = (
    "training_seed",
    "step",
    "loss",
    "learning_rate",
    "mean_reward",
    "correctness_reward",
    "format_reward",
    "reward_std",
    "mean_completion_length",
    "truncation_rate",
    "kl_proxy",
    "clip_fraction",
    "reward_zero_std_fraction",
    "policy_loss",
    "grad_norm",
    "gpu_memory_peak_bytes",
    "tokens_per_second",
    "num_tokens",
    "mode",
    "loss_type",
    "num_generations",
    "max_completion_length",
)


class Program02Error(RuntimeError):
    """Controlled Program 02 failure with an actionable message."""


@dataclass(frozen=True)
class TrainingSpec:
    seeds: tuple[int, ...]
    max_steps: int
    save_steps: int
    logging_steps: int
    num_generations: int
    max_completion_length: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    beta: float
    epsilon: float
    scale_rewards: str | bool
    loss_type: str
    num_iterations: int
    use_vllm: bool
    gradient_checkpointing: bool
    max_grad_norm: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    precision: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    lora_target_modules: str | tuple[str, ...]
    format_reward_weight: float
    shuffle_dataset: bool


# ---------------------------------------------------------------------------
# Generic integrity utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        raise Program02Error(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program02Error(f"Expected a JSON object in {path}.")
    return obj


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Program02Error(f"Refusing to overwrite immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise Program02Error(f"Immutable manifest appeared concurrently: {path}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def payload_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise Program02Error(f"Asset directory is missing: {root}")
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in HASH_EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.as_posix())


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def tree_hash(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    for p in payload_files(root):
        size = p.stat().st_size
        records.append({"path": rel(p, root), "size_bytes": size, "sha256": sha256_file(p)})
        total += size
    if not records:
        raise Program02Error(f"No files found in asset directory: {root}")
    return {
        "tree_sha256": sha256_bytes(canonical_bytes(records)),
        "file_count": len(records),
        "total_bytes": total,
        "files": records,
    }


def directory_digest_fast(root: Path) -> str:
    """Content digest for checkpoints/adapters; excludes transport/cache metadata."""
    return tree_hash(root)["tree_sha256"]


def make_read_only_tree(root: Path) -> None:
    failures: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            mode = stat.S_IMODE(p.stat().st_mode)
            p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except OSError as exc:
            failures.append(f"{p}: {exc}")
    if failures:
        print("[WARN] Could not mark every permanent adapter file read-only; hashes remain authoritative.")
        for msg in failures[:5]:
            print(f"       {msg}")


def git_commit(root: Path) -> str | None:
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        value = cp.stdout.strip()
        if cp.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", value):
            return value.lower()
    except Exception:
        pass
    return None


def fsync_directory(path: Path) -> None:
    """Best-effort directory fsync on POSIX; harmless no-op where unsupported."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Frozen environment / Program 00 / Program 01 verification
# ---------------------------------------------------------------------------


def current_environment_fingerprint() -> tuple[str, dict[str, Any]]:
    packages = {p: package_version(p) for p in CORE_PACKAGES + EXTRA_PACKAGES}
    missing = [p for p in CORE_PACKAGES if packages[p] is None]
    if missing:
        raise Program02Error(
            "Current environment is missing packages required/frozen by Program 00: " + ", ".join(missing)
        )
    if packages.get("pyarrow") is None:
        raise Program02Error("pyarrow is required to read the immutable split registry.")

    try:
        import torch  # type: ignore

        torch_cuda_build = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        raise Program02Error(f"Cannot inspect PyTorch environment: {exc}") from exc

    basis = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "torch_cuda_build": torch_cuda_build,
        "cudnn_version": cudnn_version,
    }
    return sha256_bytes(canonical_bytes(basis)), basis


def verify_environment_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Program02Error(f"Missing {path}. Program 00 must complete before Program 02.")
    env = read_json(path)
    if env.get("manifest_type") != "environment":
        raise Program02Error(f"Wrong manifest type in {path}.")
    expected = env.get("environment_fingerprint_sha256")
    observed, _ = current_environment_fingerprint()
    if expected != observed:
        raise Program02Error(
            "Current software environment differs from Program 00's frozen environment.\n"
            f"expected={expected}\nobserved={observed}"
        )
    return env


def verify_data_manifest(path: Path, gsm8k_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program02Error(f"Missing {path}. Program 00 must freeze GSM8K first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "data":
        raise Program02Error(f"Invalid Program 00 data manifest: {path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("gsm8k"), dict):
        raise Program02Error("data_manifest.json lacks datasets.gsm8k.")
    gsm = datasets["gsm8k"]
    if gsm.get("repo_id") != GSM8K_REPO or gsm.get("role") != "primary_dataset":
        raise Program02Error("Frozen primary dataset is not openai/gsm8k.")

    expected_fp = manifest.get("content_fingerprint_sha256")
    observed_fp = sha256_bytes(
        canonical_bytes({"research_scope": manifest.get("research_scope"), "datasets": datasets})
    )
    if expected_fp != observed_fp:
        raise Program02Error("Program 00 data manifest content fingerprint mismatch.")

    observed_tree = tree_hash(gsm8k_dir)["tree_sha256"]
    if observed_tree != gsm.get("tree_sha256"):
        raise Program02Error(
            "Pinned GSM8K local files differ from Program 00.\n"
            f"expected={gsm.get('tree_sha256')}\nobserved={observed_tree}"
        )
    return manifest, gsm


def tokenizer_template_hash(model_dir: Path) -> tuple[str, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program02Error(f"Cannot load pinned tokenizer locally: {exc}") from exc
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program02Error("Pinned tokenizer has no chat_template.")
    if isinstance(template, str):
        text = template
    else:
        text = json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(text), tok.__class__.__name__


def verify_model_manifest(path: Path, model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program02Error(f"Missing {path}. Program 00 must freeze the model first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "model":
        raise Program02Error(f"Invalid Program 00 model manifest: {path}")
    models = manifest.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
        raise Program02Error("model_manifest.json lacks models.primary.")
    model = models["primary"]
    if model.get("repo_id") != MODEL_REPO:
        raise Program02Error(f"Frozen primary model is not {MODEL_REPO}.")

    expected_fp = manifest.get("content_fingerprint_sha256")
    observed_fp = sha256_bytes(canonical_bytes(models))
    if expected_fp != observed_fp:
        raise Program02Error("Program 00 model manifest content fingerprint mismatch.")

    observed_tree = tree_hash(model_dir)["tree_sha256"]
    if observed_tree != model.get("tree_sha256"):
        raise Program02Error(
            "Pinned model local files differ from Program 00.\n"
            f"expected={model.get('tree_sha256')}\nobserved={observed_tree}"
        )

    validation = model.get("validation")
    if not isinstance(validation, dict) or not isinstance(validation.get("tokenizer"), dict):
        raise Program02Error("Frozen model manifest lacks tokenizer validation metadata.")
    observed_chat_hash, tok_class = tokenizer_template_hash(model_dir)
    expected_chat_hash = validation["tokenizer"].get("chat_template_sha256")
    if observed_chat_hash != expected_chat_hash:
        raise Program02Error(
            "Pinned tokenizer chat template differs from Program 00 manifest.\n"
            f"expected={expected_chat_hash}\nobserved={observed_chat_hash}"
        )
    if validation["tokenizer"].get("class") != tok_class:
        raise Program02Error("Tokenizer class differs from Program 00 validation.")
    return manifest, model


def verify_split_registry(
    manifest_path: Path,
    registry_path: Path,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.exists() or not registry_path.exists():
        raise Program02Error("Program 01 split registry/manifest is missing.")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "split_registry":
        raise Program02Error("Invalid Program 01 split registry manifest.")
    source = manifest.get("source")
    registry = manifest.get("registry")
    policy = manifest.get("split_policy")
    firewall = manifest.get("research_firewall")
    if not all(isinstance(x, dict) for x in (source, registry, policy, firewall)):
        raise Program02Error("Split registry manifest is missing required sections.")
    if source.get("dataset_revision") != gsm_record.get("resolved_revision"):
        raise Program02Error("Split registry dataset revision differs from Program 00.")
    if source.get("gsm8k_tree_sha256") != gsm_record.get("tree_sha256"):
        raise Program02Error("Split registry source tree differs from Program 00.")
    if source.get("program00_data_manifest_content_fingerprint_sha256") != data_manifest.get(
        "content_fingerprint_sha256"
    ):
        raise Program02Error("Split registry was created from a different Program 00 data manifest.")
    if firewall.get("official_test_is_never_used_for_split_tuning") is not True:
        raise Program02Error("Split registry does not preserve the official-test firewall.")
    if registry.get("file_sha256") != sha256_file(registry_path):
        raise Program02Error("Immutable split registry file SHA-256 mismatch.")

    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(registry_path)
        rows = table.to_pylist()
    except Exception as exc:
        raise Program02Error(f"Cannot read immutable split registry: {exc}") from exc

    counts: dict[str, int] = {"training": 0, "development": 0, "test": 0}
    prompt_ids: set[str] = set()
    for r in rows:
        split = r.get("research_split")
        if split not in counts:
            raise Program02Error(f"Unexpected research_split={split!r} in registry.")
        counts[str(split)] += 1
        pid = r.get("prompt_id")
        if not isinstance(pid, str) or not re.fullmatch(r"[0-9a-f]{64}", pid):
            raise Program02Error("Split registry contains invalid prompt_id.")
        if pid in prompt_ids:
            raise Program02Error("Split registry contains duplicate prompt_id.")
        prompt_ids.add(pid)
        if r.get("original_split") == "test" and split != "test":
            raise Program02Error("Official GSM8K test row leaked into training/development.")
    if counts != EXPECTED_RESEARCH_ROWS:
        raise Program02Error(f"Unexpected split counts: {counts}; expected {EXPECTED_RESEARCH_ROWS}.")

    # Reproduce Program 01's semantic content hash when its canonical columns are available.
    canonical_columns = registry.get("columns")
    if isinstance(canonical_columns, list) and canonical_columns:
        h = hashlib.sha256()
        for row in rows:
            try:
                canonical = {k: row[k] for k in canonical_columns}
            except KeyError as exc:
                raise Program02Error(f"Split registry missing canonical column {exc}.") from exc
            h.update(canonical_bytes(canonical))
            h.update(b"\n")
        if h.hexdigest() != registry.get("content_sha256"):
            raise Program02Error("Immutable split registry semantic content SHA-256 mismatch.")
    return manifest, rows


# ---------------------------------------------------------------------------
# Protocol configuration and lock
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program02Error(
            f"Missing protocol config {path}. Program 02 requires the pilot-resolved training protocol."
        )
    if package_version("PyYAML") is None:
        raise Program02Error("PyYAML is required to read protocol.yaml.")
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise Program02Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program02Error("protocol.yaml must contain a YAML mapping.")
    project = cfg.get("project") or {}
    if not isinstance(project, dict):
        raise Program02Error("protocol.yaml: project must be a mapping.")
    if project.get("name") not in (None, PROJECT_NAME):
        raise Program02Error(f"Unexpected project.name={project.get('name')!r}.")
    return cfg, sha256_file(path)


def _get_number(mapping: Mapping[str, Any], key: str, default: Any, *, integer: bool = False) -> Any:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        raise Program02Error(f"training.{key} must be numeric, not bool.")
    if integer:
        if not isinstance(value, int):
            raise Program02Error(f"training.{key} must be an integer.")
    else:
        if not isinstance(value, (int, float)):
            raise Program02Error(f"training.{key} must be numeric.")
    return value


def parse_training_spec(cfg: Mapping[str, Any]) -> TrainingSpec:
    t = cfg.get("training") or {}
    if not isinstance(t, dict):
        raise Program02Error("protocol.yaml: training must be a mapping.")
    lora = t.get("lora") or cfg.get("lora") or {}
    if not isinstance(lora, dict):
        raise Program02Error("protocol.yaml: training.lora/lora must be a mapping.")

    raw_seeds = t.get("seeds", [20260826, 20260827, 20260828])
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise Program02Error("training.seeds must be a non-empty list of integers.")
    seeds: list[int] = []
    for s in raw_seeds:
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise Program02Error("training.seeds must contain non-negative integers.")
        seeds.append(s)
    if len(set(seeds)) != len(seeds):
        raise Program02Error("training.seeds contains duplicates.")

    num_generations = int(_get_number(t, "num_generations", 4, integer=True))
    grad_acc = int(_get_number(t, "gradient_accumulation_steps", max(4, num_generations), integer=True))
    per_batch = int(_get_number(t, "per_device_train_batch_size", 1, integer=True))

    target_modules_raw = lora.get("target_modules", "all-linear")
    if isinstance(target_modules_raw, str):
        target_modules: str | tuple[str, ...] = target_modules_raw
    elif isinstance(target_modules_raw, list) and all(isinstance(x, str) and x for x in target_modules_raw):
        target_modules = tuple(target_modules_raw)
    else:
        raise Program02Error("lora.target_modules must be 'all-linear' or a list of module names.")

    scale_rewards = t.get("scale_rewards", "group")
    if scale_rewards not in (True, False, "group", "batch", "none"):
        raise Program02Error("training.scale_rewards must be group, batch, none, true, or false.")

    spec = TrainingSpec(
        seeds=tuple(seeds),
        max_steps=int(_get_number(t, "max_steps", 400, integer=True)),
        save_steps=int(_get_number(t, "save_steps", 20, integer=True)),
        logging_steps=int(_get_number(t, "logging_steps", 5, integer=True)),
        num_generations=num_generations,
        max_completion_length=int(_get_number(t, "max_completion_length", 128, integer=True)),
        temperature=float(_get_number(t, "temperature", 1.0)),
        top_p=float(_get_number(t, "top_p", 1.0)),
        top_k=int(_get_number(t, "top_k", 0, integer=True)),
        repetition_penalty=float(_get_number(t, "repetition_penalty", 1.0)),
        learning_rate=float(_get_number(t, "learning_rate", 5e-6)),
        per_device_train_batch_size=per_batch,
        gradient_accumulation_steps=grad_acc,
        beta=float(_get_number(t, "beta", 0.02)),
        epsilon=float(_get_number(t, "epsilon", 0.2)),
        scale_rewards=scale_rewards,
        loss_type=str(t.get("loss_type", "grpo")),
        num_iterations=int(_get_number(t, "num_iterations", 1, integer=True)),
        use_vllm=bool(t.get("use_vllm", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        max_grad_norm=float(_get_number(t, "max_grad_norm", 1.0)),
        warmup_ratio=float(_get_number(t, "warmup_ratio", 0.0)),
        weight_decay=float(_get_number(t, "weight_decay", 0.0)),
        lr_scheduler_type=str(t.get("lr_scheduler_type", "linear")),
        precision=str(t.get("precision", "auto")).lower(),
        lora_r=int(_get_number(lora, "r", 16, integer=True)),
        lora_alpha=int(_get_number(lora, "alpha", lora.get("lora_alpha", 32), integer=True)),
        lora_dropout=float(_get_number(lora, "dropout", lora.get("lora_dropout", 0.05))),
        lora_bias=str(lora.get("bias", "none")),
        lora_target_modules=target_modules,
        format_reward_weight=float(_get_number(t, "format_reward_weight", 0.05)),
        shuffle_dataset=bool(t.get("shuffle_dataset", True)),
    )
    validate_training_spec(spec)
    return spec


def validate_training_spec(spec: TrainingSpec) -> None:
    if spec.max_steps <= 0 or spec.save_steps <= 0 or spec.logging_steps <= 0:
        raise Program02Error("max_steps, save_steps, and logging_steps must all be positive.")
    if spec.max_steps % spec.save_steps != 0:
        raise Program02Error("training.max_steps must be divisible by training.save_steps.")
    if spec.num_generations <= 1:
        raise Program02Error("GRPO num_generations must be >= 2.")
    if not (spec.temperature > 0):
        raise Program02Error("training.temperature must be > 0.")
    if not (0 < spec.top_p <= 1):
        raise Program02Error("training.top_p must be in (0,1].")
    if spec.top_k < 0:
        raise Program02Error("training.top_k must be >= 0.")
    if spec.repetition_penalty <= 0:
        raise Program02Error("training.repetition_penalty must be > 0.")
    if spec.max_completion_length <= 0:
        raise Program02Error("training.max_completion_length must be positive.")
    if spec.learning_rate <= 0:
        raise Program02Error("training.learning_rate must be positive.")
    if spec.per_device_train_batch_size <= 0 or spec.gradient_accumulation_steps <= 0:
        raise Program02Error("Training batch sizes must be positive.")
    if spec.beta < 0 or spec.epsilon <= 0:
        raise Program02Error("training.beta must be >=0 and epsilon must be >0.")
    if spec.loss_type not in {"grpo", "dapo", "dr_grpo", "bnpo", "cispo", "sapo", "vespo"}:
        raise Program02Error(f"Unsupported/unknown explicit loss_type={spec.loss_type!r}.")
    if spec.use_vllm:
        raise Program02Error(
            "Main Program 02 intentionally requires use_vllm=false to keep generation and training "
            "inside the same Transformers path. Use a new protocol version to change this research choice."
        )
    if spec.precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise Program02Error("training.precision must be auto|bf16|fp16|fp32.")
    if spec.lora_r <= 0 or spec.lora_alpha <= 0 or not (0 <= spec.lora_dropout < 1):
        raise Program02Error("Invalid LoRA r/alpha/dropout.")
    if spec.lora_bias not in {"none", "all", "lora_only"}:
        raise Program02Error("LoRA bias must be none|all|lora_only.")
    if spec.format_reward_weight < 0 or spec.format_reward_weight > 0.2:
        raise Program02Error("format_reward_weight should be a small shaping term in [0,0.2].")


def spec_fingerprint(spec: TrainingSpec) -> str:
    return sha256_bytes(canonical_bytes(asdict(spec)))


def validate_paper_contract(spec: TrainingSpec) -> None:
    if len(spec.seeds) < PAPER_REQUIRED_MIN_SEEDS:
        raise Program02Error(f"Paper mode requires at least {PAPER_REQUIRED_MIN_SEEDS} training seeds.")
    if spec.max_steps != PAPER_REQUIRED_MAX_STEPS:
        raise Program02Error(
            f"Paper protocol requires max_steps={PAPER_REQUIRED_MAX_STEPS}; found {spec.max_steps}. "
            "If the pilot changed this, create a new protocol version and update the code-design contract."
        )
    if spec.save_steps != PAPER_REQUIRED_SAVE_STEPS:
        raise Program02Error(f"Paper protocol requires save_steps={PAPER_REQUIRED_SAVE_STEPS}.")
    if not (
        math.isclose(spec.temperature, 1.0)
        and math.isclose(spec.top_p, 1.0)
        and spec.top_k == 0
        and math.isclose(spec.repetition_penalty, 1.0)
    ):
        raise Program02Error(
            "Paper mode sampling must be temperature=1, top_p=1, top_k=0, repetition_penalty=1."
        )
    if spec.loss_type != "grpo":
        raise Program02Error(
            "The current paper's primary training path is original GRPO, so paper mode requires "
            "loss_type='grpo'. DAPO/Dr.GRPO are optional later sensitivity analyses."
        )


def _dig(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for p in paths:
        value = _dig(mapping, p)
        if value is not None:
            return value
    return None


def verify_protocol_lock(
    lock_path: Path,
    *,
    config_sha256: str,
    data_manifest: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    spec: TrainingSpec,
) -> dict[str, Any]:
    if not lock_path.exists():
        raise Program02Error(
            f"Paper mode requires frozen {lock_path}. Run the development GPU pilot and freeze the protocol first."
        )
    lock = read_json(lock_path)

    locked_config = first_present(
        lock,
        (
            ("config_sha256",),
            ("protocol_config_sha256",),
            ("protocol", "config_sha256"),
            ("inputs", "config_sha256"),
        ),
    )
    if not isinstance(locked_config, str):
        raise Program02Error(
            "protocol_lock.json must contain config_sha256 (or protocol_config_sha256 / protocol.config_sha256)."
        )
    if locked_config != config_sha256:
        raise Program02Error(
            "protocol.yaml differs from frozen protocol_lock.json.\n"
            f"locked={locked_config}\nobserved={config_sha256}"
        )

    # If a lock records the resolved training fingerprint, enforce it.  This is
    # stronger than raw YAML hashing because it also freezes explicit code defaults.
    locked_training = first_present(
        lock,
        (("training_config_sha256",), ("training", "resolved_config_sha256"), ("frozen", "training_config_sha256")),
    )
    if not isinstance(locked_training, str):
        raise Program02Error("protocol_lock.json must freeze training_config_sha256.")
    if locked_training != spec_fingerprint(spec):
        raise Program02Error("Resolved training configuration differs from protocol_lock.json.")

    expected_data = data_manifest.get("content_fingerprint_sha256")
    locked_data = first_present(
        lock,
        (("data_manifest_sha256",), ("inputs", "data_manifest_sha256"), ("data", "content_fingerprint_sha256")),
    )
    if not isinstance(locked_data, str):
        raise Program02Error("protocol_lock.json must freeze Program 00 data_manifest_sha256.")
    if locked_data != expected_data:
        raise Program02Error("Program 00 data manifest differs from protocol_lock.json.")

    primary = (model_manifest.get("models") or {}).get("primary") or {}
    model_revision = primary.get("resolved_revision")
    locked_model = first_present(
        lock,
        (("model_sha",), ("model_revision",), ("model", "resolved_revision"), ("inputs", "model_revision")),
    )
    if not isinstance(locked_model, str):
        raise Program02Error("protocol_lock.json must freeze the pinned model revision.")
    if locked_model != model_revision:
        raise Program02Error("Pinned Qwen revision differs from protocol_lock.json.")

    split_fp = split_manifest.get("content_fingerprint_sha256")
    locked_split = first_present(
        lock,
        (("split_registry_sha256",), ("inputs", "split_registry_sha256"), ("split", "content_fingerprint_sha256")),
    )
    if not isinstance(locked_split, str):
        raise Program02Error("protocol_lock.json must freeze Program 01 split_registry_sha256.")
    if locked_split != split_fp:
        raise Program02Error("Program 01 split registry differs from protocol_lock.json.")
    source_hashes = lock.get("source_code_sha256") or {}
    expected_source = source_hashes.get(PROGRAM) if isinstance(source_hashes, Mapping) else None
    if not isinstance(expected_source, str) or expected_source != sha256_file(Path(__file__).resolve()):
        raise Program02Error("Program 02 source code differs from the frozen protocol lock.")

    return lock


# ---------------------------------------------------------------------------
# GSM8K loading, no-test-leakage, prompt and rewards
# ---------------------------------------------------------------------------


def discover_gsm8k_parquet(root: Path) -> dict[str, list[Path]]:
    train = sorted((root / GSM8K_CONFIG).glob("train*.parquet"))
    test = sorted((root / GSM8K_CONFIG).glob("test*.parquet"))
    if not train:
        train = sorted(p for p in root.rglob("train*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts)
    if not test:
        test = sorted(p for p in root.rglob("test*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts)
    if not train or not test:
        raise Program02Error("Pinned GSM8K snapshot lacks main/train*.parquet or main/test*.parquet.")
    return {"train": train, "test": test}


def load_upstream_train_rows(gsm8k_dir: Path) -> list[dict[str, str]]:
    files = discover_gsm8k_parquet(gsm8k_dir)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("parquet", data_files={"train": [str(p) for p in files["train"]]}, split="train")
    except Exception as exc:
        raise Program02Error(f"Cannot load pinned local GSM8K train parquet: {exc}") from exc
    if len(ds) != EXPECTED_UPSTREAM_ROWS["train"]:
        raise Program02Error(f"Expected 7473 upstream train rows, found {len(ds)}.")
    if not {"question", "answer"}.issubset(set(ds.column_names)):
        raise Program02Error("Pinned GSM8K train data lacks question/answer fields.")
    rows: list[dict[str, str]] = []
    for i in range(len(ds)):
        q, a = ds[i]["question"], ds[i]["answer"]
        if not isinstance(q, str) or not isinstance(a, str):
            raise Program02Error(f"GSM8K train row {i}: question/answer must be strings.")
        rows.append({"question": q, "answer": a})
    return rows


def normalize_numeric_text(text: str) -> Fraction | None:
    cleaned = text.strip().replace(",", "").replace("$", "")
    if NUMBER_RE.fullmatch(cleaned) is None:
        return None
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(Decimal(cleaned))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def extract_final_answer(completion: str) -> Fraction | None:
    matches = ANSWER_RE.findall(completion)
    if len(matches) != 1:
        return None
    return normalize_numeric_text(matches[0])


def extract_gsm8k_gold(answer_field: str) -> Fraction | None:
    match = GSM8K_GOLD_RE.search(answer_field)
    if match is None:
        return None
    return normalize_numeric_text(match.group(1))


def canonical_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Conversational GRPO format: usually one assistant message.
        texts: list[str] = []
        for item in completion:
            if isinstance(item, Mapping) and isinstance(item.get("content"), str):
                texts.append(str(item["content"]))
        return "\n".join(texts)
    if isinstance(completion, Mapping) and isinstance(completion.get("content"), str):
        return str(completion["content"])
    return str(completion)


def parse_canonical_fraction(text: str) -> Fraction | None:
    return normalize_numeric_text(text)


def correctness_reward(completions: Sequence[Any], gold_answer: Sequence[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, gold in zip(completions, gold_answer):
        predicted = extract_final_answer(completion_to_text(completion))
        expected = parse_canonical_fraction(str(gold))
        rewards.append(float(predicted is not None and expected is not None and predicted == expected))
    if len(rewards) != len(completions):
        raise RuntimeError("TRL reward input length mismatch for correctness_reward.")
    return rewards


def format_reward(completions: Sequence[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion)
        matches = ANSWER_RE.findall(text)
        opening_tags = re.findall(r"<answer\s*>", text, flags=re.IGNORECASE)
        closing_tags = re.findall(r"</answer\s*>", text, flags=re.IGNORECASE)
        valid = (
            len(matches) == 1
            and len(opening_tags) == 1
            and len(closing_tags) == 1
            and normalize_numeric_text(matches[0]) is not None
        )
        rewards.append(float(valid))
    return rewards


def build_training_examples(
    *,
    registry_rows: Sequence[Mapping[str, Any]],
    raw_train: Sequence[Mapping[str, str]],
    research_split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if research_split not in {"training", "development"}:
        raise Program02Error("Program 02 may only dereference training or development rows, never test.")
    selected = [r for r in registry_rows if r.get("research_split") == research_split]
    selected = sorted(selected, key=lambda r: str(r.get("prompt_id")))
    if limit is not None:
        selected = selected[:limit]
    expected = EXPECTED_RESEARCH_ROWS[research_split] if limit is None else min(limit, EXPECTED_RESEARCH_ROWS[research_split])
    if len(selected) != expected:
        raise Program02Error(f"Expected {expected} {research_split} registry rows, found {len(selected)}.")

    examples: list[dict[str, Any]] = []
    for r in selected:
        if r.get("original_split") != "train":
            raise Program02Error("No-test-leakage failure: Program 02 encountered an upstream test row.")
        idx = r.get("source_row_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= len(raw_train):
            raise Program02Error("Split registry contains invalid source_row_index.")
        raw = raw_train[idx]
        q, a = raw["question"], raw["answer"]
        if sha256_text(q) != r.get("question_hash"):
            raise Program02Error(f"Question hash mismatch for prompt_id={r.get('prompt_id')}.")
        if sha256_text(a) != r.get("gold_answer_hash"):
            raise Program02Error(f"Gold answer hash mismatch for prompt_id={r.get('prompt_id')}.")
        gold = extract_gsm8k_gold(a)
        if gold is None:
            raise Program02Error(f"Cannot parse GSM8K gold answer for prompt_id={r.get('prompt_id')}.")
        content = PROMPT_TEMPLATE.format(question=q)
        examples.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "gold_answer": canonical_fraction(gold),
                "prompt_id": str(r["prompt_id"]),
            }
        )
    return examples


def build_hf_dataset(examples: Sequence[Mapping[str, Any]]) -> Any:
    try:
        from datasets import Dataset  # type: ignore

        return Dataset.from_list([dict(x) for x in examples])
    except Exception as exc:
        raise Program02Error(f"Cannot construct local training Dataset: {exc}") from exc


# ---------------------------------------------------------------------------
# Device, precision, TRL API preflight
# ---------------------------------------------------------------------------


def resolve_precision(requested: str, device: str) -> tuple[str, dict[str, bool]]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program02Error(f"Cannot import torch: {exc}") from exc
    support = {
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    if device == "cuda" and not torch.cuda.is_available():
        raise Program02Error("--device cuda requested, but torch.cuda.is_available() is False.")
    if device == "cpu":
        if requested in {"bf16", "fp16"}:
            raise Program02Error(f"precision={requested} is not allowed in this CPU training path.")
        return "fp32", support
    if requested == "auto":
        return ("bf16" if support["bf16_supported"] else "fp16"), support
    if requested == "bf16" and not support["bf16_supported"]:
        raise Program02Error("precision=bf16 requested but this CUDA device/PyTorch build reports no bf16 support.")
    return requested, support


def validate_trl_api() -> dict[str, Any]:
    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except Exception as exc:
        raise Program02Error(f"Cannot import TRL GRPOConfig/GRPOTrainer: {exc}") from exc

    cfg_sig = inspect.signature(GRPOConfig)
    trainer_sig = inspect.signature(GRPOTrainer)
    cfg_params = set(cfg_sig.parameters)
    trainer_params = set(trainer_sig.parameters)
    required_cfg = {
        "num_generations",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "beta",
        "epsilon",
        "scale_rewards",
        "loss_type",
        "reward_weights",
        "use_vllm",
    }
    required_trainer = {"model", "args", "train_dataset", "reward_funcs", "peft_config"}
    missing_cfg = sorted(required_cfg - cfg_params)
    missing_trainer = sorted(required_trainer - trainer_params)
    if missing_cfg or missing_trainer:
        raise Program02Error(
            "Frozen TRL API is incompatible with the Program 02 contract. "
            f"Missing GRPOConfig fields={missing_cfg}, GRPOTrainer fields={missing_trainer}."
        )
    return {
        "trl_version": package_version("trl"),
        "grpo_config_fields_checked": sorted(required_cfg),
        "grpo_trainer_fields_checked": sorted(required_trainer),
        "processing_class_supported": "processing_class" in trainer_params,
    }


def batch_divisibility_check(spec: TrainingSpec, world_size: int) -> int:
    effective = world_size * spec.per_device_train_batch_size * spec.gradient_accumulation_steps
    if effective % spec.num_generations != 0:
        raise Program02Error(
            "TRL GRPO batch divisibility failure: "
            f"world_size({world_size}) * per_device_train_batch_size({spec.per_device_train_batch_size}) * "
            f"gradient_accumulation_steps({spec.gradient_accumulation_steps}) = {effective}, "
            f"which is not divisible by num_generations={spec.num_generations}."
        )
    return effective


# ---------------------------------------------------------------------------
# Metrics and checkpoint publication
# ---------------------------------------------------------------------------


def normalize_metric_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def pick_metric(logs: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for name in aliases:
        if name in logs:
            return normalize_metric_value(logs[name])
    return None


def read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception as exc:
        raise Program02Error(f"Cannot read metrics CSV {path}: {exc}") from exc


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in columns})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def merge_global_t01(metrics_dir: Path, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("seed_*_metrics.csv")):
        rows.extend(read_metrics_csv(p))
    def key(r: Mapping[str, Any]) -> tuple[int, int]:
        try:
            return int(r.get("training_seed", 0)), int(float(r.get("step", 0)))
        except Exception:
            return (0, 0)
    rows.sort(key=key)
    atomic_write_csv(out_path, rows, T01_COLUMNS)


def save_permanent_adapter(
    *,
    model: Any,
    adapter_dir: Path,
    step: int,
    base_model_revision: str,
    training_config_sha256: str,
    seed: int,
) -> dict[str, Any]:
    final = adapter_dir / f"step_{step:04d}"
    if final.exists():
        manifest_path = final / "adapter_manifest.json"
        if not manifest_path.exists():
            raise Program02Error(f"Existing permanent adapter lacks adapter_manifest.json: {final}")
        manifest = read_json(manifest_path)
        expected_tree = manifest.get("adapter_payload_tree_sha256")
        # Exclude adapter_manifest itself from the payload digest to avoid self-reference.
        observed = adapter_payload_hash(final)
        if expected_tree != observed:
            raise Program02Error(
                f"Existing permanent adapter step {step} has a content-hash mismatch; refusing overwrite."
            )
        if manifest.get("training_seed") != seed or manifest.get("step") != step:
            raise Program02Error(f"Existing permanent adapter metadata mismatch at {final}.")
        return manifest

    adapter_dir.mkdir(parents=True, exist_ok=True)
    stage = adapter_dir / f".step_{step:04d}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=False, exist_ok=False)
        try:
            model.save_pretrained(str(stage), safe_serialization=True)
        except TypeError:
            model.save_pretrained(str(stage))
        if not (stage / "adapter_config.json").exists():
            raise Program02Error(
                "Permanent save did not produce adapter_config.json. The trainer model does not appear to be a PEFT model."
            )
        if not any((stage / x).exists() for x in ("adapter_model.safetensors", "adapter_model.bin")):
            raise Program02Error("Permanent LoRA save did not produce adapter weights.")
        payload_sha = adapter_payload_hash(stage)
        manifest = {
            "schema_version": TRAINING_MANIFEST_SCHEMA,
            "manifest_type": "permanent_lora_adapter",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            "training_seed": seed,
            "step": step,
            "base_model_repo": MODEL_REPO,
            "base_model_revision": base_model_revision,
            "training_config_sha256": training_config_sha256,
            "adapter_payload_tree_sha256": payload_sha,
        }
        atomic_write_json_once(stage / "adapter_manifest.json", manifest)
        if final.exists():
            raise Program02Error(f"Permanent adapter appeared concurrently: {final}")
        os.replace(stage, final)
        fsync_directory(adapter_dir)
        make_read_only_tree(final)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def adapter_payload_hash(root: Path) -> str:
    records: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name == "adapter_manifest.json":
            continue
        records.append(
            {
                "path": p.relative_to(root).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    records.sort(key=lambda x: x["path"])
    if not records:
        raise Program02Error(f"Adapter directory has no payload files: {root}")
    return sha256_bytes(canonical_bytes(records))


def copytree_with_fsync(src: Path, dst: Path) -> None:
    if dst.exists():
        raise Program02Error(f"Destination already exists during checkpoint copy: {dst}")
    shutil.copytree(src, dst)
    for p in dst.rglob("*"):
        if p.is_file():
            try:
                with p.open("rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass
    fsync_directory(dst)


def publish_resume_snapshot(src_checkpoint: Path, resume_latest: Path) -> dict[str, Any]:
    """Crash-resilient rolling update; at most one official resume_latest is retained."""
    if not src_checkpoint.is_dir():
        raise Program02Error(f"Trainer checkpoint directory is missing: {src_checkpoint}")
    parent = resume_latest.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".resume_next-{uuid.uuid4().hex}"
    backup = parent / f".resume_prev-{uuid.uuid4().hex}"
    try:
        copytree_with_fsync(src_checkpoint, stage)
        # Hash before publication, then add a tiny integrity sidecar into stage.
        snapshot_payload_sha = directory_digest_fast(stage)
        atomic_write_json_once(
            stage / "program02_resume_manifest.json",
            {
                "schema_version": CHECKPOINT_INDEX_SCHEMA,
                "manifest_type": "rolling_trainer_resume",
                "created_at_utc": now_utc(),
                "source_checkpoint_name": src_checkpoint.name,
                "payload_tree_sha256_before_sidecar": snapshot_payload_sha,
            },
        )
        if resume_latest.exists():
            os.replace(resume_latest, backup)
        os.replace(stage, resume_latest)
        fsync_directory(parent)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        # Remove the Trainer's numbered duplicate so only resume_latest remains.
        if src_checkpoint.exists():
            shutil.rmtree(src_checkpoint, ignore_errors=True)
        return {
            "path": str(resume_latest),
            "source_checkpoint_name": src_checkpoint.name,
            "tree_sha256": directory_digest_fast(resume_latest),
        }
    except Exception:
        # Recovery preference: keep the previously published latest if possible.
        if not resume_latest.exists() and backup.exists():
            try:
                os.replace(backup, resume_latest)
            except OSError:
                pass
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def recover_resume_artifacts(seed_root: Path) -> None:
    latest = seed_root / "resume_latest"
    nexts = sorted(seed_root.glob(".resume_next-*"))
    prevs = sorted(seed_root.glob(".resume_prev-*"))
    if latest.exists():
        for p in nexts + prevs:
            shutil.rmtree(p, ignore_errors=True)
        return
    # If a crash happened between moving old->prev and next->latest, prefer a complete next.
    candidates = nexts or prevs
    if candidates:
        chosen = candidates[-1]
        os.replace(chosen, latest)
        for p in candidates[:-1] + [p for p in nexts + prevs if p != chosen]:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)


def update_checkpoint_index(
    path: Path,
    *,
    seed: int,
    training_config_sha256: str,
    adapter_manifest: Mapping[str, Any] | None = None,
    resume_info: Mapping[str, Any] | None = None,
) -> None:
    if path.exists():
        idx = read_json(path)
    else:
        idx = {
            "schema_version": CHECKPOINT_INDEX_SCHEMA,
            "manifest_type": "training_checkpoint_index",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            "training_seed": seed,
            "training_config_sha256": training_config_sha256,
            "permanent_adapters": {},
            "resume_latest": None,
        }
    if idx.get("training_seed") != seed or idx.get("training_config_sha256") != training_config_sha256:
        raise Program02Error("Existing checkpoint index belongs to a different seed/configuration.")
    adapters = idx.setdefault("permanent_adapters", {})
    if not isinstance(adapters, dict):
        raise Program02Error("checkpoint_index permanent_adapters is corrupt.")
    if adapter_manifest is not None:
        step = str(adapter_manifest.get("step"))
        new_record = {
            "step": adapter_manifest.get("step"),
            "payload_sha256": adapter_manifest.get("adapter_payload_tree_sha256"),
        }
        if step in adapters and adapters[step] != new_record:
            raise Program02Error(f"Checkpoint index already has conflicting adapter metadata for step {step}.")
        adapters[step] = new_record
    if resume_info is not None:
        idx["resume_latest"] = dict(resume_info)
    idx["updated_at_utc"] = now_utc()
    atomic_write_json(path, idx)


class MetricsStore:
    def __init__(
        self,
        *,
        seed: int,
        mode: str,
        spec: TrainingSpec,
        per_seed_path: Path,
        global_metrics_dir: Path,
        global_t01_path: Path,
    ) -> None:
        self.seed = seed
        self.mode = mode
        self.spec = spec
        self.per_seed_path = per_seed_path
        self.global_metrics_dir = global_metrics_dir
        self.global_t01_path = global_t01_path
        self.rows_by_step: dict[int, dict[str, Any]] = {}
        for row in read_metrics_csv(per_seed_path):
            try:
                self.rows_by_step[int(float(row["step"]))] = row
            except Exception:
                continue
        self.last_wall = time.time()
        self.last_num_tokens: float | None = None

    def record(self, step: int, logs: Mapping[str, Any]) -> None:
        now = time.time()
        num_tokens = pick_metric(logs, ("num_tokens", "train_num_tokens"))
        if num_tokens is not None:
            num_tokens = float(num_tokens)
        tokens_per_second = pick_metric(logs, ("tokens_per_second", "train_tokens_per_second"))
        if tokens_per_second is None and num_tokens is not None and self.last_num_tokens is not None:
            dt = max(now - self.last_wall, 1e-9)
            tokens_per_second = max(0.0, num_tokens - self.last_num_tokens) / dt
        self.last_wall = now
        if num_tokens is not None:
            self.last_num_tokens = num_tokens

        gpu_peak: int | None = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                gpu_peak = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            gpu_peak = None

        row = {
            "training_seed": self.seed,
            "step": step,
            "loss": pick_metric(logs, ("loss",)),
            "learning_rate": pick_metric(logs, ("learning_rate",)),
            "mean_reward": pick_metric(logs, ("reward", "rewards/mean")),
            "correctness_reward": pick_metric(
                logs,
                (
                    "reward/correctness_reward/mean",
                    "reward/correctness_reward_fn/mean",
                    "rewards/correctness_reward/mean",
                    "rewards/correctness_reward_fn/mean",
                    "correctness_reward",
                ),
            ),
            "format_reward": pick_metric(
                logs,
                (
                    "reward/format_reward/mean", "reward/format_reward_fn/mean",
                    "rewards/format_reward/mean", "rewards/format_reward_fn/mean",
                    "format_reward",
                ),
            ),
            "reward_std": pick_metric(logs, ("reward_std",)),
            "mean_completion_length": pick_metric(logs, ("completions/mean_length", "completion_length")),
            "truncation_rate": pick_metric(logs, ("completions/clipped_ratio", "truncation_rate")),
            "kl_proxy": pick_metric(logs, ("kl",)),
            "clip_fraction": pick_metric(logs, ("clip_ratio/region_mean", "clip_ratio")),
            "reward_zero_std_fraction": pick_metric(logs, ("frac_reward_zero_std",)),
            "policy_loss": pick_metric(logs, ("policy_loss",)),
            "grad_norm": pick_metric(logs, ("grad_norm",)),
            "gpu_memory_peak_bytes": gpu_peak,
            "tokens_per_second": tokens_per_second,
            "num_tokens": num_tokens,
            "mode": self.mode,
            "loss_type": self.spec.loss_type,
            "num_generations": self.spec.num_generations,
            "max_completion_length": self.spec.max_completion_length,
        }
        # Ignore pure Trainer housekeeping log events without any research metric.
        research_values = [
            row["loss"],
            row["mean_reward"],
            row["mean_completion_length"],
            row["correctness_reward"],
        ]
        if all(v is None for v in research_values):
            return
        self.rows_by_step[step] = row
        rows = [self.rows_by_step[k] for k in sorted(self.rows_by_step)]
        atomic_write_csv(self.per_seed_path, rows, T01_COLUMNS)
        merge_global_t01(self.global_metrics_dir, self.global_t01_path)


# ---------------------------------------------------------------------------
# Runtime trainer construction
# ---------------------------------------------------------------------------


def mode_spec(spec: TrainingSpec, mode: str) -> tuple[TrainingSpec, str, int | None]:
    if mode == "paper":
        return spec, "training", None
    if mode == "pilot":
        pilot = TrainingSpec(
            **{
                **asdict(spec),
                "seeds": (spec.seeds[0],),
                "max_steps": min(spec.max_steps, 50),
                "save_steps": min(spec.save_steps, 20),
            }
        )
        # Ensure pilot max_steps is divisible by its save cadence.
        if pilot.max_steps % pilot.save_steps != 0:
            adjusted = max(pilot.save_steps, (pilot.max_steps // pilot.save_steps) * pilot.save_steps)
            pilot = TrainingSpec(**{**asdict(pilot), "max_steps": adjusted})
        return pilot, "development", 50
    smoke_steps = min(2, spec.max_steps)
    smoke = TrainingSpec(
        **{
            **asdict(spec),
            "seeds": (spec.seeds[0],),
            "max_steps": smoke_steps,
            "save_steps": smoke_steps,
            "logging_steps": 1,
            "num_generations": min(spec.num_generations, 4),
            "gradient_accumulation_steps": max(spec.gradient_accumulation_steps, min(spec.num_generations, 4)),
        }
    )
    return smoke, "development", 8


def build_grpo_objects(
    *,
    model_dir: Path,
    dataset: Any,
    seed: int,
    spec: TrainingSpec,
    device: str,
    work_dir: Path,
    resolved_precision: str,
    callbacks: Sequence[Any],
) -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
        from peft import LoraConfig  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except Exception as exc:
        raise Program02Error(f"Cannot import training stack: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise Program02Error("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    # TRL GRPOTrainer requires the processing tokenizer to left-pad.
    tokenizer.padding_side = "left"
    if tokenizer.padding_side != "left" or tokenizer.pad_token_id is None:
        raise Program02Error("GRPO processing tokenizer must have padding_side='left' and a pad token.")
    # Do not modify tokenizer.chat_template: Program 00 pinned it.

    if resolved_precision == "bf16":
        dtype = torch.bfloat16
    elif resolved_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
    except TypeError:
        # Transformers 4.x spelling.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = not spec.gradient_checkpointing

    lora_targets: Any = spec.lora_target_modules
    if isinstance(lora_targets, tuple):
        lora_targets = list(lora_targets)
    peft_config = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias=spec.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    bf16 = resolved_precision == "bf16"
    fp16 = resolved_precision == "fp16"
    work_dir.mkdir(parents=True, exist_ok=True)

    args_kwargs: dict[str, Any] = {
        "output_dir": str(work_dir),
        "learning_rate": spec.learning_rate,
        "per_device_train_batch_size": spec.per_device_train_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "max_steps": spec.max_steps,
        "max_grad_norm": spec.max_grad_norm,
        "weight_decay": spec.weight_decay,
        "lr_scheduler_type": spec.lr_scheduler_type,
        "warmup_ratio": spec.warmup_ratio,
        "num_generations": spec.num_generations,
        "max_completion_length": spec.max_completion_length,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "repetition_penalty": spec.repetition_penalty,
        "beta": spec.beta,
        "epsilon": spec.epsilon,
        "scale_rewards": spec.scale_rewards,
        "reward_weights": [1.0, spec.format_reward_weight],
        "loss_type": spec.loss_type,
        "num_iterations": spec.num_iterations,
        "use_vllm": False,
        "gradient_checkpointing": spec.gradient_checkpointing,
        "bf16": bf16,
        "fp16": fp16,
        "use_cpu": device == "cpu",
        "logging_strategy": "steps",
        "logging_steps": spec.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": spec.save_steps,
        "save_total_limit": 1,
        "save_only_model": False,
        "eval_strategy": "no",
        "report_to": "none",
        "push_to_hub": False,
        "remove_unused_columns": False,
        "shuffle_dataset": spec.shuffle_dataset,
        "seed": seed,
        "data_seed": seed,
        "full_determinism": False,
        "restore_callback_states_from_checkpoint": True,
        "log_completions": False,
    }

    # Handle frozen TRL versions where inherited TrainingArguments fields differ.
    sig = inspect.signature(GRPOConfig)
    supported = set(sig.parameters)
    filtered = {k: v for k, v in args_kwargs.items() if k in supported}
    essential = {
        "output_dir",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "num_generations",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "beta",
        "epsilon",
        "scale_rewards",
        "loss_type",
        "reward_weights",
        "use_vllm",
        "seed",
        "data_seed",
    }
    missing = sorted(essential - supported)
    if missing:
        raise Program02Error(f"Frozen TRL GRPOConfig lacks required fields: {missing}")
    training_args = GRPOConfig(**filtered)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "reward_funcs": [correctness_reward, format_reward],
        "peft_config": peft_config,
        "callbacks": list(callbacks),
    }
    trainer_sig = inspect.signature(GRPOTrainer)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise Program02Error("Frozen GRPOTrainer supports neither processing_class nor tokenizer.")

    trainer = GRPOTrainer(**trainer_kwargs)
    # reward_weights was passed into GRPOConfig before trainer construction so
    # TRL cannot cache an unweighted reward vector during __init__.
    if not hasattr(trainer.args, "reward_weights"):
        raise Program02Error("Frozen TRL GRPOConfig has no reward_weights field; cannot enforce reward semantics.")
    return trainer, tokenizer


def make_callbacks(
    *,
    seed: int,
    mode: str,
    spec: TrainingSpec,
    seed_root: Path,
    adapter_dir: Path,
    work_dir: Path,
    checkpoint_index_path: Path,
    metrics_store: MetricsStore,
    model_revision: str,
    training_config_sha256: str,
) -> list[Any]:
    try:
        from transformers import TrainerCallback  # type: ignore
    except Exception as exc:
        raise Program02Error(f"Cannot import TrainerCallback: {exc}") from exc

    class ResearchCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
            if bool(getattr(state, "is_world_process_zero", True)) and logs:
                metrics_store.record(int(state.global_step), logs)
            return control

        def on_save(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> Any:
            if not bool(getattr(state, "is_world_process_zero", True)):
                return control
            step = int(state.global_step)
            if step <= 0 or step % spec.save_steps != 0:
                return control
            if model is None:
                raise Program02Error("Trainer callback did not receive model during on_save.")
            adapter_manifest = save_permanent_adapter(
                model=model,
                adapter_dir=adapter_dir,
                step=step,
                base_model_revision=model_revision,
                training_config_sha256=training_config_sha256,
                seed=seed,
            )
            update_checkpoint_index(
                checkpoint_index_path,
                seed=seed,
                training_config_sha256=training_config_sha256,
                adapter_manifest=adapter_manifest,
            )

            checkpoint_dir = work_dir / f"checkpoint-{step}"
            if not checkpoint_dir.exists():
                # Some Transformers versions use PREFIX_CHECKPOINT_DIR, but the canonical name is checkpoint-N.
                candidates = sorted(work_dir.glob(f"checkpoint-*"), key=lambda p: p.stat().st_mtime)
                if not candidates:
                    raise Program02Error(f"Trainer on_save fired at step {step}, but no full checkpoint directory exists.")
                checkpoint_dir = candidates[-1]
            resume_info = publish_resume_snapshot(checkpoint_dir, seed_root / "resume_latest")
            update_checkpoint_index(
                checkpoint_index_path,
                seed=seed,
                training_config_sha256=training_config_sha256,
                resume_info=resume_info,
            )
            return control

    return [ResearchCallback()]


# ---------------------------------------------------------------------------
# Seed lifecycle and manifests
# ---------------------------------------------------------------------------


def seed_final_manifest_path(manifests_dir: Path, mode: str, seed: int) -> Path:
    if mode == "paper":
        return manifests_dir / "training" / f"seed_{seed}.json"
    return manifests_dir / f"_{mode}" / "training" / f"seed_{seed}.json"


def verify_final_seed_manifest(path: Path, seed_root: Path, spec: TrainingSpec, seed: int) -> dict[str, Any]:
    m = read_json(path)
    if m.get("schema_version") != TRAINING_MANIFEST_SCHEMA or m.get("manifest_type") != "grpo_training_seed":
        raise Program02Error(f"Invalid final seed manifest: {path}")
    if m.get("training_seed") != seed or m.get("training_config_sha256") != spec_fingerprint(spec):
        raise Program02Error(f"Final seed manifest does not match requested seed/config: {path}")
    if m.get("status") != "complete" or m.get("final_step") != spec.max_steps:
        raise Program02Error(f"Final seed manifest is not complete: {path}")
    adapters = m.get("permanent_adapters")
    if not isinstance(adapters, list):
        raise Program02Error("Final seed manifest lacks permanent_adapters.")
    expected_steps = list(range(0, spec.max_steps + 1, spec.save_steps))
    observed_steps = [int(x.get("step")) for x in adapters if isinstance(x, dict) and "step" in x]
    if observed_steps != expected_steps:
        raise Program02Error(f"Final adapter grid mismatch for seed {seed}: {observed_steps}")
    for rec in adapters:
        step = int(rec["step"])
        path_adapter = seed_root / "adapters" / f"step_{step:04d}"
        if not path_adapter.exists():
            raise Program02Error(f"Permanent adapter missing: {path_adapter}")
        if adapter_payload_hash(path_adapter) != rec.get("payload_sha256"):
            raise Program02Error(f"Permanent adapter hash mismatch at step {step}, seed {seed}.")
    return m


def build_final_seed_manifest(
    *,
    root: Path,
    mode: str,
    seed: int,
    spec: TrainingSpec,
    seed_root: Path,
    metrics_path: Path,
    model_record: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    protocol_config_sha256: str,
    protocol_lock: Mapping[str, Any] | None,
    resolved_precision: str,
    dataset_size: int,
) -> dict[str, Any]:
    adapters: list[dict[str, Any]] = []
    for step in range(0, spec.max_steps + 1, spec.save_steps):
        p = seed_root / "adapters" / f"step_{step:04d}"
        if not p.exists():
            raise Program02Error(f"Cannot finalize seed {seed}: missing permanent adapter step {step}.")
        adapters.append(
            {
                "step": step,
                "local_path": rel(p, root),
                "payload_sha256": adapter_payload_hash(p),
            }
        )
    if not metrics_path.exists():
        raise Program02Error(f"Cannot finalize seed {seed}: missing metrics file {metrics_path}.")
    resume_latest = seed_root / "resume_latest"
    if not resume_latest.exists():
        raise Program02Error(f"Cannot finalize seed {seed}: missing resume_latest snapshot.")

    inputs = {
        "dataset_repo": GSM8K_REPO,
        "dataset_revision": (split_manifest.get("source") or {}).get("dataset_revision"),
        "data_manifest_content_fingerprint_sha256": data_manifest.get("content_fingerprint_sha256"),
        "split_registry_content_fingerprint_sha256": split_manifest.get("content_fingerprint_sha256"),
        "model_repo": MODEL_REPO,
        "model_revision": model_record.get("resolved_revision"),
        "model_tree_sha256": model_record.get("tree_sha256"),
        "chat_template_sha256": ((model_record.get("validation") or {}).get("tokenizer") or {}).get(
            "chat_template_sha256"
        ),
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "protocol_config_sha256": protocol_config_sha256,
        "protocol_lock_present": protocol_lock is not None,
    }
    config_dict = asdict(spec)
    fingerprint_basis = {
        "training_seed": seed,
        "training_config_sha256": spec_fingerprint(spec),
        "inputs": inputs,
        "permanent_adapters": adapters,
        "metrics_sha256": sha256_file(metrics_path),
    }
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA,
        "manifest_type": "grpo_training_seed",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "status": "complete",
        "mode": mode,
        "training_seed": seed,
        "final_step": spec.max_steps,
        "dataset_size": dataset_size,
        "research_purpose": "produce a dense reproducible GRPO policy-drift path for downstream OPE reuse experiments",
        "training_config": config_dict,
        "training_config_sha256": spec_fingerprint(spec),
        "resolved_precision": resolved_precision,
        "reward_definition": {
            "training_reward": "correctness + format_reward_weight * format",
            "format_reward_weight": spec.format_reward_weight,
            "final_ope_reward_in_later_programs": "correctness_only",
        },
        "inputs": inputs,
        "permanent_adapters": adapters,
        "resume_latest": {
            "local_path": rel(resume_latest, root),
            "tree_sha256": directory_digest_fast(resume_latest),
        },
        "training_metrics": {
            "local_path": rel(metrics_path, root),
            "sha256": sha256_file(metrics_path),
        },
        "content_fingerprint_sha256": sha256_bytes(canonical_bytes(fingerprint_basis)),
    }


def expected_adapter_steps(spec: TrainingSpec) -> list[int]:
    return list(range(0, spec.max_steps + 1, spec.save_steps))


def ensure_step0_adapter(
    trainer: Any,
    *,
    adapter_dir: Path,
    seed: int,
    model_revision: str,
    training_config_sha256: str,
    checkpoint_index_path: Path,
) -> None:
    manifest = save_permanent_adapter(
        model=trainer.model,
        adapter_dir=adapter_dir,
        step=0,
        base_model_revision=model_revision,
        training_config_sha256=training_config_sha256,
        seed=seed,
    )
    update_checkpoint_index(
        checkpoint_index_path,
        seed=seed,
        training_config_sha256=training_config_sha256,
        adapter_manifest=manifest,
    )


def run_one_seed(
    *,
    root: Path,
    mode: str,
    seed: int,
    spec: TrainingSpec,
    dataset: Any,
    model_dir: Path,
    model_record: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    protocol_config_sha256: str,
    protocol_lock: Mapping[str, Any] | None,
    resolved_precision: str,
    device: str,
    resume: bool,
    manifests_dir: Path,
    outputs_root: Path,
    checkpoints_root: Path,
) -> None:
    seed_root = checkpoints_root / f"seed_{seed}"
    adapter_dir = seed_root / "adapters"
    work_dir = seed_root / "_trainer_work"
    checkpoint_index_path = seed_root / "checkpoint_index.json"
    metrics_dir = outputs_root / "diagnostics" / "training"
    metrics_path = metrics_dir / f"seed_{seed}_metrics.csv"
    global_t01 = outputs_root / "tables" / "T01_training_trajectory.csv"
    final_manifest = seed_final_manifest_path(manifests_dir, mode, seed)
    config_sha = spec_fingerprint(spec)

    if final_manifest.exists():
        verify_final_seed_manifest(final_manifest, seed_root, spec, seed)
        print(f"[SKIP] seed={seed} already complete and verified: {final_manifest}")
        merge_global_t01(metrics_dir, global_t01)
        return

    recover_resume_artifacts(seed_root)
    resume_latest = seed_root / "resume_latest"
    if seed_root.exists() and any(seed_root.iterdir()) and not resume and not adapter_dir.joinpath("step_0000").exists():
        raise Program02Error(
            f"Seed directory already contains partial state: {seed_root}. Re-run with --resume or reset explicitly."
        )
    if resume_latest.exists() and not resume:
        raise Program02Error(
            f"A resumable checkpoint already exists for seed {seed}: {resume_latest}. Use --resume."
        )

    seed_root.mkdir(parents=True, exist_ok=True)
    metrics_store = MetricsStore(
        seed=seed,
        mode=mode,
        spec=spec,
        per_seed_path=metrics_path,
        global_metrics_dir=metrics_dir,
        global_t01_path=global_t01,
    )
    callbacks = make_callbacks(
        seed=seed,
        mode=mode,
        spec=spec,
        seed_root=seed_root,
        adapter_dir=adapter_dir,
        work_dir=work_dir,
        checkpoint_index_path=checkpoint_index_path,
        metrics_store=metrics_store,
        model_revision=str(model_record.get("resolved_revision")),
        training_config_sha256=config_sha,
    )

    trainer, _tokenizer = build_grpo_objects(
        model_dir=model_dir,
        dataset=dataset,
        seed=seed,
        spec=spec,
        device=device,
        work_dir=work_dir,
        resolved_precision=resolved_precision,
        callbacks=callbacks,
    )
    ensure_step0_adapter(
        trainer,
        adapter_dir=adapter_dir,
        seed=seed,
        model_revision=str(model_record.get("resolved_revision")),
        training_config_sha256=config_sha,
        checkpoint_index_path=checkpoint_index_path,
    )

    resume_from = str(resume_latest) if resume and resume_latest.exists() else None
    print(f"[TRAIN] seed={seed} resume_from={resume_from or 'none'}")
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user. Completed checkpoints remain resumable.")
        raise

    # Ensure the final adapter exists even if a frozen Trainer version skipped a final save event.
    final_step = int(getattr(trainer.state, "global_step", -1))
    if final_step != spec.max_steps:
        raise Program02Error(
            f"Seed {seed} ended at global_step={final_step}; expected {spec.max_steps}. Do not mark this run complete."
        )
    final_adapter = save_permanent_adapter(
        model=trainer.model,
        adapter_dir=adapter_dir,
        step=spec.max_steps,
        base_model_revision=str(model_record.get("resolved_revision")),
        training_config_sha256=config_sha,
        seed=seed,
    )
    update_checkpoint_index(
        checkpoint_index_path,
        seed=seed,
        training_config_sha256=config_sha,
        adapter_manifest=final_adapter,
    )

    missing = [s for s in expected_adapter_steps(spec) if not (adapter_dir / f"step_{s:04d}").exists()]
    if missing:
        raise Program02Error(f"Seed {seed} is missing permanent adapter steps: {missing}")

    merge_global_t01(metrics_dir, global_t01)
    manifest = build_final_seed_manifest(
        root=root,
        mode=mode,
        seed=seed,
        spec=spec,
        seed_root=seed_root,
        metrics_path=metrics_path,
        model_record=model_record,
        data_manifest=data_manifest,
        split_manifest=split_manifest,
        protocol_config_sha256=protocol_config_sha256,
        protocol_lock=protocol_lock,
        resolved_precision=resolved_precision,
        dataset_size=len(dataset),
    )
    atomic_write_json_once(final_manifest, manifest)
    print(f"[DONE] seed={seed} final manifest: {final_manifest}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO-OPE Program 02: train dense multi-seed GRPO policy paths with resumable LoRA checkpoints."
    )
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--split", choices=("development", "test"), default="development", help="Compatibility CLI field. Program 02 paper training never dereferences test; passing --split test is rejected.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-root", default=".")
    p.add_argument("--seed", type=int, default=None, help="Run/repair one configured seed only; default runs all configured seeds.")
    p.add_argument("--reset-checkpoints", action="store_true", help="Dangerous: remove Program 02 state for the selected mode. Requires GRPO_OPE_ALLOW_RESET=YES.")
    return p.parse_args(argv)


def safe_reset(path_list: Sequence[Path]) -> None:
    if os.environ.get("GRPO_OPE_ALLOW_RESET") != "YES":
        raise Program02Error(
            "--reset-checkpoints requires environment variable GRPO_OPE_ALLOW_RESET=YES. "
            "This prevents accidental deletion of expensive GPU assets."
        )
    for p in path_list:
        if p.exists():
            print(f"[RESET] removing {p}")
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def print_header(
    *,
    root: Path,
    mode: str,
    config_path: Path,
    config_sha: str,
    spec: TrainingSpec,
    resolved_precision: str,
    data_record: Mapping[str, Any],
    model_record: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    seeds: Sequence[int],
    dataset_split: str,
    dataset_size: int,
    checkpoint_root: Path,
    outputs_root: Path,
) -> None:
    print("=" * 78)
    print("GRPO-OPE Program 02 — multi-seed GRPO training")
    print(f"program version       : {PROGRAM_VERSION}")
    print(f"project root          : {root}")
    print(f"mode                  : {mode}")
    print(f"config                : {config_path}")
    print(f"config SHA-256        : {config_sha}")
    print(f"training config hash  : {spec_fingerprint(spec)}")
    print(f"model revision        : {model_record.get('resolved_revision')}")
    print(f"data revision         : {data_record.get('resolved_revision')}")
    print(f"split registry hash   : {split_manifest.get('content_fingerprint_sha256')}")
    print(f"training seeds        : {list(seeds)}")
    print(f"input split           : {dataset_split} ({dataset_size} prompts)")
    print(f"max steps/save        : {spec.max_steps}/{spec.save_steps}")
    print(f"G / max completion    : {spec.num_generations}/{spec.max_completion_length}")
    print(f"sampling              : T={spec.temperature}, top_p={spec.top_p}, top_k={spec.top_k}, rep={spec.repetition_penalty}")
    print(f"loss / beta / eps     : {spec.loss_type} / {spec.beta} / {spec.epsilon}")
    print(f"precision             : {resolved_precision}")
    print(f"checkpoint root       : {checkpoint_root}")
    print(f"outputs root          : {outputs_root}")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()

    manifests_dir = root / "manifests"
    gsm8k_dir = root / "data" / "raw" / "gsm8k"
    model_dir = root / "models" / "qwen25_05b"
    registry_path = root / "data" / "splits" / "gsm8k_split_registry.parquet"

    env_manifest_path = manifests_dir / "environment_manifest.json"
    data_manifest_path = manifests_dir / "data_manifest.json"
    model_manifest_path = manifests_dir / "model_manifest.json"
    split_manifest_path = manifests_dir / "split_registry_manifest.json"
    protocol_lock_path = manifests_dir / "protocol_lock.json"

    # Formal and engineering assets never share directories.
    if args.mode == "paper":
        checkpoints_root = root / "checkpoints"
        outputs_root = root / "outputs"
    else:
        checkpoints_root = root / "checkpoints" / f"_{args.mode}" / "program02"
        outputs_root = root / "outputs" / f"_{args.mode}" / "program02"

    if args.reset_checkpoints:
        safe_reset(
            [
                checkpoints_root,
                outputs_root,
                manifests_dir / ("training" if args.mode == "paper" else f"_{args.mode}/training"),
            ]
        )

    # No official-test path is legal in Program 02.
    if args.split == "test":
        raise Program02Error(
            "No-test-leakage: Program 02 never trains on or reads official-test rewards. "
            "--split test is invalid for this program."
        )

    verify_environment_manifest(env_manifest_path)
    data_manifest, gsm_record = verify_data_manifest(data_manifest_path, gsm8k_dir)
    model_manifest, model_record = verify_model_manifest(model_manifest_path, model_dir)
    split_manifest, registry_rows = verify_split_registry(
        split_manifest_path, registry_path, data_manifest, gsm_record
    )
    cfg, config_sha = load_protocol(config_path)
    base_spec = parse_training_spec(cfg)
    if args.mode == "paper":
        validate_paper_contract(base_spec)
    spec, research_split, limit = mode_spec(base_spec, args.mode)

    protocol_lock: dict[str, Any] | None = None
    if args.mode == "paper":
        protocol_lock = verify_protocol_lock(
            protocol_lock_path,
            config_sha256=config_sha,
            data_manifest=data_manifest,
            model_manifest=model_manifest,
            split_manifest=split_manifest,
            spec=base_spec,
        )

    validate_trl_api()
    resolved_precision, precision_support = resolve_precision(spec.precision, args.device)

    # One-process-per-GPU is the supported primary path.  accelerate launch can
    # still set WORLD_SIZE; the divisibility check then uses it explicitly.
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise Program02Error("WORLD_SIZE environment variable is not an integer.") from exc
    if world_size <= 0:
        raise Program02Error("WORLD_SIZE must be positive.")
    effective_batch = batch_divisibility_check(spec, world_size)

    raw_train = load_upstream_train_rows(gsm8k_dir)
    examples = build_training_examples(
        registry_rows=registry_rows,
        raw_train=raw_train,
        research_split=research_split,
        limit=limit,
    )
    dataset = build_hf_dataset(examples)

    seeds = list(spec.seeds)
    if args.seed is not None:
        if args.seed not in seeds:
            raise Program02Error(f"--seed {args.seed} is not in frozen training.seeds={seeds}.")
        seeds = [args.seed]

    print_header(
        root=root,
        mode=args.mode,
        config_path=config_path,
        config_sha=config_sha,
        spec=spec,
        resolved_precision=resolved_precision,
        data_record=gsm_record,
        model_record=model_record,
        split_manifest=split_manifest,
        seeds=seeds,
        dataset_split=research_split,
        dataset_size=len(dataset),
        checkpoint_root=checkpoints_root,
        outputs_root=outputs_root,
    )
    print(f"effective train batch : {effective_batch} (world_size={world_size})")
    print(f"precision support     : {precision_support}")
    print(f"prompt template hash  : {sha256_text(PROMPT_TEMPLATE)}")
    print(f"git commit            : {git_commit(root) or 'not-a-git-checkout'}")

    for seed in seeds:
        run_one_seed(
            root=root,
            mode=args.mode,
            seed=seed,
            spec=spec,
            dataset=dataset,
            model_dir=model_dir,
            model_record=model_record,
            data_manifest=data_manifest,
            split_manifest=split_manifest,
            protocol_config_sha256=config_sha,
            protocol_lock=protocol_lock,
            resolved_precision=resolved_precision,
            device=args.device,
            resume=args.resume,
            manifests_dir=manifests_dir,
            outputs_root=outputs_root,
            checkpoints_root=checkpoints_root,
        )

    print("\nPROGRAM 02 PASSED")
    print(f"mode                  : {args.mode}")
    print(f"seeds completed       : {seeds}")
    print(f"T01                    : {outputs_root / 'tables' / 'T01_training_trajectory.csv'}")
    if args.mode == "paper":
        print("permanent adapter grid : step 0,20,...,400 for every completed seed")
    else:
        print("engineering assets only: formal paper checkpoints were not touched")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Program02Error as exc:
        print(f"\nPROGRAM 02 FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print("\nPROGRAM 02 FAILED with an unexpected exception:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(3)
