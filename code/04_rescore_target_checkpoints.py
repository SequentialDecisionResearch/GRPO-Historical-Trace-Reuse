#!/usr/bin/env python3
"""
Program 04 -- canonical symmetric FP32 teacher-forcing rescoring.

Research role
-------------
This program is the mathematical bridge between immutable behavior traces
(Program 03) and OPE/overlap analysis (Program 06). For the *exact same*
prompt-token sequence x and completion-token sequence y saved by Program 03,
it evaluates both policies with one canonical probability engine:

    ell_b^TF = log mu_b^TF(y | x)      [behavior checkpoint, teacher-forced]
    ell_e^TF = log pi_e^TF(y | x)      [target checkpoint, teacher-forced]
    log W    = ell_e^TF - ell_b^TF.

Program 03's generation-time selected-token log-probabilities remain immutable
and are still verified as provenance/audit evidence. They are NOT mixed with
teacher-forced target probabilities in the importance ratio. Instead, identity
pairs (b=e) first create the canonical FP32 teacher-forced behavior denominator
once per behavior shard. Every distant fixed/rolling pair then reuses that
verified canonical denominator.

This change was triggered pre-analysis by the identity firewall: two very long
formal development trajectories showed tiny per-token FP32 differences between
Hugging Face generation-time scores and a separate full-sequence forward pass,
whose sums barely exceeded the original sequence tolerance. Exact regeneration
reproduced Program 03 tokens/log-probabilities bit-for-bit, while alternative
forced replay paths retained the same tiny numerical bridge discrepancy. The
scientifically clean correction is therefore to use one deterministic canonical
teacher-forced scoring engine symmetrically on both sides of the ratio, while
recording generation-vs-canonical bridge diagnostics separately.

No string re-tokenization and no target-policy re-sampling is allowed.
The causal-LM shift is explicit: logits at position t-1 score token t.
EOS, when present in the saved completion, is scored. Padding introduced only
for the scoring batch is masked and never scored.

Safety / reproducibility invariants
-----------------------------------
* Program 00 data/model manifests and Program 01 split registry are verified.
* Program 02 permanent adapters are hash-verified before use.
* Program 03 immutable behavior shards are file/content-hash verified.
* A deterministic immutable pair registry prevents a 4 x 21 Cartesian product.
* Paper mode uses only the predeclared pairs; test mode additionally requires
  the frozen gate and an already-existing pair registry.
* All identity pairs are completed first and create canonical behavior scores.
* No non-identity pair may run without its verified identity-denominator shard.
* Rescore IDs are stable hashes of trajectory_id + target_step + adapter hash.
* Every rescore shard is written staging -> fsync -> atomic rename, with a
  content hash; resume skips only units that re-verify successfully.
* All sequence aggregations and log-weights are Python/float64.
* No clipping of log-weights is performed here.
* In paper mode, a source-code change relative to the original protocol lock
  requires manifests/protocol_amendment_program04.json that cryptographically
  binds the old lock, old Program-04 hash and this amended Program-04 hash.

Expected project layout
-----------------------
  grpo_ope_reuse/
    configs/protocol.yaml
    data/behavior_logs/...
    data/target_rescores/...
    checkpoints/seed_<seed>/adapters/step_<step>/...
    manifests/data_manifest.json
    manifests/model_manifest.json
    manifests/split_registry_manifest.json
    manifests/protocol_lock.json
    manifests/protocol_amendment_program04.json   # required after this amendment
    manifests/pair_registry.parquet
    outputs/frozen_gate.json                      # additionally required for test

Typical commands
----------------
Development paper rescoring (after the amendment manifest exists):
  python 04_rescore_target_checkpoints.py       --config configs/protocol.yaml --mode paper --split development       --resume --device cuda --output-root .

Official test (only after frozen gate exists):
  python 04_rescore_target_checkpoints.py       --config configs/protocol.yaml --mode paper --split test       --resume --device cuda --output-root .

This file intentionally does not import Program 03 as a Python module: every
expensive artifact is re-verified from its immutable sidecar instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROGRAM = "04_rescore_target_checkpoints.py"
PROGRAM_VERSION = "1.3"
RESCORE_SCHEMA = "1.0"
PAIR_SCHEMA = "1.0"
BEHAVIOR_SCHEMA = "1.0"
PROBABILITY_EVIDENCE_PRECISION = "fp32"
SCORING_CONTRACT = "symmetric_fp32_teacher_forced_v1"
AMENDMENT_MANIFEST = "protocol_amendment_program04.json"

GSM8K_REPO = "openai/gsm8k"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_NAME = "grpo_ope_reuse"
PAPER_ANCHORS = (0, 100, 200, 300)
PAPER_TARGET_STEPS = tuple(range(0, 401, 20))
PAPER_FIXED_TARGETS = tuple(range(20, 401, 20))
PAPER_ROLLING = {
    100: (120, 140, 160, 180, 200),
    200: (220, 240, 260, 280, 300),
    300: (320, 340, 360, 380, 400),
}
PAPER_SEEDS_DEFAULT = (20260826, 20260827, 20260828)
PAPER_SPLITS = ("development", "test")

# Identity tolerances are deliberately strict enough to catch token shift,
# EOS/mask, wrong-adapter and logits-processing bugs while allowing tiny
# numerical differences between incremental generation and batched full-sequence
# forward passes on mixed-precision GPU kernels.  They are recorded in every
# output manifest.  A frozen protocol.yaml may override them under rescore.*.
DEFAULT_IDENTITY_TOKEN_ATOL = 1.0e-2
DEFAULT_IDENTITY_SEQUENCE_ATOL = 5.0e-2
DEFAULT_SCORING_BATCH_SIZE = 8

HASH_EXCLUDE_DIRS = {".git", ".cache", "__pycache__"}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
CORE_PACKAGES = (
    "torch", "transformers", "trl", "peft", "accelerate", "datasets", "huggingface_hub"
)
EXTRA_PACKAGES = ("tokenizers", "safetensors", "pyarrow", "numpy", "PyYAML")

BEHAVIOR_ROW_COLUMNS = (
    "trajectory_id", "dataset", "dataset_revision", "training_seed", "behavior_step", "split",
    "prompt_id", "source_row_index", "sample_index", "prompt_token_ids", "completion_token_ids",
    "behavior_token_logprobs", "behavior_sequence_logprob", "completion_length", "terminated_with_eos",
    "was_truncated", "parsed_answer", "correctness_reward", "parser_status", "temperature", "top_p",
    "top_k", "repetition_penalty", "max_completion_length", "generation_seed", "generation_call_id",
    "model_revision", "tokenizer_revision", "chat_template_hash", "prompt_template_hash",
    "behavior_adapter_sha256", "generation_config_sha256", "behavior_logprob_source", "completion_text",
)

RESCORE_COLUMNS = (
    "rescore_id",
    "trajectory_id",
    "dataset",
    "dataset_revision",
    "protocol_version",
    "training_seed",
    "split",
    "behavior_step",
    "target_step",
    "purpose",
    "prompt_id",
    "sample_index",
    "behavior_adapter_sha256",
    "target_adapter_sha256",
    "behavior_sequence_logprob",
    "target_sequence_logprob",
    "log_weight",
    "mean_log_ratio_per_token",
    "completion_length",
    "correctness_reward",
    "terminated_with_eos",
    "was_truncated",
    "identity_abs_sequence_logprob_diff",
    "identity_max_abs_token_logprob_diff",
    "identity_mean_abs_token_logprob_diff",
    "identity_pass",
    "scoring_precision",
    "logprob_definition",
    "source_behavior_content_sha256",
)

PAIR_COLUMNS = (
    "pair_id",
    "training_seed",
    "dataset",
    "split",
    "behavior_step",
    "target_step",
    "purpose",
    "protocol_version",
)


class Program04Error(RuntimeError):
    """Controlled Program 04 failure with an actionable message."""


@dataclass(frozen=True)
class RescoreSpec:
    batch_size: int = DEFAULT_SCORING_BATCH_SIZE
    identity_token_atol: float = DEFAULT_IDENTITY_TOKEN_ATOL
    identity_sequence_atol: float = DEFAULT_IDENTITY_SEQUENCE_ATOL


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    training_seed: int
    dataset: str
    split: str
    behavior_step: int
    target_step: int
    purpose: str
    protocol_version: str


@dataclass(frozen=True)
class AdapterRecord:
    step: int
    path: Path
    payload_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class BehaviorShard:
    manifest_path: Path
    parquet_path: Path
    manifest_sha256: str
    content_sha256: str
    training_seed: int
    behavior_step: int
    split: str
    sample_start: int
    sample_end: int
    shard_index: int
    behavior_adapter_sha256: str
    row_count: int


# ---------------------------------------------------------------------------
# Generic deterministic integrity utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


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
    if not path.exists():
        raise Program04Error(f"Missing required JSON file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Program04Error(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Program04Error(f"Expected JSON object in {path}")
    return data


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def payload_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            parts = p.relative_to(root).parts
        except Exception:
            parts = p.parts
        if any(x in HASH_EXCLUDE_DIRS for x in parts):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def tree_hash(root: Path, *, exclude_names: Iterable[str] = ()) -> dict[str, Any]:
    """Program-00-compatible immutable directory hash."""
    excluded = set(exclude_names)
    files: list[dict[str, Any]] = []
    total = 0
    for p in payload_files(root):
        if p.name in excluded:
            continue
        rp = p.relative_to(root).as_posix()
        size = p.stat().st_size
        files.append({"path": rp, "size_bytes": size, "sha256": sha256_file(p)})
        total += size
    if not files:
        raise Program04Error(f"No payload files found in asset directory: {root}")
    return {
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
        "tree_sha256": sha256_bytes(canonical_bytes(files)),
    }


def adapter_payload_hash(root: Path) -> str:
    records: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name == "adapter_manifest.json":
            continue
        records.append({
            "path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    records.sort(key=lambda x: x["path"])
    if not records:
        raise Program04Error(f"Adapter contains no payload files: {root}")
    return sha256_bytes(canonical_bytes(records))


def make_read_only_tree(root: Path) -> None:
    # Best-effort UX guard. Hashes remain the authority.
    if not root.exists():
        return
    for p in sorted(root.rglob("*"), reverse=True):
        try:
            mode = p.stat().st_mode
            if p.is_file():
                p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except OSError:
            pass


def git_commit(root: Path) -> str | None:
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = cp.stdout.strip()
        return value if re.fullmatch(r"[0-9a-fA-F]{40}", value) else None
    except Exception:
        return None


def package_version(name: str) -> str | None:
    aliases = {"torch": "torch", "PyYAML": "PyYAML"}
    try:
        return importlib.metadata.version(aliases.get(name, name))
    except importlib.metadata.PackageNotFoundError:
        return None




# ---------------------------------------------------------------------------
# Protocol parsing / frozen evaluation configuration
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program04Error(f"Missing protocol config {path}.")
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise Program04Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program04Error("protocol.yaml must contain a YAML mapping.")
    project = cfg.get("project") or {}
    if not isinstance(project, Mapping):
        raise Program04Error("protocol.yaml project must be a mapping.")
    if project.get("name") not in (None, PROJECT_NAME):
        raise Program04Error(f"Unexpected project.name={project.get('name')!r}.")
    return cfg, sha256_file(path)


def _dig(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _dig(mapping, path)
        if value is not None:
            return value
    return None


def protocol_version(cfg: Mapping[str, Any]) -> str:
    project = cfg.get("project") or {}
    value = project.get("protocol_version") if isinstance(project, Mapping) else None
    return str(value if value is not None else "1.0")


def configured_seeds(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    training = cfg.get("training") or {}
    raw = training.get("seeds", list(PAPER_SEEDS_DEFAULT)) if isinstance(training, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise Program04Error("training.seeds must be a non-empty list of integers.")
    out: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Program04Error("training.seeds must contain unique non-negative integers.")
        out.append(int(value))
    if len(out) != len(set(out)):
        raise Program04Error("training.seeds contains duplicates.")
    return tuple(out)


def parse_target_steps(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    ope = cfg.get("ope") or {}
    if not isinstance(ope, Mapping):
        raise Program04Error("protocol.yaml ope section must be a mapping.")
    raw = ope.get("target_steps", list(PAPER_TARGET_STEPS))
    if not isinstance(raw, list) or not raw:
        raise Program04Error("ope.target_steps must be a non-empty integer list.")
    out: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Program04Error("ope.target_steps must contain non-negative integers.")
        out.append(int(value))
    if len(out) != len(set(out)) or tuple(sorted(out)) != tuple(out):
        raise Program04Error("ope.target_steps must contain unique strictly increasing checkpoints.")
    return tuple(out)


def parse_rescore_spec(cfg: Mapping[str, Any], batch_override: int | None) -> RescoreSpec:
    section = cfg.get("rescore") or {}
    if not isinstance(section, Mapping):
        raise Program04Error("protocol.yaml rescore section must be a mapping when present.")
    batch = batch_override if batch_override is not None else section.get("batch_size", DEFAULT_SCORING_BATCH_SIZE)
    token_atol = section.get("identity_token_atol", DEFAULT_IDENTITY_TOKEN_ATOL)
    sequence_atol = section.get("identity_sequence_atol", DEFAULT_IDENTITY_SEQUENCE_ATOL)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise Program04Error("rescore.batch_size/--batch-size must be a positive integer.")
    for name, value in (("identity_token_atol", token_atol), ("identity_sequence_atol", sequence_atol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise Program04Error(f"rescore.{name} must be a positive finite number.")
    return RescoreSpec(int(batch), float(token_atol), float(sequence_atol))


def current_environment_fingerprint() -> tuple[str, dict[str, Any]]:
    packages = {p: package_version(p) for p in CORE_PACKAGES + EXTRA_PACKAGES}
    missing = [p for p in CORE_PACKAGES if packages[p] is None]
    if missing:
        raise Program04Error(
            "Current environment is missing packages frozen/required by Program 00: " + ", ".join(missing)
        )
    if packages.get("pyarrow") is None:
        raise Program04Error("pyarrow is required for immutable Program 04 Parquet assets.")
    try:
        import torch  # type: ignore
        torch_cuda_build = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        raise Program04Error(f"Cannot inspect PyTorch environment: {exc}") from exc
    basis = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "torch_cuda_build": torch_cuda_build,
        "cudnn_version": cudnn_version,
    }
    return sha256_bytes(canonical_bytes(basis)), basis


def verify_environment_manifest(path: Path) -> dict[str, Any]:
    m = read_json(path)
    if m.get("manifest_type") != "environment":
        raise Program04Error(f"Unexpected environment manifest type: {path}")
    expected = m.get("environment_fingerprint_sha256")
    observed, _ = current_environment_fingerprint()
    if expected != observed:
        raise Program04Error(
            "Current software environment differs from Program 00's frozen environment.\n"
            f"expected={expected}\nobserved={observed}"
        )
    return m


def verify_data_manifest(path: Path, gsm8k_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    m = read_json(path)
    if m.get("schema_version") != "1.0" or m.get("manifest_type") != "data":
        raise Program04Error(f"Invalid Program 00 data manifest: {path}")
    datasets = m.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("gsm8k"), dict):
        raise Program04Error("data_manifest.json lacks datasets.gsm8k.")
    record = datasets["gsm8k"]
    if record.get("repo_id") != GSM8K_REPO:
        raise Program04Error(f"Unexpected GSM8K repo in manifest: {record.get('repo_id')}")
    observed_fp = sha256_bytes(canonical_bytes({"research_scope": m.get("research_scope"), "datasets": datasets}))
    if m.get("content_fingerprint_sha256") != observed_fp:
        raise Program04Error("Program 00 data manifest content fingerprint mismatch.")
    if not gsm8k_dir.exists():
        raise Program04Error(f"Pinned GSM8K directory is missing: {gsm8k_dir}")
    observed_tree = tree_hash(gsm8k_dir)["tree_sha256"]
    if observed_tree != record.get("tree_sha256"):
        raise Program04Error(
            "Pinned GSM8K tree hash differs from Program 00 manifest.\n"
            f"expected={record.get('tree_sha256')}\nobserved={observed_tree}"
        )
    return m, dict(record)


def tokenizer_template_hash(model_dir: Path) -> tuple[str, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program04Error(f"Cannot load pinned tokenizer locally: {exc}") from exc
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program04Error("Pinned tokenizer has no chat_template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(text), tok.__class__.__name__


def verify_model_manifest(path: Path, model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    m = read_json(path)
    if m.get("schema_version") != "1.0" or m.get("manifest_type") != "model":
        raise Program04Error(f"Invalid Program 00 model manifest: {path}")
    models = m.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
        raise Program04Error("model_manifest.json lacks models.primary.")
    record = models["primary"]
    if record.get("repo_id") != MODEL_REPO:
        raise Program04Error(f"Unexpected primary model in manifest: {record.get('repo_id')}")
    if m.get("content_fingerprint_sha256") != sha256_bytes(canonical_bytes(models)):
        raise Program04Error("Program 00 model manifest content fingerprint mismatch.")
    if not model_dir.exists():
        raise Program04Error(f"Pinned model directory is missing: {model_dir}")
    observed_tree = tree_hash(model_dir)["tree_sha256"]
    if observed_tree != record.get("tree_sha256"):
        raise Program04Error(
            "Pinned Qwen tree hash differs from Program 00 manifest.\n"
            f"expected={record.get('tree_sha256')}\nobserved={observed_tree}"
        )
    validation = record.get("validation") or {}
    tok_validation = validation.get("tokenizer") if isinstance(validation, dict) else None
    if not isinstance(tok_validation, dict):
        raise Program04Error("Frozen model manifest lacks tokenizer validation metadata.")
    observed_chat, tok_class = tokenizer_template_hash(model_dir)
    if observed_chat != tok_validation.get("chat_template_sha256"):
        raise Program04Error("Pinned tokenizer chat template differs from Program 00 manifest.")
    if tok_class != tok_validation.get("class"):
        raise Program04Error("Tokenizer class differs from Program 00 validation.")
    return m, dict(record)


def verify_split_registry(
    manifest_path: Path,
    registry_path: Path,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
) -> dict[str, Any]:
    m = read_json(manifest_path)
    if m.get("schema_version") != "1.0" or m.get("manifest_type") != "split_registry":
        raise Program04Error(f"Unexpected split registry manifest type: {manifest_path}")
    source = m.get("source")
    registry = m.get("registry")
    firewall = m.get("research_firewall")
    if not all(isinstance(x, dict) for x in (source, registry, firewall)):
        raise Program04Error("Split registry manifest is missing required sections.")
    if source.get("dataset_revision") != gsm_record.get("resolved_revision"):
        raise Program04Error("Program 01 split registry points to a different GSM8K revision.")
    if source.get("gsm8k_tree_sha256") != gsm_record.get("tree_sha256"):
        raise Program04Error("Program 01 split registry source tree differs from Program 00.")
    if source.get("program00_data_manifest_content_fingerprint_sha256") != data_manifest.get("content_fingerprint_sha256"):
        raise Program04Error("Program 01 split registry points to a different Program 00 data manifest.")
    if firewall.get("official_test_is_never_used_for_split_tuning") is not True:
        raise Program04Error("Split registry does not preserve the official-test firewall.")
    if not registry_path.exists():
        raise Program04Error(f"Missing Program 01 split registry: {registry_path}")
    if registry.get("file_sha256") != sha256_file(registry_path):
        raise Program04Error("Program 01 split-registry Parquet hash mismatch.")
    try:
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(registry_path).to_pylist()
    except Exception as exc:
        raise Program04Error(f"Cannot read immutable split registry: {exc}") from exc
    counts = {"training": 0, "development": 0, "test": 0}
    seen: set[str] = set()
    cols = registry.get("columns")
    h = hashlib.sha256()
    for r in rows:
        rs = r.get("research_split")
        if rs not in counts:
            raise Program04Error(f"Unexpected research_split={rs!r} in split registry.")
        counts[str(rs)] += 1
        pid = r.get("prompt_id")
        if not isinstance(pid, str) or not HEX64_RE.fullmatch(pid) or pid in seen:
            raise Program04Error("Split registry contains invalid/duplicate prompt_id.")
        seen.add(pid)
        if r.get("original_split") == "test" and rs != "test":
            raise Program04Error("Official GSM8K test row leaked into another research split.")
        if isinstance(cols, list) and cols:
            try:
                projected = {k: r[k] for k in cols}
            except KeyError as exc:
                raise Program04Error(f"Split registry missing canonical column {exc}.") from exc
            h.update(canonical_bytes(projected)); h.update(b"\n")
    if counts != {"training": 6000, "development": 1473, "test": 1319}:
        raise Program04Error(f"Unexpected split counts: {counts}")
    if isinstance(cols, list) and cols and h.hexdigest() != registry.get("content_sha256"):
        raise Program04Error("Immutable split registry semantic content SHA-256 mismatch.")
    return m


def verify_protocol_lock(
    path: Path,
    *,
    config_sha256: str,
    data_manifest: Mapping[str, Any],
    model_record: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        raise Program04Error(f"Paper mode requires frozen {path}")
    lock = read_json(path)
    locked_config = first_present(
        lock,
        (("config_sha256",), ("protocol_config_sha256",), ("protocol", "config_sha256"), ("inputs", "config_sha256")),
    )
    if not isinstance(locked_config, str):
        raise Program04Error("protocol_lock.json must contain a frozen config SHA-256.")
    if locked_config != config_sha256:
        raise Program04Error(
            "protocol.yaml differs from protocol_lock.json.\n"
            f"locked={locked_config}\nobserved={config_sha256}"
        )
    locked_data = first_present(
        lock,
        (("data_manifest_sha256",), ("inputs", "data_manifest_sha256"), ("data", "content_fingerprint_sha256")),
    )
    if not isinstance(locked_data, str):
        raise Program04Error("protocol_lock.json must freeze Program 00 data_manifest_sha256.")
    if locked_data != data_manifest.get("content_fingerprint_sha256"):
        raise Program04Error("Program 00 data manifest differs from protocol lock.")
    locked_model = first_present(
        lock,
        (("model_sha",), ("model_revision",), ("model", "resolved_revision"), ("inputs", "model_revision")),
    )
    if not isinstance(locked_model, str):
        raise Program04Error("protocol_lock.json must freeze the pinned model revision.")
    if locked_model != model_record.get("resolved_revision"):
        raise Program04Error("Pinned Qwen revision differs from protocol lock.")
    locked_split = first_present(
        lock,
        (("split_registry_sha256",), ("inputs", "split_registry_sha256"), ("split", "content_fingerprint_sha256")),
    )
    if not isinstance(locked_split, str):
        raise Program04Error("protocol_lock.json must freeze Program 01 split_registry_sha256.")
    if locked_split != split_manifest.get("content_fingerprint_sha256"):
        raise Program04Error("Program 01 split registry differs from protocol lock.")
    source_hashes = lock.get("source_code_sha256") or {}
    expected_source = source_hashes.get(PROGRAM) if isinstance(source_hashes, Mapping) else None
    observed_source = sha256_file(Path(__file__).resolve())
    if not isinstance(expected_source, str):
        raise Program04Error("protocol_lock.json does not freeze Program 04 source code.")

    if expected_source != observed_source:
        amendment_path = path.parent / AMENDMENT_MANIFEST
        if not amendment_path.exists():
            raise Program04Error(
                "Program 04 source code differs from the original frozen protocol lock. "
                "This amended symmetric scorer is legal only with an explicit pre-analysis "
                f"amendment manifest at {amendment_path}. Do not bypass the lock."
            )
        amendment = read_json(amendment_path)
        if amendment.get("schema_version") != "1.0" or amendment.get("manifest_type") != "preanalysis_program04_scoring_amendment":
            raise Program04Error(f"Invalid Program 04 amendment manifest: {amendment_path}")
        required_equal = {
            "program": PROGRAM,
            "old_program04_source_sha256": expected_source,
            "new_program04_source_sha256": observed_source,
            "original_protocol_lock_sha256": sha256_file(path),
            "protocol_config_sha256": config_sha256,
            "scoring_contract": SCORING_CONTRACT,
        }
        for key, expected in required_equal.items():
            if amendment.get(key) != expected:
                raise Program04Error(
                    f"Program 04 amendment manifest field {key!r} mismatch. "
                    f"expected={expected!r}, observed={amendment.get(key)!r}"
                )
        required_true = (
            "identity_firewall_triggered_before_nonidentity",
            "nonidentity_pairs_evaluated_before_amendment_is_false",
            "program03_behavior_logs_unchanged",
            "pair_registry_unchanged",
            "identity_tolerances_unchanged",
        )
        for key in required_true:
            if amendment.get(key) is not True:
                raise Program04Error(
                    f"Program 04 amendment manifest must explicitly certify {key}=true."
                )
        lock = dict(lock)
        lock["_program04_amendment"] = amendment
    return lock


def verify_frozen_gate_for_test(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    gate_path = root / "outputs" / "frozen_gate.json"
    if not gate_path.exists():
        raise Program04Error(
            "Official test rescoring is forbidden before outputs/frozen_gate.json exists. "
            "Finish development calibration first."
        )
    observed = sha256_file(gate_path)
    expected = first_present(
        lock,
        (("frozen_gate_sha256",), ("gate", "file_sha256"), ("inputs", "frozen_gate_sha256"), ("frozen", "gate_sha256")),
    )
    sidecar = root / "manifests" / "frozen_gate_manifest.json"
    if expected is None and sidecar.exists():
        sm = read_json(sidecar)
        expected = first_present(sm, (("file_sha256",), ("frozen_gate_sha256",), ("gate", "file_sha256")))
    if expected is None:
        raise Program04Error(
            "Test mode must verify the frozen gate hash, but neither protocol_lock.json nor "
            "manifests/frozen_gate_manifest.json contains it."
        )
    if observed != expected:
        raise Program04Error("outputs/frozen_gate.json hash differs from its frozen recorded hash.")
    return read_json(gate_path)


# ---------------------------------------------------------------------------
# Program 02 adapter verification
# ---------------------------------------------------------------------------


def training_manifest_path(root: Path, mode: str, seed: int) -> Path:
    if mode == "paper":
        return root / "manifests" / "training" / f"seed_{seed}.json"
    return root / "manifests" / f"_{mode}" / "training" / f"seed_{seed}.json"


def checkpoint_seed_root(root: Path, mode: str, seed: int) -> Path:
    if mode == "paper":
        return root / "checkpoints" / f"seed_{seed}"
    return root / "checkpoints" / f"_{mode}" / "program02" / f"seed_{seed}"


def verify_training_seed(
    *,
    root: Path,
    mode: str,
    seed: int,
    model_record: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    path = training_manifest_path(root, mode, seed)
    if not path.exists():
        raise Program04Error(f"Missing completed Program 02 training manifest: {path}")
    m = read_json(path)
    if m.get("manifest_type") != "grpo_training_seed" or m.get("status") != "complete":
        raise Program04Error(f"Program 02 seed manifest is not complete: {path}")
    if m.get("training_seed") != seed or m.get("mode") != mode:
        raise Program04Error(f"Program 02 seed manifest metadata mismatch: {path}")
    inputs = m.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise Program04Error(f"Program 02 seed manifest lacks inputs: {path}")
    if inputs.get("dataset_revision") != split_manifest.get("source", {}).get("dataset_revision"):
        raise Program04Error(f"Seed {seed} was trained against a different dataset revision.")
    if inputs.get("data_manifest_content_fingerprint_sha256") != data_manifest.get("content_fingerprint_sha256"):
        raise Program04Error(f"Seed {seed} was trained against a different data manifest.")
    if inputs.get("split_registry_content_fingerprint_sha256") != split_manifest.get("content_fingerprint_sha256"):
        raise Program04Error(f"Seed {seed} was trained against a different split registry.")
    if inputs.get("model_revision") != model_record.get("resolved_revision"):
        raise Program04Error(f"Seed {seed} was trained from a different base model revision.")
    if mode == "paper" and inputs.get("protocol_config_sha256") != config_sha256:
        raise Program04Error(f"Seed {seed} was trained under a different frozen protocol.yaml.")

    seed_root = checkpoint_seed_root(root, mode, seed)
    adapters = m.get("permanent_adapters")
    if not isinstance(adapters, list) or not adapters:
        raise Program04Error(f"Seed {seed} manifest lacks permanent adapters.")
    by_step: dict[int, AdapterRecord] = {}
    for rec in adapters:
        if not isinstance(rec, Mapping) or "step" not in rec:
            continue
        step = int(rec["step"])
        p = seed_root / "adapters" / f"step_{step:04d}"
        if not p.exists():
            raise Program04Error(f"Missing permanent adapter: {p}")
        observed = adapter_payload_hash(p)
        if observed != rec.get("payload_sha256"):
            raise Program04Error(f"Permanent adapter hash mismatch: seed={seed}, step={step}")
        amp = p / "adapter_manifest.json"
        if not amp.exists():
            raise Program04Error(f"Permanent adapter lacks adapter_manifest.json: {p}")
        am = read_json(amp)
        if am.get("training_seed") != seed or am.get("step") != step:
            raise Program04Error(f"Adapter manifest metadata mismatch: {p}")
        if am.get("base_model_revision") != model_record.get("resolved_revision"):
            raise Program04Error(f"Adapter base revision mismatch: {p}")
        if am.get("adapter_payload_tree_sha256") != observed:
            raise Program04Error(f"Adapter sidecar hash mismatch: {p}")
        by_step[step] = AdapterRecord(step, p, observed, am)
    out = dict(m)
    out["_verified_adapters"] = by_step
    return out


# ---------------------------------------------------------------------------
# Immutable pair registry
# ---------------------------------------------------------------------------


def pair_id(
    *,
    protocol_version_value: str,
    training_seed: int,
    dataset: str,
    split: str,
    behavior_step: int,
    target_step: int,
    purpose: str,
) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "protocol_version": protocol_version_value,
                "training_seed": training_seed,
                "dataset": dataset,
                "split": split,
                "behavior_step": behavior_step,
                "target_step": target_step,
                "purpose": purpose,
            }
        )
    )


def _pair(
    *, pversion: str, seed: int, split: str, b: int, e: int, purpose: str
) -> PairRecord:
    return PairRecord(
        pair_id=pair_id(
            protocol_version_value=pversion,
            training_seed=seed,
            dataset="GSM8K",
            split=split,
            behavior_step=b,
            target_step=e,
            purpose=purpose,
        ),
        training_seed=seed,
        dataset="GSM8K",
        split=split,
        behavior_step=b,
        target_step=e,
        purpose=purpose,
        protocol_version=pversion,
    )


def build_paper_pair_registry(
    *, seeds: Sequence[int], pversion: str, target_steps: Sequence[int]
) -> list[PairRecord]:
    if tuple(target_steps) != PAPER_TARGET_STEPS:
        raise Program04Error(
            "Paper pair registry requires target checkpoints 0,20,...,400 exactly. "
            f"Observed {list(target_steps)}"
        )
    records: list[PairRecord] = []
    for seed in seeds:
        for split in PAPER_SPLITS:
            # Identity first in ordering for human audit; processing order is
            # independently enforced later.
            for b in PAPER_ANCHORS:
                records.append(_pair(pversion=pversion, seed=seed, split=split, b=b, e=b, purpose="identity"))
            for e in PAPER_FIXED_TARGETS:
                records.append(_pair(pversion=pversion, seed=seed, split=split, b=0, e=e, purpose="fixed"))
            for b, targets in PAPER_ROLLING.items():
                for e in targets:
                    records.append(_pair(pversion=pversion, seed=seed, split=split, b=b, e=e, purpose="rolling"))
    keys = [(r.training_seed, r.split, r.behavior_step, r.target_step) for r in records]
    if len(keys) != len(set(keys)):
        raise Program04Error("Internal paper pair registry construction produced duplicate behavior-target pairs.")
    # 39 pairs per seed/split: 4 identity + 20 fixed + 15 rolling.
    expected = len(seeds) * len(PAPER_SPLITS) * 39
    if len(records) != expected:
        raise Program04Error(f"Internal paper pair count error: expected {expected}, got {len(records)}")
    return records


def build_engineering_pair_registry(
    *, seeds: Sequence[int], pversion: str, split: str, available_steps: Mapping[int, Sequence[int]]
) -> list[PairRecord]:
    records: list[PairRecord] = []
    for seed in seeds:
        steps = sorted(set(int(x) for x in available_steps[seed]))
        if not steps:
            raise Program04Error(f"No Program 02 checkpoints for seed {seed}.")
        # Program 03 engineering mode uses anchors 0 and final (or only 0).
        anchors = [0] if max(steps) == 0 else [0, max(steps)]
        for b in anchors:
            records.append(_pair(pversion=pversion, seed=seed, split=split, b=b, e=b, purpose="identity"))
        if max(steps) != 0:
            records.append(_pair(pversion=pversion, seed=seed, split=split, b=0, e=max(steps), purpose="fixed"))
    return records


def pair_rows(records: Sequence[PairRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]


def pair_content_sha256(records: Sequence[PairRecord]) -> str:
    ordered = sorted(
        pair_rows(records),
        key=lambda r: (
            int(r["training_seed"]), str(r["split"]), 0 if r["purpose"] == "identity" else 1,
            int(r["target_step"]), int(r["behavior_step"]), str(r["purpose"]),
        ),
    )
    return sha256_bytes(canonical_bytes(ordered))


def pair_registry_paths(root: Path, mode: str) -> tuple[Path, Path]:
    if mode == "paper":
        return root / "manifests" / "pair_registry.parquet", root / "manifests" / "pair_registry_manifest.json"
    base = root / "manifests" / f"_{mode}"
    return base / "pair_registry.parquet", base / "pair_registry_manifest.json"


def pair_parquet_schema() -> Any:
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise Program04Error(f"pyarrow is required for pair registry: {exc}") from exc
    return pa.schema(
        [
            pa.field("pair_id", pa.string(), nullable=False),
            pa.field("training_seed", pa.int64(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("behavior_step", pa.int32(), nullable=False),
            pa.field("target_step", pa.int32(), nullable=False),
            pa.field("purpose", pa.string(), nullable=False),
            pa.field("protocol_version", pa.string(), nullable=False),
        ]
    )


def write_pair_registry_once(
    *,
    root: Path,
    mode: str,
    records: Sequence[PairRecord],
    config_sha256: str,
    dataset_revision: str,
    model_revision: str,
) -> tuple[Path, dict[str, Any]]:
    parquet_path, manifest_path = pair_registry_paths(root, mode)
    if parquet_path.exists() or manifest_path.exists():
        raise Program04Error("Pair registry write-once path already exists; use verification instead of overwrite.")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = parquet_path.parent / f".pair-registry.staging-{uuid.uuid4().hex}"
    try:
        stage_dir.mkdir(parents=False, exist_ok=False)
        stage_parquet = stage_dir / parquet_path.name
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            table = pa.Table.from_pylist(pair_rows(records), schema=pair_parquet_schema())
            pq.write_table(table, stage_parquet, compression="zstd", use_dictionary=True)
        except Exception as exc:
            raise Program04Error(f"Cannot write immutable pair registry: {exc}") from exc
        with stage_parquet.open("rb+") as f:
            f.flush(); os.fsync(f.fileno())
        file_sha = sha256_file(stage_parquet)
        content_sha = pair_content_sha256(records)
        manifest = {
            "schema_version": PAIR_SCHEMA,
            "manifest_type": "behavior_target_pair_registry",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            "mode": mode,
            "protocol_config_sha256": config_sha256,
            "protocol_version": records[0].protocol_version if records else None,
            "dataset": "GSM8K",
            "dataset_revision": dataset_revision,
            "model_revision": model_revision,
            "pair_count": len(records),
            "columns": list(PAIR_COLUMNS),
            "content_sha256": content_sha,
            "parquet": {
                "local_path": rel(parquet_path, root),
                "file_sha256": file_sha,
                "row_count": len(records),
            },
            "design": {
                "paper_fixed": "(0,e), e=20,40,...,400",
                "paper_rolling": {
                    "100": [120, 140, 160, 180, 200],
                    "200": [220, 240, 260, 280, 300],
                    "300": [320, 340, 360, 380, 400],
                },
                "identity": [[0, 0], [100, 100], [200, 200], [300, 300]] if mode == "paper" else "engineering-mode anchors",
                "anti_cartesian_product": True,
                "created_before_official_test_rewards": mode == "paper",
            },
        }
        atomic_write_json(stage_dir / manifest_path.name, manifest)
        fsync_directory(stage_dir)
        os.replace(stage_parquet, parquet_path)
        os.replace(stage_dir / manifest_path.name, manifest_path)
        fsync_directory(parquet_path.parent)
        try:
            stage_dir.rmdir()
        except OSError:
            pass
        for frozen_file in (parquet_path, manifest_path):
            try:
                frozen_file.chmod(frozen_file.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
            except OSError:
                pass
        return parquet_path, manifest
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def load_and_verify_pair_registry(
    *,
    root: Path,
    mode: str,
    expected_records: Sequence[PairRecord],
    config_sha256: str,
    dataset_revision: str,
    model_revision: str,
) -> tuple[list[PairRecord], dict[str, Any]]:
    parquet_path, manifest_path = pair_registry_paths(root, mode)
    if not parquet_path.exists() or not manifest_path.exists():
        raise Program04Error("Pair registry or its manifest is missing.")
    m = read_json(manifest_path)
    if m.get("schema_version") != PAIR_SCHEMA or m.get("manifest_type") != "behavior_target_pair_registry":
        raise Program04Error("Invalid pair registry manifest schema/type.")
    if m.get("protocol_config_sha256") != config_sha256:
        raise Program04Error("Pair registry was frozen under a different protocol.yaml.")
    if m.get("dataset_revision") != dataset_revision or m.get("model_revision") != model_revision:
        raise Program04Error("Pair registry data/model revision mismatch.")
    if m.get("parquet", {}).get("file_sha256") != sha256_file(parquet_path):
        raise Program04Error("Pair registry Parquet hash mismatch.")
    try:
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(parquet_path).to_pylist()
    except Exception as exc:
        raise Program04Error(f"Cannot read pair registry: {exc}") from exc
    observed: list[PairRecord] = []
    for r in rows:
        pr = PairRecord(
            pair_id=str(r["pair_id"]), training_seed=int(r["training_seed"]), dataset=str(r["dataset"]),
            split=str(r["split"]), behavior_step=int(r["behavior_step"]), target_step=int(r["target_step"]),
            purpose=str(r["purpose"]), protocol_version=str(r["protocol_version"]),
        )
        expected_id = pair_id(
            protocol_version_value=pr.protocol_version,
            training_seed=pr.training_seed,
            dataset=pr.dataset,
            split=pr.split,
            behavior_step=pr.behavior_step,
            target_step=pr.target_step,
            purpose=pr.purpose,
        )
        if pr.pair_id != expected_id:
            raise Program04Error("Pair registry contains a non-reproducible pair_id.")
        if pr.dataset != "GSM8K" or pr.purpose not in {"identity", "fixed", "rolling", "robustness"}:
            raise Program04Error("Pair registry contains invalid dataset/purpose values.")
        observed.append(pr)
    if pair_content_sha256(observed) != m.get("content_sha256"):
        raise Program04Error("Pair registry semantic content hash mismatch.")
    if pair_content_sha256(observed) != pair_content_sha256(expected_records):
        raise Program04Error(
            "Existing pair registry differs from the deterministic protocol-defined registry. "
            "Do not silently alter behavior-target pairs."
        )
    if len(observed) != len(expected_records):
        raise Program04Error("Pair registry row count differs from expected protocol-defined pairs.")
    return observed, m


def ensure_pair_registry(
    *,
    root: Path,
    mode: str,
    split: str,
    expected_records: Sequence[PairRecord],
    config_sha256: str,
    dataset_revision: str,
    model_revision: str,
) -> tuple[list[PairRecord], dict[str, Any]]:
    parquet_path, manifest_path = pair_registry_paths(root, mode)
    if parquet_path.exists() or manifest_path.exists():
        return load_and_verify_pair_registry(
            root=root, mode=mode, expected_records=expected_records, config_sha256=config_sha256,
            dataset_revision=dataset_revision, model_revision=model_revision,
        )
    if mode == "paper" and split == "test":
        raise Program04Error(
            "Official test cannot create a new pair registry. Run Program 04 on development first so pair selection is frozen before test."
        )
    write_pair_registry_once(
        root=root, mode=mode, records=expected_records, config_sha256=config_sha256,
        dataset_revision=dataset_revision, model_revision=model_revision,
    )
    return load_and_verify_pair_registry(
        root=root, mode=mode, expected_records=expected_records, config_sha256=config_sha256,
        dataset_revision=dataset_revision, model_revision=model_revision,
    )


# ---------------------------------------------------------------------------
# Program 03 immutable behavior-shard verification
# ---------------------------------------------------------------------------


def behavior_data_root(root: Path, mode: str) -> Path:
    return root / "data" / "behavior_logs" if mode == "paper" else root / "data" / "behavior_logs" / f"_{mode}"


def behavior_manifest_root(root: Path, mode: str, split: str) -> Path:
    if mode == "paper":
        return root / "manifests" / "behavior_logs" / split
    return root / "manifests" / f"_{mode}" / "behavior_logs" / split


def verify_behavior_collection_index(
    *,
    root: Path,
    mode: str,
    split: str,
    config_sha256: str,
    dataset_revision: str,
    model_revision: str,
    selected_seeds: Sequence[int],
) -> dict[str, Any]:
    path = behavior_manifest_root(root, mode, split) / "collection_index.json"
    if not path.exists():
        raise Program04Error(f"Missing Program 03 collection index: {path}")
    m = read_json(path)
    if m.get("manifest_type") != "behavior_collection_index" or m.get("schema_version") != BEHAVIOR_SCHEMA:
        raise Program04Error(f"Invalid Program 03 behavior collection index: {path}")
    if m.get("mode") != mode or m.get("split") != split:
        raise Program04Error("Program 03 collection index mode/split mismatch.")
    if m.get("dataset_revision") != dataset_revision or m.get("model_revision") != model_revision:
        raise Program04Error("Program 03 collection index data/model revision mismatch.")
    if m.get("probability_evidence_precision") != PROBABILITY_EVIDENCE_PRECISION:
        raise Program04Error(
            "Program 03 collection index does not use the required FP32 "
            "probability-evidence precision. Regenerate Program 03 outputs."
        )
    if mode == "paper" and m.get("protocol_config_sha256") != config_sha256:
        raise Program04Error("Program 03 collection index was generated under a different frozen protocol.")
    recorded_seeds = set(int(x) for x in (m.get("seeds") or []))
    if not set(selected_seeds).issubset(recorded_seeds):
        raise Program04Error("Program 03 collection index does not cover all requested training seeds.")
    return m


def read_behavior_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise Program04Error(f"Cannot read immutable behavior Parquet {path}: {exc}") from exc
    return [dict(r) for r in rows]


def behavior_rows_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    # Byte-for-byte semantic hash contract used by Program 03: each row is
    # projected to Program 03's frozen ROW_COLUMNS, canonical-JSON encoded,
    # newline-delimited, and sorted by (prompt_id, sample_index).
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (str(r.get("prompt_id")), int(r.get("sample_index", -1)))):
        projected = {k: row.get(k) for k in BEHAVIOR_ROW_COLUMNS}
        h.update(canonical_bytes(projected))
        h.update(b"\n")
    return h.hexdigest()


def validate_behavior_row(
    row: Mapping[str, Any],
    *,
    seed: int,
    behavior_step: int,
    split: str,
    dataset_revision: str,
    model_revision: str,
    behavior_adapter_sha256: str,
    eos_set: set[int] | None = None,
    vocab_size: int | None = None,
) -> None:
    if row.get("training_seed") != seed or row.get("behavior_step") != behavior_step or row.get("split") != split:
        raise Program04Error("Behavior row seed/step/split mismatch.")
    if row.get("dataset_revision") != dataset_revision or row.get("model_revision") != model_revision:
        raise Program04Error("Behavior row data/model revision mismatch.")
    if row.get("behavior_adapter_sha256") != behavior_adapter_sha256:
        raise Program04Error("Behavior row adapter hash mismatch.")
    if row.get("dataset") != "GSM8K":
        raise Program04Error("Program 04 only accepts GSM8K main-paper behavior logs.")
    # This Program 04 implementation scores the target policy from raw causal-LM
    # logits. That is exactly the same sampling distribution as Program 03 only
    # under the paper's explicit no-truncation/no-penalty decoding contract.
    # If this contract changes, Program 04 must reproduce the same logits
    # processors/warpers rather than silently pretending raw softmax is pi_e.
    if not math.isclose(float(row.get("temperature")), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Program04Error("Exact target rescoring currently requires behavior temperature=1.0.")
    if not math.isclose(float(row.get("top_p")), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Program04Error("Exact target rescoring currently requires behavior top_p=1.0.")
    if int(row.get("top_k")) != 0:
        raise Program04Error("Exact target rescoring currently requires behavior top_k=0.")
    if not math.isclose(float(row.get("repetition_penalty")), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Program04Error("Exact target rescoring currently requires behavior repetition_penalty=1.0.")
    if row.get("behavior_logprob_source") not in (
        None, "generate_scores+compute_transition_scores(normalize_logits=True)"
    ):
        raise Program04Error("Unexpected Program 03 behavior log-probability source.")
    prompt = row.get("prompt_token_ids")
    completion = row.get("completion_token_ids")
    blogps = row.get("behavior_token_logprobs")
    if not isinstance(prompt, list) or not prompt or not all(isinstance(x, int) for x in prompt):
        raise Program04Error("Behavior row has invalid prompt_token_ids.")
    if not isinstance(completion, list) or not completion or not all(isinstance(x, int) for x in completion):
        raise Program04Error("Behavior row has invalid completion_token_ids.")
    if not isinstance(blogps, list) or len(blogps) != len(completion):
        raise Program04Error("Behavior token IDs/logprobs are not aligned.")
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in blogps):
        raise Program04Error("Behavior row contains non-finite token logprob.")
    if int(row.get("completion_length", -1)) != len(completion):
        raise Program04Error("Behavior completion_length mismatch.")
    recomputed = math.fsum(float(x) for x in blogps)
    if not math.isclose(recomputed, float(row.get("behavior_sequence_logprob")), rel_tol=1e-12, abs_tol=1e-10):
        raise Program04Error("Behavior sequence logprob differs from float64 sum of token logprobs.")
    eos = bool(row.get("terminated_with_eos"))
    trunc = bool(row.get("was_truncated"))
    if eos == trunc:
        raise Program04Error("Behavior trajectory must be exactly one of EOS-terminated or max-length truncated.")
    if eos_set is not None:
        hits = [i for i, tok in enumerate(completion) if int(tok) in eos_set]
        if eos:
            if hits != [len(completion) - 1]:
                raise Program04Error("EOS-terminated behavior trajectory must contain EOS exactly at the final saved token.")
        elif hits:
            raise Program04Error("Truncated behavior trajectory unexpectedly contains EOS.")
    if trunc and int(row.get("max_completion_length", -1)) != len(completion):
        raise Program04Error("Truncated trajectory length does not equal saved max_completion_length.")
    if vocab_size is not None:
        if any(int(x) < 0 or int(x) >= vocab_size for x in prompt + completion):
            raise Program04Error("Behavior trajectory contains a token ID outside target model vocabulary.")
    if not isinstance(row.get("trajectory_id"), str) or not HEX64_RE.fullmatch(str(row.get("trajectory_id"))):
        raise Program04Error("Behavior trajectory_id is missing or malformed.")


def discover_behavior_shards(
    *,
    root: Path,
    mode: str,
    split: str,
    seed: int,
    behavior_step: int,
    behavior_adapter_sha256: str,
) -> list[BehaviorShard]:
    base = (
        behavior_data_root(root, mode)
        / "gsm8k"
        / f"split={split}"
        / f"seed={seed}"
        / f"behavior_step={behavior_step:04d}"
    )
    if not base.exists():
        raise Program04Error(f"Program 03 behavior directory is missing: {base}")
    shards: list[BehaviorShard] = []
    for mp in sorted(base.rglob("manifest.json")):
        if not mp.parent.name.startswith("shard="):
            continue
        m = read_json(mp)
        if m.get("manifest_type") != "behavior_trajectory_shard" or m.get("schema_version") != BEHAVIOR_SCHEMA:
            continue
        if m.get("training_seed") != seed or m.get("behavior_step") != behavior_step or m.get("split") != split:
            raise Program04Error(f"Behavior shard manifest metadata mismatch: {mp}")
        if m.get("behavior_adapter_sha256") != behavior_adapter_sha256:
            raise Program04Error(f"Behavior shard adapter hash mismatch: {mp}")
        if m.get("probability_evidence_precision") != PROBABILITY_EVIDENCE_PRECISION:
            raise Program04Error(
                f"Behavior shard does not use required FP32 probability-evidence precision: {mp}"
            )
        parquet_rel = m.get("parquet", {}).get("local_path")
        if not isinstance(parquet_rel, str):
            raise Program04Error(f"Behavior shard manifest lacks parquet path: {mp}")
        pp = root / Path(parquet_rel)
        if not pp.exists():
            raise Program04Error(f"Behavior shard Parquet is missing: {pp}")
        if sha256_file(pp) != m.get("parquet", {}).get("file_sha256"):
            raise Program04Error(f"Behavior shard Parquet file hash mismatch: {pp}")
        rows = read_behavior_rows(pp)
        content_sha = behavior_rows_content_sha256(rows)
        if content_sha != m.get("parquet", {}).get("content_sha256"):
            raise Program04Error(f"Behavior shard semantic content hash mismatch: {pp}")
        if len(rows) != int(m.get("parquet", {}).get("row_count", -1)):
            raise Program04Error(f"Behavior shard row count mismatch: {pp}")
        shards.append(
            BehaviorShard(
                manifest_path=mp,
                parquet_path=pp,
                manifest_sha256=sha256_file(mp),
                content_sha256=content_sha,
                training_seed=seed,
                behavior_step=behavior_step,
                split=split,
                sample_start=int(m.get("sample_start")),
                sample_end=int(m.get("sample_end_exclusive")),
                shard_index=int(m.get("shard_index")),
                behavior_adapter_sha256=behavior_adapter_sha256,
                row_count=len(rows),
            )
        )
    if not shards:
        raise Program04Error(f"No immutable Program 03 shards found under {base}")
    shards.sort(key=lambda s: (s.sample_start, s.shard_index))
    # Reject duplicate source units.
    keys = [(s.sample_start, s.sample_end, s.shard_index) for s in shards]
    if len(keys) != len(set(keys)):
        raise Program04Error(f"Duplicate Program 03 behavior shard units under {base}")
    return shards


# ---------------------------------------------------------------------------
# Tokenizer/model loading and mathematically exact teacher forcing
# ---------------------------------------------------------------------------


def load_tokenizer(model_dir: Path, model_record: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program04Error(f"Cannot load pinned tokenizer: {exc}") from exc
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise Program04Error("Tokenizer has neither pad_token_id nor eos_token_id.")
        tok.pad_token = tok.eos_token
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program04Error("Pinned tokenizer has no chat template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    expected = ((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256")
    if expected is not None and sha256_text(text) != expected:
        raise Program04Error("Loaded tokenizer chat template hash differs from Program 00.")
    return tok


def eos_ids(tokenizer: Any) -> set[int]:
    raw = tokenizer.eos_token_id
    if raw is None:
        raise Program04Error("Tokenizer has no eos_token_id.")
    if isinstance(raw, int):
        return {int(raw)}
    if isinstance(raw, (list, tuple, set)):
        vals = {int(x) for x in raw}
        if not vals:
            raise Program04Error("Tokenizer eos_token_id collection is empty.")
        return vals
    raise Program04Error(f"Unsupported eos_token_id value: {raw!r}")


def resolve_dtype(device: str, preferred: str | None) -> tuple[Any, str]:
    """Return the fixed FP32 precision used for OPE probability evidence.

    Program 02 training precision is intentionally not inherited here.
    ``preferred`` is retained only for backward-compatible call signatures.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program04Error(f"Cannot import torch: {exc}") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise Program04Error("--device cuda requested, but torch.cuda.is_available() is False.")
    return torch.float32, PROBABILITY_EVIDENCE_PRECISION


def load_target_model(
    *,
    model_dir: Path,
    adapter_dir: Path,
    device: str,
    preferred_precision: str | None,
) -> tuple[Any, str]:
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as exc:
        raise Program04Error(f"Cannot import Transformers/PEFT inference stack: {exc}") from exc
    dtype, resolved = resolve_dtype(device, preferred_precision)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": dtype,
    }
    try:
        base = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        peft_model.eval()
        peft_model.to(device)
    except Exception as exc:
        raise Program04Error(f"Cannot load target model + LoRA adapter {adapter_dir}: {exc}") from exc
    return peft_model, resolved


def _model_vocab_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    value = getattr(cfg, "vocab_size", None)
    if not isinstance(value, int) or value <= 0:
        # PEFT wrappers sometimes expose base_model.config differently.
        base = getattr(model, "base_model", None)
        value = getattr(getattr(base, "config", None), "vocab_size", None)
    if not isinstance(value, int) or value <= 0:
        raise Program04Error("Cannot determine target model vocabulary size.")
    return int(value)


def _device_from_model(model: Any, fallback: str) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return fallback


def teacher_force_batch(
    *,
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    pad_token_id: int,
    device: str,
) -> list[list[float]]:
    """Return selected-token log probabilities for each saved completion.

    If full tokens are [x_0,...,x_{P-1}, y_0,...,y_{T-1}], a causal LM's
    logits at positions P-1,...,P+T-2 score y_0,...,y_{T-1}.  This is the
    only shift used here.  Right-padding exists solely to form a batch and is
    excluded by attention_mask and by the per-row gather slice.
    """
    if not rows:
        return []
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program04Error(f"Cannot import torch during teacher forcing: {exc}") from exc

    fulls: list[list[int]] = []
    prompt_lens: list[int] = []
    completion_lens: list[int] = []
    for r in rows:
        p = [int(x) for x in r["prompt_token_ids"]]
        c = [int(x) for x in r["completion_token_ids"]]
        if not p or not c:
            raise Program04Error("Teacher forcing received an empty prompt or completion.")
        full = p + c
        if len(full) < 2:
            raise Program04Error("Full causal sequence is too short to score.")
        fulls.append(full)
        prompt_lens.append(len(p))
        completion_lens.append(len(c))

    max_len = max(len(x) for x in fulls)
    input_ids = torch.full((len(fulls), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(fulls), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(fulls):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, : len(seq)] = 1

    with torch.inference_mode():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = getattr(out, "logits", None)
    if logits is None or logits.ndim != 3:
        raise Program04Error("Target causal LM forward pass did not return [batch,seq,vocab] logits.")
    if int(logits.shape[0]) != len(rows) or int(logits.shape[1]) < max_len:
        raise Program04Error("Target logits shape is inconsistent with teacher-forcing inputs.")

    results: list[list[float]] = []
    for i, (p_len, c_len) in enumerate(zip(prompt_lens, completion_lens)):
        # Position p_len-1 predicts completion token 0; position
        # p_len+c_len-2 predicts the final completion token (including EOS).
        pred_logits = logits[i, p_len - 1 : p_len + c_len - 1, :]
        if int(pred_logits.shape[0]) != c_len:
            raise Program04Error("Causal shift produced wrong number of target-token logits.")
        targets = input_ids[i, p_len : p_len + c_len]
        # Programs 03 and 04 use a shared FP32 probability-evidence contract.
        # Keep normalization explicitly in FP32, then promote selected values
        # to float64 only for deterministic aggregation/storage.
        pred_logits = pred_logits.to(dtype=torch.float32)
        log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        selected64 = selected.detach().to(dtype=torch.float64, device="cpu")
        vals = [float(x) for x in selected64.tolist()]
        if len(vals) != c_len or not all(math.isfinite(x) for x in vals):
            raise Program04Error("Teacher-forced target token logprobs are non-finite or misaligned.")
        results.append(vals)
    return results


def batched(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    if n <= 0:
        raise Program04Error("Batch size must be positive.")
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def rescore_id(trajectory_id_value: str, target_step: int, target_adapter_sha256: str) -> str:
    # Appendix-A contract: hash(trajectory_id, target_step, target_adapter_hash).
    return sha256_bytes(
        canonical_bytes(
            {
                "trajectory_id": trajectory_id_value,
                "target_step": int(target_step),
                "target_adapter_hash": target_adapter_sha256,
            }
        )
    )


def score_behavior_rows(
    *,
    model: Any,
    behavior_rows: Sequence[Mapping[str, Any]],
    pair: PairRecord,
    source_content_sha256: str,
    target_adapter_sha256: str,
    behavior_adapter_sha256: str,
    dataset_revision: str,
    model_revision: str,
    eos_set: set[int],
    pad_token_id: int,
    device: str,
    scoring_precision: str,
    spec: RescoreSpec,
    canonical_behavior_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score saved trajectories under the target and a canonical behavior denominator.

    Identity pairs (b=e) run the behavior checkpoint once with the canonical
    FP32 teacher-forced engine.  The same resulting sequence log-probability is
    assigned to both behavior and target, hence logW=0 exactly.

    Non-identity pairs receive ``canonical_behavior_rows`` from the already
    published and verified identity shard for the same behavior checkpoint and
    immutable Program-03 source shard.  Only the target numerator is newly
    teacher-forced.
    """
    vocab_size = _model_vocab_size(model)
    for r in behavior_rows:
        validate_behavior_row(
            r,
            seed=pair.training_seed,
            behavior_step=pair.behavior_step,
            split=pair.split,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
            behavior_adapter_sha256=behavior_adapter_sha256,
            eos_set=eos_set,
            vocab_size=vocab_size,
        )

    identity = pair.purpose == "identity"
    if identity and canonical_behavior_rows is not None:
        raise Program04Error("Identity scoring must create, not consume, canonical behavior rows.")
    if not identity and canonical_behavior_rows is None:
        raise Program04Error(
            "Non-identity scoring requires canonical behavior rows from the verified identity shard."
        )

    source_ids = {str(r["trajectory_id"]) for r in behavior_rows}
    if canonical_behavior_rows is not None and set(canonical_behavior_rows) != source_ids:
        raise Program04Error(
            "Canonical behavior denominator does not cover exactly the immutable source trajectories."
        )

    output: list[dict[str, Any]] = []
    for chunk in batched(list(behavior_rows), spec.batch_size):
        target_token_lists = teacher_force_batch(
            model=model, rows=chunk, pad_token_id=pad_token_id, device=device
        )
        if len(target_token_lists) != len(chunk):
            raise Program04Error("Teacher-forcing batch returned wrong number of trajectories.")

        for r, tlogps in zip(chunk, target_token_lists):
            tid = str(r["trajectory_id"])
            source_generation_logps = [float(x) for x in r["behavior_token_logprobs"]]
            if len(tlogps) != len(source_generation_logps):
                raise Program04Error("Target/source token-logprob vectors have different lengths.")

            target_seq = math.fsum(float(x) for x in tlogps)
            if not math.isfinite(target_seq):
                raise Program04Error("Non-finite canonical target sequence log-probability.")

            if identity:
                # One deterministic engine, one adapter, one saved token sequence:
                # define both sides from the exact same computed value.
                behavior_seq = float(target_seq)
                logw = 0.0
                abs_seq: float | None = 0.0
                max_token: float | None = 0.0
                mean_token: float | None = 0.0
                id_pass: bool | None = True
            else:
                assert canonical_behavior_rows is not None
                brow = canonical_behavior_rows.get(tid)
                if brow is None:
                    raise Program04Error(f"Missing canonical behavior denominator for trajectory {tid}.")
                if brow.get("purpose") != "identity":
                    raise Program04Error("Canonical behavior denominator row is not from an identity pair.")
                if int(brow.get("behavior_step", -1)) != pair.behavior_step or int(brow.get("target_step", -2)) != pair.behavior_step:
                    raise Program04Error("Canonical behavior denominator step mismatch.")
                if brow.get("behavior_adapter_sha256") != behavior_adapter_sha256 or brow.get("target_adapter_sha256") != behavior_adapter_sha256:
                    raise Program04Error("Canonical behavior denominator adapter hash mismatch.")
                if brow.get("source_behavior_content_sha256") != source_content_sha256:
                    raise Program04Error("Canonical behavior denominator source-content hash mismatch.")
                if bool(brow.get("identity_pass")) is not True:
                    raise Program04Error("Canonical behavior denominator comes from a failed identity row.")
                if not math.isclose(float(brow.get("log_weight")), 0.0, rel_tol=0.0, abs_tol=1e-12):
                    raise Program04Error("Canonical identity denominator has non-zero log_weight.")
                behavior_seq = float(brow["behavior_sequence_logprob"])
                if not math.isclose(
                    behavior_seq,
                    float(brow["target_sequence_logprob"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise Program04Error("Canonical identity row does not have behavior==target log-probability.")
                logw = float(target_seq - behavior_seq)
                abs_seq = None
                max_token = None
                mean_token = None
                id_pass = None

            if not math.isfinite(behavior_seq) or not math.isfinite(logw):
                raise Program04Error("Non-finite canonical behavior log-probability/log-weight.")

            clen = int(r["completion_length"])
            if clen <= 0:
                raise Program04Error("Completion length must be positive.")

            output.append(
                {
                    "rescore_id": rescore_id(tid, pair.target_step, target_adapter_sha256),
                    "trajectory_id": tid,
                    "dataset": "GSM8K",
                    "dataset_revision": dataset_revision,
                    "protocol_version": pair.protocol_version,
                    "training_seed": pair.training_seed,
                    "split": pair.split,
                    "behavior_step": pair.behavior_step,
                    "target_step": pair.target_step,
                    "purpose": pair.purpose,
                    "prompt_id": str(r["prompt_id"]),
                    "sample_index": int(r["sample_index"]),
                    "behavior_adapter_sha256": behavior_adapter_sha256,
                    "target_adapter_sha256": target_adapter_sha256,
                    "behavior_sequence_logprob": behavior_seq,
                    "target_sequence_logprob": target_seq,
                    "log_weight": logw,
                    "mean_log_ratio_per_token": logw / clen,
                    "completion_length": clen,
                    "correctness_reward": float(r["correctness_reward"]),
                    "terminated_with_eos": bool(r["terminated_with_eos"]),
                    "was_truncated": bool(r["was_truncated"]),
                    "identity_abs_sequence_logprob_diff": abs_seq,
                    "identity_max_abs_token_logprob_diff": max_token,
                    "identity_mean_abs_token_logprob_diff": mean_token,
                    "identity_pass": id_pass,
                    "scoring_precision": scoring_precision,
                    "logprob_definition": (
                        "symmetric canonical teacher-forced FP32 causal shift; "
                        "behavior denominator reused from verified identity shard; "
                        "selected-token log_softmax(model logits); no clipping"
                    ),
                    "source_behavior_content_sha256": source_content_sha256,
                }
            )
    return output

def rescore_parquet_schema() -> Any:
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise Program04Error(f"pyarrow is required for rescore outputs: {exc}") from exc
    return pa.schema(
        [
            pa.field("rescore_id", pa.string(), nullable=False),
            pa.field("trajectory_id", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("dataset_revision", pa.string(), nullable=False),
            pa.field("protocol_version", pa.string(), nullable=False),
            pa.field("training_seed", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("behavior_step", pa.int32(), nullable=False),
            pa.field("target_step", pa.int32(), nullable=False),
            pa.field("purpose", pa.string(), nullable=False),
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("sample_index", pa.int32(), nullable=False),
            pa.field("behavior_adapter_sha256", pa.string(), nullable=False),
            pa.field("target_adapter_sha256", pa.string(), nullable=False),
            pa.field("behavior_sequence_logprob", pa.float64(), nullable=False),
            pa.field("target_sequence_logprob", pa.float64(), nullable=False),
            pa.field("log_weight", pa.float64(), nullable=False),
            pa.field("mean_log_ratio_per_token", pa.float64(), nullable=False),
            pa.field("completion_length", pa.int32(), nullable=False),
            pa.field("correctness_reward", pa.float64(), nullable=False),
            pa.field("terminated_with_eos", pa.bool_(), nullable=False),
            pa.field("was_truncated", pa.bool_(), nullable=False),
            pa.field("identity_abs_sequence_logprob_diff", pa.float64(), nullable=True),
            pa.field("identity_max_abs_token_logprob_diff", pa.float64(), nullable=True),
            pa.field("identity_mean_abs_token_logprob_diff", pa.float64(), nullable=True),
            pa.field("identity_pass", pa.bool_(), nullable=True),
            pa.field("scoring_precision", pa.string(), nullable=False),
            pa.field("logprob_definition", pa.string(), nullable=False),
            pa.field("source_behavior_content_sha256", pa.string(), nullable=False),
        ]
    )


def rescore_rows_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted([dict(r) for r in rows], key=lambda r: str(r["trajectory_id"]))
    return sha256_bytes(canonical_bytes(ordered))


def validate_rescore_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair: PairRecord,
    source_rows: Sequence[Mapping[str, Any]],
    source_content_sha256: str,
    behavior_adapter_sha256: str,
    target_adapter_sha256: str,
    spec: RescoreSpec,
) -> dict[str, Any]:
    if len(rows) != len(source_rows):
        raise Program04Error("Rescore row count does not match source behavior shard.")
    source_by_tid = {str(r["trajectory_id"]): r for r in source_rows}
    if len(source_by_tid) != len(source_rows):
        raise Program04Error("Source behavior shard contains duplicate trajectory IDs.")

    seen: set[str] = set()
    max_abs_logw = 0.0
    max_token_diff = 0.0
    mean_abs_logw_acc = 0.0
    failed_identity = 0

    # Audit only: how far the canonical TF behavior denominator is from the
    # immutable Program-03 generation-time sequence score. This bridge is NOT
    # mixed into logW and is NOT the canonical identity gate.
    bridge_max_seq = 0.0
    bridge_sum_seq = 0.0
    bridge_over_original_seq_tol = 0

    for r in rows:
        tid = str(r.get("trajectory_id"))
        if tid in seen or tid not in source_by_tid:
            raise Program04Error("Rescore shard has duplicate/unexpected trajectory_id.")
        seen.add(tid)
        src = source_by_tid[tid]

        if (
            r.get("training_seed") != pair.training_seed
            or r.get("split") != pair.split
            or r.get("behavior_step") != pair.behavior_step
            or r.get("target_step") != pair.target_step
            or r.get("purpose") != pair.purpose
        ):
            raise Program04Error("Rescore row pair metadata mismatch.")
        if (
            r.get("behavior_adapter_sha256") != behavior_adapter_sha256
            or r.get("target_adapter_sha256") != target_adapter_sha256
        ):
            raise Program04Error("Rescore adapter hash mismatch.")
        if r.get("source_behavior_content_sha256") != source_content_sha256:
            raise Program04Error("Rescore row source-behavior content hash mismatch.")
        if str(r.get("rescore_id")) != rescore_id(tid, pair.target_step, target_adapter_sha256):
            raise Program04Error("Rescore ID is not reproducible from Appendix-A key.")
        if int(r.get("completion_length", -1)) != int(src["completion_length"]):
            raise Program04Error("Rescore completion length differs from source behavior row.")
        if float(r.get("correctness_reward")) != float(src["correctness_reward"]):
            raise Program04Error("Rescore correctness reward differs from immutable source behavior row.")

        b = float(r.get("behavior_sequence_logprob"))
        t = float(r.get("target_sequence_logprob"))
        lw = float(r.get("log_weight"))
        mean = float(r.get("mean_log_ratio_per_token"))
        if not all(math.isfinite(x) for x in (b, t, lw, mean)):
            raise Program04Error("Rescore row contains non-finite log probability/weight.")
        if not math.isclose(lw, t - b, rel_tol=1e-12, abs_tol=1e-10):
            raise Program04Error("log_weight != target_sequence_logprob - behavior_sequence_logprob.")
        if not math.isclose(mean, lw / int(r["completion_length"]), rel_tol=1e-12, abs_tol=1e-12):
            raise Program04Error("mean_log_ratio_per_token is inconsistent with log_weight/length.")

        bridge = abs(b - float(src["behavior_sequence_logprob"]))
        bridge_max_seq = max(bridge_max_seq, bridge)
        bridge_sum_seq += bridge
        bridge_over_original_seq_tol += int(bridge > spec.identity_sequence_atol)

        max_abs_logw = max(max_abs_logw, abs(lw))
        mean_abs_logw_acc += abs(lw)

        if pair.purpose == "identity":
            vals = (
                r.get("identity_abs_sequence_logprob_diff"),
                r.get("identity_max_abs_token_logprob_diff"),
                r.get("identity_mean_abs_token_logprob_diff"),
            )
            if any(v is None or not math.isfinite(float(v)) for v in vals):
                raise Program04Error("Identity rescore row lacks finite canonical identity diagnostics.")
            seq_diff, token_diff, mean_token_diff = (float(x) for x in vals)

            # Canonical symmetric identity is stronger than a tolerance test:
            # the exact same computed FP32 TF value is assigned to both sides.
            if not math.isclose(b, t, rel_tol=0.0, abs_tol=1e-12):
                raise Program04Error("Canonical identity row has behavior != target.")
            if not math.isclose(lw, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise Program04Error("Canonical identity row has non-zero log_weight.")
            expected_pass = (
                token_diff <= spec.identity_token_atol
                and seq_diff <= spec.identity_sequence_atol
                and math.isclose(seq_diff, 0.0, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(token_diff, 0.0, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(mean_token_diff, 0.0, rel_tol=0.0, abs_tol=1e-12)
            )
            if bool(r.get("identity_pass")) != expected_pass:
                raise Program04Error("Canonical identity pass flag is inconsistent with symmetric scoring.")
            failed_identity += int(not expected_pass)
            max_token_diff = max(max_token_diff, token_diff)
        else:
            if any(
                r.get(k) is not None
                for k in (
                    "identity_abs_sequence_logprob_diff",
                    "identity_max_abs_token_logprob_diff",
                    "identity_mean_abs_token_logprob_diff",
                    "identity_pass",
                )
            ):
                raise Program04Error("Non-identity rescore row unexpectedly contains identity diagnostics.")

    if set(source_by_tid) != seen:
        raise Program04Error("Rescore shard does not cover exactly the source behavior trajectories.")

    return {
        "row_count": len(rows),
        "max_abs_log_weight": max_abs_logw,
        "mean_abs_log_weight": (mean_abs_logw_acc / len(rows)) if rows else None,
        "identity_failed_rows": failed_identity if pair.purpose == "identity" else None,
        "identity_max_abs_token_logprob_diff": max_token_diff if pair.purpose == "identity" else None,
        "identity_pass": (failed_identity == 0) if pair.purpose == "identity" else None,
        "generation_bridge_max_abs_sequence_logprob_diff": bridge_max_seq,
        "generation_bridge_mean_abs_sequence_logprob_diff": (
            bridge_sum_seq / len(rows) if rows else None
        ),
        "generation_bridge_rows_over_original_sequence_atol": bridge_over_original_seq_tol,
        "generation_bridge_is_audit_only": True,
    }

def rescore_data_root(root: Path, mode: str) -> Path:
    return root / "data" / "target_rescores" if mode == "paper" else root / "data" / "target_rescores" / f"_{mode}"


def rescore_manifest_root(root: Path, mode: str, split: str) -> Path:
    if mode == "paper":
        return root / "manifests" / "target_rescores" / split
    return root / "manifests" / f"_{mode}" / "target_rescores" / split


def rescore_unit_dir(
    data_root: Path,
    *,
    pair: PairRecord,
    source: BehaviorShard,
) -> Path:
    return (
        data_root
        / "gsm8k"
        / f"split={pair.split}"
        / f"seed={pair.training_seed}"
        / f"behavior_step={pair.behavior_step:04d}"
        / f"target_step={pair.target_step:04d}"
        / f"sample_block={source.sample_start:04d}-{source.sample_end - 1:04d}"
        / f"shard={source.shard_index:05d}"
    )


def verify_rescore_unit(
    unit_dir: Path,
    *,
    pair: PairRecord,
    source: BehaviorShard,
    source_rows: Sequence[Mapping[str, Any]],
    behavior_adapter_sha256: str,
    target_adapter_sha256: str,
    spec: RescoreSpec,
    scoring_engine_sha256: str,
) -> dict[str, Any]:
    pp = unit_dir / "rescored.parquet"
    mp = unit_dir / "manifest.json"
    if not pp.exists() or not mp.exists():
        raise Program04Error(f"Incomplete published rescore unit: {unit_dir}")
    m = read_json(mp)
    if m.get("schema_version") != RESCORE_SCHEMA or m.get("manifest_type") != "target_rescore_shard":
        raise Program04Error(f"Invalid rescore shard manifest: {mp}")
    if m.get("pair_id") != pair.pair_id:
        raise Program04Error(f"Existing rescore unit belongs to a different pair: {unit_dir}")
    if m.get("scoring_engine_sha256") != scoring_engine_sha256:
        raise Program04Error(
            f"Existing rescore unit was created under a different scoring-engine specification: {unit_dir}"
        )
    if m.get("source_behavior", {}).get("content_sha256") != source.content_sha256:
        raise Program04Error(f"Rescore unit source behavior hash mismatch: {unit_dir}")
    if m.get("target_adapter_sha256") != target_adapter_sha256:
        raise Program04Error(f"Rescore unit target adapter hash mismatch: {unit_dir}")
    if m.get("parquet", {}).get("file_sha256") != sha256_file(pp):
        raise Program04Error(f"Rescore Parquet file hash mismatch: {pp}")
    try:
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(pp).to_pylist()
    except Exception as exc:
        raise Program04Error(f"Cannot read rescore shard {pp}: {exc}") from exc
    validation = validate_rescore_rows(
        rows,
        pair=pair,
        source_rows=source_rows,
        source_content_sha256=source.content_sha256,
        behavior_adapter_sha256=behavior_adapter_sha256,
        target_adapter_sha256=target_adapter_sha256,
        spec=spec,
    )
    if rescore_rows_content_sha256(rows) != m.get("parquet", {}).get("content_sha256"):
        raise Program04Error(f"Rescore shard semantic content hash mismatch: {pp}")
    if validation.get("row_count") != m.get("validation", {}).get("row_count"):
        raise Program04Error(f"Rescore shard validation summary mismatch: {unit_dir}")
    return m



def read_rescore_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise Program04Error(f"Cannot read immutable rescore Parquet {path}: {exc}") from exc
    return [dict(r) for r in rows]


def canonical_behavior_rows_for_source(
    *,
    root: Path,
    mode: str,
    pair: PairRecord,
    source: BehaviorShard,
    source_rows: Sequence[Mapping[str, Any]],
    behavior_adapter: AdapterRecord,
    spec: RescoreSpec,
    scoring_precision: str,
) -> dict[str, dict[str, Any]]:
    """Load and verify the identity-pair canonical behavior denominator shard."""
    if pair.purpose == "identity":
        raise Program04Error("Identity pair cannot consume its own canonical denominator.")

    identity_pair = _pair(
        pversion=pair.protocol_version,
        seed=pair.training_seed,
        split=pair.split,
        b=pair.behavior_step,
        e=pair.behavior_step,
        purpose="identity",
    )
    droot = rescore_data_root(root, mode)
    unit = rescore_unit_dir(droot, pair=identity_pair, source=source)
    if not unit.exists():
        raise Program04Error(
            "Missing canonical identity denominator shard before distant scoring: "
            f"seed={pair.training_seed}, b={pair.behavior_step}, source={source.parquet_path}. "
            "All identity pairs must finish before any non-identity pair."
        )

    scoring_hash = scoring_engine_fingerprint(spec, scoring_precision)
    verify_rescore_unit(
        unit,
        pair=identity_pair,
        source=source,
        source_rows=source_rows,
        behavior_adapter_sha256=behavior_adapter.payload_sha256,
        target_adapter_sha256=behavior_adapter.payload_sha256,
        spec=spec,
        scoring_engine_sha256=scoring_hash,
    )
    rows = read_rescore_rows(unit / "rescored.parquet")
    out = {str(r["trajectory_id"]): r for r in rows}
    if len(out) != len(rows):
        raise Program04Error("Canonical identity denominator shard contains duplicate trajectory IDs.")
    source_ids = {str(r["trajectory_id"]) for r in source_rows}
    if set(out) != source_ids:
        raise Program04Error("Canonical identity denominator shard does not match immutable source trajectories.")
    for r in out.values():
        if r.get("purpose") != "identity" or bool(r.get("identity_pass")) is not True:
            raise Program04Error("Canonical denominator shard contains a non-passing identity row.")
        if not math.isclose(float(r["log_weight"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise Program04Error("Canonical denominator identity row has non-zero log_weight.")
    return out

def publish_rescore_unit(
    *,
    root: Path,
    unit_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    pair: PairRecord,
    source: BehaviorShard,
    source_rows: Sequence[Mapping[str, Any]],
    behavior_adapter_sha256: str,
    target_adapter_sha256: str,
    dataset_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_hash: str,
    spec: RescoreSpec,
    scoring_precision: str,
    scoring_engine_sha256: str,
) -> dict[str, Any]:
    if unit_dir.exists():
        return verify_rescore_unit(
            unit_dir,
            pair=pair,
            source=source,
            source_rows=source_rows,
            behavior_adapter_sha256=behavior_adapter_sha256,
            target_adapter_sha256=target_adapter_sha256,
            spec=spec,
            scoring_engine_sha256=scoring_engine_sha256,
        )
    validation = validate_rescore_rows(
        rows,
        pair=pair,
        source_rows=source_rows,
        source_content_sha256=source.content_sha256,
        behavior_adapter_sha256=behavior_adapter_sha256,
        target_adapter_sha256=target_adapter_sha256,
        spec=spec,
    )
    unit_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = unit_dir.parent / f".{unit_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=False, exist_ok=False)
        pp = stage / "rescored.parquet"
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            table = pa.Table.from_pylist([dict(r) for r in rows], schema=rescore_parquet_schema())
            pq.write_table(table, pp, compression="zstd", use_dictionary=True)
        except Exception as exc:
            raise Program04Error(f"Cannot write immutable rescore Parquet shard: {exc}") from exc
        with pp.open("rb+") as f:
            f.flush(); os.fsync(f.fileno())
        content_sha = rescore_rows_content_sha256(rows)
        file_sha = sha256_file(pp)
        manifest = {
            "schema_version": RESCORE_SCHEMA,
            "manifest_type": "target_rescore_shard",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION, "git_commit": git_commit(root)},
            "pair_id": pair.pair_id,
            "purpose": pair.purpose,
            "dataset": "GSM8K",
            "dataset_revision": dataset_revision,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_hash": chat_template_hash,
            "protocol_version": pair.protocol_version,
            "training_seed": pair.training_seed,
            "split": pair.split,
            "behavior_step": pair.behavior_step,
            "target_step": pair.target_step,
            "behavior_adapter_sha256": behavior_adapter_sha256,
            "target_adapter_sha256": target_adapter_sha256,
            "scoring_precision": scoring_precision,
            "scoring_spec": asdict(spec),
            "scoring_engine_sha256": scoring_engine_sha256,
            "source_behavior": {
                "manifest_path": rel(source.manifest_path, root),
                "manifest_sha256": source.manifest_sha256,
                "parquet_path": rel(source.parquet_path, root),
                "content_sha256": source.content_sha256,
                "sample_start": source.sample_start,
                "sample_end_exclusive": source.sample_end,
                "shard_index": source.shard_index,
            },
            "parquet": {
                "local_path": rel(unit_dir / "rescored.parquet", root),
                "file_sha256": file_sha,
                "content_sha256": content_sha,
                "row_count": len(rows),
                "columns": list(RESCORE_COLUMNS),
            },
            "math_contract": {
                "same_saved_token_sequence": True,
                "retokenization": False,
                "resampling": False,
                "causal_shift": "logits[prompt_len-1 : prompt_len+completion_len-1] score completion tokens",
                "eos_included_when_saved": True,
                "padding_scored": False,
                "log_weight": "canonical_target_TF_sequence_logprob - canonical_behavior_TF_sequence_logprob",
                "weight_clipping": False,
                "aggregation": "float64/Python math.fsum for sequence sums",
                "behavior_denominator": "reused from verified identity-pair canonical FP32 teacher-forced shard",
                "program03_generation_logprob": "immutable provenance/audit only; not mixed into logW",
                "scoring_contract": SCORING_CONTRACT,
            },
            "validation": validation,
        }
        atomic_write_json(stage / "manifest.json", manifest)
        fsync_directory(stage)
        if unit_dir.exists():
            raise Program04Error(f"Rescore unit appeared concurrently: {unit_dir}")
        os.replace(stage, unit_dir)
        fsync_directory(unit_dir.parent)
        make_read_only_tree(unit_dir)
        return verify_rescore_unit(
            unit_dir,
            pair=pair,
            source=source,
            source_rows=source_rows,
            behavior_adapter_sha256=behavior_adapter_sha256,
            target_adapter_sha256=target_adapter_sha256,
            spec=spec,
            scoring_engine_sha256=scoring_engine_sha256,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def cleanup_staging(data_root: Path, resume: bool) -> None:
    stale = [p for p in data_root.rglob(".*.staging-*") if p.is_dir()] if data_root.exists() else []
    if not stale:
        return
    if not resume:
        raise Program04Error(
            f"Found {len(stale)} stale Program 04 staging directories. Re-run with --resume to discard only unpublished staging state."
        )
    for p in stale:
        print(f"[RESUME] removing unpublished staging directory {p}")
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Identity gate / pair processing
# ---------------------------------------------------------------------------


def scoring_engine_fingerprint(spec: RescoreSpec, precision: str) -> str:
    payload = {
        "program": PROGRAM,
        "program_version": PROGRAM_VERSION,
        "rescore_spec": asdict(spec),
        "precision": precision,
        "probability_evidence_precision": PROBABILITY_EVIDENCE_PRECISION,
        "scoring_contract": SCORING_CONTRACT,
        "causal_shift": "p-1_to_p+t-2",
        "normalization": "torch.nn.functional.log_softmax in float32",
        "use_cache": False,
        "target_token_source": "saved Program03 completion_token_ids",
        "behavior_denominator": "canonical identity-pair FP32 teacher-forced sequence logprob",
        "generation_time_behavior_logprob_role": "immutable provenance/audit only",
        "symmetric_behavior_target_engine": True,
    }
    return sha256_bytes(canonical_bytes(payload))

def process_pair(
    *,
    root: Path,
    mode: str,
    pair: PairRecord,
    model: Any,
    scoring_precision: str,
    tokenizer: Any,
    target_adapter: AdapterRecord,
    behavior_adapter: AdapterRecord,
    spec: RescoreSpec,
    dataset_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_hash: str,
    resume: bool,
) -> dict[str, Any]:
    sources = discover_behavior_shards(
        root=root,
        mode=mode,
        split=pair.split,
        seed=pair.training_seed,
        behavior_step=pair.behavior_step,
        behavior_adapter_sha256=behavior_adapter.payload_sha256,
    )
    droot = rescore_data_root(root, mode)
    existing_flags = [rescore_unit_dir(droot, pair=pair, source=src).exists() for src in sources]
    if any(existing_flags) and not all(existing_flags) and not resume:
        raise Program04Error(
            f"Partial Program 04 outputs already exist for seed={pair.training_seed}, "
            f"b={pair.behavior_step}, e={pair.target_step}. Re-run with --resume."
        )

    scoring_hash = scoring_engine_fingerprint(spec, scoring_precision)
    eos_set = eos_ids(tokenizer)
    pad = int(tokenizer.pad_token_id)
    completed = 0
    total_rows = 0
    max_identity_token = 0.0
    max_identity_seq = 0.0
    identity_failed = 0
    bridge_max_seq = 0.0
    bridge_over_seq_tol = 0

    for source in sources:
        unit = rescore_unit_dir(droot, pair=pair, source=source)
        source_rows = read_behavior_rows(source.parquet_path)
        if behavior_rows_content_sha256(source_rows) != source.content_sha256:
            raise Program04Error(f"Behavior source changed after discovery: {source.parquet_path}")

        if unit.exists():
            m = verify_rescore_unit(
                unit,
                pair=pair,
                source=source,
                source_rows=source_rows,
                behavior_adapter_sha256=behavior_adapter.payload_sha256,
                target_adapter_sha256=target_adapter.payload_sha256,
                spec=spec,
                scoring_engine_sha256=scoring_hash,
            )
        else:
            canonical_rows: Mapping[str, Mapping[str, Any]] | None = None
            if pair.purpose != "identity":
                canonical_rows = canonical_behavior_rows_for_source(
                    root=root,
                    mode=mode,
                    pair=pair,
                    source=source,
                    source_rows=source_rows,
                    behavior_adapter=behavior_adapter,
                    spec=spec,
                    scoring_precision=scoring_precision,
                )

            rows = score_behavior_rows(
                model=model,
                behavior_rows=source_rows,
                pair=pair,
                source_content_sha256=source.content_sha256,
                target_adapter_sha256=target_adapter.payload_sha256,
                behavior_adapter_sha256=behavior_adapter.payload_sha256,
                dataset_revision=dataset_revision,
                model_revision=model_revision,
                eos_set=eos_set,
                pad_token_id=pad,
                device=_device_from_model(model, "cpu"),
                scoring_precision=scoring_precision,
                spec=spec,
                canonical_behavior_rows=canonical_rows,
            )
            m = publish_rescore_unit(
                root=root,
                unit_dir=unit,
                rows=rows,
                pair=pair,
                source=source,
                source_rows=source_rows,
                behavior_adapter_sha256=behavior_adapter.payload_sha256,
                target_adapter_sha256=target_adapter.payload_sha256,
                dataset_revision=dataset_revision,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                chat_template_hash=chat_template_hash,
                spec=spec,
                scoring_precision=scoring_precision,
                scoring_engine_sha256=scoring_hash,
            )
            print(
                f"[SAVE] seed={pair.training_seed} b={pair.behavior_step} e={pair.target_step} "
                f"purpose={pair.purpose} sample={source.sample_start}:{source.sample_end} "
                f"shard={source.shard_index:05d} rows={len(rows)}"
            )

        completed += 1
        vr = m.get("validation") or {}
        total_rows += int(vr.get("row_count") or 0)
        bridge_max_seq = max(
            bridge_max_seq,
            float(vr.get("generation_bridge_max_abs_sequence_logprob_diff") or 0.0),
        )
        bridge_over_seq_tol += int(
            vr.get("generation_bridge_rows_over_original_sequence_atol") or 0
        )
        if pair.purpose == "identity":
            max_identity_token = max(
                max_identity_token,
                float(vr.get("identity_max_abs_token_logprob_diff") or 0.0),
            )
            # Canonical identity max_abs_log_weight is exactly zero by construction.
            max_identity_seq = max(
                max_identity_seq,
                float(vr.get("max_abs_log_weight") or 0.0),
            )
            identity_failed += int(vr.get("identity_failed_rows") or 0)

    return {
        "pair_id": pair.pair_id,
        "training_seed": pair.training_seed,
        "split": pair.split,
        "behavior_step": pair.behavior_step,
        "target_step": pair.target_step,
        "purpose": pair.purpose,
        "source_shards": len(sources),
        "completed_shards": completed,
        "rows": total_rows,
        "identity_failed_rows": identity_failed if pair.purpose == "identity" else None,
        "identity_max_abs_token_logprob_diff": max_identity_token if pair.purpose == "identity" else None,
        "identity_max_abs_sequence_logprob_diff": max_identity_seq if pair.purpose == "identity" else None,
        "identity_pass": (identity_failed == 0) if pair.purpose == "identity" else None,
        "generation_bridge_max_abs_sequence_logprob_diff": bridge_max_seq,
        "generation_bridge_rows_over_original_sequence_atol": bridge_over_seq_tol,
        "generation_bridge_is_audit_only": True,
    }

def verify_identity_summary(summary: Mapping[str, Any], spec: RescoreSpec) -> None:
    if summary.get("purpose") != "identity":
        raise Program04Error("Internal identity-gate call received non-identity summary.")
    if not bool(summary.get("identity_pass")):
        raise Program04Error(
            "CANONICAL IDENTITY GATE FAILED. The same FP32 teacher-forced scoring "
            "engine and same adapter did not produce behavior==target exactly enough. "
            "Do NOT run distant OPE pairs.\n"
            f"seed={summary.get('training_seed')} b=e={summary.get('behavior_step')} "
            f"failed_rows={summary.get('identity_failed_rows')} "
            f"max_token_diff={summary.get('identity_max_abs_token_logprob_diff')} "
            f"max_sequence_diff={summary.get('identity_max_abs_sequence_logprob_diff')} "
            f"tolerances(token={spec.identity_token_atol}, sequence={spec.identity_sequence_atol})"
        )

def source_unit_count_from_behavior_index(
    index: Mapping[str, Any], pairs: Sequence[PairRecord]
) -> int:
    shards = index.get("shards") or []
    if not isinstance(shards, list):
        raise Program04Error("Program 03 collection index has invalid shards list.")
    counts: dict[tuple[int, int], int] = {}
    for r in shards:
        if not isinstance(r, Mapping):
            continue
        key = (int(r.get("training_seed", -1)), int(r.get("behavior_step", -1)))
        counts[key] = counts.get(key, 0) + 1
    total = 0
    for p in pairs:
        n = counts.get((p.training_seed, p.behavior_step), 0)
        if n <= 0:
            raise Program04Error(
                f"Program 03 collection index has no source shards for seed={p.training_seed}, b={p.behavior_step}."
            )
        total += n
    return total


def rebuild_rescore_collection_index(
    *,
    root: Path,
    mode: str,
    split: str,
    manifest_root: Path,
    pairs: Sequence[PairRecord],
    pair_registry_manifest: Mapping[str, Any],
    config_sha256: str,
    dataset_revision: str,
    model_revision: str,
    spec: RescoreSpec,
) -> Path:
    droot = rescore_data_root(root, mode)
    records: list[dict[str, Any]] = []
    if droot.exists():
        for mp in sorted(droot.rglob("manifest.json")):
            if not mp.parent.name.startswith("shard="):
                continue
            m = read_json(mp)
            if m.get("manifest_type") != "target_rescore_shard" or m.get("split") != split:
                continue
            records.append(
                {
                    "manifest_path": rel(mp, root),
                    "manifest_sha256": sha256_file(mp),
                    "pair_id": m.get("pair_id"),
                    "training_seed": m.get("training_seed"),
                    "behavior_step": m.get("behavior_step"),
                    "target_step": m.get("target_step"),
                    "purpose": m.get("purpose"),
                    "source_behavior_content_sha256": m.get("source_behavior", {}).get("content_sha256"),
                    "target_adapter_sha256": m.get("target_adapter_sha256"),
                    "parquet_path": m.get("parquet", {}).get("local_path"),
                    "parquet_sha256": m.get("parquet", {}).get("file_sha256"),
                    "content_sha256": m.get("parquet", {}).get("content_sha256"),
                    "row_count": m.get("parquet", {}).get("row_count"),
                    "identity_pass": m.get("validation", {}).get("identity_pass"),
                }
            )
    records.sort(key=lambda x: (
        int(x.get("training_seed", -1)), 0 if x.get("purpose") == "identity" else 1,
        int(x.get("target_step", -1)), int(x.get("behavior_step", -1)), str(x.get("manifest_path")),
    ))
    allowed_pair_ids = {p.pair_id for p in pairs if p.split == split}
    records = [r for r in records if r.get("pair_id") in allowed_pair_ids]
    payload = {
        "schema_version": RESCORE_SCHEMA,
        "manifest_type": "target_rescore_collection_index",
        "updated_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "mode": mode,
        "split": split,
        "protocol_config_sha256": config_sha256,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "pair_registry_content_sha256": pair_registry_manifest.get("content_sha256"),
        "rescore_spec": asdict(spec),
        "scoring_contract": SCORING_CONTRACT,
        "pair_count": len([p for p in pairs if p.split == split]),
        "shard_count": len(records),
        "row_count": sum(int(r.get("row_count") or 0) for r in records),
        "shards": records,
        "shard_set_sha256": sha256_bytes(canonical_bytes(records)),
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    path = manifest_root / "collection_index.json"
    atomic_write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# CLI / main orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Program 04: symmetric canonical FP32 teacher-forcing rescoring for GRPO-OPE."
    )
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-root", default=".")
    p.add_argument("--seed", type=int, default=None, help="Engineering/debug restriction; must be a configured legal seed.")
    p.add_argument("--batch-size", type=int, default=None, help="Scoring batch size. Recorded in shard fingerprint.")
    p.add_argument("--verify-only", action="store_true", help="Verify existing pair/source/rescore assets; never load target model.")
    p.add_argument("--reset-outputs", action="store_true", help="DANGEROUS: delete Program 04 outputs for selected mode/split.")
    return p.parse_args(argv)


def safe_reset(paths: Sequence[Path]) -> None:
    if os.environ.get("GRPO_OPE_CONFIRM_RESET") != "YES":
        raise Program04Error(
            "--reset-outputs requires environment variable GRPO_OPE_CONFIRM_RESET=YES. "
            "This prevents accidental deletion of expensive GPU assets."
        )

    def _remove_readonly(func: Any, path: str, exc_info: Any) -> None:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            func(path)
        except Exception:
            raise

    for p in paths:
        if p.exists():
            print(f"[RESET] deleting {p}")
            if p.is_dir():
                shutil.rmtree(p, onerror=_remove_readonly)
            else:
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
                p.unlink()


def print_header(
    *,
    root: Path,
    mode: str,
    split: str,
    cfg_path: Path,
    config_sha: str,
    pversion: str,
    dataset_revision: str,
    model_revision: str,
    seeds: Sequence[int],
    pair_count: int,
    source_units: int,
    spec: RescoreSpec,
    data_root: Path,
) -> None:
    print("=" * 78)
    print("Program 04 -- symmetric canonical FP32 teacher-forcing rescoring")
    print("=" * 78)
    print(f"protocol version       : {pversion}")
    print(f"config                  : {cfg_path}")
    print(f"config SHA-256          : {config_sha}")
    print(f"dataset revision        : {dataset_revision}")
    print(f"model revision          : {model_revision}")
    print(f"mode / split            : {mode} / {split}")
    print(f"training seeds          : {list(seeds)}")
    print(f"selected pairs          : {pair_count}")
    print(f"source shard units      : {source_units}")
    print(f"scoring batch size      : {spec.batch_size}")
    print(f"identity token atol     : {spec.identity_token_atol}")
    print(f"identity sequence atol  : {spec.identity_sequence_atol}")
    print(f"write root              : {data_root}")
    print(f"git commit              : {git_commit(root) or 'not-a-git-checkout'}")
    print(f"torch/transformers/peft : {package_version('torch')} / {package_version('transformers')} / {package_version('peft')}")
    print("=" * 78)


def _available_steps(tm: Mapping[str, Any]) -> list[int]:
    a = tm.get("_verified_adapters")
    if not isinstance(a, Mapping):
        return []
    return sorted(int(x) for x in a.keys())


def _required_steps_for_pairs(pairs: Sequence[PairRecord], seed: int) -> set[int]:
    out: set[int] = set()
    for p in pairs:
        if p.training_seed == seed:
            out.add(p.behavior_step); out.add(p.target_step)
    return out


def _pair_sort_key(p: PairRecord) -> tuple[int, int, int, int]:
    return (
        0 if p.purpose == "identity" else 1,
        p.target_step,
        p.behavior_step,
        0 if p.purpose == "fixed" else 1,
    )


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    manifests = root / "manifests"
    gsm8k_dir = root / "data" / "raw" / "gsm8k"
    model_dir = root / "models" / "qwen25_05b"
    split_registry_path = root / "data" / "splits" / "gsm8k_split_registry.parquet"

    env_manifest = verify_environment_manifest(manifests / "environment_manifest.json")
    data_manifest, gsm_record = verify_data_manifest(manifests / "data_manifest.json", gsm8k_dir)
    model_manifest, model_record = verify_model_manifest(manifests / "model_manifest.json", model_dir)
    split_manifest = verify_split_registry(
        manifests / "split_registry_manifest.json", split_registry_path, data_manifest, gsm_record
    )
    cfg, config_sha = load_protocol(cfg_path)
    pversion = protocol_version(cfg)
    target_steps = parse_target_steps(cfg)
    spec = parse_rescore_spec(cfg, args.batch_size)
    base_seeds = configured_seeds(cfg)
    dataset_revision = str(gsm_record.get("resolved_revision"))
    model_revision = str(model_record.get("resolved_revision"))
    tokenizer_revision = str(model_record.get("tokenizer_revision") or model_revision)
    chat_hash = str(((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256"))

    if args.mode == "paper":
        if tuple(base_seeds) != PAPER_SEEDS_DEFAULT:
            # The memo says at least 3; allowing >3 is scientifically legal, but
            # current paper registry design assumes the configured seeds. We do
            # not force exact numeric seeds, only at least three unique seeds.
            if len(base_seeds) < 3:
                raise Program04Error("Paper mode requires at least three GRPO training seeds.")
        lock = verify_protocol_lock(
            manifests / "protocol_lock.json",
            config_sha256=config_sha,
            data_manifest=data_manifest,
            model_record=model_record,
            split_manifest=split_manifest,
        )
        if tuple(target_steps) != PAPER_TARGET_STEPS:
            raise Program04Error("Paper target checkpoint grid must be 0,20,...,400.")
    else:
        lock = {}
        if args.split == "test":
            raise Program04Error("smoke/pilot modes are development-only; official test is forbidden.")

    if args.split == "test":
        if args.mode != "paper":
            raise Program04Error("Official test is legal only in paper mode.")
        verify_frozen_gate_for_test(root, lock)

    selected_seeds = list(base_seeds if args.mode == "paper" else (base_seeds[0],))
    if args.seed is not None:
        if args.seed not in selected_seeds:
            raise Program04Error(f"--seed {args.seed} is not legal for mode={args.mode}; expected one of {selected_seeds}")
        selected_seeds = [args.seed]

    # Verify Program 02 paths before pair registry construction.
    training: dict[int, dict[str, Any]] = {}
    adapters_by_seed: dict[int, Mapping[int, AdapterRecord]] = {}
    available: dict[int, list[int]] = {}
    for seed in selected_seeds:
        tm = verify_training_seed(
            root=root,
            mode=args.mode,
            seed=seed,
            model_record=model_record,
            data_manifest=data_manifest,
            split_manifest=split_manifest,
            config_sha256=config_sha,
        )
        training[seed] = tm
        adapters = tm.get("_verified_adapters")
        if not isinstance(adapters, Mapping):
            raise Program04Error(f"Internal verified adapters missing for seed {seed}")
        adapters_by_seed[seed] = adapters  # type: ignore[assignment]
        available[seed] = _available_steps(tm)

    # Pair registry in paper mode is defined for ALL configured seeds and BOTH
    # dev/test splits on the first development run, before any official-test
    # reward is observed. If --seed restricts processing, registry still remains
    # the complete frozen design.
    if args.mode == "paper":
        expected_registry = build_paper_pair_registry(seeds=base_seeds, pversion=pversion, target_steps=target_steps)
    else:
        expected_registry = build_engineering_pair_registry(
            seeds=selected_seeds, pversion=pversion, split="development", available_steps=available
        )
    registry, pair_manifest = ensure_pair_registry(
        root=root,
        mode=args.mode,
        split=args.split,
        expected_records=expected_registry,
        config_sha256=config_sha,
        dataset_revision=dataset_revision,
        model_revision=model_revision,
    )

    selected_pairs = [p for p in registry if p.split == args.split and p.training_seed in selected_seeds]
    selected_pairs.sort(key=lambda p: (p.training_seed,) + _pair_sort_key(p))
    if not selected_pairs:
        raise Program04Error("Pair registry contains no pairs for requested mode/split/seeds.")

    # Every referenced behavior/target adapter must exist.
    for seed in selected_seeds:
        required = _required_steps_for_pairs(selected_pairs, seed)
        have = set(adapters_by_seed[seed].keys())
        missing = sorted(required - have)
        if missing:
            raise Program04Error(f"Seed {seed} lacks Program 02 permanent adapters needed by pair registry: {missing}")

    behavior_index = verify_behavior_collection_index(
        root=root,
        mode=args.mode,
        split=args.split,
        config_sha256=config_sha,
        dataset_revision=dataset_revision,
        model_revision=model_revision,
        selected_seeds=selected_seeds,
    )

    droot = rescore_data_root(root, args.mode)
    mroot = rescore_manifest_root(root, args.mode, args.split)
    if args.reset_outputs:
        safe_reset([droot / "gsm8k" / f"split={args.split}", mroot])
    cleanup_staging(droot, args.resume)

    tokenizer = load_tokenizer(model_dir, model_record)
    source_units = source_unit_count_from_behavior_index(behavior_index, selected_pairs)
    print_header(
        root=root,
        mode=args.mode,
        split=args.split,
        cfg_path=cfg_path,
        config_sha=config_sha,
        pversion=pversion,
        dataset_revision=dataset_revision,
        model_revision=model_revision,
        seeds=selected_seeds,
        pair_count=len(selected_pairs),
        source_units=source_units,
        spec=spec,
        data_root=droot,
    )
    print(f"environment fingerprint : {env_manifest.get('environment_fingerprint_sha256')}")
    print(f"pair registry hash      : {pair_manifest.get('content_sha256')}")
    print(f"split registry hash     : {split_manifest.get('content_fingerprint_sha256')}")

    if args.verify_only:
        # Verify only existing rescore units; missing units are reported as an
        # error rather than silently loading models.
        missing = 0
        for p in selected_pairs:
            behavior_adapter = adapters_by_seed[p.training_seed][p.behavior_step]
            target_adapter = adapters_by_seed[p.training_seed][p.target_step]
            sources = discover_behavior_shards(
                root=root, mode=args.mode, split=p.split, seed=p.training_seed,
                behavior_step=p.behavior_step, behavior_adapter_sha256=behavior_adapter.payload_sha256,
            )
            for source in sources:
                unit = rescore_unit_dir(droot, pair=p, source=source)
                if not unit.exists():
                    missing += 1
                    continue
                srows = read_behavior_rows(source.parquet_path)
                # Existing units record their actual precision/fingerprint;
                # verify structural hashes first without guessing that precision.
                manifest = read_json(unit / "manifest.json")
                precision = str(manifest.get("scoring_precision"))
                expected_hash = scoring_engine_fingerprint(spec, precision)
                verify_rescore_unit(
                    unit, pair=p, source=source, source_rows=srows,
                    behavior_adapter_sha256=behavior_adapter.payload_sha256,
                    target_adapter_sha256=target_adapter.payload_sha256,
                    spec=spec, scoring_engine_sha256=expected_hash,
                )
        if missing:
            raise Program04Error(f"--verify-only found {missing} missing rescore shard units.")
        print("PROGRAM 04 VERIFY-ONLY PASSED")
        return 0

    # GLOBAL PHASE A: every selected identity pair creates and validates the
    # canonical behavior denominator before *any* distant target rescoring.
    all_summaries: list[dict[str, Any]] = []
    start = time.time()
    identity_pairs = sorted(
        [p for p in selected_pairs if p.purpose == "identity"],
        key=lambda p: (p.training_seed, p.target_step),
    )
    other_pairs = sorted(
        [p for p in selected_pairs if p.purpose != "identity"],
        key=lambda p: (p.training_seed, p.target_step, p.behavior_step, p.purpose),
    )
    print(f"\n[GLOBAL IDENTITY GATE] {len(identity_pairs)} pairs across seeds {selected_seeds}")
    for p in identity_pairs:
        seed = p.training_seed
        target_adapter = adapters_by_seed[seed][p.target_step]
        behavior_adapter = adapters_by_seed[seed][p.behavior_step]
        if target_adapter.payload_sha256 != behavior_adapter.payload_sha256:
            raise Program04Error(f"Identity pair b=e={p.target_step} resolved to different adapter hashes.")
        model, precision = load_target_model(
            model_dir=model_dir,
            adapter_dir=target_adapter.path,
            device=args.device,
            preferred_precision=PROBABILITY_EVIDENCE_PRECISION,
        )
        if precision != PROBABILITY_EVIDENCE_PRECISION:
            raise Program04Error(
                f"Probability-evidence precision contract violated: expected "
                f"{PROBABILITY_EVIDENCE_PRECISION}, got {precision}."
            )
        try:
            summary = process_pair(
                root=root, mode=args.mode, pair=p, model=model, scoring_precision=precision,
                tokenizer=tokenizer, target_adapter=target_adapter, behavior_adapter=behavior_adapter,
                spec=spec, dataset_revision=dataset_revision, model_revision=model_revision,
                tokenizer_revision=tokenizer_revision, chat_template_hash=chat_hash, resume=args.resume,
            )
            verify_identity_summary(summary, spec)
            all_summaries.append(summary)
            print(
                f"[IDENTITY PASS] seed={seed} step={p.target_step} rows={summary['rows']} "
                f"canonical_max_token_diff={summary['identity_max_abs_token_logprob_diff']:.6g} "
                f"canonical_max_seq_diff={summary['identity_max_abs_sequence_logprob_diff']:.6g} "
                f"generation_bridge_max_seq={summary['generation_bridge_max_abs_sequence_logprob_diff']:.6g} "
                f"bridge_rows_over_old_seq_atol={summary['generation_bridge_rows_over_original_sequence_atol']}"
            )
        finally:
            try:
                import torch  # type: ignore
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        rebuild_rescore_collection_index(
            root=root, mode=args.mode, split=args.split, manifest_root=mroot, pairs=selected_pairs,
            pair_registry_manifest=pair_manifest, config_sha256=config_sha,
            dataset_revision=dataset_revision, model_revision=model_revision, spec=spec,
        )

    print("\n[GLOBAL CANONICAL IDENTITY GATE PASSED] Canonical behavior denominators are frozen before distant scoring.")

    # GLOBAL PHASE B: distant pairs. Group by (seed,target) so the target model
    # is loaded once and can score both fixed-old and recent-rolling behavior
    # logs for the same target checkpoint.
    groups: dict[tuple[int, int], list[PairRecord]] = {}
    for p in other_pairs:
        groups.setdefault((p.training_seed, p.target_step), []).append(p)
    for seed, target_step in sorted(groups):
        target_adapter = adapters_by_seed[seed][target_step]
        print(f"\n[LOAD TARGET] seed={seed} target_step={target_step} adapter={target_adapter.path}")
        model, precision = load_target_model(
            model_dir=model_dir,
            adapter_dir=target_adapter.path,
            device=args.device,
            preferred_precision=PROBABILITY_EVIDENCE_PRECISION,
        )
        if precision != PROBABILITY_EVIDENCE_PRECISION:
            raise Program04Error(
                f"Probability-evidence precision contract violated: expected "
                f"{PROBABILITY_EVIDENCE_PRECISION}, got {precision}."
            )
        try:
            for p in sorted(groups[(seed, target_step)], key=lambda x: (x.behavior_step, x.purpose)):
                behavior_adapter = adapters_by_seed[seed][p.behavior_step]
                summary = process_pair(
                    root=root, mode=args.mode, pair=p, model=model, scoring_precision=precision,
                    tokenizer=tokenizer, target_adapter=target_adapter, behavior_adapter=behavior_adapter,
                    spec=spec, dataset_revision=dataset_revision, model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision, chat_template_hash=chat_hash, resume=args.resume,
                )
                all_summaries.append(summary)
                print(
                    f"[PAIR DONE] seed={seed} b={p.behavior_step} -> e={p.target_step} "
                    f"purpose={p.purpose} rows={summary['rows']}"
                )
        finally:
            try:
                import torch  # type: ignore
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        rebuild_rescore_collection_index(
            root=root, mode=args.mode, split=args.split, manifest_root=mroot, pairs=selected_pairs,
            pair_registry_manifest=pair_manifest, config_sha256=config_sha,
            dataset_revision=dataset_revision, model_revision=model_revision, spec=spec,
        )

    index_path = rebuild_rescore_collection_index(
        root=root, mode=args.mode, split=args.split, manifest_root=mroot, pairs=selected_pairs,
        pair_registry_manifest=pair_manifest, config_sha256=config_sha,
        dataset_revision=dataset_revision, model_revision=model_revision, spec=spec,
    )
    summary_path = mroot / "pair_summaries.json"
    atomic_write_json(
        summary_path,
        {
            "schema_version": RESCORE_SCHEMA,
            "manifest_type": "target_rescore_pair_summaries",
            "updated_at_utc": now_utc(),
            "mode": args.mode,
            "split": args.split,
            "pair_registry_content_sha256": pair_manifest.get("content_sha256"),
            "rescore_spec": asdict(spec),
            "scoring_contract": SCORING_CONTRACT,
            "summaries": all_summaries,
            "summary_set_sha256": sha256_bytes(canonical_bytes(all_summaries)),
        },
    )
    elapsed = time.time() - start
    print("\n" + "=" * 78)
    print("PROGRAM 04 PASSED")
    print(f"pairs processed/verified : {len(all_summaries)}")
    print(f"collection index         : {index_path}")
    print(f"pair summaries           : {summary_path}")
    print(f"elapsed                  : {elapsed/60:.1f} min")
    print("All canonical identity gates passed before non-identity OPE rescoring.")
    print("=" * 78)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except Program04Error as exc:
        print(f"\nPROGRAM 04 FAILED\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nPROGRAM 04 INTERRUPTED. Published shards are safe; rerun with --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
