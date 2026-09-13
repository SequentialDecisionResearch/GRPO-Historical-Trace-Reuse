#!/usr/bin/env python3
"""GRPO-OPE Program 03: collect immutable behavior-policy trajectory logs.

Research boundary
-----------------
Program 03 turns the frozen GRPO checkpoints produced by Program 02 into the
logged reasoning traces used as the denominator of all later OPE calculations.
It does NOT rescore trajectories under target checkpoints, compute IS/pWIS/DR,
generate on-policy references, fit a reuse gate, or make paper figures.

The paper-level object created here is, for behavior checkpoint b and prompt x_j,

    Y^(b)_{j,k} ~ mu_b(. | x_j),  k = 1,...,K,

with the exact generated token path and the exact token-level behavior-policy
log probability saved at generation time.

Paper invariants
----------------
* Program 00 data/model/environment manifests are re-verified.
* Program 01's immutable split registry is re-verified.
* Program 02's completed training manifests and permanent LoRA adapters are
  re-verified before any generation.
* Main behavior anchors are b in {0,100,200,300}.
* Development is generated first. Official test generation is forbidden until
  protocol_lock.json exists AND a frozen gate hash can be verified.
* Sampling is explicit and auditable (paper default: temperature=1, top_p=1,
  top_k=0, repetition_penalty=1).
* Prompt token IDs are produced directly from the pinned tokenizer/chat template.
* Completion token IDs and token-level behavior log-probabilities are captured
  directly from generate(..., output_scores=True); strings are never re-tokenized
  to reconstruct probabilities.
* EOS is included in the sequence likelihood when generated. Padding after EOS
  is excluded. A max-length completion with no EOS is marked truncated.
* Stable trajectory IDs follow the code-design memo:

      hash(protocol_version, dataset_revision, model_revision, training_seed,
           behavior_step, split, prompt_id, sample_index)

* K is append-only. If development grows from K=16 to K=32, only the new sample
  indices are generated; existing trajectories are never overwritten.
* Each prompt shard is published by staging-directory -> fsync -> atomic rename,
  with an immutable shard manifest and SHA-256 content provenance.
* Resume skips only units whose Parquet and semantic hashes verify.

Engineering modes
-----------------
--mode smoke and --mode pilot are isolated from paper assets.  They use the
matching Program 02 engineering checkpoints, small development subsets, and
small K. They never touch official test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
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
from typing import Any, Iterable, Mapping, Sequence


PROGRAM = "03_collect_behavior_logs.py"
PROGRAM_VERSION = "1.1.0"
MANIFEST_SCHEMA = "1.0"
BEHAVIOR_SCHEMA = "1.0"
PROBABILITY_EVIDENCE_PRECISION = "fp32"

PROJECT_NAME = "grpo_ope_reuse"
GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

EXPECTED_RESEARCH_ROWS = {"training": 6000, "development": 1473, "test": 1319}
PAPER_ANCHORS = (0, 100, 200, 300)
PAPER_ALLOWED_K = (16, 32)
DEFAULT_SAMPLE_BLOCK_SIZE = 16
DEFAULT_GENERATION_BATCH_SIZE = 4
DEFAULT_PROMPTS_PER_SHARD = 8

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
HEX64_RE = re.compile(r"[0-9a-f]{64}")

ROW_COLUMNS = (
    "trajectory_id",
    "dataset",
    "dataset_revision",
    "training_seed",
    "behavior_step",
    "split",
    "prompt_id",
    "source_row_index",
    "sample_index",
    "prompt_token_ids",
    "completion_token_ids",
    "behavior_token_logprobs",
    "behavior_sequence_logprob",
    "completion_length",
    "terminated_with_eos",
    "was_truncated",
    "parsed_answer",
    "correctness_reward",
    "parser_status",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_completion_length",
    "generation_seed",
    "generation_call_id",
    "model_revision",
    "tokenizer_revision",
    "chat_template_hash",
    "prompt_template_hash",
    "behavior_adapter_sha256",
    "generation_config_sha256",
    "behavior_logprob_source",
    "completion_text",
)


class Program03Error(RuntimeError):
    """Controlled Program 03 failure with an actionable message."""


@dataclass(frozen=True)
class BehaviorSpec:
    anchors: tuple[int, ...]
    k_main: int
    sample_block_size: int
    generation_batch_size: int
    prompts_per_shard: int
    max_completion_length: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    original_split: str
    research_split: str
    source_row_index: int
    question: str
    raw_gold_answer: str
    gold_answer: str


# ---------------------------------------------------------------------------
# Generic integrity utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


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
        raise Program03Error(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program03Error(f"Expected a JSON object in {path}.")
    return obj


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def fsync_directory(path: Path) -> None:
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


def payload_files(root: Path) -> list[Path]:
    if not root.exists():
        raise Program03Error(f"Missing asset directory: {root}")
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if parts & HASH_EXCLUDE_DIRS:
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def tree_hash(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    for p in payload_files(root):
        size = p.stat().st_size
        records.append({"path": rel(p, root), "size_bytes": size, "sha256": sha256_file(p)})
        total += size
    if not records:
        raise Program03Error(f"No files found in asset directory: {root}")
    return {
        "tree_sha256": sha256_bytes(canonical_bytes(records)),
        "file_count": len(records),
        "total_bytes": total,
    }


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
        raise Program03Error(f"Adapter contains no payload files: {root}")
    return sha256_bytes(canonical_bytes(records))


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
        print("[WARN] Could not mark every shard file read-only; hashes remain authoritative.")
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


def package_version(name: str) -> str | None:
    aliases = {"torch": "torch", "PyYAML": "PyYAML"}
    try:
        return metadata.version(aliases.get(name, name))
    except metadata.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Program 00 / 01 integrity verification
# ---------------------------------------------------------------------------


def current_environment_fingerprint() -> tuple[str, dict[str, Any]]:
    packages = {p: package_version(p) for p in CORE_PACKAGES + EXTRA_PACKAGES}
    missing = [p for p in CORE_PACKAGES if packages[p] is None]
    if missing:
        raise Program03Error(
            "Current environment is missing packages frozen/required by Program 00: " + ", ".join(missing)
        )
    if packages.get("pyarrow") is None:
        raise Program03Error("pyarrow is required for immutable Parquet behavior shards.")
    try:
        import torch  # type: ignore

        torch_cuda_build = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        raise Program03Error(f"Cannot inspect PyTorch environment: {exc}") from exc
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
        raise Program03Error(f"Missing {path}. Program 00 must complete first.")
    env = read_json(path)
    if env.get("manifest_type") != "environment":
        raise Program03Error(f"Wrong manifest type in {path}.")
    expected = env.get("environment_fingerprint_sha256")
    observed, _ = current_environment_fingerprint()
    if expected != observed:
        raise Program03Error(
            "Current software environment differs from Program 00's frozen environment.\n"
            f"expected={expected}\nobserved={observed}"
        )
    return env


def verify_data_manifest(path: Path, gsm8k_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program03Error(f"Missing {path}. Program 00 must freeze GSM8K first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "data":
        raise Program03Error(f"Invalid Program 00 data manifest: {path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("gsm8k"), dict):
        raise Program03Error("data_manifest.json lacks datasets.gsm8k.")
    gsm = datasets["gsm8k"]
    if gsm.get("repo_id") != GSM8K_REPO:
        raise Program03Error("Frozen primary dataset is not openai/gsm8k.")
    expected_fp = manifest.get("content_fingerprint_sha256")
    observed_fp = sha256_bytes(canonical_bytes({"research_scope": manifest.get("research_scope"), "datasets": datasets}))
    if expected_fp != observed_fp:
        raise Program03Error("Program 00 data manifest content fingerprint mismatch.")
    observed_tree = tree_hash(gsm8k_dir)["tree_sha256"]
    if observed_tree != gsm.get("tree_sha256"):
        raise Program03Error(
            "Pinned GSM8K local files differ from Program 00.\n"
            f"expected={gsm.get('tree_sha256')}\nobserved={observed_tree}"
        )
    return manifest, gsm


def tokenizer_template_hash(model_dir: Path) -> tuple[str, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program03Error(f"Cannot load pinned tokenizer locally: {exc}") from exc
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program03Error("Pinned tokenizer has no chat_template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(text), tok.__class__.__name__


def verify_model_manifest(path: Path, model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program03Error(f"Missing {path}. Program 00 must freeze the model first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "model":
        raise Program03Error(f"Invalid Program 00 model manifest: {path}")
    models = manifest.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
        raise Program03Error("model_manifest.json lacks models.primary.")
    model = models["primary"]
    if model.get("repo_id") != MODEL_REPO:
        raise Program03Error(f"Frozen primary model is not {MODEL_REPO}.")
    if manifest.get("content_fingerprint_sha256") != sha256_bytes(canonical_bytes(models)):
        raise Program03Error("Program 00 model manifest content fingerprint mismatch.")
    observed_tree = tree_hash(model_dir)["tree_sha256"]
    if observed_tree != model.get("tree_sha256"):
        raise Program03Error(
            "Pinned model local files differ from Program 00.\n"
            f"expected={model.get('tree_sha256')}\nobserved={observed_tree}"
        )
    validation = model.get("validation") or {}
    tok_validation = validation.get("tokenizer") if isinstance(validation, dict) else None
    if not isinstance(tok_validation, dict):
        raise Program03Error("Frozen model manifest lacks tokenizer validation metadata.")
    observed_chat, tok_class = tokenizer_template_hash(model_dir)
    if observed_chat != tok_validation.get("chat_template_sha256"):
        raise Program03Error("Pinned tokenizer chat template differs from Program 00 manifest.")
    if tok_class != tok_validation.get("class"):
        raise Program03Error("Tokenizer class differs from Program 00 validation.")
    return manifest, model


def verify_split_registry(
    manifest_path: Path,
    registry_path: Path,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.exists() or not registry_path.exists():
        raise Program03Error("Program 01 split registry/manifest is missing.")
    m = read_json(manifest_path)
    if m.get("schema_version") != MANIFEST_SCHEMA or m.get("manifest_type") != "split_registry":
        raise Program03Error("Invalid Program 01 split registry manifest.")
    source = m.get("source")
    registry = m.get("registry")
    firewall = m.get("research_firewall")
    if not all(isinstance(x, dict) for x in (source, registry, firewall)):
        raise Program03Error("Split registry manifest is missing required sections.")
    if source.get("dataset_revision") != gsm_record.get("resolved_revision"):
        raise Program03Error("Split registry dataset revision differs from Program 00.")
    if source.get("gsm8k_tree_sha256") != gsm_record.get("tree_sha256"):
        raise Program03Error("Split registry source tree differs from Program 00.")
    if source.get("program00_data_manifest_content_fingerprint_sha256") != data_manifest.get("content_fingerprint_sha256"):
        raise Program03Error("Split registry was created from a different Program 00 data manifest.")
    if firewall.get("official_test_is_never_used_for_split_tuning") is not True:
        raise Program03Error("Split registry does not preserve the official-test firewall.")
    if registry.get("file_sha256") != sha256_file(registry_path):
        raise Program03Error("Immutable split registry file SHA-256 mismatch.")
    try:
        import pyarrow.parquet as pq  # type: ignore

        rows = pq.read_table(registry_path).to_pylist()
    except Exception as exc:
        raise Program03Error(f"Cannot read immutable split registry: {exc}") from exc
    counts = {"training": 0, "development": 0, "test": 0}
    seen: set[str] = set()
    for r in rows:
        split = r.get("research_split")
        if split not in counts:
            raise Program03Error(f"Unexpected research_split={split!r} in registry.")
        counts[str(split)] += 1
        pid = r.get("prompt_id")
        if not isinstance(pid, str) or HEX64_RE.fullmatch(pid) is None or pid in seen:
            raise Program03Error("Split registry contains invalid/duplicate prompt_id.")
        seen.add(pid)
        if r.get("original_split") == "test" and split != "test":
            raise Program03Error("Official GSM8K test row leaked into another research split.")
    if counts != EXPECTED_RESEARCH_ROWS:
        raise Program03Error(f"Unexpected split counts: {counts}; expected {EXPECTED_RESEARCH_ROWS}.")
    columns = registry.get("columns")
    if isinstance(columns, list) and columns:
        h = hashlib.sha256()
        for row in rows:
            try:
                canonical = {k: row[k] for k in columns}
            except KeyError as exc:
                raise Program03Error(f"Split registry missing canonical column {exc}.") from exc
            h.update(canonical_bytes(canonical))
            h.update(b"\n")
        if h.hexdigest() != registry.get("content_sha256"):
            raise Program03Error("Immutable split registry semantic content SHA-256 mismatch.")
    return m, rows


# ---------------------------------------------------------------------------
# Protocol and test firewall
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program03Error(f"Missing protocol config {path}.")
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise Program03Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program03Error("protocol.yaml must contain a YAML mapping.")
    project = cfg.get("project") or {}
    if not isinstance(project, dict):
        raise Program03Error("protocol.yaml project must be a mapping.")
    if project.get("name") not in (None, PROJECT_NAME):
        raise Program03Error(f"Unexpected project.name={project.get('name')!r}.")
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


def parse_behavior_spec(cfg: Mapping[str, Any]) -> BehaviorSpec:
    behavior = cfg.get("behavior") or {}
    training = cfg.get("training") or {}
    if not isinstance(behavior, Mapping) or not isinstance(training, Mapping):
        raise Program03Error("protocol.yaml behavior/training sections must be mappings.")

    raw_anchors = behavior.get("anchors", list(PAPER_ANCHORS))
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise Program03Error("behavior.anchors must be a non-empty list of integer checkpoint steps.")
    anchors: list[int] = []
    for x in raw_anchors:
        if isinstance(x, bool) or not isinstance(x, int) or x < 0:
            raise Program03Error("behavior.anchors must contain non-negative integers.")
        anchors.append(x)
    if len(set(anchors)) != len(anchors):
        raise Program03Error("behavior.anchors contains duplicates.")

    def i(name: str, default: int) -> int:
        v = behavior.get(name, default)
        if isinstance(v, bool) or not isinstance(v, int):
            raise Program03Error(f"behavior.{name} must be an integer.")
        return v

    def f(name: str, default: float) -> float:
        v = behavior.get(name, training.get(name, default))
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise Program03Error(f"behavior.{name} must be numeric.")
        return float(v)

    k_main = i("K_main", 16)
    sample_block_size = i("sample_block_size", DEFAULT_SAMPLE_BLOCK_SIZE)
    generation_batch_size = i("generation_batch_size", DEFAULT_GENERATION_BATCH_SIZE)
    prompts_per_shard = i("prompts_per_shard", DEFAULT_PROMPTS_PER_SHARD)
    mcl_raw = behavior.get("max_completion_length", training.get("max_completion_length", 128))
    if isinstance(mcl_raw, bool) or not isinstance(mcl_raw, int):
        raise Program03Error("behavior.max_completion_length/training.max_completion_length must be an integer.")

    spec = BehaviorSpec(
        anchors=tuple(anchors),
        k_main=k_main,
        sample_block_size=sample_block_size,
        generation_batch_size=generation_batch_size,
        prompts_per_shard=prompts_per_shard,
        max_completion_length=mcl_raw,
        temperature=f("temperature", 1.0),
        top_p=f("top_p", 1.0),
        top_k=int(f("top_k", 0)),
        repetition_penalty=f("repetition_penalty", 1.0),
    )
    if spec.k_main <= 0 or spec.sample_block_size <= 0 or spec.generation_batch_size <= 0 or spec.prompts_per_shard <= 0:
        raise Program03Error("K/sample_block_size/generation_batch_size/prompts_per_shard must be positive.")
    if spec.max_completion_length <= 0:
        raise Program03Error("max_completion_length must be positive.")
    if spec.generation_batch_size > spec.sample_block_size:
        raise Program03Error("generation_batch_size may not exceed sample_block_size.")
    if spec.temperature <= 0 or not (0 < spec.top_p <= 1) or spec.top_k < 0 or spec.repetition_penalty <= 0:
        raise Program03Error("Invalid behavior sampling configuration.")
    return spec


def behavior_spec_fingerprint(spec: BehaviorSpec) -> str:
    return sha256_bytes(canonical_bytes(asdict(spec)))


def validate_paper_contract(spec: BehaviorSpec) -> None:
    if spec.anchors != PAPER_ANCHORS:
        raise Program03Error(f"Paper behavior anchors must be {list(PAPER_ANCHORS)}, observed {list(spec.anchors)}.")
    if spec.k_main not in PAPER_ALLOWED_K:
        raise Program03Error(f"Paper K_main must be one of {PAPER_ALLOWED_K}, observed {spec.k_main}.")
    if not math.isclose(spec.temperature, 1.0) or not math.isclose(spec.top_p, 1.0):
        raise Program03Error("Paper behavior sampling requires temperature=1 and top_p=1.")
    if spec.top_k != 0 or not math.isclose(spec.repetition_penalty, 1.0):
        raise Program03Error("Paper behavior sampling requires top_k=0 and repetition_penalty=1.")


def verify_protocol_lock(
    path: Path,
    *,
    config_sha256: str,
    data_manifest: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    behavior_spec: BehaviorSpec,
) -> dict[str, Any]:
    if not path.exists():
        raise Program03Error(f"Paper mode requires frozen {path}.")
    lock = read_json(path)
    locked_config = first_present(
        lock,
        (("config_sha256",), ("protocol_config_sha256",), ("protocol", "config_sha256"), ("inputs", "config_sha256")),
    )
    if not isinstance(locked_config, str):
        raise Program03Error("protocol_lock.json must contain a frozen config SHA-256.")
    if locked_config != config_sha256:
        raise Program03Error(
            "protocol.yaml differs from protocol_lock.json.\n"
            f"locked={locked_config}\nobserved={config_sha256}"
        )
    locked_data = first_present(
        lock,
        (("data_manifest_sha256",), ("inputs", "data_manifest_sha256"), ("data", "content_fingerprint_sha256")),
    )
    if locked_data is not None and locked_data != data_manifest.get("content_fingerprint_sha256"):
        raise Program03Error("Program 00 data manifest differs from protocol lock.")
    model_record = ((model_manifest.get("models") or {}).get("primary") or {})
    locked_model = first_present(
        lock,
        (("model_sha",), ("model_revision",), ("model", "resolved_revision"), ("inputs", "model_revision")),
    )
    if locked_model is not None and locked_model != model_record.get("resolved_revision"):
        raise Program03Error("Pinned Qwen revision differs from protocol lock.")
    locked_split = first_present(
        lock,
        (("split_registry_sha256",), ("inputs", "split_registry_sha256"), ("split", "content_fingerprint_sha256")),
    )
    if locked_split is not None and locked_split != split_manifest.get("content_fingerprint_sha256"):
        raise Program03Error("Program 01 split registry differs from protocol lock.")
    locked_behavior = first_present(
        lock,
        (("behavior_config_sha256",), ("behavior", "resolved_config_sha256"), ("frozen", "behavior_config_sha256")),
    )
    if locked_behavior is not None and locked_behavior != behavior_spec_fingerprint(behavior_spec):
        raise Program03Error("Resolved behavior configuration differs from protocol lock.")
    return lock


def verify_frozen_gate_for_test(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    gate_path = root / "outputs" / "frozen_gate.json"
    if not gate_path.exists():
        raise Program03Error(
            "Official test generation is forbidden before outputs/frozen_gate.json exists. "
            "Calibrate/freeze the gate on development first."
        )
    observed = sha256_file(gate_path)
    expected = first_present(
        lock,
        (("frozen_gate_sha256",), ("gate", "file_sha256"), ("inputs", "frozen_gate_sha256"), ("frozen", "gate_sha256")),
    )
    sidecar_path = root / "manifests" / "frozen_gate_manifest.json"
    sidecar: dict[str, Any] | None = None
    if expected is None and sidecar_path.exists():
        sidecar = read_json(sidecar_path)
        expected = first_present(sidecar, (("gate_file_sha256",), ("file_sha256",), ("gate", "file_sha256")))
    if not isinstance(expected, str) or HEX64_RE.fullmatch(expected) is None:
        raise Program03Error(
            "Test mode must verify the frozen gate hash, but neither protocol_lock.json nor "
            "manifests/frozen_gate_manifest.json contains a valid gate file SHA-256."
        )
    if observed != expected:
        raise Program03Error(f"Frozen gate hash mismatch: expected={expected}, observed={observed}")
    return {"path": str(gate_path), "sha256": observed, "sidecar": sidecar}


# ---------------------------------------------------------------------------
# Program 02 training-path verification
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
        raise Program03Error(f"Missing completed Program 02 training manifest: {path}")
    m = read_json(path)
    if m.get("manifest_type") != "grpo_training_seed" or m.get("status") != "complete":
        raise Program03Error(f"Program 02 seed manifest is not complete: {path}")
    if m.get("training_seed") != seed or m.get("mode") != mode:
        raise Program03Error(f"Program 02 seed manifest metadata mismatch: {path}")
    inputs = m.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise Program03Error(f"Program 02 seed manifest lacks inputs: {path}")
    if inputs.get("dataset_revision") != split_manifest.get("source", {}).get("dataset_revision"):
        raise Program03Error(f"Seed {seed} was trained against a different dataset revision.")
    if inputs.get("data_manifest_content_fingerprint_sha256") != data_manifest.get("content_fingerprint_sha256"):
        raise Program03Error(f"Seed {seed} was trained against a different data manifest.")
    if inputs.get("split_registry_content_fingerprint_sha256") != split_manifest.get("content_fingerprint_sha256"):
        raise Program03Error(f"Seed {seed} was trained against a different split registry.")
    if inputs.get("model_revision") != model_record.get("resolved_revision"):
        raise Program03Error(f"Seed {seed} was trained from a different base model revision.")
    if inputs.get("chat_template_sha256") != ((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256"):
        raise Program03Error(f"Seed {seed} chat template differs from Program 00.")
    if inputs.get("prompt_template_sha256") != sha256_text(PROMPT_TEMPLATE):
        raise Program03Error(f"Seed {seed} prompt template differs from Program 03's frozen template.")
    if mode == "paper" and inputs.get("protocol_config_sha256") != config_sha256:
        raise Program03Error(f"Seed {seed} was trained under a different frozen protocol.yaml.")

    seed_root = checkpoint_seed_root(root, mode, seed)
    adapters = m.get("permanent_adapters")
    if not isinstance(adapters, list) or not adapters:
        raise Program03Error(f"Seed {seed} manifest lacks permanent adapters.")
    by_step: dict[int, dict[str, Any]] = {}
    for rec in adapters:
        if not isinstance(rec, dict) or "step" not in rec:
            continue
        step = int(rec["step"])
        p = seed_root / "adapters" / f"step_{step:04d}"
        if not p.exists():
            raise Program03Error(f"Missing permanent adapter: {p}")
        observed = adapter_payload_hash(p)
        if observed != rec.get("payload_sha256"):
            raise Program03Error(f"Permanent adapter hash mismatch: seed={seed}, step={step}")
        amp = p / "adapter_manifest.json"
        if not amp.exists():
            raise Program03Error(f"Permanent adapter lacks adapter_manifest.json: {p}")
        am = read_json(amp)
        if am.get("training_seed") != seed or am.get("step") != step:
            raise Program03Error(f"Adapter manifest metadata mismatch: {p}")
        if am.get("base_model_revision") != model_record.get("resolved_revision"):
            raise Program03Error(f"Adapter base revision mismatch: {p}")
        if am.get("adapter_payload_tree_sha256") != observed:
            raise Program03Error(f"Adapter sidecar hash mismatch: {p}")
        by_step[step] = {"path": p, "payload_sha256": observed, "manifest": am}
    out = dict(m)
    out["_verified_adapters"] = by_step
    return out


def configured_seeds(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    t = cfg.get("training") or {}
    raw = t.get("seeds", [20260826, 20260827, 20260828]) if isinstance(t, Mapping) else []
    if not isinstance(raw, list) or not raw:
        raise Program03Error("training.seeds must be a non-empty list.")
    seeds: list[int] = []
    for s in raw:
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise Program03Error("training.seeds must contain non-negative integers.")
        seeds.append(s)
    if len(set(seeds)) != len(seeds):
        raise Program03Error("training.seeds contains duplicates.")
    return tuple(seeds)


# ---------------------------------------------------------------------------
# GSM8K dereference and frozen parser/reward
# ---------------------------------------------------------------------------


def discover_gsm8k_parquet(root: Path) -> dict[str, list[Path]]:
    train = sorted((root / GSM8K_CONFIG).glob("train*.parquet"))
    test = sorted((root / GSM8K_CONFIG).glob("test*.parquet"))
    if not train:
        train = sorted(p for p in root.rglob("train*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts)
    if not test:
        test = sorted(p for p in root.rglob("test*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts)
    if not train or not test:
        raise Program03Error("Pinned GSM8K snapshot lacks main/train*.parquet or main/test*.parquet.")
    return {"train": train, "test": test}


def load_upstream_rows(gsm8k_dir: Path, original_split: str) -> list[dict[str, str]]:
    files = discover_gsm8k_parquet(gsm8k_dir)
    if original_split not in files:
        raise Program03Error(f"Unknown upstream split {original_split!r}.")
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            "parquet",
            data_files={original_split: [str(p) for p in files[original_split]]},
            split=original_split,
        )
    except Exception as exc:
        raise Program03Error(f"Cannot load pinned GSM8K {original_split} Parquet locally: {exc}") from exc
    expected = 7473 if original_split == "train" else 1319
    if len(ds) != expected:
        raise Program03Error(f"Pinned GSM8K {original_split}: expected {expected} rows, observed {len(ds)}.")
    rows: list[dict[str, str]] = []
    for x in ds:
        q, a = x.get("question"), x.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            raise Program03Error("GSM8K row lacks string question/answer fields.")
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


def canonical_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def extract_gsm8k_gold(answer_field: str) -> Fraction | None:
    match = GSM8K_GOLD_RE.search(answer_field)
    return None if match is None else normalize_numeric_text(match.group(1))


def parse_completion_answer(text: str) -> tuple[Fraction | None, str]:
    matches = ANSWER_RE.findall(text)
    if len(matches) == 0:
        return None, "missing_answer_tag"
    if len(matches) > 1:
        return None, "multiple_answer_tags"
    value = normalize_numeric_text(matches[0])
    if value is None:
        return None, "invalid_answer_value"
    return value, "ok"


def build_prompt_records(
    *,
    registry_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, str]],
    research_split: str,
    limit: int | None,
) -> list[PromptRecord]:
    if research_split not in {"development", "test"}:
        raise Program03Error("Program 03 behavior logs are defined only on development or official test prompts.")
    selected = [r for r in registry_rows if r.get("research_split") == research_split]
    selected.sort(key=lambda r: str(r.get("prompt_id")))
    if limit is not None:
        selected = selected[:limit]
    expected = EXPECTED_RESEARCH_ROWS[research_split] if limit is None else min(limit, EXPECTED_RESEARCH_ROWS[research_split])
    if len(selected) != expected:
        raise Program03Error(f"Expected {expected} {research_split} prompts, found {len(selected)}.")

    expected_original = "test" if research_split == "test" else "train"
    records: list[PromptRecord] = []
    for r in selected:
        if r.get("original_split") != expected_original:
            raise Program03Error(
                f"No-test-leakage/source mismatch: research_split={research_split} contains original_split={r.get('original_split')}"
            )
        idx = r.get("source_row_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= len(raw_rows):
            raise Program03Error("Split registry contains invalid source_row_index.")
        raw = raw_rows[idx]
        q, a = raw["question"], raw["answer"]
        if sha256_text(q) != r.get("question_hash"):
            raise Program03Error(f"Question hash mismatch for prompt_id={r.get('prompt_id')}.")
        if sha256_text(a) != r.get("gold_answer_hash"):
            raise Program03Error(f"Gold answer hash mismatch for prompt_id={r.get('prompt_id')}.")
        gold = extract_gsm8k_gold(a)
        if gold is None:
            raise Program03Error(f"Cannot parse GSM8K gold answer for prompt_id={r.get('prompt_id')}.")
        records.append(
            PromptRecord(
                prompt_id=str(r["prompt_id"]),
                original_split=expected_original,
                research_split=research_split,
                source_row_index=idx,
                question=q,
                raw_gold_answer=a,
                gold_answer=canonical_fraction(gold),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Stable IDs, sample blocks, prompt tokenization
# ---------------------------------------------------------------------------


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


def trajectory_id(
    *,
    protocol_version_value: str,
    dataset_revision: str,
    model_revision: str,
    training_seed: int,
    behavior_step: int,
    split: str,
    prompt_id: str,
    sample_index: int,
) -> str:
    basis = {
        "protocol_version": protocol_version_value,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "training_seed": training_seed,
        "behavior_step": behavior_step,
        "split": split,
        "prompt_id": prompt_id,
        "sample_index": sample_index,
    }
    return sha256_bytes(canonical_bytes(basis))


def sample_blocks(target_k: int, block_size: int) -> list[tuple[int, int]]:
    if target_k <= 0 or block_size <= 0:
        raise Program03Error("target_k and block_size must be positive.")
    return [(start, min(target_k, start + block_size)) for start in range(0, target_k, block_size)]


def generation_subblocks(start: int, end: int, batch_size: int) -> list[tuple[int, int]]:
    if not (0 <= start < end) or batch_size <= 0:
        raise Program03Error("Invalid generation subblock bounds.")
    return [(s, min(end, s + batch_size)) for s in range(start, end, batch_size)]


def prompt_shards(prompts: Sequence[PromptRecord], prompts_per_shard: int) -> list[list[PromptRecord]]:
    return [list(prompts[i : i + prompts_per_shard]) for i in range(0, len(prompts), prompts_per_shard)]


def load_tokenizer(model_dir: Path, model_record: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program03Error(f"Cannot load pinned tokenizer: {exc}") from exc
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise Program03Error("Tokenizer has neither pad_token_id nor eos_token_id.")
        tok.pad_token = tok.eos_token
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program03Error("Pinned tokenizer has no chat template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    expected = ((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256")
    if sha256_text(text) != expected:
        raise Program03Error("Loaded tokenizer chat template hash differs from Program 00.")
    return tok


def prompt_content(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def render_prompt_ids(tokenizer: Any, question: str) -> tuple[list[int], str]:
    messages = [{"role": "user", "content": prompt_content(question)}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise Program03Error("Unexpected batched prompt IDs from single conversation.")
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise Program03Error("Tokenizer.apply_chat_template did not return a flat non-empty token ID list.")
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not isinstance(rendered, str):
        rendered = str(rendered)
    return [int(x) for x in ids], rendered


# ---------------------------------------------------------------------------
# Model loading and exact generation-time behavior log probabilities
# ---------------------------------------------------------------------------


def resolve_dtype(device: str, preferred: str | None) -> tuple[Any, str]:
    """Return the fixed OPE probability-evidence dtype.

    GRPO training precision is intentionally decoupled from the precision used
    to create likelihood-ratio evidence.  The FP32 diagnostic showed that BF16
    generation-time scores and full-sequence teacher forcing can differ
    materially on the current CUDA stack, while FP32 reduces the discrepancy
    far below the predeclared identity tolerances.  Therefore Programs 03/04
    use FP32 for probability evidence even when Program 02 trained in BF16.

    ``preferred`` is retained only for backward-compatible call signatures and
    is deliberately ignored.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program03Error(f"Cannot import torch: {exc}") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise Program03Error("--device cuda requested but torch.cuda.is_available() is False.")
    return torch.float32, PROBABILITY_EVIDENCE_PRECISION


def load_behavior_model(
    *,
    model_dir: Path,
    adapter_dir: Path,
    device: str,
    preferred_precision: str | None,
) -> tuple[Any, str]:
    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as exc:
        raise Program03Error(f"Cannot import inference stack: {exc}") from exc
    dtype, resolved = resolve_dtype(device, preferred_precision)
    try:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False, dtype=dtype
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False, torch_dtype=dtype
            )
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        model.to(device)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad_(False)
        return model, resolved
    except Exception as exc:
        raise Program03Error(f"Cannot load base model + LoRA adapter {adapter_dir}: {exc}") from exc


def eos_ids(tokenizer: Any) -> set[int]:
    raw = tokenizer.eos_token_id
    if raw is None:
        raise Program03Error("Tokenizer has no eos_token_id; OPE termination semantics would be undefined.")
    if isinstance(raw, int):
        return {int(raw)}
    if isinstance(raw, (list, tuple, set)) and raw:
        return {int(x) for x in raw}
    raise Program03Error(f"Unsupported eos_token_id value: {raw!r}")


def completion_cut_length(tokens: Sequence[int], eos_set: set[int], n_steps: int, max_new_tokens: int) -> tuple[int, bool, bool]:
    if n_steps < 0 or n_steps > len(tokens):
        raise Program03Error("Generated token/scores alignment is invalid.")
    for i, token in enumerate(tokens[:n_steps]):
        if int(token) in eos_set:
            return i + 1, True, False
    if n_steps == max_new_tokens:
        return n_steps, False, True
    raise Program03Error(
        "Generation ended before max_new_tokens without producing EOS. "
        "Refusing to guess termination semantics."
    )


def generation_config_record(spec: BehaviorSpec, tokenizer: Any) -> dict[str, Any]:
    return {
        "generation_config_source": "fresh_transformers.GenerationConfig_not_model_generation_config",
        "do_sample": True,
        "num_beams": 1,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "repetition_penalty": spec.repetition_penalty,
        "max_new_tokens": spec.max_completion_length,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": sorted(eos_ids(tokenizer)),
        "bos_token_id": None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id),
        "return_dict_in_generate": True,
        "output_scores": True,
        "transition_score_normalization": True,
        "probability_evidence_precision": PROBABILITY_EVIDENCE_PRECISION,
        "logprob_definition": "selected-token log softmax of generation-time processed scores",
    }


def _seeded_generate(model: Any, input_ids: Any, attention_mask: Any, kwargs: Mapping[str, Any], seed: int, device: str) -> Any:
    import torch  # type: ignore
    try:
        from transformers import GenerationConfig  # type: ignore
    except Exception as exc:
        raise Program03Error(f"Cannot import transformers.GenerationConfig: {exc}") from exc

    # Use a fresh GenerationConfig instead of inheriting model.generation_config.
    # This prevents hidden model-card defaults (e.g. alternative sampling filters
    # or forced tokens) from silently changing the behavior policy.
    generation_config = GenerationConfig(**dict(kwargs))

    devices: list[int] = []
    if device == "cuda":
        devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )


def generate_prompt_block(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    sample_start: int,
    sample_end: int,
    spec: BehaviorSpec,
    protocol_version_value: str,
    dataset_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    training_seed: int,
    behavior_step: int,
    split: str,
    adapter_sha256: str,
    device: str,
) -> list[dict[str, Any]]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program03Error(f"Cannot import torch during generation: {exc}") from exc

    prompt_ids, rendered_prompt = render_prompt_ids(tokenizer, prompt.question)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(prompt_tensor)
    eos_set = eos_ids(tokenizer)
    gen_record = generation_config_record(spec, tokenizer)
    gen_hash = sha256_bytes(canonical_bytes(gen_record))
    chat_hash = sha256_text(
        tokenizer.chat_template
        if isinstance(tokenizer.chat_template, str)
        else json.dumps(tokenizer.chat_template, ensure_ascii=False, sort_keys=True, default=str)
    )
    prompt_template_hash = sha256_text(PROMPT_TEMPLATE)

    rows: list[dict[str, Any]] = []
    for sub_start, sub_end in generation_subblocks(sample_start, sample_end, spec.generation_batch_size):
        n = sub_end - sub_start
        call_seed = stable_seed(
            "behavior_generation",
            protocol_version_value,
            dataset_revision,
            model_revision,
            training_seed,
            behavior_step,
            split,
            prompt.prompt_id,
            sub_start,
            sub_end,
            gen_hash,
        )
        call_id = sha256_bytes(
            canonical_bytes(
                {
                    "seed": call_seed,
                    "prompt_id": prompt.prompt_id,
                    "sample_start": sub_start,
                    "sample_end": sub_end,
                    "behavior_step": behavior_step,
                    "training_seed": training_seed,
                    "split": split,
                }
            )
        )
        kwargs = {
            "do_sample": True,
            "num_beams": 1,
            "num_return_sequences": n,
            "max_new_tokens": spec.max_completion_length,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k,
            "repetition_penalty": spec.repetition_penalty,
            "pad_token_id": int(tokenizer.pad_token_id),
            "eos_token_id": sorted(eos_set) if len(eos_set) > 1 else next(iter(eos_set)),
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        outputs = _seeded_generate(model, prompt_tensor, attention, kwargs, call_seed, device)
        if not hasattr(outputs, "sequences") or not hasattr(outputs, "scores"):
            raise Program03Error("model.generate did not return sequences + scores; exact behavior logprobs unavailable.")
        scores_tuple = outputs.scores
        if scores_tuple is None:
            raise Program03Error("model.generate returned no scores despite output_scores=True.")
        n_steps = len(scores_tuple)
        if n_steps <= 0 or n_steps > spec.max_completion_length:
            raise Program03Error(f"Unexpected number of generation score steps: {n_steps}")
        try:
            transition = model.compute_transition_scores(outputs.sequences, scores_tuple, normalize_logits=True)
        except Exception as exc:
            raise Program03Error(f"compute_transition_scores failed: {exc}") from exc
        if int(outputs.sequences.shape[0]) != n or int(transition.shape[0]) != n:
            raise Program03Error("Generation returned an unexpected number of sequences/transition-score rows.")
        if int(transition.shape[1]) != n_steps:
            raise Program03Error(
                f"Transition-score length {int(transition.shape[1])} != generation score steps {n_steps}."
            )
        generated = outputs.sequences[:, len(prompt_ids) : len(prompt_ids) + n_steps]
        if int(generated.shape[1]) != n_steps:
            raise Program03Error("Prompt/completion token alignment failed during generation.")

        generated_cpu = generated.detach().to("cpu")
        transition_cpu = transition.detach().to(dtype=torch.float64, device="cpu")
        gold = Fraction(prompt.gold_answer)
        for local_i in range(n):
            sample_index = sub_start + local_i
            token_all = [int(x) for x in generated_cpu[local_i].tolist()]
            cut, ended_eos, truncated = completion_cut_length(token_all, eos_set, n_steps, spec.max_completion_length)
            comp_ids = token_all[:cut]
            logps = [float(x) for x in transition_cpu[local_i, :cut].tolist()]
            if len(comp_ids) != len(logps) or not comp_ids:
                raise Program03Error("Completion token/logprob alignment failed.")
            if not all(math.isfinite(x) for x in logps):
                raise Program03Error(
                    f"Non-finite behavior log-probability at seed={training_seed}, step={behavior_step}, "
                    f"prompt={prompt.prompt_id}, sample={sample_index}."
                )
            seq_logp = math.fsum(logps)
            text = tokenizer.decode(comp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parsed, parser_status = parse_completion_answer(text)
            parsed_str = None if parsed is None else canonical_fraction(parsed)
            correct = float(parsed is not None and parsed == gold)
            tid = trajectory_id(
                protocol_version_value=protocol_version_value,
                dataset_revision=dataset_revision,
                model_revision=model_revision,
                training_seed=training_seed,
                behavior_step=behavior_step,
                split=split,
                prompt_id=prompt.prompt_id,
                sample_index=sample_index,
            )
            rows.append(
                {
                    "trajectory_id": tid,
                    "dataset": "GSM8K",
                    "dataset_revision": dataset_revision,
                    "training_seed": training_seed,
                    "behavior_step": behavior_step,
                    "split": split,
                    "prompt_id": prompt.prompt_id,
                    "source_row_index": prompt.source_row_index,
                    "sample_index": sample_index,
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": comp_ids,
                    "behavior_token_logprobs": logps,
                    "behavior_sequence_logprob": seq_logp,
                    "completion_length": len(comp_ids),
                    "terminated_with_eos": ended_eos,
                    "was_truncated": truncated,
                    "parsed_answer": parsed_str,
                    "correctness_reward": correct,
                    "parser_status": parser_status,
                    "temperature": spec.temperature,
                    "top_p": spec.top_p,
                    "top_k": spec.top_k,
                    "repetition_penalty": spec.repetition_penalty,
                    "max_completion_length": spec.max_completion_length,
                    "generation_seed": call_seed,
                    "generation_call_id": call_id,
                    "model_revision": model_revision,
                    "tokenizer_revision": tokenizer_revision,
                    "chat_template_hash": chat_hash,
                    "prompt_template_hash": prompt_template_hash,
                    "behavior_adapter_sha256": adapter_sha256,
                    "generation_config_sha256": gen_hash,
                    "behavior_logprob_source": "generate_scores+compute_transition_scores(normalize_logits=True)",
                    "completion_text": text,
                }
            )
    if [r["sample_index"] for r in rows] != list(range(sample_start, sample_end)):
        raise Program03Error("Generated sample indices are not contiguous/deterministic.")
    return rows


# ---------------------------------------------------------------------------
# Immutable Parquet shard publication
# ---------------------------------------------------------------------------


def parquet_schema() -> Any:
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise Program03Error(f"Cannot import pyarrow: {exc}") from exc
    return pa.schema(
        [
            pa.field("trajectory_id", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("dataset_revision", pa.string(), nullable=False),
            pa.field("training_seed", pa.int64(), nullable=False),
            pa.field("behavior_step", pa.int32(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("source_row_index", pa.int64(), nullable=False),
            pa.field("sample_index", pa.int32(), nullable=False),
            pa.field("prompt_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("completion_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("behavior_token_logprobs", pa.list_(pa.float64()), nullable=False),
            pa.field("behavior_sequence_logprob", pa.float64(), nullable=False),
            pa.field("completion_length", pa.int32(), nullable=False),
            pa.field("terminated_with_eos", pa.bool_(), nullable=False),
            pa.field("was_truncated", pa.bool_(), nullable=False),
            pa.field("parsed_answer", pa.string(), nullable=True),
            pa.field("correctness_reward", pa.float64(), nullable=False),
            pa.field("parser_status", pa.string(), nullable=False),
            pa.field("temperature", pa.float64(), nullable=False),
            pa.field("top_p", pa.float64(), nullable=False),
            pa.field("top_k", pa.int32(), nullable=False),
            pa.field("repetition_penalty", pa.float64(), nullable=False),
            pa.field("max_completion_length", pa.int32(), nullable=False),
            pa.field("generation_seed", pa.int64(), nullable=False),
            pa.field("generation_call_id", pa.string(), nullable=False),
            pa.field("model_revision", pa.string(), nullable=False),
            pa.field("tokenizer_revision", pa.string(), nullable=False),
            pa.field("chat_template_hash", pa.string(), nullable=False),
            pa.field("prompt_template_hash", pa.string(), nullable=False),
            pa.field("behavior_adapter_sha256", pa.string(), nullable=False),
            pa.field("generation_config_sha256", pa.string(), nullable=False),
            pa.field("behavior_logprob_source", pa.string(), nullable=False),
            pa.field("completion_text", pa.string(), nullable=False),
        ]
    )


def row_for_hash(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ROW_COLUMNS}


def rows_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (str(r["prompt_id"]), int(r["sample_index"]))):
        h.update(canonical_bytes(row_for_hash(row)))
        h.update(b"\n")
    return h.hexdigest()


def validate_behavior_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_prompts: Sequence[PromptRecord],
    sample_start: int,
    sample_end: int,
    training_seed: int,
    behavior_step: int,
    split: str,
    protocol_version_value: str,
    dataset_revision: str,
    model_revision: str,
    adapter_sha256: str,
) -> dict[str, Any]:
    expected_count = len(expected_prompts) * (sample_end - sample_start)
    if len(rows) != expected_count:
        raise Program03Error(f"Behavior shard row count mismatch: expected {expected_count}, observed {len(rows)}.")
    expected_ids = {p.prompt_id for p in expected_prompts}
    seen_keys: set[tuple[str, int]] = set()
    parser_counts: dict[str, int] = {}
    trunc = 0
    eos = 0
    correct = 0.0
    for r in rows:
        pid = r.get("prompt_id")
        idx = r.get("sample_index")
        if pid not in expected_ids or isinstance(idx, bool) or not isinstance(idx, int) or not (sample_start <= idx < sample_end):
            raise Program03Error("Behavior shard contains an unexpected prompt/sample index.")
        key = (str(pid), int(idx))
        if key in seen_keys:
            raise Program03Error("Behavior shard contains a duplicate prompt/sample key.")
        seen_keys.add(key)
        if r.get("training_seed") != training_seed or r.get("behavior_step") != behavior_step or r.get("split") != split:
            raise Program03Error("Behavior shard seed/step/split metadata mismatch.")
        if r.get("dataset_revision") != dataset_revision or r.get("model_revision") != model_revision:
            raise Program03Error("Behavior shard data/model revision mismatch.")
        if r.get("behavior_adapter_sha256") != adapter_sha256:
            raise Program03Error("Behavior shard adapter hash mismatch.")
        expected_tid = trajectory_id(
            protocol_version_value=protocol_version_value,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
            training_seed=training_seed,
            behavior_step=behavior_step,
            split=split,
            prompt_id=str(pid),
            sample_index=int(idx),
        )
        if r.get("trajectory_id") != expected_tid:
            raise Program03Error("Behavior shard trajectory_id is not reproducible from the stable key.")
        pt = r.get("prompt_token_ids")
        ct = r.get("completion_token_ids")
        lp = r.get("behavior_token_logprobs")
        if not isinstance(pt, list) or not pt or not isinstance(ct, list) or not ct or not isinstance(lp, list):
            raise Program03Error("Behavior shard lacks token ID/logprob lists.")
        if len(ct) != len(lp) or len(ct) != r.get("completion_length"):
            raise Program03Error("Completion token/logprob/length mismatch in behavior shard.")
        if not all(isinstance(x, int) for x in pt + ct):
            raise Program03Error("Token ID list contains a non-integer.")
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in lp):
            raise Program03Error("Token logprob list contains a non-finite/non-numeric value.")
        seq = math.fsum(float(x) for x in lp)
        if not math.isclose(seq, float(r.get("behavior_sequence_logprob")), rel_tol=1e-12, abs_tol=1e-10):
            raise Program03Error("behavior_sequence_logprob does not equal the float64 sum of token logprobs.")
        if bool(r.get("terminated_with_eos")) and bool(r.get("was_truncated")):
            raise Program03Error("A trajectory cannot be both EOS-terminated and max-length truncated.")
        parser_status = str(r.get("parser_status"))
        parser_counts[parser_status] = parser_counts.get(parser_status, 0) + 1
        trunc += int(bool(r.get("was_truncated")))
        eos += int(bool(r.get("terminated_with_eos")))
        correct += float(r.get("correctness_reward"))
    return {
        "row_count": len(rows),
        "prompt_count": len(expected_prompts),
        "sample_count_per_prompt": sample_end - sample_start,
        "unique_prompt_sample_keys": len(seen_keys),
        "parser_status_counts": dict(sorted(parser_counts.items())),
        "truncation_rate": trunc / len(rows) if rows else None,
        "eos_termination_rate": eos / len(rows) if rows else None,
        "mean_correctness_reward": correct / len(rows) if rows else None,
    }


def unit_dir_path(
    data_root: Path,
    *,
    split: str,
    seed: int,
    behavior_step: int,
    sample_start: int,
    sample_end: int,
    shard_index: int,
) -> Path:
    return (
        data_root
        / "gsm8k"
        / f"split={split}"
        / f"seed={seed}"
        / f"behavior_step={behavior_step:04d}"
        / f"sample_block={sample_start:04d}-{sample_end - 1:04d}"
        / f"shard={shard_index:05d}"
    )


def verify_shard_unit(
    unit_dir: Path,
    *,
    root: Path,
    expected_prompts: Sequence[PromptRecord],
    sample_start: int,
    sample_end: int,
    training_seed: int,
    behavior_step: int,
    split: str,
    protocol_version_value: str,
    dataset_revision: str,
    model_revision: str,
    adapter_sha256: str,
    behavior_config_sha256: str,
) -> dict[str, Any]:
    parquet_path = unit_dir / "trajectories.parquet"
    manifest_path = unit_dir / "manifest.json"
    if not parquet_path.exists() or not manifest_path.exists():
        raise Program03Error(f"Incomplete published behavior shard unit: {unit_dir}")
    m = read_json(manifest_path)
    if m.get("schema_version") != BEHAVIOR_SCHEMA or m.get("manifest_type") != "behavior_trajectory_shard":
        raise Program03Error(f"Invalid behavior shard manifest: {manifest_path}")
    if m.get("behavior_config_sha256") != behavior_config_sha256:
        raise Program03Error(f"Existing behavior shard was generated under a different behavior config: {unit_dir}")
    if m.get("probability_evidence_precision") != PROBABILITY_EVIDENCE_PRECISION:
        raise Program03Error(
            f"Existing behavior shard does not use the required "
            f"{PROBABILITY_EVIDENCE_PRECISION} probability-evidence precision: {unit_dir}"
        )
    if m.get("parquet", {}).get("file_sha256") != sha256_file(parquet_path):
        raise Program03Error(f"Behavior Parquet file hash mismatch: {parquet_path}")
    try:
        import pyarrow.parquet as pq  # type: ignore

        rows = pq.read_table(parquet_path).to_pylist()
    except Exception as exc:
        raise Program03Error(f"Cannot read behavior shard {parquet_path}: {exc}") from exc
    validation = validate_behavior_rows(
        rows,
        expected_prompts=expected_prompts,
        sample_start=sample_start,
        sample_end=sample_end,
        training_seed=training_seed,
        behavior_step=behavior_step,
        split=split,
        protocol_version_value=protocol_version_value,
        dataset_revision=dataset_revision,
        model_revision=model_revision,
        adapter_sha256=adapter_sha256,
    )
    observed_content = rows_content_sha256(rows)
    if observed_content != m.get("parquet", {}).get("content_sha256"):
        raise Program03Error(f"Behavior shard semantic content hash mismatch: {unit_dir}")
    if m.get("validation", {}).get("row_count") != validation.get("row_count"):
        raise Program03Error(f"Behavior shard manifest validation mismatch: {unit_dir}")
    return m


def publish_shard_unit(
    *,
    unit_dir: Path,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    expected_prompts: Sequence[PromptRecord],
    sample_start: int,
    sample_end: int,
    training_seed: int,
    behavior_step: int,
    split: str,
    shard_index: int,
    protocol_version_value: str,
    dataset_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_hash: str,
    adapter_sha256: str,
    behavior_config_sha256: str,
    generation_config_sha256: str,
) -> dict[str, Any]:
    if unit_dir.exists():
        return verify_shard_unit(
            unit_dir,
            root=root,
            expected_prompts=expected_prompts,
            sample_start=sample_start,
            sample_end=sample_end,
            training_seed=training_seed,
            behavior_step=behavior_step,
            split=split,
            protocol_version_value=protocol_version_value,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
            adapter_sha256=adapter_sha256,
            behavior_config_sha256=behavior_config_sha256,
        )

    validation = validate_behavior_rows(
        rows,
        expected_prompts=expected_prompts,
        sample_start=sample_start,
        sample_end=sample_end,
        training_seed=training_seed,
        behavior_step=behavior_step,
        split=split,
        protocol_version_value=protocol_version_value,
        dataset_revision=dataset_revision,
        model_revision=model_revision,
        adapter_sha256=adapter_sha256,
    )
    unit_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = unit_dir.parent / f".{unit_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=False, exist_ok=False)
        parquet_path = stage / "trajectories.parquet"
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pylist([dict(r) for r in rows], schema=parquet_schema())
            pq.write_table(table, parquet_path, compression="zstd", use_dictionary=True)
        except Exception as exc:
            raise Program03Error(f"Cannot write immutable behavior Parquet shard: {exc}") from exc
        with parquet_path.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        content_sha = rows_content_sha256(rows)
        file_sha = sha256_file(parquet_path)
        manifest = {
            "schema_version": BEHAVIOR_SCHEMA,
            "manifest_type": "behavior_trajectory_shard",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            "dataset": "GSM8K",
            "dataset_revision": dataset_revision,
            "protocol_version": protocol_version_value,
            "training_seed": training_seed,
            "behavior_step": behavior_step,
            "split": split,
            "sample_start": sample_start,
            "sample_end_exclusive": sample_end,
            "shard_index": shard_index,
            "prompt_ids": [p.prompt_id for p in expected_prompts],
            "behavior_config_sha256": behavior_config_sha256,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_hash": chat_template_hash,
            "behavior_adapter_sha256": adapter_sha256,
            "generation_config_sha256": generation_config_sha256,
            "probability_evidence_precision": PROBABILITY_EVIDENCE_PRECISION,
            "parquet": {
                "local_path": rel(unit_dir / "trajectories.parquet", root),
                "file_sha256": file_sha,
                "content_sha256": content_sha,
                "row_count": len(rows),
                "columns": list(ROW_COLUMNS),
            },
            "validation": validation,
        }
        atomic_write_json(stage / "manifest.json", manifest)
        fsync_directory(stage)
        if unit_dir.exists():
            raise Program03Error(f"Behavior shard appeared concurrently: {unit_dir}")
        os.replace(stage, unit_dir)
        fsync_directory(unit_dir.parent)
        make_read_only_tree(unit_dir)
        return verify_shard_unit(
            unit_dir,
            root=root,
            expected_prompts=expected_prompts,
            sample_start=sample_start,
            sample_end=sample_end,
            training_seed=training_seed,
            behavior_step=behavior_step,
            split=split,
            protocol_version_value=protocol_version_value,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
            adapter_sha256=adapter_sha256,
            behavior_config_sha256=behavior_config_sha256,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def cleanup_staging(data_root: Path, resume: bool) -> None:
    stale = [p for p in data_root.rglob(".*.staging-*") if p.is_dir()] if data_root.exists() else []
    if not stale:
        return
    if not resume:
        raise Program03Error(
            f"Found {len(stale)} stale Program 03 staging directories. Re-run with --resume to discard only unpublished staging state."
        )
    for p in stale:
        print(f"[RESUME] removing unpublished staging directory {p}")
        shutil.rmtree(p, ignore_errors=True)


def rebuild_collection_index(
    *,
    root: Path,
    data_root: Path,
    manifest_root: Path,
    mode: str,
    split: str,
    target_k: int,
    seeds: Sequence[int],
    anchors: Sequence[int],
    behavior_spec: BehaviorSpec,
    dataset_revision: str,
    model_revision: str,
    protocol_version_value: str,
    config_sha256: str,
) -> Path:
    records: list[dict[str, Any]] = []
    if data_root.exists():
        for mp in sorted(data_root.rglob("manifest.json")):
            if mp.parent.name.startswith("shard="):
                m = read_json(mp)
                if m.get("manifest_type") == "behavior_trajectory_shard":
                    records.append(
                        {
                            "manifest_path": rel(mp, root),
                            "manifest_sha256": sha256_file(mp),
                            "parquet_path": m.get("parquet", {}).get("local_path"),
                            "parquet_sha256": m.get("parquet", {}).get("file_sha256"),
                            "content_sha256": m.get("parquet", {}).get("content_sha256"),
                            "training_seed": m.get("training_seed"),
                            "behavior_step": m.get("behavior_step"),
                            "sample_start": m.get("sample_start"),
                            "sample_end_exclusive": m.get("sample_end_exclusive"),
                            "shard_index": m.get("shard_index"),
                            "row_count": m.get("parquet", {}).get("row_count"),
                            "probability_evidence_precision": m.get("probability_evidence_precision"),
                        }
                    )
    records.sort(key=lambda x: (
        int(x.get("training_seed", -1)),
        int(x.get("behavior_step", -1)),
        int(x.get("sample_start", -1)),
        int(x.get("shard_index", -1)),
    ))
    total_rows = sum(int(x.get("row_count") or 0) for x in records)
    payload = {
        "schema_version": BEHAVIOR_SCHEMA,
        "manifest_type": "behavior_collection_index",
        "updated_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "mode": mode,
        "split": split,
        "protocol_version": protocol_version_value,
        "protocol_config_sha256": config_sha256,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "target_k": target_k,
        "seeds": list(seeds),
        "anchors": list(anchors),
        "behavior_spec": asdict(behavior_spec),
        "behavior_config_sha256": behavior_spec_fingerprint(behavior_spec),
        "probability_evidence_precision": PROBABILITY_EVIDENCE_PRECISION,
        "shard_count": len(records),
        "trajectory_rows": total_rows,
        "shards": records,
        "shard_set_sha256": sha256_bytes(canonical_bytes(records)),
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    path = manifest_root / "collection_index.json"
    atomic_write_json(path, payload)
    return path


# ---------------------------------------------------------------------------
# Planning and collection
# ---------------------------------------------------------------------------


def engineering_spec(base: BehaviorSpec, mode: str, available_steps: Sequence[int]) -> tuple[BehaviorSpec, int | None]:
    if mode == "paper":
        return base, None
    steps = sorted(set(int(x) for x in available_steps))
    if not steps or 0 not in steps:
        raise Program03Error(f"Program 02 {mode} checkpoints do not contain step 0.")
    final = max(steps)
    anchors = (0,) if final == 0 else (0, final)
    if mode == "pilot":
        return BehaviorSpec(
            **{
                **asdict(base),
                "anchors": anchors,
                "k_main": min(4, base.k_main),
                "sample_block_size": min(4, base.sample_block_size),
                "generation_batch_size": min(4, base.generation_batch_size),
                "prompts_per_shard": min(8, base.prompts_per_shard),
            }
        ), 50
    return BehaviorSpec(
        **{
            **asdict(base),
            "anchors": anchors,
            "k_main": min(2, base.k_main),
            "sample_block_size": min(2, base.sample_block_size),
            "generation_batch_size": min(2, base.generation_batch_size),
            "prompts_per_shard": min(4, base.prompts_per_shard),
            "max_completion_length": min(64, base.max_completion_length),
        }
    ), 4


def expected_unit_count(prompt_count: int, prompts_per_shard: int, k: int, sample_block_size: int, n_seeds: int, n_anchors: int) -> int:
    n_prompt_shards = math.ceil(prompt_count / prompts_per_shard)
    n_sample_blocks = math.ceil(k / sample_block_size)
    return n_prompt_shards * n_sample_blocks * n_seeds * n_anchors


def count_missing_units(
    *,
    data_root: Path,
    prompts: Sequence[PromptRecord],
    seeds: Sequence[int],
    anchors: Sequence[int],
    target_k: int,
    spec: BehaviorSpec,
) -> int:
    pshards = prompt_shards(prompts, spec.prompts_per_shard)
    missing = 0
    for seed in seeds:
        for step in anchors:
            for s0, s1 in sample_blocks(target_k, spec.sample_block_size):
                for shard_idx, _ in enumerate(pshards):
                    if not unit_dir_path(
                        data_root,
                        split=prompts[0].research_split if prompts else "development",
                        seed=seed,
                        behavior_step=step,
                        sample_start=s0,
                        sample_end=s1,
                        shard_index=shard_idx,
                    ).exists():
                        missing += 1
    return missing


def collect_seed_anchor(
    *,
    root: Path,
    mode: str,
    split: str,
    seed: int,
    behavior_step: int,
    spec: BehaviorSpec,
    target_k: int,
    prompts: Sequence[PromptRecord],
    data_root: Path,
    model_dir: Path,
    model_record: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    protocol_version_value: str,
    dataset_revision: str,
    device: str,
    resume: bool,
) -> None:
    verified = training_manifest.get("_verified_adapters")
    if not isinstance(verified, Mapping) or behavior_step not in verified:
        raise Program03Error(f"Seed {seed} does not contain required behavior adapter step {behavior_step}.")
    adapter_rec = verified[behavior_step]
    if not isinstance(adapter_rec, Mapping):
        raise Program03Error("Internal verified adapter record is invalid.")
    adapter_dir = Path(str(adapter_rec["path"]))
    adapter_sha = str(adapter_rec["payload_sha256"])
    tokenizer_revision = str(model_record.get("tokenizer_revision") or model_record.get("resolved_revision"))
    model_revision = str(model_record.get("resolved_revision"))
    preferred_precision = training_manifest.get("resolved_precision")
    if preferred_precision is not None:
        preferred_precision = str(preferred_precision)

    tokenizer = load_tokenizer(model_dir, model_record)
    gen_record = generation_config_record(spec, tokenizer)
    gen_hash = sha256_bytes(canonical_bytes(gen_record))
    behavior_hash = behavior_spec_fingerprint(spec)
    chat_hash = ((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256")

    pshards = prompt_shards(prompts, spec.prompts_per_shard)
    # Fast pre-scan: if every expected unit exists, verify lazily below without loading the model.
    expected: list[tuple[int, int, int, list[PromptRecord], Path]] = []
    for s0, s1 in sample_blocks(target_k, spec.sample_block_size):
        for shard_idx, prompt_group in enumerate(pshards):
            u = unit_dir_path(
                data_root,
                split=split,
                seed=seed,
                behavior_step=behavior_step,
                sample_start=s0,
                sample_end=s1,
                shard_index=shard_idx,
            )
            expected.append((s0, s1, shard_idx, prompt_group, u))

    missing = [x for x in expected if not x[4].exists()]
    for s0, s1, shard_idx, prompt_group, u in expected:
        if u.exists():
            verify_shard_unit(
                u,
                root=root,
                expected_prompts=prompt_group,
                sample_start=s0,
                sample_end=s1,
                training_seed=seed,
                behavior_step=behavior_step,
                split=split,
                protocol_version_value=protocol_version_value,
                dataset_revision=dataset_revision,
                model_revision=model_revision,
                adapter_sha256=adapter_sha,
                behavior_config_sha256=behavior_hash,
            )
    if not missing:
        print(f"[SKIP] seed={seed} behavior_step={behavior_step}: all K={target_k} units verified")
        return
    if not resume and len(missing) != len(expected):
        raise Program03Error(
            f"Partial behavior logs already exist for seed={seed}, step={behavior_step}. Use --resume to continue append-only."
        )

    print(f"[LOAD] seed={seed} behavior_step={behavior_step} adapter={adapter_dir}")
    model, resolved_precision = load_behavior_model(
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        device=device,
        preferred_precision=PROBABILITY_EVIDENCE_PRECISION,
    )
    if resolved_precision != PROBABILITY_EVIDENCE_PRECISION:
        raise Program03Error(
            f"Probability-evidence precision contract violated: expected "
            f"{PROBABILITY_EVIDENCE_PRECISION}, got {resolved_precision}."
        )
    print(
        f"       probability evidence precision={resolved_precision}; "
        f"training precision={preferred_precision}; missing units={len(missing)}/{len(expected)}"
    )
    start_wall = time.time()
    try:
        for unit_no, (s0, s1, shard_idx, prompt_group, u) in enumerate(expected, start=1):
            if u.exists():
                continue
            rows: list[dict[str, Any]] = []
            for prompt in prompt_group:
                rows.extend(
                    generate_prompt_block(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        sample_start=s0,
                        sample_end=s1,
                        spec=spec,
                        protocol_version_value=protocol_version_value,
                        dataset_revision=dataset_revision,
                        model_revision=model_revision,
                        tokenizer_revision=tokenizer_revision,
                        training_seed=seed,
                        behavior_step=behavior_step,
                        split=split,
                        adapter_sha256=adapter_sha,
                        device=device,
                    )
                )
            publish_shard_unit(
                unit_dir=u,
                root=root,
                rows=rows,
                expected_prompts=prompt_group,
                sample_start=s0,
                sample_end=s1,
                training_seed=seed,
                behavior_step=behavior_step,
                split=split,
                shard_index=shard_idx,
                protocol_version_value=protocol_version_value,
                dataset_revision=dataset_revision,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                chat_template_hash=str(chat_hash),
                adapter_sha256=adapter_sha,
                behavior_config_sha256=behavior_hash,
                generation_config_sha256=gen_hash,
            )
            elapsed = max(time.time() - start_wall, 1e-9)
            print(
                f"[SAVE] seed={seed} b={behavior_step} samples={s0}:{s1} shard={shard_idx:05d} "
                f"rows={len(rows)} elapsed={elapsed/60:.1f}m"
            )
    finally:
        try:
            import torch  # type: ignore

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO-OPE Program 03: collect exact immutable behavior trajectories and generation-time log probabilities."
    )
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-root", default=".")
    p.add_argument("--seed", type=int, default=None, help="Collect one configured training seed only; default all seeds for this mode.")
    p.add_argument(
        "--target-k",
        type=int,
        default=None,
        help="Development-only append target (normally 16 or 32). Test forbids overrides and uses frozen K_main.",
    )
    p.add_argument(
        "--reset-checkpoints",
        "--reset-behavior-logs",
        dest="reset_outputs",
        action="store_true",
        help="Dangerous: delete Program 03 outputs for the selected mode/split. Requires GRPO_OPE_ALLOW_RESET=YES.",
    )
    return p.parse_args(argv)


def safe_reset(paths: Sequence[Path]) -> None:
    if os.environ.get("GRPO_OPE_ALLOW_RESET") != "YES":
        raise Program03Error(
            "Reset requires environment variable GRPO_OPE_ALLOW_RESET=YES. This prevents accidental deletion of expensive logs."
        )

    def _remove_readonly(func: Any, path: str, exc_info: Any) -> None:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            func(path)
        except Exception:
            raise

    for p in paths:
        if p.exists():
            print(f"[RESET] removing {p}")
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
    protocol_version_value: str,
    data_revision: str,
    model_revision: str,
    seeds: Sequence[int],
    anchors: Sequence[int],
    target_k: int,
    spec: BehaviorSpec,
    prompt_count: int,
    missing_units: int,
    data_root: Path,
) -> None:
    print("=" * 88)
    print("GRPO-OPE Program 03 — immutable behavior trajectory collection")
    print(f"program version        : {PROGRAM_VERSION}")
    print(f"project root           : {root}")
    print(f"mode / split           : {mode} / {split}")
    print(f"protocol version       : {protocol_version_value}")
    print(f"config                 : {cfg_path}")
    print(f"config SHA-256         : {config_sha}")
    print(f"behavior config hash   : {behavior_spec_fingerprint(spec)}")
    print(f"model revision         : {model_revision}")
    print(f"data revision          : {data_revision}")
    print(f"training seeds         : {list(seeds)}")
    print(f"behavior anchors       : {list(anchors)}")
    print(f"prompts / target K     : {prompt_count} / {target_k}")
    print(f"sample block / gen grp : {spec.sample_block_size} / {spec.generation_batch_size}")
    print(f"prompt shard size      : {spec.prompts_per_shard}")
    print(f"max completion         : {spec.max_completion_length}")
    print(f"probability precision  : {PROBABILITY_EVIDENCE_PRECISION} (fixed; independent of Program 02 training precision)")
    print(
        f"sampling               : T={spec.temperature}, top_p={spec.top_p}, "
        f"top_k={spec.top_k}, rep={spec.repetition_penalty}"
    )
    print(f"unfinished units       : {missing_units}")
    print(f"behavior output root   : {data_root}")
    print("research boundary      : behavior generation only; no target rescore/OPE/online reference")
    print("=" * 88)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    manifests = root / "manifests"
    gsm8k_dir = root / "data" / "raw" / "gsm8k"
    model_dir = root / "models" / "qwen25_05b"
    registry_path = root / "data" / "splits" / "gsm8k_split_registry.parquet"

    env_manifest = verify_environment_manifest(manifests / "environment_manifest.json")
    data_manifest, gsm_record = verify_data_manifest(manifests / "data_manifest.json", gsm8k_dir)
    model_manifest, model_record = verify_model_manifest(manifests / "model_manifest.json", model_dir)
    split_manifest, registry_rows = verify_split_registry(
        manifests / "split_registry_manifest.json", registry_path, data_manifest, gsm_record
    )
    cfg, config_sha = load_protocol(cfg_path)
    pversion = protocol_version(cfg)
    base_spec = parse_behavior_spec(cfg)
    base_seeds = configured_seeds(cfg)

    if args.mode == "paper":
        validate_paper_contract(base_spec)
        lock = verify_protocol_lock(
            manifests / "protocol_lock.json",
            config_sha256=config_sha,
            data_manifest=data_manifest,
            model_manifest=model_manifest,
            split_manifest=split_manifest,
            behavior_spec=base_spec,
        )
    else:
        lock = {}
        if args.split == "test":
            raise Program03Error("smoke/pilot modes are development-only; official test is forbidden.")

    if args.split == "test":
        if args.mode != "paper":
            raise Program03Error("Official test is only legal in paper mode.")
        if args.target_k is not None:
            raise Program03Error("Official test forbids --target-k overrides; use the frozen K_main.")
        verify_frozen_gate_for_test(root, lock)

    requested_seeds = list(base_seeds)
    if args.mode != "paper":
        requested_seeds = [base_seeds[0]]
    if args.seed is not None:
        if args.seed not in requested_seeds:
            raise Program03Error(f"--seed {args.seed} is not legal for mode={args.mode}; expected one of {requested_seeds}.")
        requested_seeds = [args.seed]

    # Verify all selected Program 02 paths first. Engineering anchors are derived
    # from their actual permanent adapter grid; paper anchors remain predeclared.
    training_manifests: dict[int, dict[str, Any]] = {}
    available_steps_union: set[int] = set()
    for seed in requested_seeds:
        tm = verify_training_seed(
            root=root,
            mode=args.mode,
            seed=seed,
            model_record=model_record,
            data_manifest=data_manifest,
            split_manifest=split_manifest,
            config_sha256=config_sha,
        )
        training_manifests[seed] = tm
        available_steps_union.update(int(x) for x in tm["_verified_adapters"].keys())

    spec, prompt_limit = engineering_spec(base_spec, args.mode, sorted(available_steps_union))
    if args.mode == "paper":
        spec = base_spec
    target_k = spec.k_main
    if args.target_k is not None:
        if args.split != "development" or args.mode != "paper":
            raise Program03Error("--target-k is allowed only for paper-mode development append experiments.")
        if args.target_k not in PAPER_ALLOWED_K:
            raise Program03Error(f"Development --target-k must be one of {PAPER_ALLOWED_K}.")
        if args.target_k < spec.k_main:
            raise Program03Error("--target-k cannot shrink below frozen/configured K_main; behavior logs are append-only.")
        target_k = args.target_k

    # Every selected seed must contain every selected anchor.
    for seed, tm in training_manifests.items():
        steps = set(int(x) for x in tm["_verified_adapters"].keys())
        missing = [b for b in spec.anchors if b not in steps]
        if missing:
            raise Program03Error(f"Seed {seed} lacks required behavior anchors {missing}.")

    original_split = "test" if args.split == "test" else "train"
    raw = load_upstream_rows(gsm8k_dir, original_split)
    prompts = build_prompt_records(
        registry_rows=registry_rows,
        raw_rows=raw,
        research_split=args.split,
        limit=prompt_limit,
    )

    if args.mode == "paper":
        data_root = root / "data" / "behavior_logs"
        manifest_root = root / "manifests" / "behavior_logs" / args.split
    else:
        data_root = root / "data" / "behavior_logs" / f"_{args.mode}"
        manifest_root = root / "manifests" / f"_{args.mode}" / "behavior_logs" / args.split

    if args.reset_outputs:
        split_data_path = data_root / "gsm8k" / f"split={args.split}"
        safe_reset([split_data_path, manifest_root])

    cleanup_staging(data_root, args.resume)
    missing_units = count_missing_units(
        data_root=data_root,
        prompts=prompts,
        seeds=requested_seeds,
        anchors=spec.anchors,
        target_k=target_k,
        spec=spec,
    )
    print_header(
        root=root,
        mode=args.mode,
        split=args.split,
        cfg_path=cfg_path,
        config_sha=config_sha,
        protocol_version_value=pversion,
        data_revision=str(gsm_record.get("resolved_revision")),
        model_revision=str(model_record.get("resolved_revision")),
        seeds=requested_seeds,
        anchors=spec.anchors,
        target_k=target_k,
        spec=spec,
        prompt_count=len(prompts),
        missing_units=missing_units,
        data_root=data_root,
    )
    print(f"git commit             : {git_commit(root) or 'not-a-git-checkout'}")
    print(f"environment manifest   : {env_manifest.get('environment_fingerprint_sha256')}")
    print(f"split registry hash    : {split_manifest.get('content_fingerprint_sha256')}")

    if missing_units > 0 and not args.resume:
        # Fresh runs are legal. Only partial existing state is rejected inside
        # collect_seed_anchor, where we can distinguish fresh from partial.
        pass

    for seed in requested_seeds:
        for behavior_step in spec.anchors:
            collect_seed_anchor(
                root=root,
                mode=args.mode,
                split=args.split,
                seed=seed,
                behavior_step=behavior_step,
                spec=spec,
                target_k=target_k,
                prompts=prompts,
                data_root=data_root,
                model_dir=model_dir,
                model_record=model_record,
                training_manifest=training_manifests[seed],
                protocol_version_value=pversion,
                dataset_revision=str(gsm_record.get("resolved_revision")),
                device=args.device,
                resume=args.resume,
            )
            rebuild_collection_index(
                root=root,
                data_root=data_root,
                manifest_root=manifest_root,
                mode=args.mode,
                split=args.split,
                target_k=target_k,
                seeds=requested_seeds,
                anchors=spec.anchors,
                behavior_spec=spec,
                dataset_revision=str(gsm_record.get("resolved_revision")),
                model_revision=str(model_record.get("resolved_revision")),
                protocol_version_value=pversion,
                config_sha256=config_sha,
            )

    index_path = rebuild_collection_index(
        root=root,
        data_root=data_root,
        manifest_root=manifest_root,
        mode=args.mode,
        split=args.split,
        target_k=target_k,
        seeds=requested_seeds,
        anchors=spec.anchors,
        behavior_spec=spec,
        dataset_revision=str(gsm_record.get("resolved_revision")),
        model_revision=str(model_record.get("resolved_revision")),
        protocol_version_value=pversion,
        config_sha256=config_sha,
    )
    final_missing = count_missing_units(
        data_root=data_root,
        prompts=prompts,
        seeds=requested_seeds,
        anchors=spec.anchors,
        target_k=target_k,
        spec=spec,
    )
    if final_missing != 0:
        raise Program03Error(f"Program finished with {final_missing} expected behavior shard units still missing.")

    expected_rows = len(prompts) * target_k * len(requested_seeds) * len(spec.anchors)
    print("\nPROGRAM 03 PASSED")
    print(f"mode / split          : {args.mode} / {args.split}")
    print(f"seeds                 : {requested_seeds}")
    print(f"behavior anchors      : {list(spec.anchors)}")
    print(f"K                     : {target_k}")
    print(f"expected trajectories : {expected_rows}")
    print(f"collection index      : {index_path}")
    print("probability evidence  : exact generated token IDs + generation-time selected-token logprobs")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPROGRAM 03 INTERRUPTED. Published immutable shards remain valid; re-run with --resume.", file=sys.stderr)
        raise SystemExit(130)
    except Program03Error as exc:
        print(f"\nPROGRAM 03 FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print("\nPROGRAM 03 FAILED with an unexpected exception:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(3)
