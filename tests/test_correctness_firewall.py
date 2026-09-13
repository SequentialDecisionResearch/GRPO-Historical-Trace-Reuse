from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / name
    modname = "testload_" + name.replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def test_exact_change_of_measure_toy():
    mu = [0.5, 0.5]
    pi = [0.75, 0.25]
    reward = [1.0, 0.2]
    w = [p / m for p, m in zip(pi, mu)]
    lhs = sum(m * wi * r for m, wi, r in zip(mu, w, reward))
    rhs = sum(p * r for p, r in zip(pi, reward))
    assert math.isclose(lhs, rhs, rel_tol=0, abs_tol=1e-15)


def test_renyi_relative_ess_identity():
    mu = [0.5, 0.5]
    pi = [0.75, 0.25]
    w = [p / m for p, m in zip(pi, mu)]
    ew2 = sum(m * wi * wi for m, wi in zip(mu, w))
    d2 = math.log(ew2)
    population_ress = 1.0 / ew2
    assert math.isclose(population_ress, math.exp(-d2), rel_tol=0, abs_tol=1e-15)
    assert math.isclose(population_ress, 0.8, rel_tol=0, abs_tol=1e-15)


def test_gspo_length_normalized_ratio_is_not_ope_weight():
    mu = [0.5, 0.5]
    pi = [0.8, 0.2]
    reward = [1.0, 0.5]
    lengths = [1, 2]
    w = [p / m for p, m in zip(pi, mu)]
    exact_target = sum(p * r for p, r in zip(pi, reward))
    normalized = sum(m * (wi ** (1.0 / t)) * r for m, wi, t, r in zip(mu, w, lengths, reward))
    assert not math.isclose(normalized, exact_target, rel_tol=0, abs_tol=1e-6)


def test_program04_protocol_helpers_and_paper_spec():
    p04 = load_script("04_rescore_target_checkpoints.py")
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    assert p04.protocol_version(cfg) == "1.0"
    assert p04.configured_seeds(cfg) == (20260826, 20260827, 20260828)
    assert p04.parse_target_steps(cfg) == tuple(range(0, 401, 20))
    spec = p04.parse_rescore_spec(cfg, None)
    assert spec.batch_size == 8
    assert math.isclose(spec.identity_token_atol, 0.01)
    assert math.isclose(spec.identity_sequence_atol, 0.05)


def test_adapter_hash_contract_is_identical_across_programs(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":16}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-adapter-payload")
    mods = [load_script(x) for x in (
        "02_train_all_seeds.py", "03_collect_behavior_logs.py", "04_rescore_target_checkpoints.py",
        "05_generate_online_reference.py", "08_run_svamp_robustness.py",
    )]
    hashes = [m.adapter_payload_hash(adapter) for m in mods]
    assert len(set(hashes)) == 1, hashes


def _write_csv(path: Path, rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def test_program06_semantic_split_hash_survives_other_split_append(tmp_path: Path):
    p06 = load_script("06_compute_ope_and_diagnostics.py")
    path = tmp_path / "T02.csv"
    dev = [
        {"split":"development","training_seed":"1","target_step":"20"},
        {"split":"development","training_seed":"1","target_step":"40"},
    ]
    _write_csv(path, dev)
    h1 = p06.semantic_csv_scope_sha256(path, "split", "development")
    rows = dev + [{"split":"test","training_seed":"1","target_step":"20"}]
    _write_csv(path, rows)
    h2 = p06.semantic_csv_scope_sha256(path, "split", "development")
    assert h1 == h2


def test_program07_role_hash_survives_test_append_and_program09_verifies(tmp_path: Path):
    p07 = load_script("07_calibrate_and_test_gate.py")
    p09 = load_script("09_make_paper_outputs.py")
    out = tmp_path / "outputs" / "tables" / "T06_gate_test.csv"
    held = [{"evaluation_role":"heldout_seed_validation","training_seed":"3","target_step":"20"}]
    _write_csv(out, held)
    h = p07.semantic_csv_scope_sha256(out, "evaluation_role", "heldout_seed_validation")
    _write_csv(out, held + [{"evaluation_role":"official_test","training_seed":"1","target_step":"20"}])
    manifest = tmp_path / "m.json"
    manifest.write_text(__import__('json').dumps({
        "manifest_type":"unit_test_manifest",
        "split":"development",
        "outputs":[{
            "path":"outputs/tables/T06_gate_test.csv",
            "semantic_scope":{"column":"evaluation_role","value":"heldout_seed_validation"},
            "semantic_sha256":h,
        }]
    }), encoding="utf-8")
    p09.verify_program_manifest_output(tmp_path, manifest, "unit_test_manifest", "development")


def test_parser_and_stable_ids_are_deterministic():
    p03 = load_script("03_collect_behavior_logs.py")
    value, status = p03.parse_completion_answer("Reasoning... <answer>1,234/2</answer>")
    assert status == "ok"
    assert value == Fraction(617, 1)
    tid1 = p03.trajectory_id(protocol_version_value="1.0", dataset_revision="d"*40, model_revision="m"*40, training_seed=7, behavior_step=100, split="development", prompt_id="prompt", sample_index=3)
    tid2 = p03.trajectory_id(protocol_version_value="1.0", dataset_revision="d"*40, model_revision="m"*40, training_seed=7, behavior_step=100, split="development", prompt_id="prompt", sample_index=3)
    assert tid1 == tid2
    assert p03.stable_seed("a", 1, 2) == p03.stable_seed("a", 1, 2)


def test_program02_current_and_legacy_reward_metric_aliases_present():
    text = (ROOT / "02_train_all_seeds.py").read_text(encoding="utf-8")
    assert 'reward/correctness_reward/mean' in text
    assert 'rewards/correctness_reward/mean' in text
    assert 'reward/format_reward/mean' in text
    assert 'rewards/format_reward/mean' in text
    assert 'tokenizer.padding_side = "left"' in text


def test_program09_reuse_horizon_and_diagnostics():
    p09 = load_script("09_make_paper_outputs.py")
    t02 = []
    for seed in (1, 2):
        for step, err, ress, d2, c, kl in [
            (20, .005, .9, .1, .1, .01),
            (40, .015, .6, .5, .2, .02),
            (60, .030, .3, 1.0, .5, .04),
        ]:
            t02.append({"dataset":"GSM8K","split":"development","training_seed":seed,"behavior_step":0,"target_step":step,
                        "estimator":"prompt_wis","absolute_error":err,"median_prompt_relative_ess":ress,"mean_d2":d2,
                        "mean_max_normalized_weight":c,"tokenwise_kl_proxy":kl,"mean_completion_length":50+step,
                        "online_reference":.5,"ope_ci_low":.45,"ope_ci_high":.55})
    h = p09.reuse_horizon_rows(t02, [], split="development", tolerances=(.01,.02,.05))
    row = next(r for r in h if r["training_seed"] == 1 and math.isclose(r["tolerance"], .02))
    assert row["horizon_steps"] == 40
    diag = p09.diagnostic_benchmark_rows(t02, split="development")
    rr = next(r for r in diag if r["scope"] == "pooled" and r["diagnostic"] == "median_prompt_relative_ess")
    assert rr["spearman_with_absolute_error"] < 0
    inc = p09.mc_reference_inclusion_rows(t02)
    assert inc[0]["mc_reference_inclusion_rate"] == 1.0
