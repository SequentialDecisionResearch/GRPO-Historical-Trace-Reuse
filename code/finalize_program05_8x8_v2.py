#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

EFFECTIVE_L = 8
EXPECTED_STEPS = [0,20,100,120,160,200,220,260,300,360,400]
AMEND_REL = Path("manifests/protocol_amendment_program05_sample_size.json")
SELECTION_REL = Path("manifests/reduced_test_pair_selection.json")
P05_REL = Path("05_generate_online_reference.py")

class FinalizeError(RuntimeError): pass
def fail(msg): raise FinalizeError(msg)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def read_json(path: Path) -> dict[str,Any]:
    if not path.exists(): fail(f"Missing: {path}")
    x=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x,dict): fail(f"Expected object: {path}")
    return x

def load_module(path: Path):
    spec=importlib.util.spec_from_file_location("program05_8x8_final_original", path)
    if spec is None or spec.loader is None: fail(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod

def plan(root: Path):
    amend_path=root/AMEND_REL; amend=read_json(amend_path)
    d=amend.get("amended_effective_design") or {}
    if int(d.get("L_main",-1))!=8 or int(d.get("L_audit",-1))!=8: fail("Amendment is not 8/8.")
    sel_path=root/SELECTION_REL; sel=read_json(sel_path)
    rows=sel.get("selected_pairs")
    if not isinstance(rows,list) or len(rows)!=36: fail("Reduced selection is not 36 pairs.")
    seeds=sorted({int(r["training_seed"]) for r in rows})
    steps=sorted({int(r["target_step"]) for r in rows})
    if len(seeds)!=3 or steps!=EXPECTED_STEPS: fail(f"Unexpected plan {seeds}/{steps}")
    expected_sel=(((amend.get("immutability_and_provenance") or {})
                  .get("reduced_pair_selection") or {}).get("sha256"))
    if expected_sel!=sha256_file(sel_path): fail("Selection hash mismatch.")
    return seeds,steps,amend,sha256_file(amend_path)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--verify-only",action="store_true")
    args=ap.parse_args()

    root=Path(args.root).resolve()
    seeds,steps,amend,amend_sha=plan(root)
    p05_path=root/P05_REL; p05=load_module(p05_path)

    cfg_path=root/"configs"/"protocol.yaml"; manifests=root/"manifests"
    gsm8k_dir=root/"data"/"raw"/"gsm8k"; model_dir=root/"models"/"qwen25_05b"
    registry_path=root/"data"/"splits"/"gsm8k_split_registry.parquet"

    env_manifest=p05.verify_environment_manifest(manifests/"environment_manifest.json")
    data_manifest,gsm_record=p05.verify_data_manifest(manifests/"data_manifest.json",gsm8k_dir)
    model_manifest,model_record=p05.verify_model_manifest(manifests/"model_manifest.json",model_dir)
    split_manifest,registry_rows=p05.verify_split_registry(
        manifests/"split_registry_manifest.json",registry_path,data_manifest,gsm_record)
    cfg,config_sha=p05.load_protocol(cfg_path)
    pv=p05.protocol_version(cfg); spec=p05.parse_online_spec(cfg)
    base_seeds=list(p05.configured_seeds(cfg))
    p05.validate_paper_contract(spec,base_seeds)
    lock=p05.verify_protocol_lock(
        manifests/"protocol_lock.json",config_sha256=config_sha,data_manifest=data_manifest,
        model_record=model_record,split_manifest=split_manifest,online_spec=spec)
    p05.verify_frozen_gate_for_test(root,lock)
    if sorted(base_seeds)!=seeds: fail("Frozen seed set differs from amended plan.")

    adapters={}
    for seed in seeds:
        tm,amap=p05.verify_training_seed(
            root=root,mode="paper",seed=seed,model_record=model_record,
            data_manifest=data_manifest,split_manifest=split_manifest,config_sha256=config_sha)
        adapters[seed]=amap
        miss=[x for x in steps if x not in amap]
        if miss: fail(f"Seed {seed} lacks adapters {miss}")

    raw=p05.load_upstream_rows(gsm8k_dir,"test")
    prompts=p05.build_prompt_records(
        registry_rows=registry_rows,raw_rows=raw,research_split="test",limit=None)
    data_root=p05.online_data_root(root,"paper")
    mroot=p05.online_manifest_root(root,"paper","test")
    p05.cleanup_staging(data_root,True)
    p04_identity=p05.verify_program04_identity_gate(root,"paper","test",seeds)
    pshards=p05.prompt_shards(prompts,spec.prompts_per_shard)

    summaries=[]
    for seed in seeds:
        for step in steps:
            for shard_index,pshard in enumerate(pshards):
                unit=p05.online_unit_dir(
                    data_root,split="test",training_seed=seed,target_step=step,
                    sample_start=0,sample_end=8,shard_index=shard_index)
                if not unit.exists(): fail(f"Missing 8/8 unit: {unit}")
                p05.verify_online_unit(
                    unit,root=root,expected_prompts=pshard,sample_start=0,sample_end=8,
                    training_seed=seed,target_step=step,split="test",
                    protocol_version_value=pv,
                    dataset_revision=str(gsm_record.get("resolved_revision")),
                    model_revision=str(model_record.get("resolved_revision")),
                    adapter_sha256=adapters[seed][step].payload_sha256,spec=spec)

            prefix=p05.target_prefix(data_root,"test",seed,step)
            rp=prefix/"reference_summary.json"; pp=prefix/"prompt_summary.parquet"
            if args.verify_only:
                if not rp.exists() or not pp.exists(): fail(f"Missing target summary {seed}/{step}")
                summary=read_json(rp)
                if int(summary.get("samples_per_prompt",-1))!=8:
                    fail(f"Target summary L !=8 for {seed}/{step}")
                if (summary.get("prompt_summary") or {}).get("file_sha256")!=sha256_file(pp):
                    fail(f"Prompt summary hash mismatch {seed}/{step}")
                sp=pp
            else:
                sp,rp,summary=p05.rebuild_target_summary(
                    root=root,data_root=data_root,split="test",seed=seed,target_step=step,
                    target_l=8,prompts=prompts,
                    dataset_revision=str(gsm_record.get("resolved_revision")),
                    adapter_sha256=adapters[seed][step].payload_sha256)

            summaries.append({
                "training_seed":seed,"target_step":step,"samples_per_prompt":8,
                "online_reference":summary["online_reference"],
                "conditional_generation_mc_se":summary["uncertainty"]["conditional_generation_mc_se"],
                "prompt_summary_path":p05.rel(sp,root),
                "reference_summary_path":p05.rel(rp,root),
                "reference_summary_sha256":sha256_file(rp),
            })

    summaries.sort(key=lambda x:(int(x["training_seed"]),int(x["target_step"])))
    expected_units=len(seeds)*len(steps)*len(pshards)

    if not args.verify_only:
        idx_path=p05.rebuild_collection_index(
            root=root,data_root=data_root,manifest_root=mroot,mode="paper",split="test",
            target_l_main=spec.l_main,spec=spec,seeds=seeds,target_steps=steps,
            dataset_revision=str(gsm_record.get("resolved_revision")),
            model_revision=str(model_record.get("resolved_revision")),
            protocol_version_value=pv,config_sha256=config_sha,program04_identity=p04_identity)
        idx=read_json(idx_path)
        kept=[r for r in (idx.get("shards") or [])
              if int(r.get("sample_start",-1))==0 and int(r.get("sample_end_exclusive",-1))==8]
        if len(kept)!=expected_units:
            fail(f"8/8 collection has {len(kept)} units; expected {expected_units}.")
        idx["shards"]=kept
        idx["shard_count"]=len(kept)
        idx["trajectory_rows"]=sum(int(r.get("row_count") or 0) for r in kept)
        idx["shard_set_sha256"]=p05.sha256_bytes(p05.canonical_bytes(kept))
        idx["effective_sample_size_amendment"]={
            "manifest_path":str(AMEND_REL).replace("\\","/"),
            "manifest_sha256":amend_sha,
            "effective_L_main":8,"effective_L_audit":8,
            "original_frozen_L_main":8,"original_frozen_L_audit":32,
            "used_sample_range":[0,8],
        }
        p05.atomic_write_json(idx_path,idx)

        summary_index=mroot/"reference_summaries.json"
        p05.atomic_write_json(summary_index,{
            "schema_version":p05.ONLINE_SCHEMA,
            "manifest_type":"online_reference_summary_index",
            "updated_at_utc":p05.now_utc(),
            "mode":"paper","split":"test",
            "reference_name":"high-precision on-policy Monte Carlo reference",
            "not_exact_truth":True,
            "frozen_target_l_main":8,"frozen_l_audit":32,
            "effective_L_main":8,"effective_L_audit":8,
            "audit_steps":list(spec.audit_steps),
            "sample_size_amendment_sha256":amend_sha,
            "summaries":summaries,
            "summary_set_sha256":p05.sha256_bytes(p05.canonical_bytes(summaries)),
        })

        runner_manifest=root/"manifests"/"program05_8x8_final_manifest.json"
        p05.atomic_write_json(runner_manifest,{
            "schema_version":"1.0",
            "manifest_type":"program05_8x8_final_manifest",
            "created_at_utc":p05.now_utc(),
            "split":"test","seeds":seeds,"target_steps":steps,
            "effective_samples_per_prompt":8,
            "seed_target_references":len(summaries),
            "online_shard_units":expected_units,
            "sample_size_amendment":{"path":str(AMEND_REL).replace("\\","/"),"sha256":amend_sha},
            "original_program05":{"path":str(P05_REL).replace("\\","/"),"sha256":sha256_file(p05_path)},
            "collection_index":{"path":p05.rel(idx_path,root),"sha256":sha256_file(idx_path)},
            "reference_summaries":{"path":p05.rel(summary_index,root),"sha256":sha256_file(summary_index)},
        })
        print("="*92)
        print("PROGRAM 05 8/8 FINALIZED")
        print(f"seed-target references : {len(summaries)}")
        print(f"online shard units      : {expected_units}")
        print(f"collection index        : {idx_path}")
        print(f"summary index           : {summary_index}")
        print("="*92)
    else:
        idx_path=mroot/"collection_index.json"; sm_path=mroot/"reference_summaries.json"
        if not idx_path.exists() or not sm_path.exists(): fail("Final 8/8 indexes missing.")
        idx=read_json(idx_path); sm=read_json(sm_path)
        eff=idx.get("effective_sample_size_amendment") or {}
        if eff.get("manifest_sha256")!=amend_sha:
            fail("Collection index amendment SHA mismatch.")
        if int(eff.get("effective_L_main",-1))!=8 or int(eff.get("effective_L_audit",-1))!=8:
            fail("Collection index effective L is not 8/8.")
        if int(idx.get("shard_count",-1))!=expected_units:
            fail("Collection index shard count mismatch.")
        if len(sm.get("summaries") or [])!=33:
            fail("Reference summary index does not contain 33 references.")
        if int(sm.get("effective_L_main",-1))!=8 or int(sm.get("effective_L_audit",-1))!=8:
            fail("Reference summary index effective L is not 8/8.")
        print("="*92)
        print("PROGRAM 05 8/8 VERIFY-ONLY PASSED")
        print("seed-target references : 33")
        print(f"online shard units      : {expected_units}")
        print("="*92)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except FinalizeError as exc:
        print(f"\nPROGRAM 05 8/8 FINALIZER FAILED\n{exc}",file=sys.stderr)
        raise SystemExit(2)
