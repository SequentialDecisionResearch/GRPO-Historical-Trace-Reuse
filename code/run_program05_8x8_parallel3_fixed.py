#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_STEPS=[0,20,100,120,160,200,220,260,300,360,400]
PARALLELISM=3
DEFAULT_CHUNK=10

class SupervisorError(RuntimeError): pass
def fail(msg): raise SupervisorError(msg)

def now_utc(): return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def read_json(path: Path):
    if not path.exists(): fail(f"Missing: {path}")
    x=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x,dict): fail(f"Expected object: {path}")
    return x

def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name("."+path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")
    tmp.replace(path)

def target_prefix(root: Path, seed: int, step: int) -> Path:
    return (root/"data"/"online_reference"/"gsm8k"/"split=test"/
            f"seed={seed}"/f"target_step={step:04d}")

def target_summary(root: Path, seed: int, step: int) -> Path:
    return target_prefix(root,seed,step)/"reference_summary.json"

def complete(root: Path, seed: int, step: int) -> bool:
    rp=target_summary(root,seed,step)
    pp=rp.parent/"prompt_summary.parquet"
    if not rp.exists() or not pp.exists(): return False
    try:
        m=read_json(rp)
        return int(m.get("samples_per_prompt",-1))==8
    except Exception:
        return False

def count_units(root: Path, seed: int, step: int) -> int:
    """Count published 0:8 online shard manifests robustly, without assuming folder spelling."""
    prefix=target_prefix(root,seed,step)
    if not prefix.exists(): return 0
    n=0
    for mp in prefix.rglob("manifest.json"):
        if not mp.parent.name.startswith("shard="):
            continue
        try:
            m=read_json(mp)
        except Exception:
            continue
        if m.get("manifest_type")!="online_reference_shard":
            continue
        if int(m.get("training_seed",-1))!=seed or int(m.get("target_step",-1))!=step:
            continue
        if int(m.get("sample_start",-1))!=0 or int(m.get("sample_end_exclusive",-1))!=8:
            continue
        n += 1
    return n

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--chunk",type=int,default=DEFAULT_CHUNK)
    args=ap.parse_args()
    if args.chunk<=0: fail("--chunk must be positive.")
    root=Path(args.root).resolve()

    worker=root/"run_program05_8x8_chunk_worker_v2.py"
    finalizer=root/"finalize_program05_8x8_v2.py"
    amendment=root/"manifests"/"protocol_amendment_program05_sample_size.json"
    selection=root/"manifests"/"reduced_test_pair_selection.json"
    test_script=root/"test_program05_8x8_parallel3.py"
    telemetry=root/"benchmarks"/"program05_parallel3_8x8"/"gpu_telemetry.csv"
    for p in (worker,finalizer,amendment,selection):
        if not p.exists(): fail(f"Missing required file: {p}")

    amend=read_json(amendment)
    d=amend.get("amended_effective_design") or {}
    if int(d.get("L_main",-1))!=8 or int(d.get("L_audit",-1))!=8:
        fail("Locked Program05 amendment is not 8/8.")
    sel=read_json(selection)
    rows=sel.get("selected_pairs")
    if not isinstance(rows,list) or len(rows)!=36: fail("Reduced selection is not 36 pairs.")
    seeds=sorted({int(r["training_seed"]) for r in rows})
    steps=sorted({int(r["target_step"]) for r in rows})
    if len(seeds)!=3 or steps!=EXPECTED_STEPS: fail(f"Unexpected plan {seeds}/{steps}")

    tasks=[(s,e) for s in seeds for e in steps]
    pending=[x for x in tasks if not complete(root,*x)]
    initial_units=sum(count_units(root,*x) for x in tasks)
    expected_total=3*11*165

    run_manifest_path=root/"manifests"/"program05_8x8_parallel3_runner_manifest.json"
    run_manifest={
        "schema_version":"1.1",
        "manifest_type":"program05_8x8_parallel3_runner_manifest",
        "created_at_utc":now_utc(),
        "status":"started",
        "parallelism":3,
        "chunk_new_shards_per_process":args.chunk,
        "scheduling":"dynamic queue; one unique seed-target owned by one lane until complete",
        "concurrency_safety":"per-target workers suppress global collection/reference index writes; one single-process finalizer writes global indexes after all targets complete",
        "sample_size_amendment":{"path":"manifests/protocol_amendment_program05_sample_size.json","sha256":sha256_file(amendment)},
        "reduced_selection":{"path":"manifests/reduced_test_pair_selection.json","sha256":sha256_file(selection)},
        "worker":{"path":worker.name,"sha256":sha256_file(worker)},
        "finalizer":{"path":finalizer.name,"sha256":sha256_file(finalizer)},
        "throughput_test":{
            "parallel_processes":3,
            "official_shards":9,
            "wall_seconds":180.8,
            "aggregate_wall_seconds_per_shard":20.09,
            "gpu_util_mean_pct":54.1,
            "gpu_util_max_pct":87.0,
            "gpu_memory_max_mib":8276,
            "gpu_temperature_max_c":63,
            "gpu_power_mean_w":75.2,
            "gpu_power_max_w":107.4,
            "test_passed":True,
            "test_script_sha256":sha256_file(test_script) if test_script.exists() else None,
            "gpu_telemetry_sha256":sha256_file(telemetry) if telemetry.exists() else None,
            "selection_reason":"3-process materially outperformed the prior 2-process test while preserving VRAM headroom; no fourth process adopted.",
        },
        "initial_8x8_online_shard_units":initial_units,
        "total_expected_8x8_online_shard_units":expected_total,
        "initial_completed_seed_targets":sum(complete(root,*x) for x in tasks),
        "pending_seed_targets":len(pending),
    }
    atomic_json(run_manifest_path,run_manifest)

    print("="*96)
    print("PROGRAM 05 FORMAL 8/8 THREE-PROCESS SUPERVISOR")
    print(f"parallelism          : {PARALLELISM}")
    print(f"chunk/process        : {args.chunk} new shard units")
    print(f"seed-target tasks    : {len(tasks)}")
    print(f"pending targets      : {len(pending)}")
    print(f"existing 8/8 units  : {initial_units} / {expected_total}")
    print(f"runner manifest      : {run_manifest_path}")
    print("Global indexes are written only once, after all 3 lanes finish.")
    print("="*96)

    q=queue.Queue()
    for x in pending: q.put(x)
    stop=threading.Event()
    err_lock=threading.Lock()
    errors=[]
    print_lock=threading.Lock()
    logdir=root/"benchmarks"/"program05_parallel3_8x8"/"formal_logs"
    logdir.mkdir(parents=True,exist_ok=True)
    flags=getattr(subprocess,"CREATE_NO_WINDOW",0)

    def lane(lane_id:int):
        nonlocal errors
        while not stop.is_set():
            try:
                seed,step=q.get_nowait()
            except queue.Empty:
                return
            try:
                while not stop.is_set() and not complete(root,seed,step):
                    before=count_units(root,seed,step)
                    with print_lock:
                        print(f"[LANE {lane_id}] start seed={seed} e={step:03d} progress={before}/165")
                    log=logdir/f"seed_{seed}_target_{step:04d}.log"
                    with log.open("a",encoding="utf-8") as fo:
                        fo.write(f"\n=== {now_utc()} lane={lane_id} before={before}/165 ===\n")
                        fo.flush()
                        cp=subprocess.run(
                            [sys.executable,str(worker),"--root",str(root),
                             "--seed",str(seed),"--target-step",str(step),
                             "--device","cuda","--max-new-shards",str(args.chunk)],
                            cwd=str(root),stdout=fo,stderr=subprocess.STDOUT,
                            creationflags=flags)
                    after=count_units(root,seed,step)
                    if cp.returncode!=0:
                        raise SupervisorError(f"worker rc={cp.returncode} seed={seed} target={step}; see {log}")
                    with print_lock:
                        print(f"[LANE {lane_id}] end   seed={seed} e={step:03d} progress={after}/165 complete={complete(root,seed,step)}")
                    if after==before and not complete(root,seed,step):
                        raise SupervisorError(f"No progress seed={seed} target={step}; see {log}")
                    time.sleep(3)
                with print_lock:
                    print(f"[LANE {lane_id}] TARGET COMPLETE seed={seed} e={step:03d}")
            except Exception as exc:
                with err_lock: errors.append(str(exc))
                stop.set()
            finally:
                q.task_done()

    threads=[threading.Thread(target=lane,args=(i+1,),daemon=False) for i in range(PARALLELISM)]
    start=time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed=time.time()-start

    if errors:
        run_manifest["status"]="failed"
        run_manifest["failed_at_utc"]=now_utc()
        run_manifest["errors"]=errors
        run_manifest["elapsed_seconds_before_failure"]=elapsed
        run_manifest["saved_units_at_failure"]=sum(count_units(root,*x) for x in tasks)
        atomic_json(run_manifest_path,run_manifest)
        print("\nPROGRAM 05 PARALLEL SUPERVISOR STOPPED ON ERROR")
        for e in errors: print("ERROR:",e)
        return 2

    if any(not complete(root,*x) for x in tasks):
        fail("Workers exited but not all 33 targets are complete.")

    print("\nAll 33 targets complete. Starting SINGLE-PROCESS finalization...")
    cp=subprocess.run([sys.executable,str(finalizer),"--root",str(root)],cwd=str(root))
    if cp.returncode!=0: return cp.returncode
    print("\nRunning SINGLE-PROCESS verify-only...")
    cp=subprocess.run([sys.executable,str(finalizer),"--root",str(root),"--verify-only"],cwd=str(root))
    if cp.returncode!=0: return cp.returncode

    final_units=sum(count_units(root,*x) for x in tasks)
    run_manifest["status"]="completed_and_verified"
    run_manifest["completed_at_utc"]=now_utc()
    run_manifest["parallel_generation_elapsed_seconds"]=elapsed
    run_manifest["final_8x8_online_shard_units"]=final_units
    atomic_json(run_manifest_path,run_manifest)

    print("\n"+"="*96)
    print("PROGRAM 05 FORMAL 8/8 THREE-PROCESS RUN COMPLETED AND VERIFIED")
    print(f"final units          : {final_units} / {expected_total}")
    print(f"generation elapsed   : {elapsed/3600:.2f} hours")
    print(f"runner manifest      : {run_manifest_path}")
    print("="*96)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except SupervisorError as exc:
        print(f"\nPROGRAM 05 PARALLEL SUPERVISOR FAILED\n{exc}",file=sys.stderr)
        raise SystemExit(2)
