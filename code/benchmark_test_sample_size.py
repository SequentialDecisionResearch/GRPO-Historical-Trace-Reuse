#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, statistics, sys
from pathlib import Path

P06_CANDIDATES=("06_compute_ope_and_diagnostics.py","scripts/06_compute_ope_and_diagnostics.py")

def die(m): raise RuntimeError(m)

def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    if s is None or s.loader is None: die(f"Cannot import {path}")
    m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m

def find_p06(root):
    for rel in P06_CANDIDATES:
        p=root/rel
        if p.exists(): return p
    die("Cannot find Program 06.")

def mean(xs): return math.fsum(float(x) for x in xs)/len(xs)

def q95(xs):
    ys=sorted(float(x) for x in xs)
    if not ys: return math.nan
    p=.95*(len(ys)-1); a=int(math.floor(p)); b=int(math.ceil(p))
    return ys[a] if a==b else ys[a]*(b-p)+ys[b]*(p-a)

def rank_hash(seed,rep,tr_seed,b,source):
    txt=f"program04-test-sample-size-v1|{seed}|{rep}|{tr_seed}|{b}|{source}"
    return hashlib.sha256(txt.encode()).hexdigest()

def choose_sources(universe,fraction,seed,rep,tr_seed,b):
    ordered=sorted(universe,key=lambda h:(rank_hash(seed,rep,tr_seed,b,h),h))
    n=max(1,min(len(ordered),int(round(fraction*len(ordered)))))
    return set(ordered[:n])

def gate_map(gate):
    out={}
    for r in gate["primary_gate"]["thresholds"]:
        out[float(r["tolerance"])]={
            "threshold":float(r["threshold"]),
            "reject_all":bool(r.get("reject_all",False)),
            "direction":str(r.get("direction","ge"))}
    return out

def accepted(ress,r):
    return False if r["reject_all"] else ress>=r["threshold"]

def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--fractions",default="0.20,0.25,0.33,0.40,0.50")
    ap.add_argument("--reps",type=int,default=100)
    ap.add_argument("--sampling-seed",type=int,default=20260909)
    ap.add_argument("--output-dir",default="benchmarks/test_sample_size")
    ap.add_argument("--min-gate-agreement",type=float,default=.97)
    ap.add_argument("--max-accept-rate-diff",type=float,default=.05)
    ap.add_argument("--max-median-error-change",type=float,default=.002)
    ap.add_argument("--max-p95-error-change",type=float,default=.005)
    args=ap.parse_args(argv)

    root=Path(args.root).resolve()
    fractions=sorted(set(float(x) for x in args.fractions.split(",")))
    p06=load_module(find_p06(root),"program06_sample_size_benchmark")
    cfg,cfg_sha=p06.load_protocol(root/"configs"/"protocol.yaml")
    spec=p06.resolve_spec(cfg,"paper",None); pv=p06.protocol_version(cfg)
    p06.verify_protocol_lock(root,cfg_sha)
    upstream=p06.verify_upstream_static_manifests(root)
    pairs,_=p06.load_pairs(root,"paper","development",spec,pv,cfg_sha,upstream)
    pairs=[p for p in pairs if p.purpose in ("fixed","rolling")]

    gate=json.loads((root/"outputs"/"frozen_gate.json").read_text(encoding="utf-8"))
    thresholds=gate_map(gate)
    cal=set(int(x) for x in gate["calibration_seeds"])
    val=set(int(x) for x in gate["validation_seeds"])
    rroot=p06.rescore_data_root(root,"paper"); oroot=p06.online_data_root(root,"paper")

    print("="*88)
    print("DEVELOPMENT-ONLY TEST SAMPLE-SIZE BENCHMARK")
    print(f"pairs={len(pairs)} fractions={fractions} reps={args.reps}")
    print("sampling unit = Program-04 source shard; TEST outcomes are not read")
    print("="*88)

    online_cache={}; universe_by_beh={}; cache=[]
    for i,p in enumerate(pairs,1):
        ok=(p.training_seed,p.target_step)
        if ok not in online_cache:
            online_cache[ok]=p06.load_online_prompt_summary(oroot,"development",p.training_seed,p.target_step)[0]
        online=online_cache[ok]
        rows,_=p06.load_rescore_rows(rroot,p,spec.k_main)
        stats=p06.pair_prompt_stats(rows)
        sby={str(s.prompt_id):s for s in stats}
        psrc={}
        for r in rows:
            pid=str(r["prompt_id"]); sh=str(r["source_behavior_content_sha256"])
            if pid in psrc and psrc[pid]!=sh: die(f"Prompt spans multiple shards: {pid}")
            psrc[pid]=sh
        uni=set(psrc.values()); bk=(p.training_seed,p.behavior_step)
        if bk in universe_by_beh and universe_by_beh[bk]!=uni:
            die(f"Source universe differs across targets for {bk}")
        universe_by_beh[bk]=uni
        full_est=float(p06.estimate_pwis(stats))
        full_online=mean([online[s.prompt_id] for s in stats])
        cache.append(dict(p=p,sby=sby,psrc=psrc,online=online,uni=sorted(uni),
                          full_est=full_est,full_online=full_online,
                          full_err=abs(full_est-full_online),
                          full_ress=float(statistics.median(float(s.relative_ess) for s in stats))))
        print(f"[{i:03d}/{len(pairs):03d}] seed={p.training_seed} b={p.behavior_step:03d} e={p.target_step:03d} loaded")

    detail=[]
    for frac in fractions:
        print(f"\n[FRACTION {frac:.2f}]")
        for rep in range(args.reps):
            if rep % max(1,args.reps//10)==0: print(f"  replicate {rep+1}/{args.reps}")
            selected={bk:choose_sources(u,frac,args.sampling_seed,rep,bk[0],bk[1]) for bk,u in universe_by_beh.items()}
            for x in cache:
                p=x["p"]; ss=selected[(p.training_seed,p.behavior_step)]
                pids=sorted(pid for pid,sh in x["psrc"].items() if sh in ss)
                st=[x["sby"][pid] for pid in pids]
                est=float(p06.estimate_pwis(st)); on=mean([x["online"][pid] for pid in pids])
                err=abs(est-on); ress=float(statistics.median(float(s.relative_ess) for s in st))
                role="calibration" if p.training_seed in cal else "validation" if p.training_seed in val else "other"
                base=dict(fraction=frac,replicate=rep,training_seed=p.training_seed,
                          behavior_step=p.behavior_step,target_step=p.target_step,purpose=p.purpose,
                          seed_role=role,n_sources_full=len(x["uni"]),n_sources_selected=len(ss),
                          n_prompts_selected=len(pids),full_estimate=x["full_est"],subset_estimate=est,
                          abs_estimate_change=abs(est-x["full_est"]),
                          full_absolute_error=x["full_err"],subset_absolute_error=err,
                          abs_error_change=abs(err-x["full_err"]),
                          full_median_relative_ess=x["full_ress"],subset_median_relative_ess=ress,
                          abs_ress_change=abs(ress-x["full_ress"]))
                for eps,tr in sorted(thresholds.items()):
                    fa=accepted(x["full_ress"],tr); sa=accepted(ress,tr)
                    detail.append({**base,"tolerance":eps,"threshold":tr["threshold"],
                                   "full_accept":int(fa),"subset_accept":int(sa),
                                   "gate_agreement":int(fa==sa),
                                   "subset_false_accept":int(sa and err>eps)})

    summary=[]
    scopes={"all":lambda r:True,"calibration":lambda r:r["seed_role"]=="calibration",
            "validation":lambda r:r["seed_role"]=="validation"}
    for frac in fractions:
        for eps in sorted(thresholds):
            for scope,pred in scopes.items():
                rr=[r for r in detail if r["fraction"]==frac and r["tolerance"]==eps and pred(r)]
                reps=[]
                for rep in range(args.reps):
                    xx=[r for r in rr if r["replicate"]==rep]
                    if not xx: continue
                    reps.append(dict(
                        agreement=mean([r["gate_agreement"] for r in xx]),
                        accept_diff=abs(mean([r["subset_accept"] for r in xx])-mean([r["full_accept"] for r in xx])),
                        median_err_change=statistics.median(r["abs_error_change"] for r in xx),
                        p95_err_change=q95([r["abs_error_change"] for r in xx])))
                summary.append(dict(
                    fraction=frac,tolerance=eps,scope=scope,
                    mean_gate_agreement=mean([r["agreement"] for r in reps]),
                    mean_accept_rate_diff=mean([r["accept_diff"] for r in reps]),
                    mean_median_error_change=mean([r["median_err_change"] for r in reps]),
                    p95_of_p95_error_change=q95([r["p95_err_change"] for r in reps])))

    target_eps=[e for e in (0.02,0.05) if e in thresholds]
    checks=[]; recommended=None
    for frac in fractions:
        rel=[r for r in summary if r["fraction"]==frac and r["tolerance"] in target_eps and r["scope"] in ("all","validation")]
        passed=bool(rel); reasons=[]
        for r in rel:
            if r["mean_gate_agreement"]<args.min_gate_agreement: passed=False; reasons.append("gate_agreement")
            if r["mean_accept_rate_diff"]>args.max_accept_rate_diff: passed=False; reasons.append("accept_rate_diff")
            if r["mean_median_error_change"]>args.max_median_error_change: passed=False; reasons.append("median_error_change")
            if r["p95_of_p95_error_change"]>args.max_p95_error_change: passed=False; reasons.append("p95_error_change")
        checks.append(dict(fraction=frac,passed=passed,reasons=sorted(set(reasons))))
        if passed and recommended is None: recommended=frac

    outdir=Path(args.output_dir)
    if not outdir.is_absolute(): outdir=root/outdir
    outdir.mkdir(parents=True,exist_ok=True)
    write_csv(outdir/"sample_size_detail.csv",detail)
    write_csv(outdir/"sample_size_summary.csv",summary)
    rec=dict(recommended_fraction=recommended,checks=checks,fractions=fractions,reps=args.reps,
             calibration_seeds=sorted(cal),validation_seeds=sorted(val),
             criteria=dict(min_gate_agreement=args.min_gate_agreement,
                           max_accept_rate_diff=args.max_accept_rate_diff,
                           max_median_error_change=args.max_median_error_change,
                           max_p95_error_change=args.max_p95_error_change))
    (outdir/"sample_size_recommendation.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")

    print("\n"+"="*88)
    print("DECISION")
    for c in checks: print(f"fraction={c['fraction']:.2f} -> {'PASS' if c['passed'] else 'FAIL'}")
    print("RECOMMENDED FRACTION:", "NONE" if recommended is None else f"{recommended:.2f}")
    print("summary:",outdir/"sample_size_summary.csv")
    print("recommendation:",outdir/"sample_size_recommendation.json")
    print("="*88)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
