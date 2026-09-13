#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OLD_SEL = Path('manifests/reduced_test_pair_selection.json')
GRID_AMEND = Path('manifests/protocol_amendment_reduced_grid.json')
NEW_SEL = Path('manifests/reduced_test_pair_selection_program06_compatible.json')
REPAIR = Path('manifests/protocol_amendment_program06_rolling_comparator_repair.json')
P04 = Path('04_rescore_target_checkpoints.py')
P06 = Path('06_compute_ope_and_diagnostics.py')
REQUIRED_TARGETS = (160,260,360)
SEEDS = (20260826,20260827,20260828)

class E(RuntimeError): pass
def fail(x): raise E(x)
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def readj(p):
    if not p.exists(): fail(f'Missing: {p}')
    x=json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(x,dict): fail(f'Expected object: {p}')
    return x
def writej(p,x):
    t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False),encoding='utf-8'); t.replace(p)
def semhash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def test_row_count(path: Path) -> int:
    if not path.exists(): return 0
    n=0
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if str(r.get('split','')).strip().lower()=='test': n+=1
    return n

def main():
    import argparse, pyarrow.parquet as pq
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root).resolve()
    oldp=root/OLD_SEL; gridp=root/GRID_AMEND; regp=root/'manifests'/'pair_registry.parquet'; newp=root/NEW_SEL; repairp=root/REPAIR
    for p in (oldp,gridp,regp,root/P04,root/P06):
        if not p.exists(): fail(f'Missing required file: {p}')
    old=readj(oldp); grid=readj(gridp)
    rows=old.get('selected_pairs')
    if not isinstance(rows,list) or len(rows)!=36: fail('Expected original reduced selection to contain 36 pairs.')
    old_ids={str(r['pair_id']) for r in rows}
    if ((grid.get('selection_manifest') or {}).get('sha256')!=sha(oldp)): fail('Original reduced-grid amendment/selection SHA mismatch.')
    reg=pq.read_table(regp).to_pylist()
    needed=[]
    for seed in SEEDS:
        for e in REQUIRED_TARGETS:
            hits=[r for r in reg if str(r.get('split'))=='test' and int(r.get('training_seed'))==seed and int(r.get('behavior_step'))==0 and int(r.get('target_step'))==e and str(r.get('purpose'))=='fixed']
            if len(hits)!=1: fail(f'Expected one fixed comparator for seed={seed}, e={e}; found {len(hits)}')
            needed.append(dict(hits[0]))
    needed_ids={str(r['pair_id']) for r in needed}
    if old_ids & needed_ids: fail('At least one structural comparator is already in the 36-pair selection; unexpected repair state.')
    repaired=list(rows)+needed
    repaired.sort(key=lambda r:(int(r['training_seed']),int(r['behavior_step']),int(r['target_step']),str(r['purpose'])))
    if len(repaired)!=45 or len({str(r['pair_id']) for r in repaired})!=45: fail('Repaired selection is not exactly 45 unique pairs.')

    new_obj=dict(old)
    new_obj['selected_pairs']=repaired
    new_obj['selected_pair_count']=45
    new_obj['structural_repair']={
        'created_at_utc':now(),
        'reason':'Program06 rolling comparison requires fixed old-behavior comparator (seed, behavior_step=0, same target_step=e) for every retained rolling pair. The 36-pair reduced grid retained rolling targets e=160,260,360 but omitted those fixed comparator pairs.',
        'added_pair_count':9,
        'added_pair_ids':sorted(needed_ids),
        'added_pairs':needed,
        'metric_values_used_for_selection':False,
    }
    new_obj['content_sha256']=semhash({k:v for k,v in new_obj.items() if k!='content_sha256'})
    writej(newp,new_obj)

    p06_manifest=root/'outputs'/'diagnostics'/'program06_test_manifest.json'
    t02=root/'outputs'/'tables'/'T02_fixed_reuse.csv'; t03=root/'outputs'/'tables'/'T03_overlap.csv'; t04=root/'outputs'/'tables'/'T04_rolling_comparison.csv'
    repair={
        'schema_version':'1.0',
        'manifest_type':'post_start_structural_program06_rolling_comparator_repair',
        'created_at_utc':now(),
        'original_reduced_selection':{'path':str(OLD_SEL).replace('\\','/'),'sha256':sha(oldp),'pair_count':36},
        'repaired_selection':{'path':str(NEW_SEL).replace('\\','/'),'sha256':sha(newp),'pair_count':45},
        'added_structural_comparators':needed,
        'deterministic_dependency':{
            'rolling_targets':[160,260,360],
            'required_old_behavior_step':0,
            'required_pairs_per_seed':3,
            'total_added_pairs':9,
            'rule':'For each retained rolling pair (seed,b,e), Program06 build_rolling_rows requires pair_results[(seed,0,e)].',
        },
        'failure_context':{
            'program06_test_manifest_present':p06_manifest.exists(),
            'T02_test_rows_present':test_row_count(t02),
            'T03_test_rows_present':test_row_count(t03),
            'T04_test_rows_present':test_row_count(t04),
            'statement':'An official-test Program06 run computed pair statistics and then failed while constructing/provenancing T04 because the old-behavior same-target comparators were absent. This repair is dictated solely by the pre-existing rolling-comparison code dependency; no test metric values are used to choose the nine added pairs.'
        },
        'provenance':{
            'original_grid_amendment_sha256':sha(gridp),
            'pair_registry_sha256':sha(regp),
            'program04_source_sha256':sha(root/P04),
            'program06_source_sha256':sha(root/P06),
        },
        'scientific_effect':'Expands the computational TEST pair set from 36 to 45 only to restore the required fixed comparator for each retained rolling case. The retained rolling cases, frozen gate, Program05 8/8 reference amendment, estimators, K, and bootstrap specification are unchanged.',
        'paper_disclosure':'After the reduced-grid computation had begun, we identified a structural dependency in the rolling comparison: each retained rolling pair requires the corresponding behavior-step-0 evaluation at the same target checkpoint. We therefore added exactly these nine comparator pairs (three targets across three seeds), without using test outcome values to select them.'
    }
    repair['content_sha256']=semhash(repair)
    writej(repairp,repair)
    print('='*94)
    print('PROGRAM 06 ROLLING-COMPARATOR STRUCTURAL REPAIR LOCKED')
    print('original pairs      : 36')
    print('added comparators   : 9')
    print('repaired pairs      : 45')
    print('added targets       : 160, 260, 360 at behavior_step=0 for all 3 seeds')
    print(f'T02 test rows exist : {test_row_count(t02)}')
    print(f'T03 test rows exist : {test_row_count(t03)}')
    print(f'T04 test rows exist : {test_row_count(t04)}')
    print(f'repaired selection  : {newp}')
    print(f'repair manifest     : {repairp}')
    print(f'repair SHA-256      : {sha(repairp)}')
    print('='*94)
    return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except E as e:
        print(f'\nREPAIR PREPARATION FAILED\n{e}',file=sys.stderr); raise SystemExit(2)
