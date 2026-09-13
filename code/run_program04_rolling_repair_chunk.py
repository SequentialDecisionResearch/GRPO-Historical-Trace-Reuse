#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, sys
from pathlib import Path

SEL=Path('manifests/reduced_test_pair_selection_program06_compatible.json')
REPAIR=Path('manifests/protocol_amendment_program06_rolling_comparator_repair.json')
P04=Path('04_rescore_target_checkpoints.py')
MORE=75
class E(RuntimeError): pass
class StopChunk(RuntimeError): pass
def fail(x): raise E(x)
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
def loadmod(p):
    s=importlib.util.spec_from_file_location('p04_rolling_repair_original',p)
    if s is None or s.loader is None: fail(f'Cannot import {p}')
    m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
def cleanup():
    try: gc.collect()
    except: pass
    try:
        import torch
        if torch.cuda.is_available():
            try: torch.cuda.synchronize()
            except: pass
            torch.cuda.empty_cache()
    except: pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--max-new-shards',type=int,default=20); ap.add_argument('--verify-only',action='store_true'); a=ap.parse_args()
    root=Path(a.root).resolve(); sp=root/SEL; rp=root/REPAIR; p04p=root/P04
    for p in (sp,rp,p04p):
        if not p.exists(): fail(f'Missing: {p}')
    sel=readj(sp); rep=readj(rp); rows=sel.get('selected_pairs')
    if not isinstance(rows,list) or len(rows)!=45: fail('Repaired selection must contain 45 pairs.')
    if ((rep.get('repaired_selection') or {}).get('sha256')!=sha(sp)): fail('Repair manifest / repaired selection SHA mismatch.')
    ids={str(r['pair_id']) for r in rows}
    p04=loadmod(p04p)
    original_ensure=p04.ensure_pair_registry
    def patched_ensure(**kw):
        registry,manifest=original_ensure(**kw)
        filtered=[p for p in registry if p.split!='test' or p.pair_id in ids]
        got={p.pair_id for p in filtered if p.split=='test'}
        if got!=ids: fail(f'Repaired Program04 registry filter mismatch; missing={sorted(ids-got)[:5]}')
        return filtered,manifest
    p04.ensure_pair_registry=patched_ensure
    state={'n':0}; original_publish=p04.publish_rescore_unit
    def counted_publish(**kw):
        out=original_publish(**kw); state['n']+=1
        if not a.verify_only and state['n']>=a.max_new_shards: raise StopChunk
        return out
    p04.publish_rescore_unit=counted_publish
    argv=['--config','configs/protocol.yaml','--mode','paper','--split','test','--resume','--device',a.device,'--output-root',str(root)]
    if a.verify_only: argv.append('--verify-only')
    print('='*92); print('PROGRAM 04 STRUCTURAL-COMPARATOR REPAIR WORKER'); print('selected TEST pairs : 45'); print(f'chunk limit         : {a.max_new_shards}'); print(f'verify only         : {a.verify_only}'); print('='*92)
    try:
        try: rc=int(p04.run(argv))
        except StopChunk:
            print(f'\nCHUNK COMPLETE: {state["n"]} new comparator/resume shard units published.'); return MORE
        return rc
    finally: cleanup()
if __name__=='__main__':
    try: raise SystemExit(main())
    except E as e:
        print(f'\nPROGRAM04 REPAIR WORKER FAILED\n{e}',file=sys.stderr); raise SystemExit(2)
