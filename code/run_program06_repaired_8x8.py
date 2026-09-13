#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any
SEL=Path('manifests/reduced_test_pair_selection_program06_compatible.json')
REPAIR=Path('manifests/protocol_amendment_program06_rolling_comparator_repair.json')
P05A=Path('manifests/protocol_amendment_program05_sample_size.json')
P05F=Path('manifests/program05_8x8_final_manifest.json')
P06=Path('06_compute_ope_and_diagnostics.py')
class E(RuntimeError): pass
def fail(x): raise E(x)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def readj(p):
 if not p.exists(): fail(f'Missing: {p}')
 x=json.loads(p.read_text(encoding='utf-8'))
 if not isinstance(x,dict): fail(f'Expected object: {p}')
 return x
def loadmod(p):
 s=importlib.util.spec_from_file_location('p06_repaired_original',p)
 if s is None or s.loader is None: fail(f'Cannot import {p}')
 m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
def main():
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root).resolve()
 sp=root/SEL; rp=root/REPAIR; p05a=root/P05A; p05f=root/P05F; p06p=root/P06
 for p in (sp,rp,p05a,p05f,p06p):
  if not p.exists(): fail(f'Missing: {p}')
 sel=readj(sp); rep=readj(rp); rows=sel.get('selected_pairs')
 if not isinstance(rows,list) or len(rows)!=45: fail('Repaired selection must contain 45 pairs.')
 if ((rep.get('repaired_selection') or {}).get('sha256')!=sha(sp)): fail('Repair manifest SHA mismatch.')
 p05=readj(p05a); d=p05.get('amended_effective_design') or {}
 if int(d.get('L_main',-1))!=8 or int(d.get('L_audit',-1))!=8: fail('Program05 amendment is not 8/8.')
 f=readj(p05f)
 if int(f.get('online_shard_units',-1))!=5445: fail('Program05 final units !=5445.')
 ids={str(r['pair_id']) for r in rows}
 p06=loadmod(p06p); orig=p06.load_pairs
 def patched(root_arg,mode,split,spec,pv,config_sha256,upstream):
  pairs,manifest=orig(root_arg,mode,split,spec,pv,config_sha256,upstream)
  if split!='test': return pairs,manifest
  out=[p for p in pairs if p.pair_id in ids]; got={p.pair_id for p in out}
  if got!=ids: fail(f'Program06 repaired selection mismatch; missing={sorted(ids-got)[:5]}')
  out.sort(key=lambda x:(x.training_seed,x.behavior_step,x.target_step,x.purpose)); return out,manifest
 p06.load_pairs=patched
 print('='*94); print('PROGRAM 06 STRUCTURALLY REPAIRED OFFICIAL TEST RUNNER'); print('selected TEST pairs : 45'); print('analysis pairs      : 33 (24 fixed + 9 rolling)'); print('Program05 reference : effective 8/8'); print('='*94)
 rc=int(p06.main(['--config','configs/protocol.yaml','--mode','paper','--split','test','--resume','--device','cpu','--output-root',str(root)]))
 if rc!=0: return rc
 mp=root/'outputs'/'diagnostics'/'program06_test_manifest.json'; m=readj(mp)
 m['structural_repair']={'manifest_path':str(REPAIR).replace('\\','/'),'manifest_sha256':sha(rp),'repaired_selection_path':str(SEL).replace('\\','/'),'repaired_selection_sha256':sha(sp),'pair_count':45}
 tmp=mp.with_name('.'+mp.name+'.tmp'); tmp.write_text(json.dumps(m,indent=2,sort_keys=True),encoding='utf-8'); tmp.replace(mp)
 counts=m.get('row_counts_current_split') or {}
 print('\n'+'='*94); print('PROGRAM 06 REPAIRED OFFICIAL TEST COMPLETED'); print('T02 rows:',counts.get('T02_fixed_reuse')); print('T03 rows:',counts.get('T03_overlap')); print('T04 rows:',counts.get('T04_rolling_comparison')); print('manifest:',mp); print('='*94)
 return 0
if __name__=='__main__':
 try: raise SystemExit(main())
 except E as e:
  print(f'\nPROGRAM06 REPAIRED RUNNER FAILED\n{e}',file=sys.stderr); raise SystemExit(2)
