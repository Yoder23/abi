"""Exact functional-embedding repair for V249."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from . import capability_compiler_phase3_direct_linear_integration_audit as base
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error,_write_immutable
FORMAT="abi-capability-compiler-phase3-direct-linear-integration-replay/1"
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_LINEAR_INTEGRATION_REPLAY_V251.json"); ap.add_argument("--output",default="results/abi_capability_compiler_phase3_direct_linear/integration_replay_v252.json"); a=ap.parse_args(); root=Path.cwd().resolve(); p=json.loads((root/a.protocol).read_text(encoding="utf-8"))
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_EXACT_EMBEDDING_REPAIR": raise Phase3Error("replay governance changed")
 for n,h in p["bindings"].items():
  q=root/n
  if not q.is_file() or sha256_file(q)!=h: raise Phase3Error(f"binding changed: {n}")
 torch.nn.Parameter.__call__=lambda self,ids:F.embedding(ids,self)
 out=root/a.output
 if out.exists(): raise Phase3Error("output exists")
 r=base.execute(root,root/p["base_protocol"]); r["repair_protocol"]={"path":a.protocol,"sha256":sha256_file(root/a.protocol),"scientific_fields_changed":False}; _write_immutable(out,json.dumps(r,indent=2,sort_keys=True).encode()+b"\n"); print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r["status"].startswith("PASS") else 1
if __name__=="__main__": raise SystemExit(main())
