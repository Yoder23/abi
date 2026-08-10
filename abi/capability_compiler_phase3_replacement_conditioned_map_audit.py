"""Exact V249 replay fitting the map on compact-attention features."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from . import capability_compiler_phase3_direct_linear_integration_audit as base
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error,_write_immutable
FORMAT="abi-capability-compiler-phase3-replacement-conditioned-map/1"
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_REPLACEMENT_CONDITIONED_MAP_PROTOCOL_V253.json"); ap.add_argument("--output",default="results/abi_capability_compiler_phase3_direct_linear/replacement_conditioned_map_v254.json"); a=ap.parse_args(); root=Path.cwd().resolve(); p=json.loads((root/a.protocol).read_text(encoding="utf-8"))
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_INTERFACE_REPAIR": raise Phase3Error("governance changed")
 for n,h in p["bindings"].items():
  q=root/n
  if not q.is_file() or sha256_file(q)!=h: raise Phase3Error(f"binding changed: {n}")
 bp=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8")); lc=(root/bp["layercake_host"]["repository"]).resolve(); sys.path.insert(0,str(lc)); from layercake.direct_linear_progressive_core import DirectLinearProgressiveCore
 model=DirectLinearProgressiveCore(fixed_vocab_size=32015,full_width=3072,bottleneck_width=192,attention_heads=2,replacement_layers=1,intermediate_size=768); sub=load_file(str(root/bp["substrate"])); model.load_state_dict({k:v for k,v in sub.items() if not k.startswith("layers.") or k.startswith("layers.0.")},strict=False); neg=load_file(str(root/bp["attention_checkpoint"])); state=model.state_dict()
 for k,v in neg.items():
  if k.startswith("layers.0.") and any(x in k for x in ("attention_","qkv_proj","o_proj")) and k in state and state[k].shape==v.shape: state[k].copy_(v.to(state[k].dtype))
 model.cuda().eval(); original=base.dual._teacher_components
 def patched(teacher,index,hidden):
  module=teacher.model.layers[0].post_attention_layernorm; saved=module.forward; target=original(teacher,index,hidden); student,_=base.dual._student_components(model.layers[0],hidden,torch.arange(hidden.shape[1],device=hidden.device)); feature=model.layers[0].post_attention_norm(student).float(); module.forward=lambda ignored:feature; patched.saved=(module,saved); return target
 def dispatch(teacher,index,hidden):
  if hasattr(patched,"saved"): patched.saved[0].forward=patched.saved[1]; del patched.saved
  return patched(teacher,index,hidden)
 base.dual._teacher_components=dispatch; torch.nn.Parameter.__call__=lambda self,ids:torch.nn.functional.embedding(ids,self); out=root/a.output
 if out.exists(): raise Phase3Error("output exists")
 r=base.execute(root,root/p["base_protocol"]); r["replacement_conditioned_features"]=True; r["protocol"]={"path":a.protocol,"sha256":sha256_file(root/a.protocol)}; _write_immutable(out,json.dumps(r,indent=2,sort_keys=True).encode()+b"\n"); print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r["status"].startswith("PASS") else 1
if __name__=="__main__": raise SystemExit(main())
