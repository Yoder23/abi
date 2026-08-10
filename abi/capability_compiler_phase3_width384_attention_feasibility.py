"""No-model final-system envelope for width-384 compact attention."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
FORMAT = "abi-capability-compiler-phase3-width384-attention-feasibility/1"
def accounting():
    copied=196_899_840; full=3072; width=384; layers=32; attention=layers*(2*full*width+4*width*width+width); residual=160_530_432; deployed=copied+attention+residual; active=307_540_992+layers*((2*full*width+4*width*width)-(2*full*192+4*192*192)); return {"copied_parameters":copied,"attention_parameters":attention,"nonlinear_rank768_residual_parameters":residual,"deployed_parameters":deployed,"fp16_payload_bytes":2*deployed,"active_incremental_macs_at_maximum_context":active,"source_to_target_active_mac_ratio":3_823_042_560/active}
def execute(root:Path,path:Path):
 p=json.loads(path.read_text(encoding="utf-8"))
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_NO_MODEL_WIDTH384_ATTENTION_FEASIBILITY" or any(p.get(n) is not False for n in ("teacher_model_loading_authorized","tensor_value_access_authorized","training_authorized","final_test_access_authorized")): raise Phase3Error("width384 governance changed")
 for n,h in p["bindings"].items():
  q=root/n
  if not q.is_file() or sha256_file(q)!=h: raise Phase3Error(f"binding changed: {n}")
 a=accounting(); gates={"payload_below_one_gib":a["fp16_payload_bytes"]<1024**3,"active_mac_margin_at_least_four":a["source_to_target_active_mac_ratio"]>=4,"zero_source_blocks":p["source_blocks"]==0}; return {"format":FORMAT,"status":"PASS_FEASIBLE_LOCAL_AUDIT_MAY_BE_DESIGNED" if all(gates.values()) else "FAIL_FEASIBILITY","protocol_sha256":sha256_file(path),"accounting":a,"gates":gates,"source_model_loaded":False,"training_performed":False,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"No-model width384 envelope only; no host, artifact, quality, runtime, certificate, or superiority claim."}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_WIDTH384_ATTENTION_FEASIBILITY_PROTOCOL_V279.json"); ap.add_argument("--output",default="results/abi_capability_compiler_phase3_width384_attention/feasibility_v280.json"); a=ap.parse_args(); root=Path.cwd().resolve(); out=root/a.output
 if out.exists(): raise Phase3Error("output exists")
 r=execute(root,root/a.protocol); _write_immutable(out,json.dumps(r,indent=2,sort_keys=True).encode()+b"\n"); print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r["status"].startswith("PASS") else 1
if __name__=="__main__": raise SystemExit(main())
