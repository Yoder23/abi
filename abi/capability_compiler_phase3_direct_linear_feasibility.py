"""No-model accounting for the direct-linear coefficient host."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT = "abi-capability-compiler-phase3-direct-linear-feasibility/1"

def accounting(full: int=3072, rank: int=192, layers: int=32):
    copied = 196_899_840
    attention_per_layer = 1_327_296
    imported_mlp_per_layer = 2 * full * rank + full
    attention_trainable = layers * attention_per_layer
    imported = layers * imported_mlp_per_layer
    deployed = copied + attention_trainable + imported
    active_macs = 199_013_376 - layers * (3 * rank * 768)
    return {"copied_parameters": copied, "trainable_attention_parameters": attention_trainable, "imported_coefficient_basis_mean_parameters": imported, "deployed_parameters": deployed, "fp16_payload_bytes": 2 * deployed, "active_incremental_macs_at_maximum_context": active_macs, "source_to_target_mac_ratio": 3_823_042_560 / active_macs}

def execute(root: Path, path: Path):
    p=json.loads(path.read_text(encoding="utf-8"))
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_NO_MODEL_FEASIBILITY": raise Phase3Error("direct-linear governance changed")
    if any(p.get(k) is not False for k in ("teacher_model_loading_authorized","tensor_value_access_authorized","training_authorized","final_test_access_authorized")): raise Phase3Error("direct-linear authorization changed")
    for n,h in p["bindings"].items():
        q=Path(n) if Path(n).is_absolute() else root/n
        if not q.is_file() or sha256_file(q)!=h: raise Phase3Error(f"binding changed: {n}")
    a=accounting(); gates={"deployed":a["deployed_parameters"]==277_220_352,"payload":a["fp16_payload_bytes"]<=629_145_600,"compute_reduced":a["active_incremental_macs_at_maximum_context"]==184_857_600,"margin":a["source_to_target_mac_ratio"]>=4,"zero_source_blocks":p["source_blocks"]==0}
    return {"format":FORMAT,"status":"PASS_FEASIBLE" if all(gates.values()) else "FAIL_FEASIBILITY","accounting":a,"gates":gates,"source_model_loaded":False,"tensor_values_read":False,"training_performed":False,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"No-model direct-linear accounting only; no artifact, quality, runtime, or superiority claim."}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_LINEAR_FEASIBILITY_PROTOCOL_V247.json"); ap.add_argument("--output",default="results/abi_capability_compiler_phase3_direct_linear/feasibility_v248.json"); a=ap.parse_args(); root=Path.cwd().resolve(); out=root/a.output
    if out.exists(): raise Phase3Error("output exists")
    r=execute(root,root/a.protocol); _write_immutable(out,json.dumps(r,indent=2,sort_keys=True).encode()+b"\n"); print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r["status"]=="PASS_FEASIBLE" else 1
if __name__=="__main__": raise SystemExit(main())
