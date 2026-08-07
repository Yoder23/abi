"""Read-only routed training-fit attribution for sealed V47."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from typing import Any, Iterable, Mapping
import torch
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_route_bridge import _collate, _examples, _json, _load_candidate

FORMAT="abi-capability-compiler-phase3-route-bridge-fit/1"

def load_protocol(root:Path,path:Path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_FIT" or p.get("final_test_access")!="PROHIBITED":raise Phase3Error("V49 governance changed")
 for rel,want in p["bindings"].items():
  target=(root/rel).resolve()
  if not target.is_file() or sha256_file(target)!=want:raise Phase3Error(f"V49 binding changed: {rel}")
 return p,sha256_file(path)

@torch.inference_mode()
def run(root:Path,path:Path,output:Path)->Mapping[str,Any]:
 p,ph=load_protocol(root,path)
 if output.exists():raise Phase3Error("V49 output exists")
 candidate=(root/p["candidate_dir"]).resolve();metadata=_json(candidate/"metadata.json")
 if sha256_file(candidate/"model.safetensors")!=p["checkpoint_sha256"]:raise Phase3Error("V47 checkpoint changed")
 model,tokenizer=_load_candidate(root,p,candidate);examples,_=_examples(root,p,tokenizer);rows=[];started=time.perf_counter();device=torch.device("cuda")
 for start in range(0,len(examples),int(p["batch_size"])):
  batch=examples[start:start+int(p["batch_size"])];source,targets=_collate(batch,device);log_probs=model(source,targets)["log_probs"].float();mask=targets.ge(0);safe=targets.clamp(min=0);predicted=log_probs.argmax(-1);correct=predicted.eq(targets)&mask;chosen=log_probs.gather(-1,safe[:,:,None]).squeeze(-1)
  for i,row in enumerate(batch):
   actions=int(mask[i].sum());right=int(correct[i].sum());rows.append({"record_id":row["record_id"],"capability":row["capability"],"actions":actions,"correct_actions":right,"exact_sequence":right==actions,"action_nll_sum":float((-chosen[i].masked_select(mask[i])).sum())})
 output.mkdir(parents=True);raw=output/"training_fit_rows.jsonl";raw.write_bytes(b"".join(canonical_json_bytes(x) for x in rows));actions=sum(x["actions"] for x in rows);right=sum(x["correct_actions"] for x in rows);exact=sum(x["exact_sequence"] for x in rows);receipt={"format":"abi-capability-compiler-phase3-route-bridge-fit-decision/1","status":"COMPLETE_ATTRIBUTION_ONLY","protocol_sha256":ph,"checkpoint_sha256":p["checkpoint_sha256"],"records":len(rows),"actions":actions,"correct_actions":right,"action_accuracy":right/actions,"exact_sequences":exact,"exact_sequence_rate":exact/len(rows),"mean_action_nll":sum(x["action_nll_sum"] for x in rows)/actions,"rows_sha256":sha256_file(raw),"wall_seconds":time.perf_counter()-started,"classification":"BRIDGE_OR_BACKBONE_FIT_LIMITED" if right/actions<float(p["fit_gates"]["action_accuracy_minimum"]) or exact/len(rows)<float(p["fit_gates"]["exact_sequence_rate_minimum"]) else "TRAINING_FIT_SUFFICIENT_HELDOUT_GENERALIZATION_LIMITED","training_authorized":False,"phase3_certified":False,"final_test_accessed":False};receipt["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(receipt)).hexdigest();_write_immutable(output/"decision.json",json.dumps(receipt,sort_keys=True,indent=2).encode()+b"\n");return receipt

def main(argv:Iterable[str]|None=None):
 a=argparse.ArgumentParser();a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_BRIDGE_FIT_PROTOCOL_V49.json");a.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_route_bridge_fit/attribution_v49");x=a.parse_args(argv);result=run(Path.cwd().resolve(),(Path.cwd()/x.protocol).resolve(),(Path.cwd()/x.output_dir).resolve());print(json.dumps(result,sort_keys=True,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
