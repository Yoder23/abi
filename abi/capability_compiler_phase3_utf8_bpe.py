"""Training-only UTF-8-concatenative BPE representation bake-off."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Iterable
from tokenizers import Tokenizer, models, trainers
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import rows
FORMAT="abi-capability-compiler-phase3-utf8-bpe/1"

def load_protocol(root,path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_REPRESENTATION_FIT" or p.get("neural_training_allowed") is not False or p.get("final_test_access")!="PROHIBITED": raise Phase3Error("V34 governance changed")
 for rel,want in p["bindings"].items():
  target=(root/rel).resolve()
  if not target.is_file() or sha256_file(target)!=want: raise Phase3Error(f"V34 binding changed: {rel}")
 return p,sha256_file(path)

def fit(strings,vocab_size):
 t=Tokenizer(models.BPE(unk_token="[UNK]")); trainer=trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,special_tokens=["[UNK]"],initial_alphabet=[chr(v) for v in range(0x20,0x7F)],show_progress=False); t.train_from_iterator(strings,trainer); return t

def evaluate(t,dataset):
 total=0; maximum=0; failing=[]; errors=[]; stream=hashlib.sha256()
 for row in dataset:
  e=t.encode(row["output"]); reconstructed="".join(e.tokens); count=len(e.tokens)+1; total+=count; maximum=max(maximum,count)
  if reconstructed!=row["output"] or "[UNK]" in e.tokens: errors.append(row["record_id"])
  if count>320:failing.append({"record_id":row["record_id"],"capability":row["capability"],"actions":count})
  stream.update(canonical_json_bytes({"record_id":row["record_id"],"tokens":e.tokens}))
 return {"records":len(dataset),"mean_actions":total/len(dataset),"maximum_actions":maximum,"over_320":len(failing),"failing":failing,"reconstruction_errors":errors,"stream_sha256":stream.hexdigest()}

def execute(root:Path,protocol_path:Path)->dict[str,Any]:
 p,ph=load_protocol(root,protocol_path); train,dev=rows(root,p); strings=[row[field] for row in train for field in ("prompt","output")]+["".join(chr(v) for v in range(0x20,0x7F))]
 candidates={}
 for budget in p["bpe"]["budgets"]:
  t=fit(strings,budget); vocab=t.get_vocab(); tr=evaluate(t,train); dv=evaluate(t,dev); actual=len(vocab)+3; ok=not tr["reconstruction_errors"] and not dv["reconstruction_errors"] and tr["over_320"]==0 and dv["over_320"]==0 and actual<=p["qualification"]["maximum_fixed_actions"]
  candidates[str(budget)]={"requested_vocabulary":budget,"actual_fixed_actions":actual,"qualifying":ok,"training":tr,"development":dv,"tokenizer_sha256":hashlib.sha256(t.to_str().encode()).hexdigest()}
 passing=[v for v in candidates.values() if v["qualifying"]]; selected=min(passing,key=lambda v:v["requested_vocabulary"])["requested_vocabulary"] if passing else None
 result={"format":"abi-capability-compiler-phase3-utf8-bpe-result/1","status":"PASS_REPRESENTATION_ONLY" if selected else "FAIL_REPRESENTATION","protocol":{"path":protocol_path.name,"sha256":ph},"representation_fit_performed":True,"neural_training_performed":False,"teacher_outputs_used":"SEALED_PHASE1_TRAINING_ONLY","final_test_accessed":False,"phase3_certified":False,"candidates":candidates,"selected_budget":selected,"host_change_authorized":False,"model_training_authorized":False,"claim_boundary":"Tokenizer representation result only."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();return result

def main(argv:Iterable[str]|None=None)->int:
 a=argparse.ArgumentParser();a.add_argument("command",choices=("execute","verify"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_UTF8_BPE_PROTOCOL_V34.json");a.add_argument("--output",default="results/abi_capability_compiler_phase3_utf8_bpe/utf8_bpe_v34.json");x=a.parse_args(argv);root=Path.cwd().resolve();e=execute(root,(root/x.protocol).resolve());o=(root/x.output).resolve()
 if x.command=="execute":
  if o.exists():raise Phase3Error(f"V34 output immutable: {o}")
  _write_immutable(o,json.dumps(e,indent=2,sort_keys=True).encode()+b"\n")
 elif _json(o)!=e:raise Phase3Error("stored V34 result differs")
 print(json.dumps({"status":e["status"],"selected_budget":e["selected_budget"],"evidence_sha256":e["evidence_sha256"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
