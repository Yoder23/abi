"""Prompt-preserving pruned-lexeme representation successor."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Iterable
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import rows, tokenizer_type
from .capability_compiler_phase3_compact_sublexeme import evaluate, ranked_sublexemes
FORMAT="abi-capability-compiler-phase3-pruned-sublexeme/1"

def load_protocol(root,path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_NO_TRAINING" or p.get("training_allowed") is not False or p.get("final_test_access")!="PROHIBITED": raise Phase3Error("V33 governance changed")
 for rel,want in p["bindings"].items():
  target=(root/rel).resolve()
  if not target.is_file() or sha256_file(target)!=want: raise Phase3Error(f"V33 binding changed: {rel}")
 return p,sha256_file(path)

def execute(root:Path,protocol_path:Path)->dict[str,Any]:
 p,ph=load_protocol(root,protocol_path); train,dev=rows(root,p); T=tokenizer_type(root,p)
 prompt={piece for row in train for piece in T.split(row["prompt"])}; output={piece for row in train for piece in T.split(row["output"])}
 ascii_set={bytes((v,)) for v in range(0x20,0x7F)}; characters={c.encode("utf-8") for row in train for c in row["output"]}|ascii_set
 full=prompt|ascii_set; ranked=ranked_sublexemes(train,T.split,full|characters,minimum_length=p["sublexemes"]["minimum_characters"],maximum_length=p["sublexemes"]["maximum_characters"])
 candidates={}
 for budget in p["sublexemes"]["budgets"]:
  added=set(ranked[:budget]); fallback=characters|added; tr=evaluate(train,T.split,full,fallback); dv=evaluate(dev,T.split,full,fallback); fixed=len(full|fallback)+4; ok=tr["over_320"]==0 and dv["over_320"]==0 and fixed<=p["qualification"]["maximum_fixed_actions"]
  candidates[str(budget)]={"budget":budget,"fixed_actions":fixed,"qualifying":ok,"training":tr,"development":dv,"vocabulary_sha256":hashlib.sha256(b"".join(x+b"\0" for x in sorted(added))).hexdigest()}
 passing=[v for v in candidates.values() if v["qualifying"]]; selected=min(passing,key=lambda v:v["budget"])["budget"] if passing else None
 result={"format":"abi-capability-compiler-phase3-pruned-sublexeme-result/1","status":"PASS_REPRESENTATION_ONLY" if selected else "FAIL_REPRESENTATION","protocol":{"path":protocol_path.name,"sha256":ph},"inventory":{"prompt_lexemes":len(prompt),"output_lexemes":len(output),"output_only_lexemes_pruned":len(output-prompt)},"candidates":candidates,"selected_budget":selected,"training_performed":False,"final_test_accessed":False,"phase3_certified":False,"host_change_authorized":False,"model_training_authorized":False,"claim_boundary":"Representation feasibility only."}; result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest(); return result

def main(argv:Iterable[str]|None=None)->int:
 a=argparse.ArgumentParser();a.add_argument("command",choices=("execute","verify"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_PRUNED_SUBLEXEME_PROTOCOL_V33.json");a.add_argument("--output",default="results/abi_capability_compiler_phase3_pruned_sublexeme/pruned_sublexeme_v33.json");x=a.parse_args(argv);root=Path.cwd().resolve();e=execute(root,(root/x.protocol).resolve());o=(root/x.output).resolve()
 if x.command=="execute":
  if o.exists(): raise Phase3Error(f"V33 output is immutable: {o}")
  _write_immutable(o,json.dumps(e,indent=2,sort_keys=True).encode()+b"\n")
 elif _json(o)!=e: raise Phase3Error("stored V33 result differs")
 print(json.dumps({"status":e["status"],"selected_budget":e["selected_budget"],"evidence_sha256":e["evidence_sha256"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
