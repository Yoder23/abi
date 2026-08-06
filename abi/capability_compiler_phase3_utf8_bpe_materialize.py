"""Materialize and verify the exact selected V34 UTF-8 BPE tokenizer."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Iterable
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import rows
from .capability_compiler_phase3_utf8_bpe import fit
FORMAT="abi-capability-compiler-phase3-utf8-bpe-materialize/1"

def load_protocol(root,path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_MATERIALIZATION" or p.get("neural_training_allowed") is not False or p.get("final_test_access")!="PROHIBITED": raise Phase3Error("V35 governance changed")
 for rel,want in p["bindings"].items():
  target=(root/rel).resolve()
  if not target.is_file() or sha256_file(target)!=want: raise Phase3Error(f"V35 binding changed: {rel}")
 return p,sha256_file(path)

def build(root,p):
 train,_=rows(root,p); strings=[row[field] for row in train for field in ("prompt","output")]+["".join(chr(v) for v in range(0x20,0x7F))]
 tokenizer=fit(strings,p["selected_budget"]); payload=tokenizer.to_str().encode("utf-8"); digest=hashlib.sha256(payload).hexdigest()
 if digest!=p["selected_tokenizer_sha256"]: raise Phase3Error("selected V34 tokenizer identity changed")
 return payload,tokenizer

def manifest(root,protocol_path):
 p,ph=load_protocol(root,protocol_path); payload,t=build(root,p); vocab=t.get_vocab();
 value={"format":"abi-capability-compiler-phase3-utf8-bpe-artifact/1","status":"MATERIALIZED_REPRESENTATION_ONLY","protocol":{"path":protocol_path.name,"sha256":ph},"tokenizer":{"path":"tokenizer.json","sha256":hashlib.sha256(payload).hexdigest(),"vocabulary_entries":len(vocab),"fixed_actions_with_host_specials":len(vocab)+3},"training_source":"PHASE1_TRAINING_PROMPTS_AND_OUTPUTS_PLUS_PRINTABLE_ASCII_SYNTAX_LINE","development_used_for_vocabulary":False,"final_test_accessed":False,"neural_training_performed":False,"teacher_present_at_inference":False,"claim_boundary":"Tokenizer identity only; no learned capability or host conformance."};value["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(value)).hexdigest();return value,payload

def main(argv:Iterable[str]|None=None)->int:
 a=argparse.ArgumentParser();a.add_argument("command",choices=("write","verify"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_UTF8_BPE_MATERIALIZATION_PROTOCOL_V35.json");a.add_argument("--directory",default="artifacts/phase3/utf8_bpe_v34");x=a.parse_args(argv);root=Path.cwd().resolve();m,payload=manifest(root,(root/x.protocol).resolve());d=(root/x.directory).resolve();tok=d/"tokenizer.json";mp=d/"manifest.json"
 if x.command=="write":
  if tok.exists() or mp.exists():raise Phase3Error("V35 artifact is immutable")
  _write_immutable(tok,payload);_write_immutable(mp,json.dumps(m,indent=2,sort_keys=True).encode()+b"\n")
 else:
  if sha256_file(tok)!=m["tokenizer"]["sha256"] or _json(mp)!=m:raise Phase3Error("V35 materialized artifact differs")
 print(json.dumps({"status":"PASS","tokenizer_sha256":m["tokenizer"]["sha256"],"evidence_sha256":m["evidence_sha256"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
