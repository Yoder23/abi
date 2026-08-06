"""Conform the exact V34 tokenizer to LayerCake direct-core ABI v3."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any, Iterable
from tokenizers import Tokenizer
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import rows
FORMAT="abi-capability-compiler-phase3-external-tokenizer-conformance/1"
def load_protocol(root,path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_CONFORMANCE" or p.get("neural_training_allowed") is not False or p.get("final_test_access")!="PROHIBITED":raise Phase3Error("V37 governance changed")
 for rel,want in p["bindings"].items():
  t=(root/rel).resolve()
  if not t.is_file() or sha256_file(t)!=want:raise Phase3Error(f"V37 binding changed: {rel}")
 return p,sha256_file(path)
def execute(root:Path,path:Path):
 p,ph=load_protocol(root,path);lc=(root/p["layercake_host"]["repository"]).resolve();sys.path.insert(0,str(lc)) if str(lc) not in sys.path else None
 from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer,BPE_DIRECT_NEURAL_CORE_ABI_VERSION,BPE_DIRECT_NEURAL_CORE_ABI_SHA256
 raw=(root/p["tokenizer"]).read_text(encoding="utf-8");native=Tokenizer.from_str(raw);host=Utf8ConcatenativeBpeTokenizer(json.loads(raw));train,dev=rows(root,p);mismatches=[];stream=hashlib.sha256()
 for split_name,dataset in (("training",train),("development",dev)):
  for row in dataset:
   for field in ("prompt","output"):
    expected=native.encode(row[field]).tokens;actual=[value.decode("utf-8") for value in host.split(row[field])]
    if actual!=expected:mismatches.append({"split":split_name,"record_id":row["record_id"],"field":field})
    stream.update(canonical_json_bytes({"split":split_name,"record_id":row["record_id"],"field":field,"tokens":actual}))
 document=host.canonical_dict();roundtrip=Utf8ConcatenativeBpeTokenizer.from_document(document)
 result={"format":"abi-capability-compiler-phase3-external-tokenizer-conformance-result/1","status":"PASS" if not mismatches and roundtrip.hash()==host.hash() else "FAIL","protocol":{"path":path.name,"sha256":ph},"layercake":{"commit":p["layercake_host"]["commit"],"interface":BPE_DIRECT_NEURAL_CORE_ABI_VERSION,"interface_sha256":BPE_DIRECT_NEURAL_CORE_ABI_SHA256},"tokenizer":{"raw_sha256":sha256_file(root/p["tokenizer"]),"canonical_sha256":host.hash(),"fixed_actions":host.vocab_size,"sequence_stream_sha256":stream.hexdigest()},"training_records":len(train),"development_records":len(dev),"fields_compared":2*(len(train)+len(dev)),"mismatches":mismatches,"neural_training_performed":False,"final_test_accessed":False,"phase3_certified":False,"next_gate":"One separately preregistered bounded neural acquisition candidate if PASS.","claim_boundary":"Exact tokenizer/host conformance only; no learned quality or performance."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();return result,document
def main(argv:Iterable[str]|None=None):
 a=argparse.ArgumentParser();a.add_argument("command",choices=("write","verify"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_EXTERNAL_TOKENIZER_CONFORMANCE_PROTOCOL_V37.json");a.add_argument("--result",default="results/abi_capability_compiler_phase3_external_tokenizer_conformance/conformance_v37.json");a.add_argument("--document",default="artifacts/phase3/utf8_bpe_v34/layercake_tokenizer_v3.json");x=a.parse_args(argv);root=Path.cwd().resolve();r,d=execute(root,(root/x.protocol).resolve());rp=(root/x.result).resolve();dp=(root/x.document).resolve()
 if x.command=="write":
  if rp.exists() or dp.exists():raise Phase3Error("V37 outputs immutable")
  _write_immutable(dp,json.dumps(d,indent=2,sort_keys=True).encode()+b"\n");_write_immutable(rp,json.dumps(r,indent=2,sort_keys=True).encode()+b"\n")
 elif _json(rp)!=r or _json(dp)!=d:raise Phase3Error("stored V37 evidence differs")
 print(json.dumps({"status":r["status"],"mismatches":len(r["mismatches"]),"evidence_sha256":r["evidence_sha256"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
