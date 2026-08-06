"""Read-only LayerCake v2 conformance check for the exact V34 tokenizer."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any, Iterable
from tokenizers import Tokenizer
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
FORMAT="abi-capability-compiler-phase3-host-conformance/1"
def load_protocol(root,path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY" or p.get("host_change_allowed") is not False:raise Phase3Error("V36 governance changed")
 for rel,want in p["bindings"].items():
  t=(root/rel).resolve()
  if not t.is_file() or sha256_file(t)!=want:raise Phase3Error(f"V36 binding changed: {rel}")
 return p,sha256_file(path)
def execute(root:Path,path:Path)->dict[str,Any]:
 p,ph=load_protocol(root,path);lc=(root/p["layercake_host"]["repository"]).resolve();sys.path.insert(0,str(lc)) if str(lc) not in sys.path else None
 from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer
 raw=(root/p["tokenizer"]).read_text(encoding="utf-8");bpe=Tokenizer.from_str(raw);doc=json.loads(raw);accepted=True;error=None
 try:UnicodeAtomicLexemePointerTokenizer.from_document(doc)
 except Exception as exc:accepted=False;error=f"{type(exc).__name__}: {exc}"
 pieces=[value.encode("utf-8") for value in bpe.get_vocab() if value!="[UNK]"];valid=all(piece.decode("utf-8",errors="strict") is not None for piece in pieces)
 result={"format":"abi-capability-compiler-phase3-host-conformance-result/1","status":"FAIL_HOST_INTERFACE_UNSUPPORTED" if not accepted else "PASS_HOST_CONFORMANCE","protocol":{"path":path.name,"sha256":ph},"tokenizer_sha256":sha256_file(root/p["tokenizer"]),"bpe_loads":True,"all_non_special_pieces_valid_utf8":valid,"layercake_v2_document_accepted":accepted,"rejection":error,"host_changed":False,"neural_training_performed":False,"phase3_certified":False,"decision":"A rejection authorizes a separate LayerCake-owned BPE tokenizer host-interface successor; ABI extraction and model training remain unchanged.","claim_boundary":"Host document conformance only."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();return result
def main(argv:Iterable[str]|None=None)->int:
 a=argparse.ArgumentParser();a.add_argument("command",choices=("execute","verify"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_HOST_CONFORMANCE_PROTOCOL_V36.json");a.add_argument("--output",default="results/abi_capability_compiler_phase3_host_conformance/host_conformance_v36.json");x=a.parse_args(argv);root=Path.cwd().resolve();e=execute(root,(root/x.protocol).resolve());o=(root/x.output).resolve()
 if x.command=="execute":
  if o.exists():raise Phase3Error("V36 output immutable")
  _write_immutable(o,json.dumps(e,indent=2,sort_keys=True).encode()+b"\n")
 elif _json(o)!=e:raise Phase3Error("stored V36 differs")
 print(json.dumps(e,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
