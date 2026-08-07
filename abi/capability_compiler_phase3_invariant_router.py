"""Train and verify one cross-header-invariant ABI capability router."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
import psutil, torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _json, _layercake_api, _tokenizer

FORMAT="abi-capability-compiler-phase3-invariant-router/1"

class InvariantRouter(nn.Module):
 def __init__(self,vocabulary:int,width:int,hidden:int,classes:int,dropout:float):
  super().__init__();self.embedding=nn.EmbeddingBag(vocabulary,width,mode="mean");self.norm=nn.LayerNorm(width);self.hidden=nn.Linear(width,hidden);self.output=nn.Linear(hidden,classes);self.dropout=nn.Dropout(dropout)
 def forward(self,ids,offsets):return self.output(self.dropout(F.gelu(self.hidden(self.norm(self.embedding(ids,offsets))))))

def load_protocol(root:Path,path:Path):
 p=_json(path)
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_ROUTER_GATE" or p.get("final_test_access")!="PROHIBITED" or p.get("training",{}).get("device")!="cuda":raise Phase3Error("V43 governance changed")
 for rel,want in p["bindings"].items():
  target=(root/rel).resolve()
  if not target.is_file() or sha256_file(target)!=want:raise Phase3Error(f"V43 binding changed: {rel}")
 return p,sha256_file(path)

def _data(root,p,tokenizer):
 raw=load_phase1_ir((root/p["phase1_ir"]).resolve());headers={c:[] for c in CAPABILITIES};rows=[]
 for row in raw:
  prompt=str(row["normalized_acquisition_prompt"]);lines=prompt.splitlines();cap=str(row["capability"])
  if len(lines)<2:raise Phase3Error("router prompt lacks header/body boundary")
  headers[cap].append(lines[0]);body="\n".join(lines[1:]);rows.append({"record_id":str(row["ir_record_id"]),"capability":cap,"body":body,"body_ids":[tokenizer.lexeme_to_id[x] for x in tokenizer.split(body)]})
 return rows,{c:tuple(sorted(set(v))) for c,v in headers.items()}

def _foreign_header(headers,cap,record_id,step):
 choices=[name for name in CAPABILITIES if name!=cap];h=hashlib.sha256(f"{record_id}\0{step}\0foreign-capability".encode()).digest();other=choices[int.from_bytes(h[:4],"big")%len(choices)];values=headers[other];return values[int.from_bytes(h[4:8],"big")%len(values)]

def _encode(tokenizer,text):return [tokenizer.lexeme_to_id[x] for x in tokenizer.split(text)]
def _collate(sequences,device):
 if any(not x for x in sequences):raise Phase3Error("empty router sequence")
 offsets=[];flat=[]
 for seq in sequences:offsets.append(len(flat));flat.extend(seq)
 return torch.tensor(flat,dtype=torch.long,device=device),torch.tensor(offsets,dtype=torch.long,device=device)

def _model(p,vocab):
 a=p["architecture"];return InvariantRouter(vocab,int(a["embedding_width"]),int(a["hidden_width"]),len(CAPABILITIES),float(a["dropout"]))

def inventory(root,path):
 p,ph=load_protocol(root,path);_,_,tt,_,_=_layercake_api(root,p);tok=_tokenizer(root,p,tt);rows,headers=_data(root,p,tok);m=_model(p,tok.vocab_size);n=sum(x.numel() for x in m.parameters())
 if n!=int(p["training"]["trainable_parameters"]):raise Phase3Error(f"router parameter count changed: {n}")
 return {"status":"PASS","protocol_sha256":ph,"records":len(rows),"capabilities":len(headers),"unique_headers":sum(len(x) for x in headers.values()),"trainable_parameters":n,"maximum_body_actions":max(len(x["body_ids"]) for x in rows),"teacher_outputs_added":0,"final_test_accessed":False}

def train(root,path,out):
 p,ph=load_protocol(root,path)
 if out.exists() or not torch.cuda.is_available():raise Phase3Error("router output exists or CUDA unavailable")
 _,_,tt,_,_=_layercake_api(root,p);tok=_tokenizer(root,p,tt);rows,headers=_data(root,p,tok);cfg=p["training"];seed=int(cfg["seed"]);set_determinism(seed);device=torch.device("cuda");model=_model(p,tok.vocab_size).to(device);n=sum(x.numel() for x in model.parameters())
 if n!=int(cfg["trainable_parameters"]):raise Phase3Error("router parameter count changed")
 opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),betas=(.9,.95),weight_decay=.01);sampler=_BalancedSampler(rows,seed);labels={c:i for i,c in enumerate(CAPABILITIES)};curves=[];started=time.perf_counter();process=psutil.Process();peak=process.memory_info().rss;sequence=hashlib.sha256();model.train()
 for step in range(1,int(cfg["steps"])+1):
  batch=sampler.batch(int(cfg["batch_size"]));texts=[];targets=[]
  for row in batch:
   foreign=_foreign_header(headers,row["capability"],row["record_id"],step);texts.extend((row["body"],foreign+"\n"+row["body"]));targets.extend((labels[row["capability"]],labels[row["capability"]]));sequence.update(row["record_id"].encode()+b"\n")
  ids,offsets=_collate([_encode(tok,x) for x in texts],device);target=torch.tensor(targets,device=device);opt.zero_grad(set_to_none=True);logits=model(ids,offsets);loss=F.cross_entropy(logits,target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();peak=max(peak,process.memory_info().rss)
  if step==1 or step%int(cfg["curve_interval"])==0:
   value={"step":step,"loss":float(loss.detach()),"accuracy":float(logits.argmax(-1).eq(target).float().mean()),"wall_seconds":time.perf_counter()-started};curves.append(value);print(json.dumps(value),flush=True)
 out.mkdir(parents=True);checkpoint=out/"router.safetensors";save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint));config=out/"config.json";_write_immutable(config,json.dumps({"vocabulary":tok.vocab_size,**p["architecture"]},sort_keys=True,indent=2).encode()+b"\n")
 meta={"format":"abi-capability-compiler-phase3-invariant-router-candidate/1","status":"TRAINED_ROUTER_GATE_ONLY","protocol_sha256":ph,"seed":seed,"checkpoint":{"path":"router.safetensors","sha256":sha256_file(checkpoint),"bytes":checkpoint.stat().st_size},"config":{"path":"config.json","sha256":sha256_file(config),"trainable_parameters":n},"training":{"steps":int(cfg["steps"]),"batch_size":int(cfg["batch_size"]),"effective_views_per_step":2*int(cfg["batch_size"]),"wall_seconds":time.perf_counter()-started,"record_sequence_sha256":sequence.hexdigest(),"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"curves":curves},"imported_information":{"records":7000,"capability_labels":7000,"teacher_outputs_added":0,"stored_logits":0,"stored_activations":0,"source_parameters_copied":0},"teacher_present_at_inference":False,"layercake_host_changed":False,"phase3_certified":False,"final_test_accessed":False,"hardware":{"machine":platform.node(),"gpu":torch.cuda.get_device_name(0)}};meta["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(meta)).hexdigest();_write_immutable(out/"metadata.json",json.dumps(meta,sort_keys=True,indent=2).encode()+b"\n");return meta

def _load(root,p,candidate):
 _,_,tt,_,_=_layercake_api(root,p);tok=_tokenizer(root,p,tt);m=_model(p,tok.vocab_size);m.load_state_dict(load_file(str(candidate/"router.safetensors"),device="cuda"),strict=True);return m.cuda().eval(),tok

@torch.inference_mode()
def evaluate(root,path,candidate,out):
 p,ph=load_protocol(root,path);meta=_json(candidate/"metadata.json")
 if out.exists() or meta.get("protocol_sha256")!=ph or sha256_file(candidate/"router.safetensors")!=meta["checkpoint"]["sha256"]:raise Phase3Error("router evaluation identity failed")
 model,tok=_load(root,p,candidate);train_rows,headers=_data(root,p,tok);probes=development_probes((root/p["development_catalog"]).resolve());rows=[]
 canonical={c:headers[c][0] for c in CAPABILITIES}
 for probe in probes:
  cap=str(probe["canonical_capability"]);prompt=str(probe["prompt"]);body="\n".join(prompt.splitlines()[1:]);variants={"original":prompt,"body":body,"matched_header":canonical[cap]+"\n"+body}
  seq=[_encode(tok,x) for x in variants.values()];ids,offsets=_collate(seq,torch.device("cuda"));pred=model(ids,offsets).argmax(-1).tolist()
  for (variant,_),index in zip(variants.items(),pred):rows.append({"probe_id":str(probe["probe_id"]),"capability":cap,"variant":variant,"predicted":CAPABILITIES[index],"correct":CAPABILITIES[index]==cap})
 out.mkdir(parents=True);raw=out/"rows.jsonl";raw.write_bytes(b"".join(canonical_json_bytes(x) for x in rows));receipt=_decision(p,ph,meta,rows,sha256_file(raw));_write_immutable(out/"decision.json",json.dumps(receipt,sort_keys=True,indent=2).encode()+b"\n");return receipt

def _wilson(k,n,z=1.959963984540054):
 q=k/n;d=1+z*z/n;c=(q+z*z/(2*n))/d;h=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/d;return {"point":q,"lower_95":c-h,"upper_95":c+h}
def _decision(p,ph,meta,rows,raw_sha):
 summaries={}
 for variant in ("original","body","matched_header"):
  v=[x for x in rows if x["variant"]==variant];summaries[variant]={"correct":sum(x["correct"] for x in v),"observations":len(v),"wilson":_wilson(sum(x["correct"] for x in v),len(v)),"per_capability":{c:_wilson(sum(x["correct"] for x in v if x["capability"]==c),100) for c in CAPABILITIES},"predicted_counts":dict(sorted(Counter(x["predicted"] for x in v).items()))}
 g=p["router_gate"];original=summaries["original"];body=summaries["body"];per=all(x["point"]>=float(g["per_capability_point_minimum"]) and x["lower_95"]>=float(g["per_capability_wilson_lower_minimum"]) for x in original["per_capability"].values());passed=original["wilson"]["point"]>=float(g["aggregate_point_minimum"]) and original["wilson"]["lower_95"]>=float(g["aggregate_wilson_lower_minimum"]) and body["wilson"]["point"]>=float(g["body_point_minimum"]) and per
 r={"format":"abi-capability-compiler-phase3-invariant-router-decision/1","status":"PASS_ROUTER_GATE_HOST_SUCCESSOR_OPEN" if passed else "FAIL_ROUTER_GATE_ARCHITECTURE_CLOSED","protocol":{"path":"ABI_CAPABILITY_COMPILER_PHASE3_INVARIANT_ROUTER_PROTOCOL_V43.json","sha256":ph},"checkpoint_sha256":meta["checkpoint"]["sha256"],"summaries":summaries,"gates":{"aggregate_original":original["wilson"]["point"]>=float(g["aggregate_point_minimum"]) and original["wilson"]["lower_95"]>=float(g["aggregate_wilson_lower_minimum"]),"aggregate_body":body["wilson"]["point"]>=float(g["body_point_minimum"]),"per_capability_original":per,"router_gate_pass":passed},"rows_sha256":raw_sha,"teacher_outputs_added":0,"layercake_host_changed":False,"phase3_certified":False,"final_test_accessed":False,"next_step":"Preregister a separate LayerCake routed-host construct only if PASS." if passed else "Preserve failure; routed host remains prohibited."};r["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(r)).hexdigest();return r

def main(argv:Iterable[str]|None=None):
 a=argparse.ArgumentParser();a.add_argument("command",choices=("inventory","train","evaluate"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_INVARIANT_ROUTER_PROTOCOL_V43.json");a.add_argument("--candidate-dir",default="results/abi_capability_compiler_phase3_invariant_router/development_v43/R0-seed240043");a.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_invariant_router/evaluation_v43/R0-seed240043");x=a.parse_args(argv);root=Path.cwd().resolve();path=(root/x.protocol).resolve();result=inventory(root,path) if x.command=="inventory" else train(root,path,(root/x.candidate_dir).resolve()) if x.command=="train" else evaluate(root,path,(root/x.candidate_dir).resolve(),(root/x.output_dir).resolve());print(json.dumps(result,sort_keys=True,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
