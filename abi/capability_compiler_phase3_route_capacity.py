"""Train the preregistered V50 routed full-generator capacity upper bound."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
import psutil, torch
import torch.nn.functional as F
from safetensors.torch import save_file
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable
from .capability_compiler_phase3_route_bridge import _base, _collate, _examples, load_protocol, BOS_ID, PAD_ID

def inventory(root:Path,path:Path)->Mapping[str,Any]:
 p,ph=load_protocol(root,path);model,tokenizer=_base(root,p,torch.device("cpu"));examples,controls=_examples(root,p,tokenizer);parameters=model.parameter_count()
 if parameters!=int(p["training"]["trainable_parameters"]):raise Phase3Error("V50 parameter count changed")
 return {"status":"PASS","protocol_sha256":ph,"records":len(examples),"trainable_parameters":parameters,"route_controls":len(controls),"maximum_source_actions":max(len(x["source_ids"]) for x in examples),"maximum_target_actions":max(len(x["target_actions"]) for x in examples),"teacher_outputs_added":0,"final_test_accessed":False}

def train(root:Path,path:Path,output:Path)->Mapping[str,Any]:
 p,ph=load_protocol(root,path)
 if output.exists() or not torch.cuda.is_available():raise Phase3Error("V50 output exists or CUDA unavailable")
 device=torch.device("cuda");model,tokenizer=_base(root,p,device);examples,controls=_examples(root,p,tokenizer);cfg=p["training"];seed=int(cfg["seed"]);set_determinism(seed);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),betas=(.9,.95),weight_decay=.1);scaler=torch.amp.GradScaler("cuda",enabled=True);sampler=_BalancedSampler(examples,seed);curves=[];sampled=Counter();sequence=hashlib.sha256();started=time.perf_counter();process=psutil.Process();peak=process.memory_info().rss;skipped=0;successful=0
 while successful<int(cfg["steps"]):
  batch=sampler.batch(int(cfg["batch_size"]));source,targets=_collate(batch,device);previous=torch.full_like(targets,PAD_ID);previous[:,0]=BOS_ID
  if targets.shape[1]>1:previous[:,1:]=torch.where(targets[:,:-1].ge(0),targets[:,:-1],torch.full_like(targets[:,:-1],PAD_ID))
  while True:
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast("cuda",dtype=torch.float16):log_probs=model.action_log_probs(source,previous)["log_probs"];loss=F.nll_loss(log_probs.float().reshape(-1,log_probs.shape[-1]),targets.reshape(-1),ignore_index=-100)
   scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);before=scaler.get_scale();scaler.step(optimizer);scaler.update()
   if scaler.get_scale()<before:skipped+=1;continue
   break
  successful+=1
  for row in batch:sampled[row["capability"]]+=1;sequence.update(row["record_id"].encode()+b"\n")
  peak=max(peak,process.memory_info().rss)
  if successful==1 or successful%int(cfg["curve_interval"])==0:
   value={"step":successful,"loss":float(loss.detach()),"wall_seconds":time.perf_counter()-started};curves.append(value);print(json.dumps(value),flush=True)
 output.mkdir(parents=True);checkpoint=output/"model.safetensors";save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint));router_source=(root/p["router"]["checkpoint_path"]).resolve();router=output/"router.safetensors";router.write_bytes(router_source.read_bytes());tokenizer_path=output/"tokenizer.json";_write_immutable(tokenizer_path,json.dumps(tokenizer.canonical_dict(),sort_keys=True,indent=2).encode()+b"\n");config=output/"model_config.json";_write_immutable(config,json.dumps({**p["architecture"],"fixed_vocab_size":tokenizer.vocab_size},sort_keys=True,indent=2).encode()+b"\n");control_doc=[{"capability":c,"token_id":controls[i][0],"piece_hex":controls[i][1].hex()} for i,c in enumerate(CAPABILITIES)];control_path=output/"route_controls.json";_write_immutable(control_path,json.dumps(control_doc,sort_keys=True,indent=2).encode()+b"\n")
 metadata={"format":"abi-capability-compiler-phase3-route-capacity-candidate/1","status":"TRAINED_NONPROMOTIONAL_CAPACITY_UPPER_BOUND","protocol_sha256":ph,"seed":seed,"checkpoint":{"path":"model.safetensors","sha256":sha256_file(checkpoint),"bytes":checkpoint.stat().st_size},"router":{"path":"router.safetensors","sha256":sha256_file(router),"bytes":router.stat().st_size,"parameters":1058040},"tokenizer":{"path":"tokenizer.json","sha256":sha256_file(tokenizer_path),"canonical_sha256":tokenizer.hash(),"vocabulary":tokenizer.vocab_size},"model_config":{"path":"model_config.json","sha256":sha256_file(config),"trainable_parameters":model.parameter_count()},"route_controls":{"path":"route_controls.json","sha256":sha256_file(control_path),"selection_sha256":p["route_controls"]["selection_sha256"]},"training":{"steps":successful,"batch_size":int(cfg["batch_size"]),"wall_seconds":time.perf_counter()-started,"skipped_amp_steps":skipped,"record_sequence_sha256":sequence.hexdigest(),"sampled_by_capability":dict(sorted(sampled.items())),"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"curves":curves},"imported_information":{"records":7000,"teacher_outputs_added":0,"stored_logits":0,"stored_activations":0,"source_parameters_copied":0},"capacity_upper_bound":True,"promotion_eligible":False,"teacher_present_at_inference":False,"source_blocks_retained":0,"layercake_host_changed":False,"phase3_certified":False,"final_test_accessed":False,"hardware":{"machine":platform.node(),"gpu":torch.cuda.get_device_name(0)}};metadata["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(metadata)).hexdigest();_write_immutable(output/"metadata.json",json.dumps(metadata,sort_keys=True,indent=2).encode()+b"\n");return metadata

def main(argv:Iterable[str]|None=None):
 a=argparse.ArgumentParser();a.add_argument("command",choices=("inventory","train"));a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_PROTOCOL_V50.json");a.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_route_capacity/development_v50/U0-seed240050");x=a.parse_args(argv);root=Path.cwd().resolve();result=inventory(root,(root/x.protocol).resolve()) if x.command=="inventory" else train(root,(root/x.protocol).resolve(),(root/x.output_dir).resolve());print(json.dumps(result,sort_keys=True,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
