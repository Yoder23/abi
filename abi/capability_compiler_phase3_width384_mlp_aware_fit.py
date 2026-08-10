"""Bounded width-384 MLP-aware compact-attention layer-1 fit."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from safetensors.torch import load_file, save_file
import torch
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
FORMAT="abi-capability-compiler-phase3-width384-mlp-aware-fit/1"
def execute(root:Path,path:Path,output:Path):
 from transformers import AutoModelForCausalLM
 p=json.loads(path.read_text(encoding="utf-8"))
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_BOUNDED_WIDTH384_MLP_AWARE_FIT" or p.get("device")!="cuda" or p.get("final_test_access")!="PROHIBITED": raise Phase3Error("width384 fit governance changed")
 for n,h in p["bindings"].items():
  q=root/n
  if not q.is_file() or sha256_file(q)!=h: raise Phase3Error(f"binding changed: {n}")
 if output.exists() or not torch.cuda.is_available(): raise Phase3Error("output exists or CUDA unavailable")
 output.mkdir(parents=True); base=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8")); device=torch.device("cuda"); prefix,tokenizer,_,_,_=sequential._model(root,base,device); state=prefix.state_dict()
 for i in (0,1):
  ck=load_file(str(root/p["checkpoints"][str(i)]["path"]),device="cpu")
  for n,v in ck.items():
   if n in state: state[n].copy_(v.to(state[n].dtype))
 prefix.eval(); source_attention=prefix.layers[1]; set_determinism(int(p["training"]["seed"])); from layercake.dual_path_progressive_core import DualPathProgressiveLayer
 layer=DualPathProgressiveLayer(3072,384,4,768,rms_epsilon=1e-5,rope_theta=10000.0)
 with torch.no_grad(): layer.input_norm.weight.copy_(source_attention.input_norm.weight.cpu()); layer.post_attention_norm.weight.copy_(source_attention.post_attention_norm.weight.cpu()); layer.attention_output_projection.weight.zero_()
 del layer.mlp_input_projection,layer.mlp_norm,layer.gate_up_proj,layer.down_proj,layer.mlp_output_projection
 layer.input_norm.weight.requires_grad_(False); layer.post_attention_norm.weight.requires_grad_(False); layer.to(device); examples=sequential.field._examples(root,base,tokenizer); cfg=base["calibration"]; train,val,tokens=dual._calibration_examples(examples,seed=int(p["training"]["seed"]),train_per_capability=int(cfg["train_records_per_capability"]),validation_per_capability=int(cfg["validation_records_per_capability"]),maximum_tokens=int(cfg["maximum_sequence_tokens"]))
 teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").to(device).eval(); source=teacher.model.layers[1]
 for parameter in teacher.parameters(): parameter.requires_grad_(False)
 params=[parameter for parameter in layer.parameters() if parameter.requires_grad]; optimizer=torch.optim.AdamW(params,lr=float(p["training"]["learning_rate"]),betas=(0.9,0.95),weight_decay=float(p["training"]["weight_decay"])); steps=int(p["training"]["steps"]); curves=[]; layer.train()
 for step in range(steps):
  row=train[(step+256)%len(train)]; ids=torch.tensor([row["input_ids"]],dtype=torch.long,device=device)
  with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): hidden=dual.base._prefix_hidden(prefix,ids,1); at,ft=dual._teacher_components(teacher,1,hidden); feature_target=source.post_attention_layernorm(at)
  optimizer.zero_grad(set_to_none=True)
  with torch.autocast("cuda",dtype=torch.bfloat16): attention=sequential._student_attention(layer,hidden,torch.arange(ids.shape[1],device=device)); feature=layer.post_attention_norm(attention); final=attention+source.mlp(feature); armse,acos=dual.base._metrics(attention,at,hidden); frmse,fcos=dual.base._metrics(final,ft,hidden); xrmse=torch.sqrt((feature.float()-feature_target.float()).square().mean()/feature_target.float().square().mean().clamp_min(1e-8)); loss=armse.square()+frmse.square()+xrmse.square()+float(p["training"]["cosine_weight"])*(2-acos-fcos)
  loss.backward(); torch.nn.utils.clip_grad_norm_(params,float(p["training"]["gradient_clip_norm"])); optimizer.step()
  if step==0 or (step+1)%int(p["training"]["curve_interval"])==0: curves.append({"step":step+1,"attention_relative_rmse":float(armse.detach()),"feature_relative_rmse":float(xrmse.detach()),"final_relative_rmse":float(frmse.detach()),"final_cosine":float(fcos.detach()),"loss":float(loss.detach())})
 layer.eval(); ar=[]; xr=[]; rs=[]; cs=[]
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
  for row in val:
   ids=torch.tensor([row["input_ids"]],dtype=torch.long,device=device); hidden=dual.base._prefix_hidden(prefix,ids,1); at,ft=dual._teacher_components(teacher,1,hidden); target=source.post_attention_layernorm(at); attention=sequential._student_attention(layer,hidden,torch.arange(ids.shape[1],device=device)); feature=layer.post_attention_norm(attention); final=attention+source.mlp(feature); armse,_=dual.base._metrics(attention,at,hidden); rmse,cos=dual.base._metrics(final,ft,hidden); xrmse=torch.sqrt((feature.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-8)); ar.append(float(armse));xr.append(float(xrmse));rs.append(float(rmse));cs.append(float(cos))
 mr=sum(rs)/len(rs);mc=sum(cs)/len(cs);gate=p["gate"];passed=mr<=gate["mean_relative_rmse_maximum"] and mc>=gate["mean_output_cosine_minimum"]; weights={n:v.detach().to(torch.float16).cpu().contiguous() for n,v in layer.named_parameters()}; wp=output/"layer1_width384_attention.safetensors";save_file(weights,str(wp),metadata={"format":FORMAT,"protocol_sha256":sha256_file(path)})
 result={"format":FORMAT,"status":"PASS_WIDTH384_MLP_AWARE_INTERFACE" if passed else "FAIL_WIDTH384_MLP_AWARE_INTERFACE","protocol_sha256":sha256_file(path),"steps":steps,"attention_width":384,"curves":curves,"validation":{"mean_attention_relative_rmse":sum(ar)/len(ar),"mean_feature_relative_rmse":sum(xr)/len(xr),"mean_relative_rmse":mr,"maximum_relative_rmse":max(rs),"mean_output_cosine":mc,"minimum_output_cosine":min(cs),"passed":passed},"checkpoint":{"path":wp.name,"sha256":sha256_file(wp),"parameters":sum(v.numel() for v in weights.values())},"complete_source_mlp_promoted":False,"artifact_promoted":False,"training_performed":True,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Bounded width384 layer1 interface fit only; no host, deployable artifact, runtime, certificate, or superiority claim."};_write_immutable(output/"metadata.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_WIDTH384_MLP_AWARE_FIT_PROTOCOL_V281.json");ap.add_argument("--output",default="results/abi_capability_compiler_phase3_width384_attention/layer1_fit_v282");a=ap.parse_args();root=Path.cwd().resolve();r=execute(root,root/a.protocol,root/a.output);print(json.dumps(r,indent=2,sort_keys=True));return 0 if r["status"].startswith("PASS") else 1
if __name__=="__main__":raise SystemExit(main())
