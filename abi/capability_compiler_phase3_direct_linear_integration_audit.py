"""Read-only layer-0 composition of qualified compact attention and analytic MLP map."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from safetensors.torch import load_file
import torch
from . import capability_compiler_phase3_causal_field_core as field
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error,_write_immutable
FORMAT="abi-capability-compiler-phase3-direct-linear-integration-audit/1"
def execute(root:Path,path:Path):
 p=json.loads(path.read_text(encoding="utf-8"))
 if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_INTEGRATION_AUDIT": raise Phase3Error("governance changed")
 for n,h in p["bindings"].items():
  q=Path(n) if Path(n).is_absolute() else root/n
  if not q.is_file() or sha256_file(q)!=h: raise Phase3Error(f"binding changed: {n}")
 if not torch.cuda.is_available(): raise Phase3Error("CUDA required")
 device=torch.device("cuda"); lc=(root/p["layercake_host"]["repository"]).resolve(); sys.path.insert(0,str(lc))
 from layercake.direct_linear_progressive_core import DirectLinearProgressiveCore
 from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
 tokenizer=field._tokenizer(p,DecoderAwareExternalTokenizer); examples=field._examples(root,p,tokenizer); c=p["calibration"]; train,val,tokens=dual._calibration_examples(examples,seed=p["seed"],train_per_capability=c["train_records_per_capability"],validation_per_capability=c["validation_records_per_capability"],maximum_tokens=c["maximum_sequence_tokens"])
 model=DirectLinearProgressiveCore(fixed_vocab_size=tokenizer.vocab_size,full_width=3072,bottleneck_width=192,attention_heads=2,replacement_layers=1,intermediate_size=768,maximum_source_actions=192,maximum_target_actions=320,maximum_sequence_actions=512).bind_tokenizer(tokenizer)
 substrate=load_file(str(root/p["substrate"]),device="cpu"); model.load_state_dict({k:v for k,v in substrate.items() if not k.startswith("layers.") or k.startswith("layers.0.")},strict=False)
 negative=load_file(str(root/p["attention_checkpoint"]),device="cpu"); state=model.state_dict(); copied=0
 for k,v in negative.items():
  if k.startswith("layers.0.") and any(x in k for x in ("attention_","qkv_proj","o_proj")) and k in state and state[k].shape==v.shape: state[k].copy_(v.to(state[k].dtype)); copied+=v.numel()
 model.to(device).eval(); embedding=model.token_embedding.weight
 from transformers import AutoModelForCausalLM
 teacher=AutoModelForCausalLM.from_pretrained(p["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").to(device).eval(); layer=teacher.model.layers[0]
 deltas=[]; features=[]
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
  for row in train:
   ids=torch.tensor(row["input_ids"],device=device); hidden=embedding(ids).unsqueeze(0); att,final=dual._teacher_components(teacher,0,hidden); deltas.append((final-att).squeeze(0).float().cpu()); features.append(layer.post_attention_layernorm(att).squeeze(0).float().cpu())
 mean,cov,obs=rank_audit.centered_covariance(deltas,3072,device); _,vectors=torch.linalg.eigh(cov); basis=vectors.flip(1)[:,:192].contiguous(); x=torch.cat(features).to(device); y=(torch.cat(deltas).to(device)-mean)@basis; weights,ridge=closed.solve_ridge(x,y,p["relative_ridge"])
 with torch.no_grad(): model.layers[0].mlp_residual_mean.copy_(mean); model.layers[0].mlp_output_projection.weight.copy_(basis); model.layers[0].mlp_coefficient_projection.weight.copy_(weights.T)
 rmses=[]; cosines=[]
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
  for row in val:
   ids=torch.tensor([row["input_ids"]],device=device); hidden=embedding(ids); _,target=dual._teacher_components(teacher,0,hidden); pred,_,_=model.layers[0].forward_with_cache(hidden,torch.arange(ids.shape[1],device=device)); rmse,cos=dual.base._metrics(pred,target,hidden); rmses.append(float(rmse)); cosines.append(float(cos))
 mr=sum(rmses)/len(rmses); mc=sum(cosines)/len(cosines); passed=mr<=p["gate"]["mean_relative_rmse_maximum"] and mc>=p["gate"]["mean_output_cosine_minimum"]
 return {"format":FORMAT,"status":"PASS_INTEGRATED_LAYER0_NO_ARTIFACT" if passed else "FAIL_INTEGRATION","compact_attention_parameters_loaded":copied,"train_observations":obs,"effective_ridge":ridge,"calibration_tokens":tokens,"validation":{"mean_relative_rmse":mr,"maximum_relative_rmse":max(rmses),"mean_output_cosine":mc,"minimum_output_cosine":min(cosines),"passed":passed},"artifact_written":False,"training_performed":False,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Read-only layer-0 component integration only; no deployable artifact, English quality, inference, certificate, or superiority claim."}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_DIRECT_LINEAR_INTEGRATION_AUDIT_PROTOCOL_V249.json"); ap.add_argument("--output",default="results/abi_capability_compiler_phase3_direct_linear/integration_audit_v250.json"); a=ap.parse_args(); root=Path.cwd().resolve(); out=root/a.output
 if out.exists(): raise Phase3Error("output exists")
 r=execute(root,root/a.protocol); _write_immutable(out,json.dumps(r,indent=2,sort_keys=True).encode()+b"\n"); print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r["status"].startswith("PASS") else 1
if __name__=="__main__": raise SystemExit(main())
