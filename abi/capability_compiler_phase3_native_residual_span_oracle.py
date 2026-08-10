"""Read-only rank-768 output-span oracle for native layer-0 residuals."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys,time
from safetensors.torch import load_file
import torch
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes,sha256_file
from .capability_compiler_phase3 import Phase3Error,_write_immutable
FORMAT="abi-capability-compiler-phase3-native-residual-span-oracle/1"

def execute(root:Path,protocol_path:Path,output:Path)->dict:
    from transformers import AutoModelForCausalLM
    p=json.loads(protocol_path.read_text(encoding="utf-8"))
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_NATIVE_RESIDUAL_OUTPUT_SPAN_ORACLE" or p.get("training")!="PROHIBITED" or p.get("artifact_write")!="PROHIBITED" or p.get("final_test_access")!="PROHIBITED":raise Phase3Error("native residual oracle governance changed")
    for name,expected in p["bindings"].items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if not path.is_file() or sha256_file(path)!=expected:raise Phase3Error(f"native residual binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():raise Phase3Error("native residual output exists or CUDA unavailable")
    output.mkdir(parents=True);device=torch.device("cuda");base=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8"));artifact=(root/p["artifact"]["directory"]).resolve();artifact_path=artifact/"model.safetensors";before=sha256_file(artifact_path);config=json.loads((artifact/"config.json").read_text(encoding="utf-8"));sys.path.insert(0,str((root/p["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer=DecoderAwareExternalTokenizer.from_document(config["tokenizer"]);model=PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer);model.load_state_dict(load_file(str(artifact_path),device="cpu"),strict=True,assign=True);layer=model.layers[0].cuda().eval();embedding=model.token_embedding
    examples=sequential.field._examples(root,base,tokenizer);cfg=base["calibration"];train,validation,tokens=dual._calibration_examples(examples,seed=int(base["training"]["seed"]),train_per_capability=int(cfg["train_records_per_capability"]),validation_per_capability=int(cfg["validation_records_per_capability"]),maximum_tokens=int(cfg["maximum_sequence_tokens"]));teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval();terminal=int(base["source"]["terminal_token_id"]);residuals=[];route_exact=0;started=time.perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row in train:
            host=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host);route_exact+=int(route==routed._route(str(row["capability"])));hidden=embedding(host).to(device);attention=routed._attention(layer,hidden,torch.arange(hidden.shape[1],device=device));source=torch.tensor([[trajectory.source_token_id(v,terminal) for v in row["input_ids"]]],device=device);teacher_hidden=teacher.model.embed_tokens(source);_,target=dual._teacher_components(teacher,0,teacher_hidden);residuals.append((target-attention).squeeze(0).float().cpu())
    mean,covariance,observations=rank_audit.centered_covariance(residuals,int(p["full_width"]),device);eigenvalues,eigenvectors=torch.linalg.eigh(covariance);eigenvalues=eigenvalues.clamp_min(0).flip(0);basis=eigenvectors.flip(1)[:,:int(p["rank"])].contiguous();energy=float(eigenvalues[:int(p["rank"])].sum()/eigenvalues.sum().clamp_min(1e-12));cosines=[];rmses=[];records=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row in validation:
            host=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host);route_exact+=int(route==routed._route(str(row["capability"])));hidden=embedding(host).to(device);attention=routed._attention(layer,hidden,torch.arange(hidden.shape[1],device=device));source=torch.tensor([[trajectory.source_token_id(v,terminal) for v in row["input_ids"]]],device=device);teacher_hidden=teacher.model.embed_tokens(source);_,target=dual._teacher_components(teacher,0,teacher_hidden);residual=target.float()-attention.float();centered=residual-mean;oracle=attention.float()+mean+(centered.reshape(-1,centered.shape[-1])@basis@basis.T).reshape_as(centered);cos,rmse=trajectory._metrics(oracle,target);cosines.append(cos);rmses.append(rmse);records.append({"record_id":row["record_id"],"capability":row["capability"],"cosine":cos,"relative_rmse":rmse})
    after=sha256_file(artifact_path);mean_cos=sum(cosines)/len(cosines);mean_rmse=sum(rmses)/len(rmses);gates={"rank_energy":energy>=float(p["gates"]["rank_energy_minimum"]),"validation_mean_cosine":mean_cos>=float(p["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":mean_rmse<=float(p["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(train)+len(validation),"artifact_unchanged":before==after};passed=all(gates.values());result={"format":FORMAT,"status":"PASS_NATIVE_RESIDUAL_SPAN_ORACLE" if passed else "FAIL_NATIVE_RESIDUAL_SPAN_ORACLE","protocol_sha256":sha256_file(protocol_path),"rank":int(p["rank"]),"train_observations":observations,"rank_energy":energy,"validation":{"records":len(validation),"mean_cosine":mean_cos,"minimum_cosine":min(cosines),"mean_relative_rmse":mean_rmse,"maximum_relative_rmse":max(rmses),"record_metrics":records},"route_correct":route_exact,"gates":gates,"passed":passed,"artifact_model_sha256_before":before,"artifact_model_sha256_after":after,"wall_seconds":time.perf_counter()-started,"training_performed":False,"artifact_written":False,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Read-only direct-coefficient output-span oracle only; no realizable map, model, autonomous quality, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result
def main():
    q=argparse.ArgumentParser();q.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_RESIDUAL_SPAN_ORACLE_PROTOCOL_V356.json");q.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/span_oracle_v357");a=q.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/a.protocol).resolve(),(root/a.output_dir).resolve()),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
