"""Read-only rank-768 residual span oracle at layer 1 on the passing replacement prefix."""

from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import sys, time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import psutil
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT="abi-capability-compiler-phase3-layer1-residual-span-oracle/1"


def execute(root:Path,protocol_path:Path,output:Path)->dict:
    from transformers import AutoModelForCausalLM
    p=json.loads(protocol_path.read_text(encoding="utf-8"))
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_LAYER1_RESIDUAL_SPAN_ORACLE" or p.get("training_authorized") is not False or p.get("artifact_write")!="PROHIBITED" or p.get("final_test_access")!="PROHIBITED":raise Phase3Error("layer1 span governance changed")
    for name,expected in p["bindings"].items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if not path.is_file() or sha256_file(path)!=expected:raise Phase3Error(f"layer1 span binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():raise Phase3Error("layer1 span output exists or CUDA unavailable")
    output.mkdir(parents=True);set_determinism(int(p["seed"]));torch.use_deterministic_algorithms(True);device=torch.device("cuda")
    base=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8"));artifact=root/p["artifact"]["directory"];artifact_path=artifact/"model.safetensors";artifact_before=sha256_file(artifact_path);config=json.loads((artifact/"config.json").read_text(encoding="utf-8"));sys.path.insert(0,str((root/p["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer=DecoderAwareExternalTokenizer.from_document(config["tokenizer"]);model=PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer);model.load_state_dict(load_file(str(artifact_path),device="cpu"),strict=True,assign=True);state=model.state_dict();prefix=load_file(str(root/p["layer0_checkpoint"]["path"]),device="cpu")
    with torch.no_grad():
        for name,value in prefix.items():state[name].copy_(value.to(state[name].dtype))
    layer0=model.layers[0].float().cuda().eval();layer1=model.layers[1].float().cuda().eval();examples=sequential.field._examples(root,base,tokenizer);train,validation=coverage.expanded_split(examples,seed=int(base["training"]["seed"]),maximum_tokens=128);teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()
    for value in teacher.parameters():value.requires_grad_(False)
    terminal=int(base["source"]["terminal_token_id"]);started=time.perf_counter();process=psutil.Process();peak=process.memory_info().rss;torch.cuda.reset_peak_memory_stats();route_exact=0
    def capture(rows):
        nonlocal route_exact,peak
        values=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for index,row in enumerate(rows):
                host=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host);route_exact+=int(route==routed._route(str(row["capability"])));candidate=model.token_embedding(host).to(device);positions=torch.arange(candidate.shape[1],device=device);candidate,_,_=layer0.forward_with_cache(candidate,positions,route);attention=routed._attention(layer1,candidate,positions);source=torch.tensor([[trajectory.source_token_id(value,terminal) for value in row["input_ids"]]],device=device);native=teacher.model.embed_tokens(source)
                for source_index in range(2):_,native=dual._teacher_components(teacher,source_index,native)
                values.append({"record_id":row["record_id"],"capability":row["capability"],"attention":attention.squeeze(0).float().cpu(),"target":native.squeeze(0).float().cpu(),"residual":(native-attention).squeeze(0).float().cpu()});peak=max(peak,process.memory_info().rss)
                if (index+1)%500==0:print(json.dumps({"capture_records":index+1}),flush=True)
        return values
    train_cache=capture(train);validation_cache=capture(validation);del teacher;torch.cuda.empty_cache();mean,covariance,observations=rank_audit.centered_covariance([row["residual"] for row in train_cache],int(p["full_width"]),device);eigenvalues,eigenvectors=torch.linalg.eigh(covariance);eigenvalues=eigenvalues.clamp_min(0).flip(0);rank=int(p["rank"]);basis=eigenvectors.flip(1)[:,:rank].contiguous();energy=float(eigenvalues[:rank].sum()/eigenvalues.sum().clamp_min(1e-12));cosines=[];rmses=[];records=[]
    with torch.inference_mode():
        for row in validation_cache:
            residual=row["residual"].to(device);coefficients=(residual-mean)@basis;prediction=row["attention"].to(device)+mean+F.linear(coefficients,basis);cosine,rmse=trajectory._metrics(prediction,row["target"].to(device));cosines.append(cosine);rmses.append(rmse);records.append({"record_id":row["record_id"],"capability":row["capability"],"cosine":cosine,"relative_rmse":rmse})
    artifact_after=sha256_file(artifact_path);mean_cosine=sum(cosines)/len(cosines);mean_rmse=sum(rmses)/len(rmses);gates={"rank_energy":energy>=float(p["gates"]["rank_energy_minimum"]),"validation_mean_cosine":mean_cosine>=float(p["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":mean_rmse<=float(p["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(train)+len(validation),"artifact_unchanged":artifact_before==artifact_after};passed=all(gates.values());result={"format":FORMAT,"status":"PASS_LAYER1_RESIDUAL_SPAN_ORACLE" if passed else "FAIL_LAYER1_RESIDUAL_SPAN_ORACLE","protocol_sha256":sha256_file(protocol_path),"rank":rank,"rank_energy":energy,"train_observations":observations,"validation":{"records":len(validation),"mean_cosine":mean_cosine,"minimum_cosine":min(cosines),"mean_relative_rmse":mean_rmse,"maximum_relative_rmse":max(rmses),"record_metrics":records},"route_correct":route_exact,"gates":gates,"passed":passed,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"training_performed":False,"artifact_written":False,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Read-only direct-coefficient layer-1 residual-span oracle on the replacement prefix only; no realizable decoder, checkpoint, autonomous quality, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_LAYER1_RESIDUAL_SPAN_ORACLE_PROTOCOL_V391.json");p.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/layer1_span_oracle_v392");a=p.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/a.protocol).resolve(),(root/a.output_dir).resolve()),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
