"""V47: train a minimal route-token bridge into the frozen V41 generator."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    set_determinism,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _BalancedSampler, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _json, _layercake_api, _model, _tokenizer
from .capability_compiler_phase3_bpe_core_analysis import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_segment_router import _semantic_segments
from . import capability_compiler_phase3_sparse_router as sparse


FORMAT = "abi-capability-compiler-phase3-route-bridge/1"
BOS_ID = 1
PAD_ID = 0


def load_protocol(root: Path, path: Path) -> tuple[Mapping[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_ROUTE_BRIDGE_SCREEN"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("training", {}).get("device") != "cuda"
    ):
        raise Phase3Error("V47 governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V47 binding changed: {relative}")
    return protocol, sha256_file(path)


def _base(root: Path, protocol: Mapping[str, Any], device: torch.device):
    _, model_type, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    candidate = (root / protocol["base_generator"]["candidate_dir"]).resolve()
    model = _model(protocol, tokenizer, model_type)
    checkpoint = candidate / "model.safetensors"
    if sha256_file(checkpoint) != protocol["base_generator"]["checkpoint_sha256"]:
        raise Phase3Error("V41 generator changed")
    model.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    return model.to(device), tokenizer


def _select_controls(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[tuple[int, bytes]]:
    used: set[int] = set()
    for row in rows:
        for field in ("normalized_acquisition_prompt", "normalized_output"):
            used.update(tokenizer.lexeme_to_id[piece] for piece in tokenizer.split(str(row[field])))
    eligible = []
    for piece, token_id in tokenizer.lexeme_to_id.items():
        text = piece.decode("utf-8")
        if (
            token_id not in used
            and 4 <= len(text) <= 16
            and text.isalpha()
            and tokenizer.split(text) == [piece]
            and tokenizer.split(text + "\n")[0] == piece
        ):
            eligible.append((hashlib.sha256(piece).hexdigest(), token_id, piece))
    selected = [(token_id, piece) for _, token_id, piece in sorted(eligible)[: len(CAPABILITIES)]]
    if len(selected) != len(CAPABILITIES):
        raise Phase3Error("insufficient training-unused route controls")
    return selected


def _examples(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], list[tuple[int, bytes]]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    controls = _select_controls(rows, tokenizer)
    control_by_capability = {capability: controls[index][0] for index, capability in enumerate(CAPABILITIES)}
    examples = []
    for row in rows:
        lines = str(row["normalized_acquisition_prompt"]).splitlines()
        body = "\n".join(lines[1:]).strip()
        source_ids = [control_by_capability[str(row["capability"])]] + [
            tokenizer.lexeme_to_id[piece] for piece in tokenizer.split("\n" + body)
        ]
        target = [tokenizer.lexeme_to_id[piece] for piece in tokenizer.split(str(row["normalized_output"]))] + [2]
        if len(source_ids) > int(protocol["architecture"]["maximum_source_lexemes"]) or len(target) > int(protocol["architecture"]["maximum_target_actions"]):
            raise Phase3Error("route-bridge example exceeds host bound")
        examples.append({"record_id":str(row["ir_record_id"]),"capability":str(row["capability"]),"source_ids":source_ids,"target_actions":target})
    return examples, controls


def _collate(rows: Sequence[Mapping[str, Any]], device: torch.device):
    source_width=max(len(row["source_ids"]) for row in rows); target_width=max(len(row["target_actions"]) for row in rows)
    source=torch.full((len(rows),source_width),PAD_ID,dtype=torch.long,device=device);targets=torch.full((len(rows),target_width),-100,dtype=torch.long,device=device)
    for index,row in enumerate(rows):
        source[index,:len(row["source_ids"])]=torch.tensor(row["source_ids"],device=device);targets[index,:len(row["target_actions"])]=torch.tensor(row["target_actions"],device=device)
    return source,targets


def _encoded(model: nn.Module, source: torch.Tensor, routes: nn.Embedding, labels: torch.Tensor):
    padding=source.eq(PAD_ID);positions=torch.arange(source.shape[1],device=source.device)
    ordinary=model.lexeme_embedding(source)
    first=routes(labels)[:,None,:]
    embedded=torch.cat((first,ordinary[:,1:]),dim=1)+model.source_position(positions)[None]
    return model.encoder(embedded,src_key_padding_mask=padding),padding


def inventory(root: Path, path: Path) -> Mapping[str, Any]:
    protocol, protocol_hash=load_protocol(root,path);model,tokenizer=_base(root,protocol,torch.device("cpu"));examples,controls=_examples(root,protocol,tokenizer)
    selection=hashlib.sha256(canonical_json_bytes([{"capability":c,"token_id":controls[i][0],"piece_hex":controls[i][1].hex()} for i,c in enumerate(CAPABILITIES)])).hexdigest()
    if selection!=protocol["route_controls"]["selection_sha256"]:raise Phase3Error("route-control selection changed")
    return {"status":"PASS","protocol_sha256":protocol_hash,"records":len(examples),"bridge_parameters":len(CAPABILITIES)*model.model_width,"deployed_parameters":model.parameter_count(),"route_control_selection_sha256":selection,"maximum_source_actions":max(len(x["source_ids"]) for x in examples),"maximum_target_actions":max(len(x["target_actions"]) for x in examples),"teacher_outputs_added":0,"final_test_accessed":False}


def train(root: Path, path: Path, output: Path) -> Mapping[str, Any]:
    protocol,protocol_hash=load_protocol(root,path)
    if output.exists() or not torch.cuda.is_available():raise Phase3Error("route-bridge output exists or CUDA unavailable")
    device=torch.device("cuda");model,tokenizer=_base(root,protocol,device);examples,controls=_examples(root,protocol,tokenizer);cfg=protocol["training"];seed=int(cfg["seed"]);set_determinism(seed)
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval();routes=nn.Embedding(len(CAPABILITIES),model.model_width).to(device)
    with torch.no_grad():routes.weight.copy_(model.lexeme_embedding.weight[torch.tensor([x[0] for x in controls],device=device)])
    optimizer=torch.optim.AdamW(routes.parameters(),lr=float(cfg["learning_rate"]),weight_decay=0.0);sampler=_BalancedSampler(examples,seed);label_ids={c:i for i,c in enumerate(CAPABILITIES)};curves=[];sequence=hashlib.sha256();started=time.perf_counter();process=psutil.Process();peak=process.memory_info().rss
    for step in range(1,int(cfg["steps"])+1):
        batch=sampler.batch(int(cfg["batch_size"]));source,targets=_collate(batch,device);labels=torch.tensor([label_ids[x["capability"]] for x in batch],device=device);previous=torch.full_like(targets,PAD_ID);previous[:,0]=BOS_ID
        if targets.shape[1]>1: previous[:,1:]=torch.where(targets[:,:-1].ge(0),targets[:,:-1],torch.full_like(targets[:,:-1],PAD_ID))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            encoded,padding=_encoded(model,source,routes,labels);log_probs=model.action_log_probs(source,previous,encoded=encoded,source_padding=padding)["log_probs"];loss=F.nll_loss(log_probs.float().reshape(-1,log_probs.shape[-1]),targets.reshape(-1),ignore_index=-100)
        loss.backward();torch.nn.utils.clip_grad_norm_(routes.parameters(),1.0);optimizer.step();peak=max(peak,process.memory_info().rss)
        for row in batch:sequence.update(row["record_id"].encode()+b"\n")
        if step==1 or step%int(cfg["curve_interval"])==0:
            value={"step":step,"loss":float(loss.detach()),"wall_seconds":time.perf_counter()-started};curves.append(value);print(json.dumps(value),flush=True)
    with torch.no_grad():model.lexeme_embedding.weight[torch.tensor([x[0] for x in controls],device=device)]=routes.weight
    output.mkdir(parents=True);checkpoint=output/"model.safetensors";save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint));router_source=(root/protocol["router"]["checkpoint_path"]).resolve();router_target=output/"router.safetensors";router_target.write_bytes(router_source.read_bytes());tokenizer_path=output/"tokenizer.json";_write_immutable(tokenizer_path,json.dumps(tokenizer.canonical_dict(),sort_keys=True,indent=2).encode()+b"\n");config_path=output/"model_config.json";_write_immutable(config_path,json.dumps({**protocol["architecture"],"fixed_vocab_size":tokenizer.vocab_size},sort_keys=True,indent=2).encode()+b"\n")
    control_doc=[{"capability":c,"token_id":controls[i][0],"piece_hex":controls[i][1].hex()} for i,c in enumerate(CAPABILITIES)];control_path=output/"route_controls.json";_write_immutable(control_path,json.dumps(control_doc,sort_keys=True,indent=2).encode()+b"\n")
    metadata={"format":"abi-capability-compiler-phase3-route-bridge-candidate/1","status":"TRAINED_CONDITIONAL_DEVELOPMENT_SCREEN","protocol_sha256":protocol_hash,"seed":seed,"checkpoint":{"path":"model.safetensors","sha256":sha256_file(checkpoint),"bytes":checkpoint.stat().st_size},"router":{"path":"router.safetensors","sha256":sha256_file(router_target),"bytes":router_target.stat().st_size,"parameters":1058040},"tokenizer":{"path":"tokenizer.json","sha256":sha256_file(tokenizer_path),"canonical_sha256":tokenizer.hash(),"vocabulary":tokenizer.vocab_size},"model_config":{"path":"model_config.json","sha256":sha256_file(config_path),"trainable_parameters":model.parameter_count()},"route_controls":{"path":"route_controls.json","sha256":sha256_file(control_path),"selection_sha256":protocol["route_controls"]["selection_sha256"]},"bridge":{"trainable_parameters":routes.weight.numel(),"materialized_model_rows":len(CAPABILITIES),"base_checkpoint_sha256":protocol["base_generator"]["checkpoint_sha256"]},"training":{"steps":int(cfg["steps"]),"batch_size":int(cfg["batch_size"]),"wall_seconds":time.perf_counter()-started,"record_sequence_sha256":sequence.hexdigest(),"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"curves":curves},"imported_information":{"records":7000,"teacher_outputs_added":0,"stored_logits":0,"stored_activations":0,"source_parameters_copied":0},"teacher_present_at_inference":False,"source_blocks_retained":0,"promotion_eligible":False,"layercake_host_changed":False,"phase3_certified":False,"final_test_accessed":False,"hardware":{"machine":platform.node(),"gpu":torch.cuda.get_device_name(0)}};metadata["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(metadata)).hexdigest();_write_immutable(output/"metadata.json",json.dumps(metadata,sort_keys=True,indent=2).encode()+b"\n");return metadata


def _load_candidate(root:Path,protocol:Mapping[str,Any],candidate:Path):
    _,model_type,tokenizer_type,_,_=_layercake_api(root,protocol);tokenizer=tokenizer_type.from_document(_json(candidate/"tokenizer.json"));model=model_type(**_json(candidate/"model_config.json")).bind_tokenizer(tokenizer);model.load_state_dict(load_file(str(candidate/"model.safetensors"),device="cuda"),strict=True);return model.cuda().eval(),tokenizer


def evaluate(root:Path,path:Path,candidate:Path,output:Path)->Mapping[str,Any]:
    protocol,protocol_hash=load_protocol(root,path);metadata=_json(candidate/"metadata.json")
    if output.exists() or metadata.get("protocol_sha256")!=protocol_hash or sha256_file(candidate/"model.safetensors")!=metadata["checkpoint"]["sha256"]:raise Phase3Error("route-bridge evaluation identity failed")
    model,tokenizer=_load_candidate(root,protocol,candidate);router_protocol=_json(root/protocol["router"]["protocol_path"]);router,router_tokenizer=sparse._load(root,router_protocol,(root/protocol["router"]["candidate_dir"]).resolve());controls=_json(candidate/"route_controls.json");control={x["capability"]:bytes.fromhex(x["piece_hex"]).decode() for x in controls};probes=development_probes((root/protocol["development_catalog"]).resolve());rows=[];started=time.perf_counter()
    for index,probe in enumerate(probes):
        prompt=str(probe["prompt"]);route,_=sparse._route(router,router_tokenizer,router_protocol,prompt);body=_semantic_segments(prompt)[-1];controlled=control[route]+"\n"+body;error=None
        try: value=model.generate_bytes(controlled,maximum_actions=min(int(probe["max_new_tokens"]),int(protocol["architecture"]["maximum_target_actions"]))).decode("utf-8",errors="strict")
        except Exception as exc:value="";error=f"{type(exc).__name__}: {exc}"
        rows.append({"probe_id":str(probe["probe_id"]),"capability":str(probe["canonical_capability"]),"predicted_route":route,"route_correct":route==str(probe["canonical_capability"]),"output":value,"generation_error":error,"functional_pass":evaluate_functional(value,probe["evaluator"]),"repetition_collapse":repetition_collapse(value)})
        if (index+1)%100==0:print(json.dumps({"evaluated":index+1}),flush=True)
    output.mkdir(parents=True);raw=output/"development_outputs.jsonl";raw.write_bytes(b"".join(canonical_json_bytes(x) for x in rows));decision=_decision(root,protocol,protocol_hash,metadata,rows,sha256_file(raw),time.perf_counter()-started);_write_immutable(output/"decision.json",json.dumps(decision,sort_keys=True,indent=2).encode()+b"\n");return decision


def _decision(root,protocol,protocol_hash,metadata,rows,raw_hash,wall):
    per={};
    for capability in CAPABILITIES:
        values=[x for x in rows if x["capability"]==capability];passes=sum(x["functional_pass"] for x in values);per[capability]={"passes":passes,"observations":len(values),"collapses":sum(x["repetition_collapse"] for x in values),"wilson":wilson(passes,len(values))}
    teacher={str(x["probe_id"]):x for x in map(json.loads,open(root/protocol["teacher_reference"],encoding="utf-8"))};probes={str(x["probe_id"]):x for x in development_probes((root/protocol["development_catalog"]).resolve())};paired=[{"capability":x["capability"],"candidate_pass":bool(x["functional_pass"]),"teacher_pass":evaluate_functional(str(teacher[x["probe_id"]]["output"]),probes[x["probe_id"]]["evaluator"])} for x in rows];comparison=paired_stratified_bootstrap(paired,replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]),seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]));g=protocol["absolute_screen"];ordinary=all(x["wilson"]["point"]>=g["per_capability_functional_point_estimate_minimum"] and x["wilson"]["lower_95"]>=g["per_capability_functional_wilson_lower_minimum"] for x in per.values());critical=all(per[c]["wilson"]["point"]>=g["critical_point_minimum"] and per[c]["wilson"]["lower_95"]>=g["critical_wilson_lower_minimum"] for c in ("prompt_grounding","instruction_following","abstention"));collapses=sum(x["repetition_collapse"] for x in rows);errors=sum(x["generation_error"] is not None for x in rows);gates={"per_capability_functional":ordinary,"critical_capabilities":critical,"zero_repetition_collapses":collapses==0,"zero_generation_errors":errors==0,"router_accuracy":sum(x["route_correct"] for x in rows)==len(rows),"teacher_relative_noninferiority":comparison["lower_95"]>=protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]};passed=all(gates.values());result={"format":"abi-capability-compiler-phase3-route-bridge-decision/1","status":"PASS_INITIAL_INTEGRATED_SCREEN_REPLICATIONS_AND_HOST_CERTIFICATION_REQUIRED" if passed else "FAIL_INITIAL_INTEGRATED_SCREEN_ROUTE_BRIDGE_CLOSED","protocol":{"path":"ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_BRIDGE_PROTOCOL_V47.json","sha256":protocol_hash},"checkpoint_sha256":metadata["checkpoint"]["sha256"],"router_sha256":metadata["router"]["sha256"],"functional_passes":sum(x["functional_pass"] for x in rows),"observations":len(rows),"per_capability":per,"repetition_collapses":collapses,"generation_errors":errors,"route_correct":sum(x["route_correct"] for x in rows),"teacher_comparison":comparison,"gates":gates,"initial_screen_pass":passed,"outputs_sha256":raw_hash,"evaluation_wall_seconds":wall,"teacher_present_at_inference":False,"layercake_host_changed":False,"phase3_certified":False,"final_test_accessed":False,"next_step":"Preregister paired seeds and LayerCake host certification." if passed else "Preserve failure; no seeds or host promotion."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();return result


def main(argv:Iterable[str]|None=None):
    p=argparse.ArgumentParser();p.add_argument("command",choices=("inventory","train","evaluate"));p.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_BRIDGE_PROTOCOL_V47.json");p.add_argument("--candidate-dir",default="results/abi_capability_compiler_phase3_route_bridge/development_v47/G0-seed240047");p.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_route_bridge/evaluation_v47/G0-seed240047");a=p.parse_args(argv);root=Path.cwd().resolve();path=(root/a.protocol).resolve();result=inventory(root,path) if a.command=="inventory" else train(root,path,(root/a.candidate_dir).resolve()) if a.command=="train" else evaluate(root,path,(root/a.candidate_dir).resolve(),(root/a.output_dir).resolve());print(json.dumps(result,sort_keys=True,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
