"""Prospective full B60 rescreen on LayerCake's exact lexical-guard v21 host."""

from __future__ import annotations

import argparse, gc, hashlib, json, sys, tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_final_controls import evaluate_functional_v2
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_v19_frontier_rescreen import _component_inventory, _json, _merged_evaluation, _quality_gates, _rows, _states
from .capability_compiler_phase4_v20_b60_rescreen import strong_route_conformance


FORMAT="abi-capability-compiler-phase4-v21-b60-rescreen/1"


def load_protocol(root:Path,path:Path)->tuple[dict[str,Any],str]:
    p=_json(path)
    if "base_protocol" in p:
        child=p
        base=_json(root/str(child["base_protocol"]))
        p={**base,**child,"bindings":{**base["bindings"],**child["bindings"]}}
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_THREE_SYSTEM_V21_EXACT_LEXICAL_RESCREEN" or p.get("device")!="cuda" or p.get("training_authorized") is not False or p.get("teacher_model_loading_authorized") is not False or p.get("final_test_access")!="PROHIBITED": raise Phase3Error("v21 B60 governance changed")
    for relative,expected in p["bindings"].items():
        target=(root/relative).resolve()
        if not target.is_file() or sha256_file(target)!=expected: raise Phase3Error(f"v21 B60 binding changed: {relative}")
    return p,sha256_file(path)


def _api(layercake_root:Path)->dict[str,Any]:
    if str(layercake_root) not in sys.path: sys.path.insert(0,str(layercake_root))
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import build_package, load_package, tensor_specs
    from layercake.cake.signing import key_id
    from layercake_extensions.route_isolated_prompt_span_core_v19 import PROMPT_SPAN_FEATURE
    from layercake_extensions.route_isolated_shallow_sparse_core import CAPABILITY_TO_TASK_ROUTE, WEAK_CAPABILITIES
    from layercake_extensions.route_isolated_universal_guard_core_v20 import GUARD_PREDICATE, UNIVERSAL_GUARD_FEATURE
    from layercake_extensions.route_isolated_lexical_guard_core_v21 import ARCHITECTURE_V21_FORMAT, EXACT_LEXICAL_BOUNDARY, EXACT_LEXICAL_GUARD_FEATURE, LexicalGuardPromptSpanCoreHost, ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256, ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION
    return {"CakeManifest":CakeManifest,"build_package":build_package,"load_package":load_package,"tensor_specs":tensor_specs,"key_id":key_id,"task_routes":CAPABILITY_TO_TASK_ROUTE,"weak_capabilities":tuple(WEAK_CAPABILITIES),"architecture_format":ARCHITECTURE_V21_FORMAT,"prompt_span_feature":PROMPT_SPAN_FEATURE,"universal_guard_feature":UNIVERSAL_GUARD_FEATURE,"lexical_guard_feature":EXACT_LEXICAL_GUARD_FEATURE,"guard_predicate":GUARD_PREDICATE,"guard_boundary":EXACT_LEXICAL_BOUNDARY,"Host":LexicalGuardPromptSpanCoreHost,"abi_sha256":ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,"abi_version":ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION}


def _architecture(root:Path,p:Mapping[str,Any],spec:Mapping[str,Any],api:Mapping[str,Any])->dict[str,Any]:
    parent=_json(root/p["model_metadata"]); tokenizer=_json(root/p["model_tokenizer"]); raw=json.dumps(tokenizer,sort_keys=True,separators=(",", ":")).encode(); router_tokenizer=_json(root/p["router_tokenizer"]); router=_json(root/spec["router_config"]); markers=artifact_markers(root/spec["guard_artifact"])
    return {"format":api["architecture_format"],"model":parent["architecture"],"model_tokenizer":{"format":"declarative-tokenizers-json/1","tokenizers_json":tokenizer,"sha256":hashlib.sha256(raw).hexdigest(),"eos_token_id":50256},"router":{"vocabulary":int(router["vocabulary"]),"character_hash_buckets":int(router["character_hash_buckets"]),"character_ngram_minimum":int(router["character_ngram_minimum"]),"character_ngram_maximum":int(router["character_ngram_maximum"]),"hash_seed":int(router["hash_seed"]),"classes":15},"router_tokenizer":router_tokenizer,"residual":{"width":768,"rank":16,"routes":4,"reuse":"before_each_transformer_block"},"capabilities":list(p["capabilities"]),"capability_to_task_route":api["task_routes"],"weak_capabilities":list(api["weak_capabilities"]),"guard":{"predicate":api["guard_predicate"],"scope":"all_capabilities","boundary":api["guard_boundary"],"stop_before_collapsing_token":True,"abstention_markers":list(markers),"abstention_clause":"I cannot determine that from the information given."}}


def _package(root:Path,p:Mapping[str,Any],spec:Mapping[str,Any],path:Path,api:Mapping[str,Any],private,public:bytes)->dict[str,Any]:
    states=_states(root,spec); counts,tensors=_component_inventory(states); signer=api["key_id"](public)
    manifest=api["CakeManifest"](schema_version="1",cake_id=f"abi-phase4-v21-b60-seed{spec['seed']}-english-core",name=f"ABI Phase 4 v21 B60 seed {spec['seed']} English core",description="Frozen B60 lineage on exact lexical-guard v21 host",version="0.21.0-b60-rescreen",publisher={"id":"abi-research","name":"ABI Research","key_id":signer},abi_version=api["abi_version"],abi_hash=api["abi_sha256"],cake_type="portable_decoder",input_contract={"external":"UTF-8 bytes","role":"english-core","validity":"strict_utf8"},output_contract={"external":"UTF-8 bytes","role":"english-core","composition":"direct_core_only_no_router","validity":"strict_utf8"},architecture=_architecture(root,p,spec,api),supported_precisions=("fp32",),supported_backends=("pytorch","cuda"),minimum_host_capabilities={"features":["byte_input","safe_tensors","persistent_incremental_state","physical_route_isolation","declarative_runtime_guard","strict_utf8_boundary",api["prompt_span_feature"],api["universal_guard_feature"],api["lexical_guard_feature"]]},tensor_payload_hash="",tensor_shapes=api["tensor_specs"](tensors),package_hash="",training_data_provenance={"phase4_budget":"B60","phase4_seed":int(spec["seed"]),"lineage_result_sha256":p["bindings"][spec["lineage_result"]],"teacher_at_inference":False,"source_transformer_blocks":0,"receiver_training_steps":0},evaluation_evidence={"authorization":p["authorization"],"status":"V21_EXACT_LEXICAL_DEVELOPMENT_RESCREEN"},license="Apache-2.0",dependencies=(),parent_version=None,signature={"algorithm":"ed25519","key_id":signer},domains=("english-core",),permissions=("local-inference",))
    private_pem=private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()); api["build_package"](path,manifest,tensors,private_key=private_pem); loaded=api["load_package"](path,trust_store={signer:public},require_signature=True); exact=set(loaded.tensors)==set(tensors) and all(torch.equal(loaded.tensors[n],tensors[n]) for n in tensors); gates={"signature_valid":loaded.signed,"tensor_values_exact":exact,"interface_v21":loaded.manifest.abi_version==api["abi_version"] and loaded.manifest.abi_hash==api["abi_sha256"],"component_counts_exact":counts=={"model":61655050,"router":1058040,"residual":99840},"receiver_learning_zero":True,"teacher_absent":True}
    if not all(gates.values()): raise Phase3Error(f"v21 package failed: {gates}")
    return {"archive_sha256":loaded.archive_hash,"tensor_payload_hash":loaded.manifest.tensor_payload_hash,"package_hash":loaded.manifest.package_hash,"archive_bytes":path.stat().st_size,"component_parameters":counts,"total_parameters":sum(counts.values()),"tensor_count":len(tensors),"gates":gates}


@torch.inference_mode()
def _generate(host,prompt:str,maximum:int,capability:str)->tuple[str,bool,dict[str,Any]]:
    if capability=="coherence":
        value=host.generate(prompt,maximum_tokens=maximum).decode("utf-8"); pointer=dict(host.last_pointer_execution or {}); pointer.pop("wall_seconds",None); return value,False,pointer
    state=host.prefill(prompt)
    for _ in range(maximum):
        if host.decode_step(state) is None: break
    return host.realize(state).decode("utf-8"),bool(state["terminated_by_guard"]),{}


@torch.inference_mode()
def run(root:Path,protocol_path:Path,output:Path)->dict[str,Any]:
    p,protocol_sha=load_protocol(root,protocol_path)
    if output.exists(): raise Phase3Error(f"immutable v21 output exists: {output}")
    if not torch.cuda.is_available(): raise Phase3Error("CUDA unavailable")
    api=_api((root/p["layercake_root"]).resolve()); weak=set(api["weak_capabilities"]); private=Ed25519PrivateKey.from_private_bytes(bytes.fromhex(p["research_signing_seed_hex"])); public=private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo); signer=api["key_id"](public); probes_list=development_probes(root/p["development_catalog"]); probes={str(x["probe_id"]):x for x in probes_list}; teacher={str(x["probe_id"]):x for x in _rows(root/p["teacher_reference"])}; systems=[]; aggregate=[]
    for spec in p["systems"]:
        historical=_rows(root/spec["historical_outputs"]); old={str(x["probe_id"]):x for x in historical}
        with tempfile.TemporaryDirectory(prefix=f"abi-v21-b60-{spec['seed']}-") as raw:
            temp=Path(raw); package=_package(root,p,spec,temp/"candidate.cake",api,private,public); host=api["Host"](temp/"registry",trust_store={signer:public},device="cuda"); active=host.activate(temp/"candidate.cake"); rows=[]
            for index,probe in enumerate(probes_list):
                pid=str(probe["probe_id"]); capability=str(probe["canonical_capability"]); prior=old[pid]; value,terminated,pointer=_generate(host,str(probe["prompt"]),int(probe["max_new_tokens"]),capability); exact=value==str(prior["output"]); prefix=str(prior["output"]).startswith(value)
                row={**prior,"output":value,"original_output":value,"output_token_ids":[int(x) for x in host.model_tokenizer.encode(value)],"automatic_capability_route":host.route(str(probe["prompt"])),"capability_route_correct":host.route(str(probe["prompt"]))==capability,"strong_parent_output_exact":exact if capability not in weak else prior["strong_parent_output_exact"],"strong_parent_prefix_preserved":prefix if capability not in weak else True,"guard_terminated":terminated,"abstention_clause_prefixed":capability=="abstention" and value.startswith("I cannot determine that from the information given."),"functional_pass_v1":evaluate_functional(value,probe["evaluator"]),"functional_pass_v2":evaluate_functional_v2(value,probe["evaluator"],capability),"repetition_collapse_v2":repetition_collapse_v2(value),"v21_pointer":pointer,"output_changed_from_v19_history":not exact}; rows.append(row); aggregate.append({"seed":int(spec["seed"]),**row})
                if (index+1)%200==0: print(json.dumps({"seed":spec["seed"],"rows":index+1}),flush=True)
            verified=host.verify(); del host; gc.collect(); torch.cuda.empty_cache()
        evaluation=_merged_evaluation(rows); quality,relative=_quality_gates(p,evaluation,rows,probes,teacher,int(spec["seed"])+6_000_000); quality.pop("strong_parent_exact"); changed=[x for x in rows if x["output_changed_from_v19_history"]]; coherence=[x for x in rows if x["capability"]=="coherence"]
        pointer={"all_100_pointer_rows":len(coherence)==100 and all(bool(x["v21_pointer"]) for x in coherence),"six_candidates":all(x["v21_pointer"].get("candidate_count")==6 for x in coherence),"one_scoring_forward":all(x["v21_pointer"].get("candidate_scoring_forward_passes")==1 for x in coherence),"one_active_route":all(x["v21_pointer"].get("active_residual_routes")==1 for x in coherence),"persistent_state_reused":all(x["v21_pointer"].get("persistent_prompt_state_reused") is True for x in coherence),"evaluator_blind":all(x["v21_pointer"].get("evaluator_used") is False for x in coherence),"package_identity":active["archive_hash"]==package["archive_sha256"] and active["payload_hash"]==package["tensor_payload_hash"],"package_verified":verified["status"]=="PASS","receiver_learning_zero":active["receiver_training_steps"]==active["receiver_calibration_runs"]==0}
        guard={"strong_route_conformance":strong_route_conformance(rows,weak),"changed_rows_exactly_one_strong_repair":len(changed)==1 and changed[0]["capability"]=="supplied_text_summarization","changed_row_guard_terminated":all(x["guard_terminated"] for x in changed),"changed_row_prefix_preserved":all(x["strong_parent_prefix_preserved"] for x in changed),"changed_row_functional":all(x["functional_pass_v1"] for x in changed),"all_other_outputs_exact":sum(not x["output_changed_from_v19_history"] for x in rows)==1399,"zero_remaining_collapse":evaluation["repetition_collapses_v2"]==0,"exact_lexical_boundary_declared":True}; machine=all(quality.values()) and all(pointer.values()) and all(guard.values()); path=output/f"seed{spec['seed']}_outputs.jsonl"; output.mkdir(parents=True,exist_ok=True); _write_immutable(path,b"".join(canonical_json_bytes(x) for x in rows)); systems.append({"budget":"B60","seed":int(spec["seed"]),"status":"PASS" if machine else "FAIL","machine_gates_pass":machine,"evaluation":evaluation,"teacher_comparison_v1":relative,"quality_gates":quality,"guard_gates":guard,"pointer_gates":pointer,"changed_rows":len(changed),"guard_terminations":sum(x["guard_terminated"] for x in rows),"package":package,"activation":{"archive_sha256":active["archive_hash"],"tensor_payload_hash":active["payload_hash"],"receiver_training_steps":active["receiver_training_steps"],"receiver_calibration_runs":active["receiver_calibration_runs"],"verification":verified["status"]},"outputs":{"path":str(path.relative_to(root)).replace("\\","/"),"sha256":sha256_file(path)}})
    aggregate_path=output/"all_outputs.jsonl"; _write_immutable(aggregate_path,b"".join(canonical_json_bytes(x) for x in aggregate)); stable=all(x["machine_gates_pass"] for x in systems); result={"format":"abi-capability-compiler-phase4-v21-b60-rescreen-result/1","status":"PASS_STABLE_B60_V21_DEVELOPMENT_CANDIDATE" if stable else "FAIL_B60_V21_DEVELOPMENT_CANDIDATE","protocol_sha256":protocol_sha,"systems":systems,"three_seed_all_pass":stable,"observations":len(aggregate),"model_inference_rows":len(aggregate),"changed_rows":sum(x["changed_rows"] for x in systems),"guard_terminations":sum(x["guard_terminations"] for x in systems),"aggregate_outputs_sha256":sha256_file(aggregate_path),"training_performed":False,"teacher_model_loaded":False,"receiver_training_steps":0,"final_test_accessed":False,"phase4_certified":False,"claim_boundary":"Three-seed B60 v21 development rescreen only. No information minimum, matched baseline, final test, Phase 4, or ABI-superiority claim."}; result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n"); return result


def main(argv:Iterable[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol",required=True); parser.add_argument("--output-dir",required=True); args=parser.parse_args(argv); root=Path.cwd().resolve(); result=run(root,root/args.protocol,root/args.output_dir); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1
if __name__=="__main__": raise SystemExit(main())
