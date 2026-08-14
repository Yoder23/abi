"""Read-only independent evaluator repair for the complete V771 v21 B60 rows."""

from __future__ import annotations

import argparse, hashlib, json, tempfile
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_final_controls import evaluate_functional_v2
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_v19_frontier_rescreen import _json, _merged_evaluation, _quality_gates, _rows
from .capability_compiler_phase4_v19_frontier_verify import _pointer_checks, _without
from .capability_compiler_phase4_v20_b60_rescreen import strong_route_conformance
from .capability_compiler_phase4_v21_b60_rescreen import _api, _package, load_protocol as load_source_protocol


FORMAT="abi-capability-compiler-phase4-v21-b60-evaluator-repair/1"


def expected_changed_rows(seed:int)->int:
    if seed not in (104729,130363,155921): raise Phase3Error("unregistered B60 seed")
    return 1 if seed==104729 else 0


def load_protocol(root:Path,path:Path)->tuple[dict[str,Any],str]:
    p=_json(path)
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_V21_EVALUATOR_REPAIR" or p.get("model_inference_authorized") is not False or p.get("training_authorized") is not False or p.get("teacher_model_loading_authorized") is not False or p.get("final_test_access")!="PROHIBITED": raise Phase3Error("v21 evaluator repair governance changed")
    for relative,expected in p["bindings"].items():
        target=(root/relative).resolve()
        if not target.is_file() or sha256_file(target)!=expected: raise Phase3Error(f"v21 evaluator repair binding changed: {relative}")
    return p,sha256_file(path)


def run(root:Path,protocol_path:Path,output:Path)->dict[str,Any]:
    repair,repair_sha=load_protocol(root,protocol_path)
    if output.exists(): raise Phase3Error(f"immutable repair output exists: {output}")
    source,source_sha=load_source_protocol(root,root/repair["source_protocol"]); recorded=_json(root/repair["source_result"]); recorded_evidence=hashlib.sha256(canonical_json_bytes(_without(recorded,"evidence_sha256"))).hexdigest(); api=_api((root/source["layercake_root"]).resolve()); weak=set(api["weak_capabilities"]); private=Ed25519PrivateKey.from_private_bytes(bytes.fromhex(source["research_signing_seed_hex"])); public=private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
    probes_list=development_probes(root/source["development_catalog"]); probes={str(x["probe_id"]):x for x in probes_list}; teacher={str(x["probe_id"]):x for x in _rows(root/source["teacher_reference"])}; specs={int(x["seed"]):x for x in source["systems"]}; verified=[]; aggregate=[]
    from layercake_extensions.route_isolated_prompt_span_core_v19 import extract_prompt_segments, render_prompt_segments
    for system in recorded["systems"]:
        seed=int(system["seed"]); spec=specs[seed]; rows=_rows(root/system["outputs"]["path"]); historical=_rows(root/spec["historical_outputs"]); history={str(x["probe_id"]):x for x in historical}; changed=[]; row_checks=[]
        for row in rows:
            pid=str(row["probe_id"]); probe=probes[pid]; prior=history[pid]; output_text=str(row["output"]); exact=output_text==str(prior["output"])
            if not exact: changed.append(row)
            checks={"probe_identity":str(probe["canonical_capability"])==str(row["capability"]),"route_record":str(row["automatic_capability_route"])==str(row["capability"]) and bool(row["capability_route_correct"]),"functional_v1":bool(row["functional_pass_v1"])==evaluate_functional(output_text,probe["evaluator"]),"functional_v2":bool(row["functional_pass_v2"])==evaluate_functional_v2(output_text,probe["evaluator"],str(row["capability"])),"collapse":bool(row["repetition_collapse_v2"])==repetition_collapse_v2(output_text),"change_flag":bool(row["output_changed_from_v19_history"])==(not exact)}
            if row["capability"]=="coherence": checks.update(_pointer_checks(str(probe["prompt"]),output_text,dict(row["v21_pointer"]),extract_prompt_segments,render_prompt_segments))
            row_checks.append(checks); aggregate.append({"seed":seed,**row})
        evaluation=_merged_evaluation(rows); quality,relative=_quality_gates(source,evaluation,rows,probes,teacher,seed+6_000_000); quality.pop("strong_parent_exact"); coherence=[x for x in rows if x["capability"]=="coherence"]
        pointer={"all_100_pointer_rows":len(coherence)==100 and all(bool(x["v21_pointer"]) for x in coherence),"six_candidates":all(x["v21_pointer"].get("candidate_count")==6 for x in coherence),"one_scoring_forward":all(x["v21_pointer"].get("candidate_scoring_forward_passes")==1 for x in coherence),"one_active_route":all(x["v21_pointer"].get("active_residual_routes")==1 for x in coherence),"persistent_state_reused":all(x["v21_pointer"].get("persistent_prompt_state_reused") is True for x in coherence),"evaluator_blind":all(x["v21_pointer"].get("evaluator_used") is False for x in coherence),"package_identity":system["activation"]["archive_sha256"]==system["package"]["archive_sha256"] and system["activation"]["tensor_payload_hash"]==system["package"]["tensor_payload_hash"],"package_verified":system["activation"]["verification"]=="PASS","receiver_learning_zero":system["activation"]["receiver_training_steps"]==system["activation"]["receiver_calibration_runs"]==0}
        expected=expected_changed_rows(seed); guard={"strong_route_conformance":strong_route_conformance(rows,weak),"changed_count_seed_specific":len(changed)==expected,"seed104_repair_identity":seed!=104729 or (len(changed)==1 and changed[0]["capability"]=="supplied_text_summarization"),"changed_rows_guard_terminated":all(x["guard_terminated"] for x in changed),"changed_rows_prefix_preserved":all(x["strong_parent_prefix_preserved"] for x in changed),"changed_rows_functional":all(x["functional_pass_v1"] for x in changed),"all_other_outputs_exact":sum(str(x["output"])==str(history[str(x["probe_id"])]["output"]) for x in rows)==1400-expected,"zero_remaining_collapse":evaluation["repetition_collapses_v2"]==0,"exact_lexical_boundary_declared":source["interface"]=="lc-direct-neural-core/21"}
        with tempfile.TemporaryDirectory(prefix=f"abi-v21-repair-{seed}-") as raw: rebuilt=_package(root,source,spec,Path(raw)/"candidate.cake",api,private,public)
        machine=all(quality.values()) and all(pointer.values()) and all(guard.values()); gates={"depth":len(rows)==len(historical)==1400,"raw_hash":sha256_file(root/system["outputs"]["path"])==system["outputs"]["sha256"],"all_row_checks":all(all(x.values()) for x in row_checks),"evaluation_recomputed":evaluation==system["evaluation"],"quality_recomputed":quality==system["quality_gates"],"teacher_comparison_recomputed":relative==system["teacher_comparison_v1"],"pointer_recomputed":pointer==system["pointer_gates"],"package_rebuilt_exact":rebuilt==system["package"],"corrected_machine_pass":machine}
        verified.append({"budget":"B60","seed":seed,"functional_passes_v1":evaluation["functional_passes_v1"],"repetition_collapses_v2":evaluation["repetition_collapses_v2"],"changed_rows":len(changed),"quality_gates":quality,"pointer_gates":pointer,"corrected_guard_gates":guard,"corrected_machine_gates_pass":machine,"verification_gates":gates,"all_pass":all(gates.values())}); print(json.dumps({"verified":seed,"all_pass":all(gates.values())}),flush=True)
    aggregate_path=root/repair["aggregate_outputs"]; aggregate_exact=b"".join(canonical_json_bytes(x) for x in aggregate)==aggregate_path.read_bytes(); stable=all(x["all_pass"] and x["corrected_machine_gates_pass"] for x in verified); top={"source_protocol_hash":source_sha==recorded["protocol_sha256"],"source_result_hash":sha256_file(root/repair["source_result"])==repair["bindings"][repair["source_result"]],"source_evidence_hash":recorded_evidence==recorded["evidence_sha256"],"three_registered_systems":len(verified)==3,"all_system_verifiers":all(x["all_pass"] for x in verified),"aggregate_exact":aggregate_exact,"aggregate_hash":sha256_file(aggregate_path)==recorded["aggregate_outputs_sha256"],"changed_rows_total":sum(x["changed_rows"] for x in verified)==1,"stable_three_seed_pass":stable,"model_inference_absent":True,"training_absent":True,"teacher_loading_absent":True,"final_test_not_accessed":True}
    result={"format":"abi-capability-compiler-phase4-v21-b60-evaluator-repair-result/1","status":"PASS_CORRECTED_STABLE_B60_V21_DEVELOPMENT_CANDIDATE" if all(top.values()) else "FAIL_V21_EVALUATOR_REPAIR","protocol_sha256":repair_sha,"source_result_sha256":sha256_file(root/repair["source_result"]),"source_evidence_sha256":recorded["evidence_sha256"],"systems":verified,"three_seed_all_pass":stable,"gates":top,"packages_deterministically_rebuilt":3,"rows_recomputed":4200,"model_inference_performed":False,"training_performed":False,"teacher_model_loaded":False,"final_test_accessed":False,"phase4_certified":False,"claim_boundary":"Corrected stable B60 v21 development candidacy only. No information minimum, matched baseline, final test, Phase 4, or ABI-superiority claim."}; result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest(); output.parent.mkdir(parents=True,exist_ok=True); _write_immutable(output,json.dumps(result,indent=2,sort_keys=True).encode()+b"\n"); return result


def main(argv:Iterable[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(argv); root=Path.cwd().resolve(); result=run(root,root/args.protocol,root/args.output); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1
if __name__=="__main__": raise SystemExit(main())
