"""Fail-closed raw recomputation of the bounded R97 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.generic_prompt_identity_r83.screen_host_v1 import _bootstrap, _jsonl
from experiments.length_invariant_span_r96.core_v1 import validate_metadata


EXPECTED = {
    "result_file": "e50c92e0da7f70ddcbc4b93a1e716b0c5fdf0ca1cdab8b45b5e10c1b7eb34f14",
    "result_evidence": "5caa7f0a46b2c0b377e08a3eb032d122a2515097ce9c6fa22720e9896ebaf9f9",
    "raw": "b4a4999228af61a509a7d22f8f6906b3075ce649a52e531b959646fbbdb40b8c",
    "source_result_file": "64855eaea22037a5051cec875d4c057ff479384715d7795c46feab976d8f5d5a",
    "source_evidence": "c1f45da421a4453141d94d406b72bd13ceff5d586e245c7f15066ad92f8155b1",
    "source_raw": "8a7de8be5f84249f2e00224363793a0ad20d92d197c2e1621ddc17a3d3443802",
    "catalog": "1fb192d66203fe3c5e29cae5d9533d2758902bd1e74f58bcaec324adafa2006b",
    "binding_file": "b013f998308e529f2e31039482523eb61b99adcc51def6b10c3d8596cbd14204",
    "binding_claim": "216f4092c2145d0ac1e63ebbb5ee28dc2ba726261c9929404d449bec11828397",
    "candidate_binding_file": "6e32708745a77f93b6e9ff83761003ec978bcda34cf14ba11701396039e98970",
    "candidate_binding_claim": "97f45c6cfb5398586ed5a27b49d580fe34530a4791a9cd78f031e3779566d2c1",
    "candidate": "9d3d6b38b1d437b5cbbed50c1470b4df1aa0371ffa51b1abd2adefbb645806e4",
    "candidate_metadata": "1cfd15fe838b903ede7e59417f837a1bf00b3664e6deb29bd02876750bcabcb9",
    "parent": "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e",
    "parent_metadata": "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e",
    "artifact": "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc",
}
TOKENIZER_FILES = {
    "merges.txt": (456_318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "special_tokens_map.json": (494, "fee4e5d22631d2d4598132f059304af341ff9c5ed7a5bde94be6a7f47a1a3817"),
    "tokenizer.json": (3_557_680, "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e"),
    "tokenizer_config.json": (534, "c8e5a90723b23e61d70174aeaa0d3688716f6a4b5a9ddf4cc1027b978e187899"),
    "vocab.json": (798_156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7"),
}


class VerificationError(RuntimeError):
    pass


def require_file(path: Path, digest: str) -> None:
    if not path.is_file() or _sha256_file(path) != digest:
        raise VerificationError(f"required file missing or changed: {path}")


def evidence_result(path: Path, file_digest: str, evidence_digest: str) -> dict[str, Any]:
    require_file(path, file_digest)
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value); claim = unsigned.pop("evidence_sha256", None)
    if claim != evidence_digest or evidence_hash(unsigned) != claim:
        raise VerificationError(f"evidence result is stale or unrecomputable: {path}")
    return value


def binding(path: Path, root: Path) -> dict[str, Any]:
    require_file(path, EXPECTED["binding_file"])
    value = json.loads(path.read_text(encoding="utf-8")); claim = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    if claim != EXPECTED["binding_claim"] or _canonical_sha(unsigned) != claim or value.get("candidate_observations_before_binding") != 0 or value.get("candidate_retrained_or_calibrated_after_r97_catalog") is not False:
        raise VerificationError("R97 binding is stale or invalid")
    for name, item in value.get("files", {}).items():
        target = root / str(item.get("path", ""))
        if not target.is_file() or target.stat().st_size != item.get("bytes") or _sha256_file(target) != item.get("sha256"):
            raise VerificationError(f"R97 bound file changed: {name}")
    return value


def verify(root: Path) -> dict[str, Any]:
    candidate = root / "results/length_invariant_span_r96/candidate_v1"
    parent = root / "results/role_invariant_choice_r76/candidate_v1"
    artifact = root / "results/role_invariant_choice_r76/reasoning-role-invariant-search-v2.abix"
    for path, digest in ((candidate/"bridge.safetensors",EXPECTED["candidate"]),(candidate/"metadata.json",EXPECTED["candidate_metadata"]),(parent/"model.safetensors",EXPECTED["parent"]),(parent/"metadata.json",EXPECTED["parent_metadata"]),(artifact,EXPECTED["artifact"])): require_file(path,digest)
    for name,(size,digest) in TOKENIZER_FILES.items():
        path=parent/name
        if not path.is_file() or path.stat().st_size!=size or _sha256_file(path)!=digest: raise VerificationError(f"parent tokenizer changed: {name}")
    metadata=json.loads((candidate/"metadata.json").read_text(encoding="utf-8")); validate_metadata(metadata,candidate)
    if metadata["bridge"]["parameter_count"]!=530_050 or metadata["training"]["successful_optimizer_steps"]!=3_000 or metadata["training"]["unique_records_seen"]!=8_129 or metadata["normalization"]["r95_exact_prompts_consumed"]!=0 or metadata["source_boundary"]["teacher_present_at_inference"] is not False: raise VerificationError("R96 package accounting changed")
    candidate_binding_path=root/"results/length_invariant_span_r96/candidate_binding_v1.json"; require_file(candidate_binding_path,EXPECTED["candidate_binding_file"])
    frozen=json.loads(candidate_binding_path.read_text(encoding="utf-8")); frozen_claim=frozen.get("binding_sha256"); frozen_unsigned={k:v for k,v in frozen.items() if k!="binding_sha256"}
    if frozen_claim!=EXPECTED["candidate_binding_claim"] or _canonical_sha(frozen_unsigned)!=frozen_claim or frozen.get("candidate_generation_observations_before_binding")!=0: raise VerificationError("R96 freeze binding changed")
    bound=binding(root/"results/prospective_length_span_r97/candidate_binding_v1.json",root)
    result=evidence_result(root/"results/prospective_length_span_r97/screen_v1/result.json",EXPECTED["result_file"],EXPECTED["result_evidence"])
    source_result=evidence_result(root/"results/prospective_length_span_r97/source_v1/result.json",EXPECTED["source_result_file"],EXPECTED["source_evidence"])
    raw_path=root/"results/prospective_length_span_r97/screen_v1/evaluation.jsonl"; source_raw_path=root/"results/prospective_length_span_r97/source_v1/prior_corrected_scores.jsonl"; catalog_path=root/"catalogs/prospective_length_span_r97_v1.json"
    require_file(raw_path,EXPECTED["raw"]); require_file(source_raw_path,EXPECTED["source_raw"]); require_file(catalog_path,EXPECTED["catalog"])
    if source_result.get("verdict")!="PASS_R97_SOURCE_CAPTURE" or source_result.get("candidate_accessed") is not False or source_result.get("layercake_loaded") is not False or not all(source_result.get("gates",{}).values()): raise VerificationError("source capture authorization invalid")
    if source_result.get("artifacts",{}).get("prior_corrected_scores",{}).get("sha256")!=EXPECTED["source_raw"]: raise VerificationError("source raw manifest is stale")
    probes=list(load_probe_catalog(catalog_path)["probes"]); rows=_jsonl(raw_path); source_rows={str(row["probe_id"]):row for row in _jsonl(source_raw_path)}; probe_rows={str(row["probe_id"]):row for row in probes}
    if len(rows)!=1400 or len(source_rows)!=1400 or len(probe_rows)!=1400 or len({str(row["probe_id"]) for row in rows})!=1400 or set(source_rows)!=set(probe_rows): raise VerificationError("R97 matrix incomplete")
    tokenizer=AutoTokenizer.from_pretrained(parent,local_files_only=True)
    for row in rows:
        probe_id=str(row["probe_id"]); probe=probe_rows.get(probe_id); source=source_rows.get(probe_id)
        if probe is None or source is None: raise VerificationError("unpaired R97 row")
        prompt=str(probe["prompt"]); evaluator=probe["evaluator"]; prompt_ids=tokenizer.encode(prompt+"\n",add_special_tokens=False); start,end=int(row["candidate_start"]),int(row["candidate_end"])
        if not 0<=start<=end<len(prompt_ids) or end-start>=16: raise VerificationError(f"illegal span: {probe_id}")
        realized=tokenizer.decode(prompt_ids[start:end+1],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip(); candidate_passed,_=evaluate_output(str(row["candidate_output"]),evaluator); parent_passed,_=evaluate_output(str(row["parent_output"]),evaluator); random_passed,_=evaluate_output(str(row["random_output"]),evaluator); collapse=_collapse_metrics(list(row["candidate_token_ids"]),str(row["candidate_output"]),prompt_ids,prompt)
        expected_retained=bool(source["raw_passed"] and candidate_passed)
        if row.get("prompt_sha256")!=hashlib.sha256(prompt.encode()).hexdigest() or source.get("prompt_sha256")!=hashlib.sha256(prompt.encode()).hexdigest() or source.get("expected_output_sha256")!=hashlib.sha256(str(evaluator["value"]).encode()).hexdigest() or bool(row.get("source_passed"))!=bool(source["raw_passed"]) or bool(row.get("candidate_passed"))!=bool(candidate_passed) or bool(row.get("parent_passed"))!=bool(parent_passed) or bool(row.get("random_passed"))!=bool(random_passed) or bool(row.get("source_passing_retained"))!=expected_retained or row.get("candidate_collapse")!=collapse or str(row.get("candidate_output"))!=realized or list(row.get("candidate_token_ids",[]))!=tokenizer.encode(str(row["candidate_output"]),add_special_tokens=False) or row.get("candidate_physical_sparse") is not True or row.get("candidate_model_invocations")!=1 or row.get("candidate_task_cake_invocations")!=1 or row.get("candidate_deep_adapter_invocations")!=6 or row.get("candidate_joint_span_bridge_invocations")!=1: raise VerificationError(f"R97 row stale or unrecomputable: {probe_id}")
    source_passing=sum(bool(row["source_passed"]) for row in rows); candidate_passing=sum(bool(row["candidate_passed"]) for row in rows); parent_passing=sum(bool(row["parent_passed"]) for row in rows); random_passing=sum(bool(row["random_passed"]) for row in rows); retained=sum(bool(row["source_passing_retained"]) for row in rows)
    families={str(i):{"source_passing":sum(row["premise_family"]==i and row["source_passed"] for row in rows),"candidate_passing":sum(row["premise_family"]==i and row["candidate_passed"] for row in rows),"parent_passing":sum(row["premise_family"]==i and row["parent_passed"] for row in rows),"random_passing":sum(row["premise_family"]==i and row["random_passed"] for row in rows)} for i in range(4)}
    metrics=result["metrics"]; recomputed={"rows":len(rows),"source_passing":source_passing,"candidate_passing":candidate_passing,"parent_passing":parent_passing,"random_bridge_passing":random_passing,"source_passing_retained":retained,"source_retention":retained/source_passing,"candidate_collapses":sum(row["candidate_collapse"]["collapse_detected"] for row in rows),"candidate_physical_sparse_rows":sum(row["candidate_physical_sparse"] for row in rows),"by_family":families,"candidate_minus_source":_bootstrap([row["candidate_passed"] for row in rows],[row["source_passed"] for row in rows],97_001),"candidate_minus_parent":_bootstrap([row["candidate_passed"] for row in rows],[row["parent_passed"] for row in rows],97_002),"candidate_minus_random":_bootstrap([row["candidate_passed"] for row in rows],[row["random_passed"] for row in rows],97_003)}
    if any(metrics.get(key)!=value for key,value in recomputed.items()): raise VerificationError("R97 aggregate metrics do not recompute")
    gates={"matrix":len(rows)==1400,"source_quality":source_passing>=0,"candidate_quality":candidate_passing>=1330,"candidate_family_floor":min(v["candidate_passing"] for v in families.values())>=315,"source_retention":retained/source_passing>=0.95,"causal_parent_gain":(candidate_passing-parent_passing)/1400>=0.50,"random_bridge_fails":random_passing<=500,"trained_beats_random":(candidate_passing-random_passing)/1400>=0.50,"zero_collapse":recomputed["candidate_collapses"]==0,"physical_sparse":recomputed["candidate_physical_sparse_rows"]==1400,"teacher_absent":metadata["source_boundary"]["teacher_present_at_inference"] is False,"artifacts_unchanged":all(_sha256_file(root/item["path"])==item["sha256"] for item in bound["files"].values()),"candidate_at_least_source_point":candidate_passing>=source_passing,"candidate_source_noninferior_bootstrap":recomputed["candidate_minus_source"]["ci95_low"]>=-0.02}
    if result.get("gates")!=gates or not all(gates.values()) or result.get("verdict")!="PASS_R97_BOUNDED_PROSPECTIVE_TRANSFER": raise VerificationError("R97 gate vector does not recompute")
    if not all(math.isfinite(float(metrics[key])) for key in ("parent_generation_seconds","candidate_and_random_generation_seconds","candidate_peak_cuda_allocated_bytes","process_rss_bytes")): raise VerificationError("R97 runtime accounting non-finite")
    return {"rows":len(rows),"candidate_passing":candidate_passing,"source_passing":source_passing,"parent_passing":parent_passing,"random_passing":random_passing,"source_retained":retained,"candidate_collapses":recomputed["candidate_collapses"],"sparse_rows":recomputed["candidate_physical_sparse_rows"],"raw_sha256":EXPECTED["raw"],"result_sha256":EXPECTED["result_file"]}


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); args=parser.parse_args()
    if args.output.exists(): parser.error("immutable R97 verification receipt exists")
    metrics=verify(args.root.resolve()); receipt={"format":"abi-r97-strict-raw-recomputation/1","verdict":"PASS_R97_STRICT_RECOMPUTATION","stored_scientific_booleans_trusted":False,"metrics":metrics,"candidate_checkpoint_sha256":EXPECTED["candidate"],"full_abi_moonshot":"OPEN","claim_boundary":"Strict verification of bounded R97 reasoning transfer only."}; receipt["evidence_sha256"]=evidence_hash(receipt); write_json_once(args.output,receipt); print(json.dumps(receipt,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
