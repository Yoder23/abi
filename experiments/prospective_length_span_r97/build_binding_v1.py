"""Bind frozen R96 and R97 source evidence before candidate execution."""

from __future__ import annotations
import argparse,json
from pathlib import Path
from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import evidence_hash,write_json_once

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","candidate-binding","catalog","source-result","source-raw","output"): p.add_argument(f"--{name}",required=True,type=Path)
    a=p.parse_args(); root=Path(__file__).resolve().parents[2]; frozen=json.loads(a.candidate_binding.read_text(encoding="utf-8")); source=json.loads(a.source_result.read_text(encoding="utf-8")); unsigned=dict(source); source_claim=unsigned.pop("evidence_sha256",None)
    if frozen.get("candidate_generation_observations_before_binding")!=0 or source.get("verdict")!="PASS_R97_SOURCE_CAPTURE" or source.get("candidate_accessed") is not False or source.get("layercake_loaded") is not False or not all(source.get("gates",{}).values()) or source_claim!=evidence_hash(unsigned): p.error("R96 freeze or R97 source invalid")
    paths={"candidate_checkpoint":a.candidate/"bridge.safetensors","candidate_metadata":a.candidate/"metadata.json","candidate_freeze_binding":a.candidate_binding,"catalog":a.catalog,"source_result":a.source_result,"source_raw":a.source_raw,"candidate_core":root/"experiments/length_invariant_span_r96/core_v1.py","candidate_trainer":root/"experiments/length_invariant_span_r96/train_v1.py","candidate_protocol":root/"experiments/length_invariant_span_r96/PROTOCOL.md","shared_screen":root/"experiments/source_qualified_span_r93/screen_v1.py","screen_wrapper":root/"experiments/prospective_length_span_r97/screen_v1.py","protocol":root/"experiments/prospective_length_span_r97/PROTOCOL.md","builder":root/"experiments/prospective_length_span_r97/build_catalog_v1.py","source_scorer":root/"experiments/prospective_length_span_r97/score_source_v1.py"}
    value={"format":"abi-r97-prospective-candidate-binding/1","candidate_observations_before_binding":0,"candidate_retrained_after_r91_development":True,"candidate_retrained_or_calibrated_after_r97_catalog":False,"source_pass_field":"raw_passed","files":{name:{"path":str(path.resolve().relative_to(root)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":_sha256_file(path)} for name,path in paths.items()},"source_evidence_sha256":source_claim,"full_abi_moonshot":"OPEN"}; value["binding_sha256"]=_canonical_sha(value); write_json_once(a.output,value); print(json.dumps(value,indent=2,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
