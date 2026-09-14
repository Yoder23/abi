"""Capture live frozen source behavior for R97."""

from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from experiments.foreign_capability_r14.core import evidence_hash
import experiments.prompt_identity_reasoning_r81.score_source_corrected_v1 as scorer

CATALOG_SHA256="TO_BE_SEALED"
SCORER_SHA256="54ded3d1e1afbbd016b1d4a9a837777d24bfe313241a4f9f76a4e915b4b9116c"

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--catalog",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--batch-size",default=4,type=int); a=p.parse_args(); implementation=Path(scorer.__file__).resolve()
    if hashlib.sha256(a.catalog.read_bytes()).hexdigest()!=CATALOG_SHA256 or hashlib.sha256(implementation.read_bytes()).hexdigest()!=SCORER_SHA256 or a.output.exists(): p.error("R97 source input, implementation, or output changed")
    scorer.CATALOG_SHA256=CATALOG_SHA256; result=scorer.run(Path.cwd(),a.catalog.resolve(),a.output.resolve(),a.batch_size)
    rows=[json.loads(line) for line in (a.output/"prior_corrected_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]; display=Counter(str(row["destination_display_index"]) for row in rows); metrics=result["metrics"]
    result.update({"campaign":"r97","format":"abi-r97-live-source-capture/1","method":"native_mean_conditional_log_likelihood; neutral_prior_scores_retained_as_diagnostic","authorization_score":"capture_integrity_only","captured_display_position_rows":dict(sorted(display.items())),"source_scoring_implementation_sha256":SCORER_SHA256})
    result["gates"]={"rows":metrics["rows"]==1400,"all_destination_display_positions_captured":set(display)=={"0","1","2"},"zero_native_ties":metrics["raw_ties"]==0,"raw_artifact_complete":len(rows)==1400 and len({row["probe_id"] for row in rows})==1400}; result["verdict"]="PASS_R97_SOURCE_CAPTURE" if all(result["gates"].values()) else "FAIL_R97_SOURCE_CAPTURE"; result.pop("evidence_sha256",None); result["evidence_sha256"]=evidence_hash(result); (a.output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
