"""Read-only V21 teacher-forced split of the failed oracle upper bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3_failure_attribution import _teacher_forced, _teacher_rows
from .capability_compiler_phase3_shared_output import load_candidate, load_protocol as load_v11_protocol
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-oracle-teacher-forced/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise Phase3Error(f"expected object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_NO_PROMOTION" or protocol.get("training_allowed") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("oracle teacher-forced governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"oracle teacher-forced binding changed: {relative}")
    return protocol, sha256_file(path)


def classify(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> str:
    if metrics["token_accuracy"] >= float(thresholds["teacher_forced_token_accuracy_minimum"]) and metrics["mean_nll"] <= float(thresholds["teacher_forced_mean_nll_maximum"]):
        return "AUTONOMOUS_STATE_DYNAMICS_LIMITATION"
    return "BRIDGE_FIT_OPTIMIZATION_OR_EXPRESSIVITY_LIMITATION"


def run(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_path.exists() or not torch.cuda.is_available(): raise Phase3Error("oracle teacher-forced output exists or GPU unavailable")
    v11, _ = load_v11_protocol(root, (root / protocol["v11_protocol"]).resolve()); candidate = (root / protocol["oracle_candidate"]).resolve(); metadata = _json(candidate / "metadata.json")
    if sha256_file(candidate / "model.safetensors") != metadata.get("checkpoint", {}).get("sha256"): raise Phase3Error("oracle candidate checkpoint changed")
    model, tokenizer = load_candidate(root=root, protocol=v11, candidate_dir=candidate, device=torch.device("cuda")); rows = _teacher_rows(root, protocol); metrics, _ = _teacher_forced(model, tokenizer, rows, torch.device("cuda"), int(protocol["execution"]["batch_size"])); attribution = classify(metrics, protocol["thresholds"])
    result = {"format": "abi-capability-compiler-phase3-oracle-teacher-forced-evidence/1", "status": "COMPLETE_DIAGNOSTIC_NO_PROMOTION", "protocol_sha256": protocol_sha, "checkpoint_sha256": metadata["checkpoint"]["sha256"], "metrics": metrics, "attribution": attribution, "development_contaminated": True, "training_performed": False, "promotion_eligible": False, "final_test_accessed": False, "host_representational_ceiling_proven": False}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ORACLE_TEACHER_FORCED_PROTOCOL_V21.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_oracle_fit/teacher_forced_v21.json"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, (root / args.protocol).resolve(), (root / args.output).resolve()); print(json.dumps({"status": result["status"], "attribution": result["attribution"], "metrics": {k: result["metrics"][k] for k in ("mean_nll", "token_accuracy", "tokens")}, "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
