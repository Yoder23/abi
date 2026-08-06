"""Measure V41's sealed training-only capability head on held-out prompts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from safetensors.torch import load_file
import torch
from torch import nn

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _load_candidate, _json, load_protocol as load_bpe_protocol


FORMAT = "abi-capability-compiler-phase3-auxiliary-router-diagnostic/1"


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY" or protocol.get("training_allowed") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("V42 governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V42 binding changed: {relative}")
    base_path = (root / protocol["base_protocol"]["path"]).resolve()
    base, base_sha = load_bpe_protocol(root, base_path)
    if base_sha != protocol["base_protocol"]["sha256"]:
        raise Phase3Error("V42 base protocol changed")
    return protocol, base, sha256_file(path)


@torch.inference_mode()
def _predict(model: Any, tokenizer: Any, auxiliary: nn.Linear, prompt: str) -> str:
    ids = [tokenizer.lexeme_to_id[value] for value in tokenizer.split(prompt)]
    source = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    encoded, _ = model.encode(source)
    pooled = encoded.mean(dim=1)
    return CAPABILITIES[int(auxiliary(pooled).argmax(dim=-1).item())]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "observations": len(rows),
        "correct": sum(row["correct"] is True for row in rows),
        "accuracy": sum(row["correct"] is True for row in rows) / len(rows),
        "predicted_counts": dict(sorted(Counter(str(row["predicted"]) for row in rows).items())),
        "per_capability_correct": {
            capability: sum(row["correct"] is True for row in rows if row["capability"] == capability)
            for capability in CAPABILITIES
        },
    }


def execute(root: Path, protocol_path: Path):
    protocol, base, protocol_sha = load_protocol(root, protocol_path)
    candidate_dir = (root / protocol["candidate_dir"]).resolve()
    metadata = _json(candidate_dir / "metadata.json")
    auxiliary_spec = metadata["training_auxiliary"]
    if sha256_file(candidate_dir / "model.safetensors") != protocol["candidate"]["checkpoint_sha256"] or sha256_file(candidate_dir / auxiliary_spec["path"]) != protocol["candidate"]["auxiliary_sha256"]:
        raise Phase3Error("V42 candidate changed")
    model, tokenizer = _load_candidate(root, base, candidate_dir, torch.device("cuda"))
    auxiliary = nn.Linear(int(base["architecture"]["model_width"]), len(CAPABILITIES)).to("cuda")
    auxiliary.load_state_dict(load_file(str(candidate_dir / auxiliary_spec["path"]), device="cuda"), strict=True)
    auxiliary.eval()
    acquisition = load_phase1_ir((root / base["phase1_ir"]).resolve())
    probes = development_probes((root / base["development_catalog"]).resolve())
    header = {}
    rows: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        values = sorted((row for row in acquisition if row["capability"] == capability), key=lambda row: str(row["ir_record_id"]))
        header[capability] = str(values[0]["normalized_acquisition_prompt"]).splitlines()[0]
        for row in values[: int(protocol["diagnostic"]["training_observations_per_capability"])]:
            prompt = str(row["normalized_acquisition_prompt"])
            body = "\n".join(prompt.splitlines()[1:])
            for variant, value in (("training_full", prompt), ("training_body", body)):
                predicted = _predict(model, tokenizer, auxiliary, value)
                rows.append({"scope": "training", "record_id": str(row["ir_record_id"]), "capability": capability, "variant": variant, "predicted": predicted, "correct": predicted == capability})
    for probe in probes:
        capability = str(probe["canonical_capability"])
        prompt = str(probe["prompt"])
        body = "\n".join(prompt.splitlines()[1:])
        variants = (("development_original", prompt), ("development_body", body), ("development_matched_header", header[capability] + "\n" + body))
        for variant, value in variants:
            predicted = _predict(model, tokenizer, auxiliary, value)
            rows.append({"scope": "development", "record_id": str(probe["probe_id"]), "capability": capability, "variant": variant, "predicted": predicted, "correct": predicted == capability})
    summaries = {variant: _summary([row for row in rows if row["variant"] == variant]) for variant in ("training_full", "training_body", "development_original", "development_body", "development_matched_header")}
    result = {
        "format": "abi-capability-compiler-phase3-auxiliary-router-diagnostic-result/1",
        "status": "PASS_READ_ONLY_DIAGNOSTIC",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "auxiliary_sha256": auxiliary_spec["sha256"],
        "summaries": summaries,
        "training_performed": False,
        "checkpoint_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "decision_rule": protocol["decision_rule"],
        "claim_boundary": "Read-only diagnostic of a training-only ABI classifier; not a routed LayerCake artifact or quality result.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result, rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_AUXILIARY_ROUTER_DIAGNOSTIC_PROTOCOL_V42.json")
    parser.add_argument("--result", default="results/abi_capability_compiler_phase3_auxiliary_router_diagnostic/diagnostic_v42.json")
    parser.add_argument("--rows", default="results/abi_capability_compiler_phase3_auxiliary_router_diagnostic/rows_v42.jsonl")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result, rows = execute(root, (root / args.protocol).resolve())
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    result["rows_sha256"] = hashlib.sha256(payload).hexdigest()
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    result_path, rows_path = (root / args.result).resolve(), (root / args.rows).resolve()
    if args.command == "write":
        _write_immutable(rows_path, payload)
        _write_immutable(result_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(result_path) != result or rows_path.read_bytes() != payload:
        raise Phase3Error("stored V42 evidence differs")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
