"""Independent hostile verifier for the immutable Phase 3 V6 result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import PHASE1_IR_SHA256, Phase3Error
from .capability_compiler_phase3_sequence_analysis import analyze
from .capability_compiler_phase3_sequence_bridge import (
    EXPECTED_TRAINABLE_PARAMETERS,
    SYSTEMS,
    _is_bridge_tensor,
    load_protocol,
)


class SequenceVerificationError(RuntimeError):
    """Raised when sealed sequence-successor evidence fails closed."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SequenceVerificationError(f"expected object: {path}")
    return value


def _self_hash(value: dict[str, Any], field: str) -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SequenceVerificationError(message)


def _verify_metadata(
    metadata: dict[str, Any], *, system: str, protocol_sha: str
) -> None:
    _require(metadata.get("manifest_sha256") == _self_hash(metadata, "manifest_sha256"),
             f"{system} manifest self hash")
    _require(metadata.get("system") == system and metadata.get("seed") == 104729,
             f"{system} identity")
    _require(metadata.get("protocol_sha256") == protocol_sha,
             f"{system} protocol identity")
    _require(metadata.get("final_test_accessed") is False,
             f"{system} final-data firewall")
    _require(metadata.get("phase2_human_gate") == "DEFERRED_NOT_PASSED",
             f"{system} Phase 2 boundary")
    source = metadata.get("source", {})
    _require(source.get("phase1_ir_sha256") == PHASE1_IR_SHA256,
             f"{system} IR identity")
    _require(source.get("teacher_present_during_training") is False,
             f"{system} teacher present during training")
    _require(source.get("teacher_present_at_inference") is False,
             f"{system} teacher present at inference")
    _require(source.get("source_parameters_copied") == 0,
             f"{system} copied source parameters")
    _require(source.get("source_blocks_retained") == 0,
             f"{system} retained source blocks")
    bridge = metadata.get("sequence_bridge", {})
    _require(bridge.get("teacher_present_at_training") is False,
             f"{system} bridge teacher presence")
    _require(bridge.get("teacher_present_at_inference") is False,
             f"{system} bridge inference teacher presence")
    _require(bridge.get("source_parameters_copied") == 0,
             f"{system} bridge copied parameters")
    _require(bridge.get("source_blocks_retained") == 0,
             f"{system} bridge retained blocks")
    _require(bridge.get("new_transformer_blocks") == 0,
             f"{system} unregistered transformer block")
    _require(bridge.get("canonical_abi_changed") is False,
             f"{system} canonical ABI mutation")
    training = metadata.get("training", {})
    _require(training.get("steps") == 7000 and training.get("batch_size") == 4,
             f"{system} exposure contract")
    _require(training.get("trainable_parameters") == EXPECTED_TRAINABLE_PARAMETERS,
             f"{system} parameter contract")
    _require(training.get("teacher_response_tokens_seen") == (0 if system == "B3" else (726062 if system == "B2" else 723739)),
             f"{system} response-token accounting")
    _require(sum(training.get("sampled_records_by_capability", {}).values()) == 28000,
             f"{system} record exposure accounting")
    isolation = metadata.get("isolation", {})
    _require(isolation.get("frozen_state_sha256_before") == isolation.get("frozen_state_sha256_after"),
             f"{system} frozen state changed")
    changed = isolation.get("changed_tensors", [])
    _require(bool(changed) and isolation.get("changed_tensor_count") == len(changed),
             f"{system} changed-tensor accounting")
    _require(all(_is_bridge_tensor(str(name)) for name in changed),
             f"{system} unregistered changed tensor")
    control = metadata.get("control", {})
    expected = {
        "B0": (True, False, True, False),
        "B1": (False, False, True, False),
        "B2": (True, True, True, False),
        "B3": (True, False, False, False),
        "B4": (False, False, True, True),
    }[system]
    observed = (
        control.get("uses_destination_labels"),
        control.get("targets_deranged"),
        control.get("teacher_payload_present"),
        control.get("monolithic_route"),
    )
    _require(observed == expected, f"{system} control identity")


def _verify_decision(decision: dict[str, Any], *, protocol_sha: str) -> None:
    _require(decision.get("evidence_sha256") == _self_hash(decision, "evidence_sha256"),
             "decision self hash")
    _require(decision.get("status") == "FAIL_INITIAL_SEED_SEQUENCE_SUCCESSOR",
             "decision status")
    _require(decision.get("protocol", {}).get("sha256") == protocol_sha,
             "decision protocol")
    _require(decision.get("phase3_certified") is False,
             "false Phase 3 certificate")
    _require(decision.get("phase4_status") == "LOCKED", "Phase 4 opened")
    _require(decision.get("final_test_accessed") is False, "final data accessed")
    _require(decision.get("decision", {}).get("branch_promoted") is False,
             "failed branch promoted")
    _require(decision.get("decision", {}).get("remaining_two_seeds_authorized") is False,
             "unauthorized seeds")
    _require(decision.get("negative_evidence_preserved") is True,
             "negative evidence not preserved")
    _require(not all(decision.get("gates", {}).values()), "failure gates rewritten")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol_path = root / "ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_SUCCESSOR_PROTOCOL_V6.json"
    protocol, protocol_sha = load_protocol(root, protocol_path)
    evidence = root / "results/abi_capability_compiler_phase3_sequence"
    sequence_hashes: set[str] = set()
    for system in SYSTEMS:
        candidate = evidence / "development_v6" / f"{system}-seed104729"
        evaluation = evidence / "evaluation_v6" / f"{system}-seed104729"
        metadata_path = candidate / "metadata.json"
        receipt_path = evaluation / "receipt.json"
        outputs_path = evaluation / "development_outputs.jsonl"
        checkpoint_path = candidate / "model.safetensors"
        metadata = _json(metadata_path)
        receipt = _json(receipt_path)
        _verify_metadata(metadata, system=system, protocol_sha=protocol_sha)
        _require(receipt.get("protocol_sha256") == protocol_sha,
                 f"{system} receipt protocol")
        _require(receipt.get("final_test_accessed") is False,
                 f"{system} receipt final-data firewall")
        _require(sha256_file(outputs_path) == receipt.get("outputs_sha256"),
                 f"{system} output hash")
        _require(sha256_file(checkpoint_path) == metadata["checkpoint"]["sha256"],
                 f"{system} checkpoint hash")
        sequence_hashes.add(str(metadata["training"]["successful_record_sequence_sha256"]))
    _require(len(sequence_hashes) == 1, "paired training sequences differ")
    decision_path = evidence / "conditional_decision_v1.json"
    decision = _json(decision_path)
    _verify_decision(decision, protocol_sha=protocol_sha)
    with tempfile.TemporaryDirectory(prefix="abi-phase3-sequence-verify-") as temp:
        recomputed_path = Path(temp) / "decision.json"
        analyze(
            root=root,
            protocol_path=protocol_path,
            evidence_root=evidence,
            output_path=recomputed_path,
        )
        _require(recomputed_path.read_bytes() == decision_path.read_bytes(),
                 "decision recomputation is not byte-identical")
    return {
        "status": "PASS",
        "systems_verified": len(SYSTEMS),
        "checkpoints_rehashed": len(SYSTEMS),
        "decision_sha256": sha256_file(decision_path),
        "evidence_sha256": decision["evidence_sha256"],
        "final_test_accessed": False,
    }


def adversarial(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _, protocol_sha = load_protocol(
        root, root / "ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_SUCCESSOR_PROTOCOL_V6.json"
    )
    metadata = _json(
        root / "results/abi_capability_compiler_phase3_sequence/development_v6/B0-seed104729/metadata.json"
    )
    decision = _json(
        root / "results/abi_capability_compiler_phase3_sequence/conditional_decision_v1.json"
    )
    cases = []
    mutations = (
        ("teacher_presence", lambda value: value["source"].__setitem__("teacher_present_during_training", True)),
        ("copied_parameters", lambda value: value["source"].__setitem__("source_parameters_copied", 1)),
        ("frozen_state", lambda value: value["isolation"].__setitem__("frozen_state_sha256_after", "0" * 64)),
        ("unregistered_tensor", lambda value: value["isolation"]["changed_tensors"].append("transformer.h.0.weight")),
    )
    for name, mutate in mutations:
        value = copy.deepcopy(metadata)
        mutate(value)
        value["isolation"]["changed_tensor_count"] = len(value["isolation"]["changed_tensors"])
        value["manifest_sha256"] = _self_hash(value, "manifest_sha256")
        try:
            _verify_metadata(value, system="B0", protocol_sha=protocol_sha)
        except SequenceVerificationError:
            cases.append(name)
        else:
            raise SequenceVerificationError(f"mutation accepted: {name}")
    for name, mutation in (
        ("false_promotion", ("branch_promoted", True)),
        ("unauthorized_seeds", ("remaining_two_seeds_authorized", True)),
    ):
        value = copy.deepcopy(decision)
        value["decision"][mutation[0]] = mutation[1]
        value["evidence_sha256"] = _self_hash(value, "evidence_sha256")
        try:
            _verify_decision(value, protocol_sha=protocol_sha)
        except SequenceVerificationError:
            cases.append(name)
        else:
            raise SequenceVerificationError(f"mutation accepted: {name}")
    return {"status": "PASS", "mutations_rejected": cases, "count": len(cases)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "adversarial"))
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root) if args.mode == "verify" else adversarial(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
