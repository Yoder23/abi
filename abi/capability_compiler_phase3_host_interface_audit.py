"""Read-only Phase 3 audit of teacher-token causality versus LayerCake hosting."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from tokenizers import Tokenizer


class HostInterfaceAuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_concatenative_v3_compatible(document: Mapping[str, Any]) -> bool:
    model = document.get("model", {})
    return (
        document.get("normalizer") is None
        and document.get("pre_tokenizer") is None
        and document.get("post_processor") is None
        and document.get("decoder") is None
        and model.get("type") == "BPE"
        and model.get("dropout") is None
        and model.get("unk_token") == "[UNK]"
        and model.get("continuing_subword_prefix") is None
        and model.get("end_of_word_suffix") is None
        and model.get("byte_fallback") is False
        and model.get("ignore_merges") is False
    )


def portable_plan_parameter_count(fixed_vocab_size: int) -> int:
    """Exact count for the frozen V50/V70 PortableTokenPlan topology."""
    if fixed_vocab_size <= 4:
        raise ValueError("fixed vocabulary is too small")
    vocabulary_independent = 2_249_665
    per_fixed_action = 385  # 192 embedding + 192 output weights + bias.
    return vocabulary_independent + fixed_vocab_size * per_fixed_action


def _load_ir(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise HostInterfaceAuditError("Phase 1 IR identity changed")
    with zipfile.ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines()]
    if len(rows) != 7000:
        raise HostInterfaceAuditError("Phase 1 record depth changed")
    counts = Counter(str(row.get("capability")) for row in rows)
    if len(counts) != 14 or set(counts.values()) != {500}:
        raise HostInterfaceAuditError("Phase 1 capability balance changed")
    return rows


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_READ_ONLY_AUDIT":
        raise HostInterfaceAuditError("audit protocol is not preregistered")
    if protocol.get("neural_training_authorized") is not False:
        raise HostInterfaceAuditError("audit unexpectedly authorizes training")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise HostInterfaceAuditError(f"bound input changed: {relative}")

    source = protocol["source"]
    tokenizer_json = Path(source["tokenizer_json"])
    document = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    rows = _load_ir((root / protocol["phase1_ir"]).resolve(), protocol["bindings"][protocol["phase1_ir"]])

    exact_roundtrips = 0
    raw_piece_concatenations = 0
    observations = 0
    input_tokens = 0
    output_tokens = 0
    maximum_input_tokens = 0
    maximum_output_tokens = 0
    for row in rows:
        for field, is_output in (("normalized_acquisition_prompt", False), ("normalized_output", True)):
            value = str(row[field])
            encoded = tokenizer.encode(value, add_special_tokens=False)
            decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
            observations += 1
            exact_roundtrips += int(decoded == value)
            raw_piece_concatenations += int("".join(encoded.tokens) == value)
            if is_output:
                output_tokens += len(encoded.ids)
                maximum_output_tokens = max(maximum_output_tokens, len(encoded.ids))
            else:
                input_tokens += len(encoded.ids)
                maximum_input_tokens = max(maximum_input_tokens, len(encoded.ids))

    model_vocab = len(document["model"]["vocab"])
    added_tokens = len(document.get("added_tokens", []))
    teacher_actions = max(
        model_vocab,
        max((int(token["id"]) + 1 for token in document.get("added_tokens", [])), default=0),
    )
    host_fixed_actions = teacher_actions + 4  # Preserve LayerCake PAD/BOS/EOS/UNK semantics.
    current_parameters = portable_plan_parameter_count(4999)
    projected_parameters = portable_plan_parameter_count(host_fixed_actions)

    decision_path = (root / protocol["layercake_host"]["construct_decision"]).resolve()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    interface_source = (root / protocol["layercake_host"]["implementation"]).resolve().read_text(encoding="utf-8")
    rejection_rule_present = all(
        fragment in interface_source
        for fragment in (
            'document.get("normalizer") is not None',
            'document.get("post_processor") is not None',
            'document.get("decoder") is not None',
            'raise ValueError("BPE tokenizer graph must be raw concatenative")',
        )
    )
    compatible = raw_concatenative_v3_compatible(document)
    exact = exact_roundtrips == observations
    construct_only = decision.get("status") == "PASS_CONSTRUCT_ONLY" and decision.get("external_artifact_used") is False
    blocked = exact and not compatible and rejection_rule_present and construct_only
    return {
        "format": "abi-capability-compiler-phase3-host-interface-audit/1",
        "status": "BLOCKED_EXTERNAL_LAYERCAKE_HOST_INTERFACE" if blocked else "AUDIT_DID_NOT_ESTABLISH_REGISTERED_BLOCKER",
        "protocol_sha256": sha256_file(protocol_path),
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "dataset": {"records": len(rows), "capabilities": len(Counter(str(row["capability"]) for row in rows)), "text_observations": observations},
        "teacher_tokenizer": {
            "model_vocab": model_vocab,
            "added_tokens": added_tokens,
            "teacher_action_ids": teacher_actions,
            "normalizer_type": None if document.get("normalizer") is None else document["normalizer"].get("type"),
            "post_processor_type": None if document.get("post_processor") is None else document["post_processor"].get("type"),
            "decoder_type": None if document.get("decoder") is None else document["decoder"].get("type"),
            "byte_fallback": document["model"].get("byte_fallback"),
            "exact_encode_decode_roundtrips": exact_roundtrips,
            "raw_piece_concatenation_matches": raw_piece_concatenations,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "maximum_input_tokens": maximum_input_tokens,
            "maximum_output_tokens": maximum_output_tokens
        },
        "causal_alignment": {"boundary_straddles_if_teacher_actions_are_native": 0, "exact_teacher_action_causality_available": exact},
        "portable_plan_projection": {
            "current_fixed_actions": 4999,
            "current_parameters": current_parameters,
            "teacher_native_fixed_actions_including_host_specials": host_fixed_actions,
            "projected_parameters": projected_parameters,
            "parameter_ratio": projected_parameters / current_parameters
        },
        "layercake_v3": {
            "interface": decision.get("interface"),
            "construct_status": decision.get("status"),
            "external_artifact_used": decision.get("external_artifact_used"),
            "teacher_tokenizer_raw_concatenative_compatible": compatible,
            "machine_rejection_rule_present": rejection_rule_present,
            "qualified_english_artifact_available": False
        },
        "ownership": {
            "abi": "Teacher-token causality is exactly representable and fully accounted; V70 proves the current retokenized bridge is insufficient.",
            "layercake": "A separately governed canonical host interface accepting decoder-bearing external tokenizers, or an independently qualified English host with another causal ABI, is required before this route can be tested fairly."
        },
        "phase3_certified": False,
        "phase4_open": False,
        "superiority_claim_allowed": False,
        "next_gate": "LayerCake must preregister and construct-certify a decoder-aware external-core interface; ABI may then preregister exactly one teacher-token-native candidate against that frozen interface."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_HOST_INTERFACE_AUDIT_PROTOCOL_V72.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_host_interface/audit_v72.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise HostInterfaceAuditError("audit output already exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
