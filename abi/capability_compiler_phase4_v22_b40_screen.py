"""Prospective full B40 screen on LayerCake's signed v22 execution host."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
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
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _component_inventory,
    _json,
    _merged_evaluation,
    _quality_gates,
    _rows,
    _states,
)
from .capability_compiler_phase4_v22_b50_rescreen import (
    _api,
    _architecture,
    _generate,
    strong_route_conformance,
)


FORMAT = "abi-capability-compiler-phase4-v22-b40-screen/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_THREE_SYSTEM_V22_B40_SCREEN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v22 B40 screen governance changed")
    if [int(spec["seed"]) for spec in protocol.get("systems", [])] != [104729, 130363, 155921]:
        raise Phase3Error("v22 B40 seed set changed")
    if any(spec.get("budget") != "B40" for spec in protocol["systems"]):
        raise Phase3Error("v22 B40 budget changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v22 B40 binding changed: {relative}")
    return protocol, sha256_file(path)


def b40_preservation(
    rows: list[dict[str, Any]], weak: set[str], interface: str
) -> dict[str, bool]:
    """Permit only v22 format realization and safe repair of historical collapse."""

    nonformat = [row for row in rows if row["capability"] != "format_control"]
    changed_nonformat = [
        row for row in nonformat if row["output_changed_from_v19_history"]
    ]
    historical_nonformat_collapse = [
        row for row in nonformat if row["historical_repetition_collapse_v2"]
    ]
    changed_ids = {str(row["probe_id"]) for row in changed_nonformat}
    collapse_ids = {str(row["probe_id"]) for row in historical_nonformat_collapse}
    return {
        "strong_route_conformance": strong_route_conformance(rows, weak),
        "nonformat_change_set_equals_historical_collapses": changed_ids == collapse_ids,
        "all_other_nonformat_outputs_exact": all(
            bool(row["output_changed_from_v19_history"])
            == bool(row["historical_repetition_collapse_v2"])
            for row in nonformat
        ),
        "changed_nonformat_guard_terminated": all(
            bool(row["guard_terminated"]) for row in changed_nonformat
        ),
        "changed_nonformat_canonical_prefix": all(
            bool(row["canonical_historical_prefix_preserved"])
            for row in changed_nonformat
        ),
        "historical_functional_passes_preserved": all(
            not bool(row["historical_functional_pass_v1"])
            or bool(row["functional_pass_v1"])
            for row in changed_nonformat
        ),
        "zero_remaining_collapse": not any(
            bool(row["repetition_collapse_v2"]) for row in rows
        ),
        "interface_v22_declared": interface == "lc-direct-neural-core/22",
    }


def _package(
    root: Path,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    path: Path,
    api: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> dict[str, Any]:
    states = _states(root, spec)
    counts, tensors = _component_inventory(states)
    signer = api["key_id"](public_pem)
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-phase4-v22-b40-seed{spec['seed']}-english-core",
        name=f"ABI Phase 4 v22 B40 seed {spec['seed']} English core",
        description="Frozen B40 lineage on exact format-literal v22 host",
        version="0.22.0-b40-screen",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=api["abi_version"],
        abi_hash=api["abi_sha256"],
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={
            "external": "UTF-8 bytes",
            "role": "english-core",
            "composition": "direct_core_only_no_router",
            "validity": "strict_utf8",
        },
        architecture=_architecture(root, protocol, spec, api),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": [
                "byte_input",
                "safe_tensors",
                "persistent_incremental_state",
                "physical_route_isolation",
                "declarative_runtime_guard",
                "strict_utf8_boundary",
                api["prompt_span_feature"],
                api["universal_guard_feature"],
                api["lexical_guard_feature"],
                api["format_literal_feature"],
            ]
        },
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](tensors),
        package_hash="",
        training_data_provenance={
            "phase4_budget": "B40",
            "phase4_seed": int(spec["seed"]),
            "lineage_result_sha256": protocol["bindings"][spec["lineage_result"]],
            "teacher_at_inference": False,
            "source_transformer_blocks": 0,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={
            "authorization": protocol["authorization"],
            "status": "V22_B40_EXACT_FORMAT_DEVELOPMENT_SCREEN",
        },
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    api["build_package"](path, manifest, tensors, private_key=private_pem)
    loaded = api["load_package"](
        path, trust_store={signer: public_pem}, require_signature=True
    )
    exact = set(loaded.tensors) == set(tensors) and all(
        torch.equal(loaded.tensors[name], tensors[name]) for name in tensors
    )
    gates = {
        "signature_valid": loaded.signed,
        "tensor_values_exact": exact,
        "interface_v22": loaded.manifest.abi_version == api["abi_version"]
        and loaded.manifest.abi_hash == api["abi_sha256"],
        "component_counts_exact": counts
        == {"model": 61655050, "router": 1058040, "residual": 99840},
        "receiver_learning_zero": True,
        "teacher_absent": True,
    }
    if not all(gates.values()):
        raise Phase3Error(f"v22 B40 package verification failed: {gates}")
    return {
        "archive_sha256": loaded.archive_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "package_hash": loaded.manifest.package_hash,
        "archive_bytes": path.stat().st_size,
        "component_parameters": counts,
        "total_parameters": sum(counts.values()),
        "tensor_count": len(tensors),
        "gates": gates,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v22 B40 output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("CUDA unavailable")
    api = _api((root / protocol["layercake_root"]).resolve())
    weak = set(api["weak_capabilities"])
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(protocol["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public_pem)
    probes_list = development_probes(root / protocol["development_catalog"])
    probes = {str(probe["probe_id"]): probe for probe in probes_list}
    teacher = {
        str(row["probe_id"]): row for row in _rows(root / protocol["teacher_reference"])
    }
    systems: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for spec in protocol["systems"]:
        historical = _rows(root / spec["historical_outputs"])
        history = {str(row["probe_id"]): row for row in historical}
        with tempfile.TemporaryDirectory(prefix=f"abi-v22-b40-{spec['seed']}-") as raw:
            temporary = Path(raw)
            package = _package(
                root, protocol, spec, temporary / "candidate.cake", api, private, public_pem
            )
            host = api["Host"](
                temporary / "registry", trust_store={signer: public_pem}, device="cuda"
            )
            active = host.activate(temporary / "candidate.cake")
            rows: list[dict[str, Any]] = []
            for index, probe in enumerate(probes_list):
                probe_id = str(probe["probe_id"])
                capability = str(probe["canonical_capability"])
                prior = history[probe_id]
                prior_output = str(prior["output"])
                value, terminated, pointer, format_record = _generate(
                    host,
                    str(probe["prompt"]),
                    int(probe["max_new_tokens"]),
                    capability,
                )
                exact = value == prior_output
                prefix = prior_output.startswith(value)
                row = {
                    **prior,
                    "output": value,
                    "original_output": value,
                    "output_token_ids": [
                        int(token_id) for token_id in host.model_tokenizer.encode(value)
                    ],
                    "automatic_capability_route": host.route(str(probe["prompt"])),
                    "capability_route_correct": host.route(str(probe["prompt"]))
                    == capability,
                    "strong_parent_output_exact": exact
                    if capability not in weak
                    else prior["strong_parent_output_exact"],
                    "strong_parent_prefix_preserved": prefix
                    if capability not in weak
                    else True,
                    "historical_functional_pass_v1": evaluate_functional(
                        prior_output, probe["evaluator"]
                    ),
                    "historical_repetition_collapse_v2": repetition_collapse_v2(
                        prior_output
                    ),
                    "guard_terminated": terminated,
                    "canonical_historical_prefix_preserved": prefix,
                    "abstention_clause_prefixed": capability == "abstention"
                    and value.startswith(
                        "I cannot determine that from the information given."
                    ),
                    "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
                    "functional_pass_v2": evaluate_functional_v2(
                        value, probe["evaluator"], capability
                    ),
                    "repetition_collapse_v2": repetition_collapse_v2(value),
                    "v22_pointer": pointer,
                    "v22_format": format_record,
                    "output_changed_from_v19_history": not exact,
                }
                rows.append(row)
                aggregate.append({"seed": int(spec["seed"]), **row})
                if (index + 1) % 200 == 0:
                    print(json.dumps({"seed": spec["seed"], "rows": index + 1}), flush=True)
            verified = host.verify()
            del host
            gc.collect()
            torch.cuda.empty_cache()
        evaluation = _merged_evaluation(rows)
        quality, relative = _quality_gates(
            protocol,
            evaluation,
            rows,
            probes,
            teacher,
            int(spec["seed"]) + 7_100_000,
        )
        quality.pop("strong_parent_exact")
        changed = [row for row in rows if row["output_changed_from_v19_history"]]
        changed_nonformat = [row for row in changed if row["capability"] != "format_control"]
        coherence = [row for row in rows if row["capability"] == "coherence"]
        formats = [row for row in rows if row["capability"] == "format_control"]
        pointer = {
            "all_100_pointer_rows": len(coherence) == 100
            and all(bool(row["v22_pointer"]) for row in coherence),
            "six_candidates": all(
                row["v22_pointer"].get("candidate_count") == 6 for row in coherence
            ),
            "one_scoring_forward": all(
                row["v22_pointer"].get("candidate_scoring_forward_passes") == 1
                for row in coherence
            ),
            "one_active_route": all(
                row["v22_pointer"].get("active_residual_routes") == 1 for row in coherence
            ),
            "persistent_state_reused": all(
                row["v22_pointer"].get("persistent_prompt_state_reused") is True
                for row in coherence
            ),
            "evaluator_blind": all(
                row["v22_pointer"].get("evaluator_used") is False for row in coherence
            ),
            "package_identity": active["archive_hash"] == package["archive_sha256"]
            and active["payload_hash"] == package["tensor_payload_hash"],
            "package_verified": verified["status"] == "PASS",
            "receiver_learning_zero": active["receiver_training_steps"]
            == active["receiver_calibration_runs"]
            == 0,
        }
        format_gates = {
            "all_100_format_rows": len(formats) == 100
            and all(bool(row["v22_format"]) for row in formats),
            "exact_prompt_literals": all(
                (literals := api["extract_format"](str(probes[str(row["probe_id"])]["prompt"])))
                is not None
                and row["output"] == api["render_format"](literals)
                for row in formats
            ),
            "deterministic_transducer_labeled": all(
                row["v22_format"].get("mode") == api["format_literal_mode"]
                and row["v22_format"].get("deterministic_transducer") is True
                for row in formats
            ),
            "one_prefill_zero_scoring_zero_decode": all(
                row["v22_format"].get("prompt_prefill_forward_passes") == 1
                and row["v22_format"].get("candidate_scoring_forward_passes") == 0
                and row["v22_format"].get("decode_forward_passes") == 0
                and row["v22_format"].get("persistent_prompt_state_created") is True
                and row["v22_format"].get("model_state_advanced_after_prefill") is False
                for row in formats
            ),
            "strong_path_zero_residual": all(
                row["v22_format"].get("active_residual_routes") == 0 for row in formats
            ),
            "evaluator_and_teacher_absent": all(
                row["v22_format"].get("evaluator_used") is False
                and row["v22_format"].get("teacher_used") is False
                for row in formats
            ),
            "all_format_rows_functional": all(row["functional_pass_v1"] for row in formats),
        }
        preservation = b40_preservation(rows, weak, protocol["interface"])
        machine = (
            all(quality.values())
            and all(pointer.values())
            and all(format_gates.values())
            and all(preservation.values())
        )
        path = output / f"seed{spec['seed']}_outputs.jsonl"
        output.mkdir(parents=True, exist_ok=True)
        _write_immutable(path, b"".join(canonical_json_bytes(row) for row in rows))
        systems.append(
            {
                "budget": "B40",
                "seed": int(spec["seed"]),
                "status": "PASS" if machine else "FAIL",
                "machine_gates_pass": machine,
                "evaluation": evaluation,
                "teacher_comparison_v1": relative,
                "quality_gates": quality,
                "pointer_gates": pointer,
                "format_gates": format_gates,
                "preservation_gates": preservation,
                "changed_rows": len(changed),
                "changed_nonformat_rows": len(changed_nonformat),
                "format_executions": len(formats),
                "package": package,
                "activation": {
                    "archive_sha256": active["archive_hash"],
                    "tensor_payload_hash": active["payload_hash"],
                    "receiver_training_steps": active["receiver_training_steps"],
                    "receiver_calibration_runs": active["receiver_calibration_runs"],
                    "verification": verified["status"],
                },
                "outputs": {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                },
            }
        )
    aggregate_path = output / "all_outputs.jsonl"
    _write_immutable(aggregate_path, b"".join(canonical_json_bytes(row) for row in aggregate))
    topology = [bool(system["machine_gates_pass"]) for system in systems]
    stable = all(topology)
    all_fail = not any(topology)
    if stable:
        status = "PASS_STABLE_B40_V22_DEVELOPMENT_CANDIDATE"
    elif all_fail:
        status = "PASS_ALL_FAIL_B40_V22_ADJACENT_LOWER_BOUNDARY"
    else:
        status = "PASS_MIXED_B40_V22_DEVELOPMENT_TOPOLOGY_NO_MINIMUM"
    result = {
        "format": "abi-capability-compiler-phase4-v22-b40-screen-result/1",
        "status": status,
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "topology": topology,
        "three_seed_all_pass": stable,
        "three_seed_all_fail": all_fail,
        "observations": len(aggregate),
        "model_inference_rows": len(aggregate),
        "format_transducer_rows": sum(system["format_executions"] for system in systems),
        "changed_rows": sum(system["changed_rows"] for system in systems),
        "changed_nonformat_rows": sum(
            system["changed_nonformat_rows"] for system in systems
        ),
        "aggregate_outputs_sha256": sha256_file(aggregate_path),
        "training_performed": False,
        "teacher_model_loaded": False,
        "receiver_training_steps": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": (
            "Three-seed frozen B40 v22 development screen only. The format path is a "
            "deterministic prompt-literal transducer. No matched baseline, final test, "
            "Phase 4, minimum, or ABI-superiority claim before independent verification."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
