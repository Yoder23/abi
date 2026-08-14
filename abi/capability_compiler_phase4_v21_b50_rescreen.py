"""Prospective full B50 rescreen on LayerCake's exact lexical-guard v21 host."""

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
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _component_inventory,
    _json,
    _merged_evaluation,
    _quality_gates,
    _rows,
    _states,
)
from .capability_compiler_phase4_v21_b60_rescreen import _api, _architecture, _generate


FORMAT = "abi-capability-compiler-phase4-v21-b50-rescreen/1"


def strong_route_conformance(rows: list[dict[str, Any]], weak: set[str]) -> bool:
    """Require exact strong routes except safe, non-regressing lexical repairs."""

    strong = [row for row in rows if row["capability"] not in weak]
    return bool(strong) and all(
        bool(row["strong_parent_output_exact"])
        or (
            bool(row["guard_terminated"])
            and bool(row["canonical_historical_prefix_preserved"])
            and not bool(row["repetition_collapse_v2"])
            and (
                not bool(row["historical_functional_pass_v1"])
                or bool(row["functional_pass_v1"])
            )
        )
        for row in strong
    )


def lexical_guard_contract(
    rows: list[dict[str, Any]], weak: set[str], interface: str
) -> dict[str, bool]:
    """Evaluate the prospective B50-only change and preservation contract."""

    changed = [row for row in rows if row["output_changed_from_v19_history"]]
    return {
        "strong_route_conformance": strong_route_conformance(rows, weak),
        "changed_rows_were_historical_collapses": all(
            row["historical_repetition_collapse_v2"] for row in changed
        ),
        "historical_noncollapsed_outputs_exact": all(
            not row["output_changed_from_v19_history"]
            for row in rows
            if not row["historical_repetition_collapse_v2"]
        ),
        "changed_rows_guard_terminated": all(
            row["guard_terminated"] for row in changed
        ),
        "changed_rows_canonical_prefixes": all(
            row["canonical_historical_prefix_preserved"] for row in changed
        ),
        "historical_functional_passes_preserved": all(
            not row["historical_functional_pass_v1"] or row["functional_pass_v1"]
            for row in changed
        ),
        "zero_remaining_collapse": not any(
            row["repetition_collapse_v2"] for row in rows
        ),
        "exact_lexical_boundary_declared": interface == "lc-direct-neural-core/21",
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_THREE_SYSTEM_V21_B50_RESCREEN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v21 B50 rescreen governance changed")
    if [int(spec["seed"]) for spec in protocol.get("systems", [])] != [104729, 130363, 155921]:
        raise Phase3Error("v21 B50 seed set changed")
    if any(spec.get("budget") != "B50" for spec in protocol["systems"]):
        raise Phase3Error("v21 B50 budget changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v21 B50 binding changed: {relative}")
    return protocol, sha256_file(path)


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
        cake_id=f"abi-phase4-v21-b50-seed{spec['seed']}-english-core",
        name=f"ABI Phase 4 v21 B50 seed {spec['seed']} English core",
        description="Frozen B50 lineage on exact lexical-guard v21 host",
        version="0.21.0-b50-rescreen",
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
            ]
        },
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](tensors),
        package_hash="",
        training_data_provenance={
            "phase4_budget": "B50",
            "phase4_seed": int(spec["seed"]),
            "lineage_result_sha256": protocol["bindings"][spec["lineage_result"]],
            "teacher_at_inference": False,
            "source_transformer_blocks": 0,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={
            "authorization": protocol["authorization"],
            "status": "V21_B50_EXACT_LEXICAL_DEVELOPMENT_RESCREEN",
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
        "interface_v21": loaded.manifest.abi_version == api["abi_version"]
        and loaded.manifest.abi_hash == api["abi_sha256"],
        "component_counts_exact": counts
        == {"model": 61655050, "router": 1058040, "residual": 99840},
        "receiver_learning_zero": True,
        "teacher_absent": True,
    }
    if not all(gates.values()):
        raise Phase3Error(f"v21 B50 package verification failed: {gates}")
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
        raise Phase3Error(f"immutable v21 B50 output exists: {output}")
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
        str(row["probe_id"]): row
        for row in _rows(root / protocol["teacher_reference"])
    }
    systems: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []

    for spec in protocol["systems"]:
        historical = _rows(root / spec["historical_outputs"])
        history = {str(row["probe_id"]): row for row in historical}
        with tempfile.TemporaryDirectory(
            prefix=f"abi-v21-b50-{spec['seed']}-"
        ) as raw:
            temporary = Path(raw)
            package = _package(
                root,
                protocol,
                spec,
                temporary / "candidate.cake",
                api,
                private,
                public_pem,
            )
            host = api["Host"](
                temporary / "registry",
                trust_store={signer: public_pem},
                device="cuda",
            )
            active = host.activate(temporary / "candidate.cake")
            rows: list[dict[str, Any]] = []
            for index, probe in enumerate(probes_list):
                probe_id = str(probe["probe_id"])
                capability = str(probe["canonical_capability"])
                prior = history[probe_id]
                prior_output = str(prior["output"])
                value, terminated, pointer = _generate(
                    host,
                    str(probe["prompt"]),
                    int(probe["max_new_tokens"]),
                    capability,
                )
                exact = value == prior_output
                canonical_prefix = prior_output.startswith(value)
                historical_functional = evaluate_functional(
                    prior_output, probe["evaluator"]
                )
                historical_collapse = repetition_collapse_v2(prior_output)
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
                    "strong_parent_prefix_preserved": canonical_prefix
                    if capability not in weak
                    else True,
                    "canonical_historical_prefix_preserved": canonical_prefix,
                    "historical_functional_pass_v1": historical_functional,
                    "historical_repetition_collapse_v2": historical_collapse,
                    "guard_terminated": terminated,
                    "abstention_clause_prefixed": capability == "abstention"
                    and value.startswith(
                        "I cannot determine that from the information given."
                    ),
                    "functional_pass_v1": evaluate_functional(
                        value, probe["evaluator"]
                    ),
                    "functional_pass_v2": evaluate_functional_v2(
                        value, probe["evaluator"], capability
                    ),
                    "repetition_collapse_v2": repetition_collapse_v2(value),
                    "v21_pointer": pointer,
                    "output_changed_from_v19_history": not exact,
                }
                rows.append(row)
                aggregate.append({"seed": int(spec["seed"]), **row})
                if (index + 1) % 200 == 0:
                    print(
                        json.dumps({"seed": spec["seed"], "rows": index + 1}),
                        flush=True,
                    )
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
            int(spec["seed"]) + 6_500_000,
        )
        quality.pop("strong_parent_exact")
        changed = [row for row in rows if row["output_changed_from_v19_history"]]
        coherence = [row for row in rows if row["capability"] == "coherence"]
        pointer = {
            "all_100_pointer_rows": len(coherence) == 100
            and all(bool(row["v21_pointer"]) for row in coherence),
            "six_candidates": all(
                row["v21_pointer"].get("candidate_count") == 6 for row in coherence
            ),
            "one_scoring_forward": all(
                row["v21_pointer"].get("candidate_scoring_forward_passes") == 1
                for row in coherence
            ),
            "one_active_route": all(
                row["v21_pointer"].get("active_residual_routes") == 1
                for row in coherence
            ),
            "persistent_state_reused": all(
                row["v21_pointer"].get("persistent_prompt_state_reused") is True
                for row in coherence
            ),
            "evaluator_blind": all(
                row["v21_pointer"].get("evaluator_used") is False
                for row in coherence
            ),
            "package_identity": active["archive_hash"] == package["archive_sha256"]
            and active["payload_hash"] == package["tensor_payload_hash"],
            "package_verified": verified["status"] == "PASS",
            "receiver_learning_zero": active["receiver_training_steps"]
            == active["receiver_calibration_runs"]
            == 0,
        }
        guard = lexical_guard_contract(rows, weak, protocol["interface"])
        machine = all(quality.values()) and all(pointer.values()) and all(guard.values())
        path = output / f"seed{spec['seed']}_outputs.jsonl"
        output.mkdir(parents=True, exist_ok=True)
        _write_immutable(path, b"".join(canonical_json_bytes(row) for row in rows))
        systems.append(
            {
                "budget": "B50",
                "seed": int(spec["seed"]),
                "status": "PASS" if machine else "FAIL",
                "machine_gates_pass": machine,
                "evaluation": evaluation,
                "teacher_comparison_v1": relative,
                "quality_gates": quality,
                "guard_gates": guard,
                "pointer_gates": pointer,
                "historical_collapses": sum(
                    row["historical_repetition_collapse_v2"] for row in rows
                ),
                "changed_rows": len(changed),
                "guard_terminations": sum(row["guard_terminated"] for row in rows),
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
    _write_immutable(
        aggregate_path,
        b"".join(canonical_json_bytes(row) for row in aggregate),
    )
    stable = all(system["machine_gates_pass"] for system in systems)
    result = {
        "format": "abi-capability-compiler-phase4-v21-b50-rescreen-result/1",
        "status": "PASS_STABLE_B50_V21_DEVELOPMENT_CANDIDATE"
        if stable
        else "FAIL_B50_V21_DEVELOPMENT_CANDIDATE",
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "three_seed_all_pass": stable,
        "observations": len(aggregate),
        "model_inference_rows": len(aggregate),
        "historical_collapses": sum(
            system["historical_collapses"] for system in systems
        ),
        "changed_rows": sum(system["changed_rows"] for system in systems),
        "guard_terminations": sum(
            system["guard_terminations"] for system in systems
        ),
        "aggregate_outputs_sha256": sha256_file(aggregate_path),
        "training_performed": False,
        "teacher_model_loaded": False,
        "receiver_training_steps": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": (
            "Three-seed B50 v21 development rescreen only. A stable pass is a "
            "sufficient candidate, not an information minimum. No matched baseline, "
            "final test, Phase 4, or ABI-superiority claim."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
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
