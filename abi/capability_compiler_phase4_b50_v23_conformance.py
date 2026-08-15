"""Repackage exact B50 tensors for LayerCake v23 and prove output conformance."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json, _rows
from .capability_compiler_phase4_v22_b50_rescreen import (
    _api as _api_v22,
    _generate,
    _package as _package_v22,
    load_protocol as _load_v22_protocol,
)


FORMAT = "abi-capability-compiler-phase4-b50-v23-conformance/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v23-conformance-result/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_EXACT_B50_V23_REPACKAGE_AND_CONFORMANCE"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or [int(row["seed"]) for row in protocol.get("systems", [])]
        != [104729, 130363, 155921]
    ):
        raise Phase3Error("exact B50 v23 conformance governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 v23 conformance binding changed: {relative}")
    return protocol, sha256_file(path)


def _api_v23(layercake_root: Path) -> dict[str, Any]:
    api = _api_v22(layercake_root)
    from layercake_extensions.route_isolated_runtime_residency_core_v23 import (
        ARCHITECTURE_V23_FORMAT,
        ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
        ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION,
        SINGLE_PARSE_ACTIVATION_FEATURE,
        RuntimeResidencyFormatLiteralCoreHost,
    )

    return {
        **api,
        "architecture_format": ARCHITECTURE_V23_FORMAT,
        "Host": RuntimeResidencyFormatLiteralCoreHost,
        "abi_sha256": ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
        "abi_version": ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION,
        "single_parse_feature": SINGLE_PARSE_ACTIVATION_FEATURE,
    }


def _v23_manifest_document(
    loaded_v22: Any,
    api: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    document = loaded_v22.manifest.canonical_dict()
    features = list(document["minimum_host_capabilities"]["features"])
    features.append(str(api["single_parse_feature"]))
    document.update(
        {
            "cake_id": f"abi-phase4-v23-b50-seed{seed}-english-core",
            "name": f"ABI Phase 4 v23 B50 seed {seed} English core",
            "description": "Exact v22 B50 tensors on isolated single-parse v23 host",
            "version": "0.23.0-b50-runtime-residency",
            "abi_version": api["abi_version"],
            "abi_hash": api["abi_sha256"],
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "package_hash": "",
            "evaluation_evidence": {
                "status": "V23_EXACT_TENSOR_REPACKAGE_OUTPUT_CONFORMANCE",
                "parent_tensor_payload_hash": loaded_v22.manifest.tensor_payload_hash,
            },
        }
    )
    return document


def _repackage(
    root: Path,
    source_protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    directory: Path,
    api_v22: Mapping[str, Any],
    api_v23: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> tuple[Path, dict[str, Any]]:
    signer = api_v23["key_id"](public_pem)
    v22_path = directory / "parent-v22.cake"
    parent = _package_v22(
        root,
        source_protocol,
        spec,
        v22_path,
        api_v22,
        private,
        public_pem,
    )
    loaded_v22 = api_v22["load_package"](
        v22_path, trust_store={signer: public_pem}, require_signature=True
    )
    document = _v23_manifest_document(
        loaded_v22, api_v23, seed=int(spec["seed"])
    )
    manifest = api_v23["CakeManifest"].from_dict(document)
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    v23_path = directory / "candidate-v23.cake"
    api_v23["build_package"](
        v23_path, manifest, loaded_v22.tensors, private_key=private_pem
    )
    loaded_v23 = api_v23["load_package"](
        v23_path, trust_store={signer: public_pem}, require_signature=True
    )
    tensors_exact = set(loaded_v23.tensors) == set(loaded_v22.tensors) and all(
        torch.equal(loaded_v23.tensors[name], loaded_v22.tensors[name])
        for name in loaded_v22.tensors
    )
    gates = {
        "signature_valid": loaded_v23.signed,
        "v23_interface": loaded_v23.manifest.abi_version == api_v23["abi_version"]
        and loaded_v23.manifest.abi_hash == api_v23["abi_sha256"],
        "tensor_values_exact_to_v22": tensors_exact,
        "tensor_payload_hash_exact_to_v22": loaded_v23.manifest.tensor_payload_hash
        == loaded_v22.manifest.tensor_payload_hash
        == parent["tensor_payload_hash"],
        "single_parse_feature_declared": api_v23["single_parse_feature"]
        in loaded_v23.manifest.minimum_host_capabilities["features"],
    }
    if not all(gates.values()):
        raise Phase3Error(f"exact B50 v23 repackage failed: {gates}")
    return v23_path, {
        "archive_sha256": loaded_v23.archive_hash,
        "archive_bytes": v23_path.stat().st_size,
        "package_hash": loaded_v23.manifest.package_hash,
        "tensor_payload_hash": loaded_v23.manifest.tensor_payload_hash,
        "parent_v22_archive_sha256": parent["archive_sha256"],
        "component_parameters": parent["component_parameters"],
        "total_parameters": parent["total_parameters"],
        "tensor_count": parent["tensor_count"],
        "gates": gates,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 v23 output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("CUDA unavailable for exact B50 v23 conformance")
    source_path = root / str(protocol["source_candidate_protocol"])
    source, _ = _load_v22_protocol(root, source_path)
    api_v22 = _api_v22((root / str(source["layercake_root"])).resolve())
    api_v23 = _api_v23((root / str(source["layercake_root"])).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(source["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api_v23["key_id"](public_pem)
    probes = development_probes(root / str(protocol["development_catalog"]))
    source_specs = {int(row["seed"]): row for row in source["systems"]}
    systems = []
    all_rows = []
    for system in protocol["systems"]:
        seed = int(system["seed"])
        reference_rows = _rows(root / str(system["quality_reference_outputs"]))
        reference = {str(row["probe_id"]): row for row in reference_rows}
        if len(reference) != 1400:
            raise Phase3Error("exact B50 v23 quality reference depth changed")
        with tempfile.TemporaryDirectory(prefix=f"abi-b50-v23-{seed}-") as raw:
            temporary = Path(raw)
            package_path, package = _repackage(
                root,
                source,
                source_specs[seed],
                temporary,
                api_v22,
                api_v23,
                private,
                public_pem,
            )
            if (
                package["tensor_payload_hash"]
                != system["tensor_payload_sha256"]
                or int(package["total_parameters"])
                != int(protocol["expected"]["parameters_per_seed"])
            ):
                raise Phase3Error("exact B50 v23 payload accounting changed")
            host = api_v23["Host"](
                temporary / "registry",
                trust_store={signer: public_pem},
                device="cuda",
            )
            active = host.activate(package_path)
            rows = []
            for index, probe in enumerate(probes):
                probe_id = str(probe["probe_id"])
                capability = str(probe["canonical_capability"])
                value, terminated, pointer, format_record = _generate(
                    host,
                    str(probe["prompt"]),
                    int(probe["max_new_tokens"]),
                    capability,
                )
                token_ids = [
                    int(value) for value in host.model_tokenizer.encode(value)
                ]
                expected = reference[probe_id]
                row = {
                    "probe_id": probe_id,
                    "capability": capability,
                    "output": value,
                    "output_token_ids": token_ids,
                    "reference_output_exact": value == str(expected["output"]),
                    "reference_tokens_exact": token_ids
                    == [int(item) for item in expected["output_token_ids"]],
                    "route_correct": host.route(str(probe["prompt"])) == capability,
                    "guard_terminated": terminated,
                    "pointer": pointer,
                    "format": format_record,
                }
                rows.append(row)
                all_rows.append({"seed": seed, **row})
                if (index + 1) % 200 == 0:
                    print(json.dumps({"seed": seed, "rows": index + 1}), flush=True)
            verified = host.verify()
            del host
            gc.collect()
            torch.cuda.empty_cache()
        coherence = [row for row in rows if row["capability"] == "coherence"]
        formats = [row for row in rows if row["capability"] == "format_control"]
        ordinary = [
            row
            for row in rows
            if row["capability"] not in {"coherence", "format_control"}
        ]
        gates = {
            "all_1400_outputs_exact": all(
                row["reference_output_exact"] for row in rows
            ),
            "all_1400_tokens_exact": all(
                row["reference_tokens_exact"] for row in rows
            ),
            "all_routes_correct": all(row["route_correct"] for row in rows),
            "pointer_physical_execution": len(coherence) == 100
            and all(
                row["pointer"].get("candidate_count") == 6
                and row["pointer"].get("candidate_scoring_forward_passes") == 1
                and row["pointer"].get("active_residual_routes") == 1
                and row["pointer"].get("persistent_prompt_state_reused") is True
                and row["pointer"].get("evaluator_used") is False
                for row in coherence
            ),
            "format_physical_execution": len(formats) == 100
            and all(
                row["format"].get("deterministic_transducer") is True
                and row["format"].get("prompt_prefill_forward_passes") == 1
                and row["format"].get("candidate_scoring_forward_passes") == 0
                and row["format"].get("decode_forward_passes") == 0
                for row in formats
            ),
            "ordinary_depth": len(ordinary) == 1200,
            "one_authenticated_parse": active["authenticated_package_parses"] == 1,
            "receiver_learning_zero": active["receiver_training_steps"]
            == active["receiver_calibration_runs"]
            == 0,
            "package_identity": active["archive_hash"] == package["archive_sha256"]
            and active["payload_hash"] == package["tensor_payload_hash"],
            "package_verifies": verified["status"] == "PASS",
        }
        path = output / f"seed{seed}_outputs.jsonl"
        output.mkdir(parents=True, exist_ok=True)
        _write_immutable(path, b"".join(canonical_json_bytes(row) for row in rows))
        systems.append(
            {
                "seed": seed,
                "status": "PASS" if all(gates.values()) else "FAIL",
                "gates": gates,
                "package": package,
                "activation": active,
                "outputs": {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                },
            }
        )
    aggregate = output / "all_outputs.jsonl"
    _write_immutable(
        aggregate, b"".join(canonical_json_bytes(row) for row in all_rows)
    )
    passed = all(system["status"] == "PASS" for system in systems) and len(
        all_rows
    ) == int(protocol["expected"]["observations"])
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_EXACT_B50_V23_THREE_SEED_OUTPUT_CONFORMANCE"
        if passed
        else "FAIL_EXACT_B50_V23_OUTPUT_CONFORMANCE",
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "observations": len(all_rows),
        "aggregate_outputs_sha256": sha256_file(aggregate),
        "training_performed": False,
        "teacher_model_loaded": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Exact three-seed v22-to-v23 tensor and development-output conformance only. No new training, teacher query, final test, runtime, Phase 4, or ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
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
