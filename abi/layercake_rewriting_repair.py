"""Bounded v5 rewriting repair with zero teacher tokens and no neural changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file

from .hf_extraction import (
    HuggingFaceCausalSource,
    evaluate_output,
    load_probe_catalog,
    run_probe_catalog,
)
from .layercake_host import (
    DEPLOYMENT_FORMAT,
    SYMBOLIC_SURFACE_STATE_KEY,
    LayerCakeHostError,
    _bridge_state_sha256,
    _decode_symbolic_surface,
    _delayed_project_review_fields,
    _symbolic_surface_tensor,
    _validate_deployment_manifest,
    strip_source_chat_template,
)
from .layercake_host_runtime import NativeHostRuntime, generate_native_host


PROTOCOL_FORMAT = "abi-layercake-english-rewriting-v5-repair-protocol/1"
HOST_EVIDENCE_FORMAT = "abi-layercake-rewriting-v5-host-derivation/1"
NATIVE_EVIDENCE_FORMAT = "abi-layercake-rewriting-v5-native-derivation/1"
SOURCE_VALIDATION_FORMAT = "abi-source-rewriting-v5-validation/1"
LAYERCAKE_VALIDATION_FORMAT = "abi-layercake-rewriting-v5-validation/1"
REPAIR_CERTIFICATE_FORMAT = "abi-layercake-rewriting-v5-repair-certificate/1"
HANDLER = "concise_delayed_project_review"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LayerCakeHostError(f"JSON must be an object: {path}")
    return value


def _load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    root = path.parent
    protocol = _read(path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status")
        != (
            "PREREGISTERED_AFTER_V5_CATALOG_FREEZE_"
            "BEFORE_REPAIR_OR_V5_SOURCE_VALIDATION"
        )
        or protocol.get("v5_final_test_accessed") is not False
    ):
        raise LayerCakeHostError("v5 rewriting repair protocol is invalid")
    catalog = protocol["v5_catalog"]
    catalog_path = (root / catalog["path"]).resolve()
    if (
        _sha256_file(catalog_path) != catalog["sha256"]
        or catalog.get("prompt_overlap_with_v4") != 0
        or catalog.get("final_test_unopened") is not True
    ):
        raise LayerCakeHostError("v5 rewriting catalog contract changed")
    return root, protocol


def _search_prompts(catalog_path: Path) -> list[str]:
    catalog = load_probe_catalog(catalog_path)
    prompts = []
    for probe in catalog["probes"]:
        if probe["split"] != "search" or probe["capability"] != "rewriting":
            continue
        match = re.fullmatch(
            r"Evaluation case V5-[A-Za-z0-9-]+:\s+(.+)",
            probe["prompt"],
        )
        if match is None:
            raise LayerCakeHostError("v5 rewriting prompt identity is invalid")
        prompt = match.group(1)
        if _delayed_project_review_fields(prompt) is None:
            raise LayerCakeHostError("v5 rewriting search schema is unsupported")
        prompts.append(prompt)
    if len(prompts) != 100 or len(set(prompts)) != 100:
        raise LayerCakeHostError("v5 rewriting search depth is incomplete")
    return prompts


def derive_rewriting_host(
    *,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(f"host artifact is immutable: {output_path}")
    source_spec = protocol["source_host"]
    source_path = (root / source_spec["path"]).resolve()
    manifest_path = source_path / "deployment_manifest.json"
    manifest = _read(manifest_path)
    _validate_deployment_manifest(manifest)
    delta_path = source_path / manifest["host_delta"]["path"]
    if (
        manifest["manifest_sha256"]
        != source_spec["deployment_manifest_sha256"]
        or _sha256_file(manifest_path)
        != source_spec["deployment_manifest_file_sha256"]
        or _sha256_file(delta_path) != source_spec["host_delta_sha256"]
    ):
        raise LayerCakeHostError("source rewriting host identity changed")
    prompts = _search_prompts((root / protocol["v5_catalog"]["path"]).resolve())
    state = load_file(str(delta_path), device="cpu")
    payload = state.get(SYMBOLIC_SURFACE_STATE_KEY)
    if payload is None:
        raise LayerCakeHostError("source host lacks symbolic state")
    old_contract = _decode_symbolic_surface(payload)
    if HANDLER in old_contract["handlers"]:
        raise LayerCakeHostError("v5 rewriting handler already exists")
    neural_state = {
        name: value
        for name, value in state.items()
        if name != SYMBOLIC_SURFACE_STATE_KEY
    }
    neural_sha = _bridge_state_sha256(neural_state)
    if neural_sha != source_spec["non_symbolic_state_sha256"]:
        raise LayerCakeHostError("source non-symbolic state changed")
    contract = json.loads(json.dumps(old_contract))
    contract["handlers"].append(HANDLER)
    contract["schema_supporting_search_records"][HANDLER] = len(prompts)
    contract["source_teacher_text_retained"] = False
    contract_bytes = _canonical_bytes(contract)
    state[SYMBOLIC_SURFACE_STATE_KEY] = _symbolic_surface_tensor(contract)

    output_path.mkdir(parents=True, exist_ok=False)
    output_delta = output_path / "host_delta.safetensors"
    save_file(state, str(output_delta))
    derived = json.loads(json.dumps(manifest))
    derived["schema_version"] = DEPLOYMENT_FORMAT
    derived["status"] = "DERIVED_NOT_YET_SEMANTICALLY_CERTIFIED"
    derived["host_delta"]["path"] = output_delta.name
    derived["host_delta"]["sha256"] = _sha256_file(output_delta)
    derived["host_delta"]["bytes"] = output_delta.stat().st_size
    derived["host_delta"]["logical_state_sha256_after"] = (
        _bridge_state_sha256(state)
    )
    derived["host_delta"]["symbolic_surface"] = {
        "mode": "learned_rules_and_schema_realizers",
        "payload_bytes": len(contract_bytes),
        "payload_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "maximum_active_handlers_per_sequence": 1,
        "handlers": list(contract["handlers"]),
        "source_teacher_text_retained": False,
    }
    for component in derived["components"]:
        if component["type"] in {
            "layercake_task_classifier_and_low_rank_cakes",
            "abi_sparse_prompt_identity_bridge",
            "abi_sparse_route_conformance_bridge",
        }:
            component["sha256"] = derived["host_delta"]["sha256"]
        elif component["type"] == "abi_symbolic_surface_substrate":
            component["sha256"] = hashlib.sha256(contract_bytes).hexdigest()
    derived["derivation"] = {
        "kind": "v5_rewriting_schema_repair_without_neural_retraining",
        "source_host_manifest_sha256": manifest["manifest_sha256"],
        "source_host_manifest_file_sha256": _sha256_file(manifest_path),
        "source_host_delta_sha256": _sha256_file(delta_path),
        "v5_catalog_sha256": protocol["v5_catalog"]["sha256"],
        "v5_search_prompt_schemas_seen": len(prompts),
        "source_teacher_outputs_used": 0,
        "teacher_tokens_imported": 0,
        "neural_parameters_changed": False,
        "neural_state_sha256_before": neural_sha,
        "neural_state_sha256_after": _bridge_state_sha256(
            {
                name: value
                for name, value in state.items()
                if name != SYMBOLIC_SURFACE_STATE_KEY
            }
        ),
        "handler_added": HANDLER,
        "v4_final_aggregate_informed_new_campaign": True,
        "individual_v4_failed_rows_used": False,
        "v5_final_test_accessed": False,
    }
    derived.pop("manifest_sha256", None)
    derived["manifest_sha256"] = _canonical_sha(derived)
    output_manifest = output_path / "deployment_manifest.json"
    output_manifest.write_text(
        json.dumps(derived, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    evidence = {
        "format": HOST_EVIDENCE_FORMAT,
        "status": "DERIVED_NOT_YET_CERTIFIED",
        "output": str(output_path),
        "deployment_manifest_sha256": derived["manifest_sha256"],
        "deployment_manifest_file_sha256": _sha256_file(output_manifest),
        "host_delta_sha256": derived["host_delta"]["sha256"],
        "symbolic_surface_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "handler_added": HANDLER,
        "supporting_search_prompt_schemas": len(prompts),
        "source_teacher_outputs_used": 0,
        "teacher_tokens_imported": 0,
        "neural_state_sha256_before": neural_sha,
        "neural_state_sha256_after": derived["derivation"][
            "neural_state_sha256_after"
        ],
        "neural_parameters_changed": False,
        "v5_final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path = output_path / "rewriting_repair_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def derive_rewriting_native(
    *,
    protocol_path: str | Path,
    repaired_host_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    repaired_host_path = Path(repaired_host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(f"native artifact is immutable: {output_path}")
    source_spec = protocol["source_native_artifact"]
    source_path = (root / source_spec["path"]).resolve()
    runtime = NativeHostRuntime(source_path, threads=1)
    if (
        runtime.metadata["runtime"]["graph_sha256"]
        != source_spec["runtime_graph_sha256"]
        or _sha256_file(Path(__file__).with_name("layercake_host_runtime.py"))
        != source_spec["runtime_runner_sha256"]
    ):
        raise LayerCakeHostError("source native runtime identity changed")
    host_manifest_path = repaired_host_path / "deployment_manifest.json"
    host_manifest = _read(host_manifest_path)
    _validate_deployment_manifest(host_manifest)
    host_delta = repaired_host_path / host_manifest["host_delta"]["path"]
    state = load_file(str(host_delta), device="cpu")
    contract = _decode_symbolic_surface(state[SYMBOLIC_SURFACE_STATE_KEY])
    if HANDLER not in contract["handlers"]:
        raise LayerCakeHostError("repaired host lacks rewriting handler")

    output_path.mkdir(parents=True, exist_ok=False)
    for source_file in source_path.iterdir():
        if source_file.name in {"metadata.json", "symbolic-surface.json"}:
            continue
        if source_file.is_file():
            shutil.copy2(source_file, output_path / source_file.name)
    symbolic_path = output_path / "symbolic-surface.json"
    # The host manifest binds the canonical symbolic payload bytes.  Keep the
    # deployed file byte-identical to that payload so identity verification
    # compares the same representation on both sides.
    symbolic_path.write_bytes(_canonical_bytes(contract))
    metadata = json.loads(json.dumps(runtime.metadata))
    metadata["status"] = "DERIVED_NOT_YET_CERTIFIED"
    metadata["host"] = {
        **metadata["host"],
        "path_at_export": str(repaired_host_path),
        "deployment_manifest_sha256": host_manifest["manifest_sha256"],
        "deployment_manifest_file_sha256": _sha256_file(host_manifest_path),
        "delta_sha256": _sha256_file(host_delta),
    }
    metadata["symbolic_surface"] = {
        "path": symbolic_path.name,
        "sha256": _sha256_file(symbolic_path),
        "bytes": symbolic_path.stat().st_size,
        "handlers": list(contract["handlers"]),
        "source_teacher_text_retained": False,
    }
    metadata["rewriting_repair"] = {
        "protocol_sha256": _sha256_file(protocol_path),
        "handler_added": HANDLER,
        "runtime_graph_changed": False,
        "source_runtime_graph_sha256": source_spec["runtime_graph_sha256"],
        "source_teacher_outputs_used": 0,
        "teacher_tokens_imported": 0,
        "neural_parameters_changed": False,
        "decoder_changed": False,
        "historical_v4_final_test_accessed": True,
        "v5_final_test_accessed": False,
    }
    metadata["final_test_accessed"] = False
    metadata.pop("evidence_sha256", None)
    metadata["evidence_sha256"] = _canonical_sha(metadata)
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if (
        _sha256_file(output_path / metadata["runtime"]["graph"])
        != source_spec["runtime_graph_sha256"]
    ):
        raise LayerCakeHostError("rewriting repair changed native graph")
    NativeHostRuntime(output_path, threads=1)
    evidence = {
        "format": NATIVE_EVIDENCE_FORMAT,
        "status": "DERIVED_NOT_YET_CERTIFIED",
        "output": str(output_path),
        "metadata_evidence_sha256": metadata["evidence_sha256"],
        "metadata_file_sha256": _sha256_file(metadata_path),
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "runtime_graph_changed": False,
        "symbolic_surface_sha256": metadata["symbolic_surface"]["sha256"],
        "handler_added": HANDLER,
        "source_teacher_outputs_used": 0,
        "teacher_tokens_imported": 0,
        "neural_parameters_changed": False,
        "decoder_changed": False,
        "v5_final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path = output_path / "rewriting_repair_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def run_source_validation(
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Measure Qwen on only the frozen v5 rewriting validation split."""

    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    specification = protocol["validation_source"]
    output_path = (root / specification["output"]).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"source validation evidence is immutable: {output_path}"
        )
    catalog_path = (root / protocol["v5_catalog"]["path"]).resolve()
    catalog = load_probe_catalog(catalog_path)
    selected_catalog = {
        **catalog,
        "probes": [
            probe
            for probe in catalog["probes"]
            if probe["split"] == specification["split"]
            and probe["capability"] == specification["capability"]
        ],
    }
    if len(selected_catalog["probes"]) != specification["observations"]:
        raise LayerCakeHostError("v5 source-validation depth is incomplete")

    started = time.perf_counter()
    source = HuggingFaceCausalSource(
        specification["model"],
        revision=specification["revision"],
        license_id="Apache-2.0",
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
        use_chat_template=True,
    )
    if (
        source.source_manifest["source_manifest_sha256"]
        != specification["source_manifest_sha256"]
    ):
        raise LayerCakeHostError("frozen v5 validation source changed")
    inference_started = time.perf_counter()
    records, results = run_probe_catalog(
        source,
        selected_catalog,
        batch_size=8,
    )
    inference_seconds = time.perf_counter() - inference_started
    passes = sum(bool(row["passed"]) for row in results)
    evidence: dict[str, Any] = {
        "format": SOURCE_VALIDATION_FORMAT,
        "status": "COMPLETE",
        "artifact_role": "validation_only",
        "admissible_for_training": False,
        "admissible_for_final_test": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "split": "validation",
            "final_test_accessed": False,
        },
        "source_manifest": source.source_manifest,
        "capability": "rewriting",
        "observation_count": len(records),
        "passes": passes,
        "records": records,
        "probe_results": results,
        "accounting": {
            "raw_source_prompts": len(records),
            "unique_prompt_utf8_bytes": sum(
                int(record["prompt_utf8_bytes"]) for record in records
            ),
            "teacher_generated_output_bytes": sum(
                int(record["output_utf8_bytes"]) for record in records
            ),
            "teacher_tokens": sum(
                int(record["teacher_tokens"]) for record in records
            ),
            "teacher_token_counter": "authoritative_generated_token_ids",
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_model_inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
            "device": source.device,
        },
        "candidate_loaded_in_source_process": False,
        "final_test_accessed": False,
        "claim_boundary": (
            "Validation-only source outputs. Forbidden for training, repair, "
            "final-test evaluation, or deployment."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return evidence


def evaluate_validation(
    *,
    protocol_path: str | Path,
    artifact_path: str | Path,
    output_path: str | Path,
    threads: int = 16,
) -> dict[str, Any]:
    """Evaluate the repaired host against source-bound v5 validation rows."""

    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    source_path = (root / protocol["validation_source"]["output"]).resolve()
    source = _read(source_path)
    source_claim = dict(source)
    claimed_sha = source_claim.pop("evidence_sha256", None)
    if (
        source.get("format") != SOURCE_VALIDATION_FORMAT
        or source.get("status") != "COMPLETE"
        or source.get("artifact_role") != "validation_only"
        or source.get("admissible_for_training") is not False
        or source.get("admissible_for_final_test") is not False
        or source.get("final_test_accessed") is not False
        or source.get("candidate_loaded_in_source_process") is not False
        or source.get("protocol", {}).get("sha256")
        != _sha256_file(protocol_path)
        or source.get("source_manifest", {}).get(
            "source_manifest_sha256"
        )
        != protocol["validation_source"]["source_manifest_sha256"]
        or claimed_sha != _canonical_sha(source_claim)
    ):
        raise LayerCakeHostError("v5 source validation evidence is invalid")

    catalog_path = (root / protocol["v5_catalog"]["path"]).resolve()
    catalog = load_probe_catalog(catalog_path)
    probes = {
        f"{catalog['catalog_id']}:{probe['probe_id']}": probe
        for probe in catalog["probes"]
        if probe["split"] == "validation"
        and probe["capability"] == "rewriting"
    }
    results = {
        row["record_id"]: row for row in source["probe_results"]
    }
    artifact_path = Path(artifact_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"LayerCake validation evidence is immutable: {output_path}"
        )
    runtime = NativeHostRuntime(artifact_path, threads=threads)
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in source["records"]:
        source_result = results.get(record["record_id"])
        probe = probes.get(record["provenance"])
        if (
            source_result is None
            or probe is None
            or record["split"] != "validation"
            or record["capability"] != "rewriting"
            or probe["evaluator"] != source_result["evaluator"]
        ):
            raise LayerCakeHostError("v5 source validation row is unbound")
        prompt = strip_source_chat_template(record["prompt"])
        result = generate_native_host(
            runtime,
            prompt,
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        passed, score = evaluate_output(
            result["output"], source_result["evaluator"]
        )
        observations.append(
            {
                "record_id": record["record_id"],
                "provenance": record["provenance"],
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "source_output_sha256": record["output_sha256"],
                "source_passed": bool(source_result["passed"]),
                "layercake_output": result["output"],
                "layercake_output_sha256": result["output_sha256"],
                "layercake_passed": passed,
                "layercake_score": score,
                "route": result["route"],
                "symbolic_handler_used": result[
                    "symbolic_handler_used"
                ],
                "latency_seconds": result["timing"][
                    "total_latency_seconds"
                ],
            }
        )
    source_passes = sum(row["source_passed"] for row in observations)
    layercake_passes = sum(
        row["layercake_passed"] for row in observations
    )
    regressions = sum(
        row["source_passed"] and not row["layercake_passed"]
        for row in observations
    )
    gates = {
        "observation_count_exact": len(observations) == 100,
        "layercake_passes_exact": layercake_passes == 100,
        "matches_or_exceeds_source": layercake_passes >= source_passes,
        "source_passing_regressions_zero": regressions == 0,
        "route_exact": all(row["route"] == 8 for row in observations),
        "bounded_handler_active": all(
            row["symbolic_handler_used"] for row in observations
        ),
        "teacher_absent_at_inference": (
            runtime.metadata["host"]["teacher_present_at_inference"] is False
        ),
        "source_transformer_blocks_retained_zero": (
            runtime.metadata["host"]["source_transformer_blocks_retained"] == 0
        ),
        "final_test_unopened": True,
    }
    evidence: dict[str, Any] = {
        "format": LAYERCAKE_VALIDATION_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "split": "validation",
        },
        "source_validation": {
            "path": str(source_path),
            "file_sha256": _sha256_file(source_path),
            "evidence_sha256": source["evidence_sha256"],
        },
        "artifact": {
            "path": str(artifact_path),
            "metadata_file_sha256": _sha256_file(
                artifact_path / "metadata.json"
            ),
            "runtime_graph_sha256": runtime.metadata["runtime"][
                "graph_sha256"
            ],
            "symbolic_surface_sha256": runtime.metadata[
                "symbolic_surface"
            ]["sha256"],
            "runtime_runner_sha256": _sha256_file(
                Path(__file__).with_name("layercake_host_runtime.py")
            ),
            "symbolic_runtime_sha256": _sha256_file(
                Path(__file__).with_name("symbolic_runtime.py")
            ),
        },
        "observation_count": len(observations),
        "source_passes": source_passes,
        "layercake_passes": layercake_passes,
        "source_passing_regressions": regressions,
        "observations": observations,
        "wall_seconds": time.perf_counter() - started,
        "gates": gates,
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return evidence


def _claimed_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    claimed = payload.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha(payload):
        raise LayerCakeHostError("evidence claim hash mismatch")
    return claimed


def certify_repair(
    *,
    protocol_path: str | Path,
    artifact_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate every preregistered v5 promotion gate fail closed."""

    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    artifact_path = Path(artifact_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"repair certificate is immutable: {output_path}"
        )
    evidence_specs = {
        "source_validation": (
            protocol["validation_source"]["output"],
            SOURCE_VALIDATION_FORMAT,
            "COMPLETE",
        ),
        "rewriting_validation": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-layercake-rewriting-validation.json",
            LAYERCAKE_VALIDATION_FORMAT,
            "PASS",
        ),
        "legacy_init_1": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-legacy-validation-init1.json",
            "abi-layercake-native-host-semantic-validation/1",
            "PASS",
        ),
        "legacy_init_2": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-legacy-validation-init2.json",
            "abi-layercake-native-host-semantic-validation/1",
            "PASS",
        ),
        "legacy_init_3": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-legacy-validation-init3.json",
            "abi-layercake-native-host-semantic-validation/1",
            "PASS",
        ),
        "general_validation": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-general-validation.json",
            "abi-layercake-native-general-preservation-validation/1",
            "PASS",
        ),
        "physical_sparse_execution": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-physical-verification.json",
            "abi-layercake-host-physical-sparse-proof/1",
            "PASS",
        ),
        "runtime_identity": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-runtime-identity.json",
            "abi-layercake-host-native-identity/1",
            "PASS",
        ),
        "headline_benchmark": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-benchmark-128.json",
            "abi-layercake-native-host-benchmark/1",
            "PASS",
        ),
        "sustained_benchmark": (
            "results/abi_moonshot/rewriting_v5/"
            "v47-benchmark-1024.json",
            "abi-layercake-native-host-benchmark/1",
            "PASS",
        ),
        "domain_package_validation": (
            "results/abi_moonshot/domain_cakes/"
            "package-certification-validation.json",
            "abi-layercake-domain-package-certification-evidence/1",
            "PASS_VALIDATION_PACKAGE_GATES_FINAL_TEST_UNOPENED",
        ),
    }
    loaded: dict[str, dict[str, Any]] = {}
    evidence_index: dict[str, Any] = {}
    for name, (relative, expected_format, expected_status) in (
        evidence_specs.items()
    ):
        path = (root / relative).resolve()
        value = _read(path)
        if (
            value.get("format", value.get("schema_version"))
            != expected_format
            or value.get("status") != expected_status
            or value.get("final_test_accessed") is not False
        ):
            raise LayerCakeHostError(f"{name} failed the repair gate")
        claimed = _claimed_hash(value)
        loaded[name] = value
        evidence_index[name] = {
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "evidence_sha256": claimed,
            "status": expected_status,
        }

    metadata_path = artifact_path / "metadata.json"
    metadata = _read(metadata_path)
    graph_sha = metadata["runtime"]["graph_sha256"]
    host_sha = metadata["host"]["deployment_manifest_sha256"]
    rewriting = loaded["rewriting_validation"]
    legacy = [
        loaded[f"legacy_init_{index}"] for index in range(1, 4)
    ]
    semantic_hashes = []
    for value in legacy:
        semantic_observations = [
            {
                key: item
                for key, item in observation.items()
                if key != "latency_seconds"
            }
            for observation in value["observations"]
        ]
        semantic_hashes.append(_canonical_sha(semantic_observations))
    general = loaded["general_validation"]
    physical = loaded["physical_sparse_execution"]
    identity = loaded["runtime_identity"]
    headline = loaded["headline_benchmark"]["aggregates"]
    sustained = loaded["sustained_benchmark"]["aggregates"]
    package_validation = loaded["domain_package_validation"]
    gates = {
        "exact_artifact_metadata_bound": (
            _sha256_file(metadata_path)
            == rewriting["artifact"]["metadata_file_sha256"]
        ),
        "runtime_graph_unchanged": (
            graph_sha
            == protocol["source_native_artifact"]["runtime_graph_sha256"]
            and metadata["rewriting_repair"]["runtime_graph_changed"]
            is False
        ),
        "neural_parameters_unchanged": (
            metadata["rewriting_repair"]["neural_parameters_changed"]
            is False
        ),
        "teacher_tokens_imported_zero": (
            metadata["rewriting_repair"]["teacher_tokens_imported"] == 0
        ),
        "v5_rewriting_validation_exact": (
            rewriting["layercake_passes"] == 100
            and rewriting["source_passing_regressions"] == 0
        ),
        "legacy_three_initializations_exact": (
            len(legacy) == 3
            and all(
                value["observation_count"] == 1400
                and value["bounded_zero_regression_pass"] is True
                and value["runtime_graph_sha256"] == graph_sha
                and value["host_manifest_sha256"] == host_sha
                and value["teacher_present_at_inference"] is False
                and value["source_transformer_blocks_retained"] == 0
                for value in legacy
            )
            and len(set(semantic_hashes)) == 1
        ),
        "general_validation_exact": (
            general["observation_count"] == 420
            and general["compared_response_tokens"] == 43089
            and general["parent_top1_agreement"] >= 0.95
            and general["candidate_runtime_graph_sha256"] == graph_sha
        ),
        "physical_sparse_execution": (
            physical["runtime_graph_sha256"] == graph_sha
            and all(physical["checks"].values())
        ),
        "runtime_identity": (
            identity["runtime_graph_sha256"] == graph_sha
            and identity["host_manifest_sha256"] == host_sha
            and all(identity["checks"].values())
        ),
        "headline_speed_quality_memory": all(
            headline["gates"].values()
        ),
        "sustained_speed_quality_memory": all(
            sustained["gates"].values()
        ),
        "phase2_throughput_retention": (
            headline["phase2_throughput_retained_ratio"] >= 0.95
            and sustained["phase2_throughput_retained_ratio"] >= 0.95
        ),
        "qwen_speed_ratio_at_least_2": (
            headline["median_throughput_ratio"] >= 2.0
            and sustained["median_throughput_ratio"] >= 2.0
        ),
        "domain_packages_unchanged_and_qualified": (
            len(package_validation["packages"]) == 3
            and {
                package["domain"]
                for package in package_validation["packages"]
            }
            == {"chemistry", "civics", "python"}
        ),
        "teacher_absent_at_inference": (
            metadata["host"]["teacher_present_at_inference"] is False
        ),
        "source_transformer_blocks_retained_zero": (
            metadata["host"]["source_transformer_blocks_retained"] == 0
        ),
        "final_test_unopened": True,
    }

    negative_specs = {
        "v4_final_failure": (
            "results/abi_moonshot/final_test/layercake-final-test.json"
        ),
        "v46_identity_failure": (
            "results/abi_moonshot/rewriting_v5/"
            "v46-runtime-identity.json"
        ),
    }
    negative_evidence = {}
    for name, relative in negative_specs.items():
        path = (root / relative).resolve()
        value = _read(path)
        if value.get("status") != "FAIL":
            raise LayerCakeHostError(
                f"historical negative evidence changed: {name}"
            )
        negative_evidence[name] = {
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "evidence_sha256": _claimed_hash(value),
            "status": "FAIL",
        }

    certificate: dict[str, Any] = {
        "format": REPAIR_CERTIFICATE_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "candidate": {
            "artifact": str(artifact_path),
            "metadata_file_sha256": _sha256_file(metadata_path),
            "metadata_evidence_sha256": metadata["evidence_sha256"],
            "runtime_graph_sha256": graph_sha,
            "runtime_runner_sha256": _sha256_file(
                Path(__file__).with_name("layercake_host_runtime.py")
            ),
            "symbolic_runtime_sha256": _sha256_file(
                Path(__file__).with_name("symbolic_runtime.py")
            ),
            "symbolic_surface_sha256": metadata[
                "symbolic_surface"
            ]["sha256"],
            "host_manifest_sha256": host_sha,
        },
        "evidence": evidence_index,
        "legacy_semantic_observation_payload_sha256": semantic_hashes[0],
        "performance": {
            "headline": headline,
            "sustained": sustained,
        },
        "gates": gates,
        "negative_evidence_preserved": negative_evidence,
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "final_test_accessed": False,
        "claim_boundary": (
            "This promotes one bounded LayerCake English host on the locked "
            "v2-v5 functional, token-preservation, physical-sparsity, "
            "identity, and CPU performance suites. It is not universal "
            "semantic equivalence, exhaustive domain discovery, or a proof "
            "of a global minimum."
        ),
    }
    certificate["evidence_sha256"] = _canonical_sha(certificate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return certificate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    host = subparsers.add_parser("derive-host")
    host.add_argument("--protocol", required=True)
    host.add_argument("--output", required=True)
    native = subparsers.add_parser("derive-native")
    native.add_argument("--protocol", required=True)
    native.add_argument("--host", required=True)
    native.add_argument("--output", required=True)
    source_validation = subparsers.add_parser("source-validation")
    source_validation.add_argument("--protocol", required=True)
    validation = subparsers.add_parser("evaluate-validation")
    validation.add_argument("--protocol", required=True)
    validation.add_argument("--artifact", required=True)
    validation.add_argument("--output", required=True)
    validation.add_argument("--threads", type=int, default=16)
    certificate = subparsers.add_parser("certify-repair")
    certificate.add_argument("--protocol", required=True)
    certificate.add_argument("--artifact", required=True)
    certificate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "derive-host":
        result = derive_rewriting_host(
            protocol_path=args.protocol,
            output_path=args.output,
        )
    elif args.command == "derive-native":
        result = derive_rewriting_native(
            protocol_path=args.protocol,
            repaired_host_path=args.host,
            output_path=args.output,
        )
    elif args.command == "source-validation":
        result = run_source_validation(protocol_path=args.protocol)
    elif args.command == "evaluate-validation":
        result = evaluate_validation(
            protocol_path=args.protocol,
            artifact_path=args.artifact,
            output_path=args.output,
            threads=args.threads,
        )
    else:
        result = certify_repair(
            protocol_path=args.protocol,
            artifact_path=args.artifact,
            output_path=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
