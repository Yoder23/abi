"""Build a partial multi-source vault from complete semantic-judge evidence.

The full semantic survey may fail because one capability misses its gates while
other capability inventories pass exactly.  This builder verifies the entire
survey and durable journal, then materializes only the capability set declared
available by a preregistered multi-source protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capability_pipeline import (
    build_capability_inventory,
    build_extraction_bundle,
    build_inventory_survey_plan,
    build_probe_result,
    read_extraction_bundle,
    validate_source_model_manifest,
    verify_extraction_bundle,
)
from .contrastive_source_artifact import _budgets
from .hf_extraction import load_probe_catalog


class PartialSemanticSourceArtifactError(RuntimeError):
    """Raised when partial semantic material cannot be proven exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validated_complete_observations(
    *,
    evidence: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
    results_by_probe: Mapping[str, Mapping[str, Any]],
    probes_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if (
        evidence.get("format")
        != "abi-independent-semantic-source-qualification/1"
        or evidence.get("mode") != "full"
        or evidence.get("status") not in {"PASS", "FAIL"}
    ):
        raise PartialSemanticSourceArtifactError(
            "semantic evidence must be a complete full qualification"
        )
    if evidence.get("evidence_sha256") != _canonical_sha(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    ):
        raise PartialSemanticSourceArtifactError("semantic evidence self-hash is stale")
    judge_manifest = evidence.get("judge", {}).get("source_manifest")
    if not isinstance(judge_manifest, Mapping):
        raise PartialSemanticSourceArtifactError("semantic judge manifest is missing")
    validate_source_model_manifest(judge_manifest)
    runtime = evidence.get("judge", {}).get("runtime", {})
    if (
        runtime.get("device") != "cuda"
        or runtime.get("weight_execution_precision") != "bitsandbytes_int8"
        or runtime.get("cpu_offload_enabled") is not False
    ):
        raise PartialSemanticSourceArtifactError("semantic judge runtime changed")
    observations: dict[str, Mapping[str, Any]] = {}
    seen_records: set[str] = set()
    for raw in evidence.get("observations", []):
        if not isinstance(raw, Mapping):
            raise PartialSemanticSourceArtifactError("invalid semantic observation")
        observation = dict(raw)
        if observation.get("observation_sha256") != _canonical_sha(
            {
                key: value
                for key, value in observation.items()
                if key != "observation_sha256"
            }
        ):
            raise PartialSemanticSourceArtifactError(
                "semantic observation hash is stale"
            )
        probe_id = str(observation.get("probe_id", ""))
        record_id = str(observation.get("record_id", ""))
        result = results_by_probe.get(probe_id)
        record = records_by_id.get(record_id)
        probe = probes_by_id.get(probe_id)
        if (
            probe_id in observations
            or record_id in seen_records
            or result is None
            or record is None
            or probe is None
        ):
            raise PartialSemanticSourceArtifactError(
                "semantic observation lacks unique frozen source evidence"
            )
        raw_prompt_hash = hashlib.sha256(
            str(probe["prompt"]).encode("utf-8")
        ).hexdigest()
        if (
            str(result["record_id"]) != record_id
            or observation.get("raw_prompt_sha256") != raw_prompt_hash
            or observation.get("source_response_sha256")
            != record["output_sha256"]
            or observation.get("capability") != probe["capability"]
            or observation.get("split") != probe["split"]
        ):
            raise PartialSemanticSourceArtifactError(
                "semantic observation content binding changed"
            )
        ids = observation.get("authoritative_judge_token_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != observation.get("judge_tokens")
            or observation.get("judge_token_counter")
            != "authoritative_generated_token_ids"
            or observation.get("judge_finish_reason") != "eos_token"
            or observation.get("parsed") is not True
        ):
            raise PartialSemanticSourceArtifactError(
                "semantic observation runtime evidence is incomplete"
            )
        observations[probe_id] = observation
        seen_records.add(record_id)
    if set(observations) != set(probes_by_id):
        raise PartialSemanticSourceArtifactError(
            "semantic evidence does not cover the exact frozen catalog"
        )
    if (
        int(evidence.get("observation_count", -1)) != len(observations)
        or int(evidence.get("parse_count", -1)) != len(observations)
        or int(evidence.get("eos_count", -1)) != len(observations)
        or int(evidence.get("semantic_passes", -1))
        != sum(bool(row["passed"]) for row in observations.values())
        or int(evidence.get("judge", {}).get("generated_tokens", -1))
        != sum(int(row["judge_tokens"]) for row in observations.values())
    ):
        raise PartialSemanticSourceArtifactError(
            "semantic aggregate accounting is stale"
        )
    return observations


def build_partial_semantic_source_vault(
    *,
    protocol_path: Path,
    amendment_path: Path,
    decision_path: Path,
    source_bundle_path: Path,
    source_runtime_audit_path: Path,
    catalog_path: Path,
    semantic_evidence_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise PartialSemanticSourceArtifactError(
            f"partial semantic source vault is immutable: {output_path}"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    source_runtime_audit = json.loads(
        source_runtime_audit_path.read_text(encoding="utf-8")
    )
    source_bundle = read_extraction_bundle(source_bundle_path)
    catalog = load_probe_catalog(catalog_path)
    evidence = json.loads(semantic_evidence_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "abi-english-partial-semantic-source-artifact-protocol/1":
        raise PartialSemanticSourceArtifactError("unsupported partial protocol")
    if (
        amendment.get("format")
        != "abi-english-partial-semantic-v70-runtime-ledger-amendment/1"
        or amendment["parent_protocol"]["sha256"] != _sha256_file(protocol_path)
        or amendment["source_runtime_audit"]["file_sha256"]
        != _sha256_file(source_runtime_audit_path)
        or amendment["source_runtime_audit"]["evidence_sha256"]
        != source_runtime_audit.get("evidence_sha256")
        or source_runtime_audit.get("status")
        != "PASS_SOURCE_RUNTIME_EVIDENCE_PREFLIGHT"
        or source_runtime_audit.get("bundle", {}).get("sha256")
        != source_bundle["verification"]["archive_sha256"]
        or source_runtime_audit.get("bundle", {}).get("source_manifest_sha256")
        != [source_bundle["sources"][0]["source_manifest_sha256"]]
    ):
        raise PartialSemanticSourceArtifactError(
            "runtime-ledger amendment or source audit identity changed"
        )
    if (
        protocol["decision"]["sha256"] != _sha256_file(decision_path)
        or protocol["source_bundle"]["sha256"]
        != source_bundle["verification"]["archive_sha256"]
        or protocol["catalog"]["sha256"] != _sha256_file(catalog_path)
        or protocol["semantic_evidence"]["file_sha256"]
        != _sha256_file(semantic_evidence_path)
        or protocol["semantic_evidence"]["evidence_sha256"]
        != evidence.get("evidence_sha256")
    ):
        raise PartialSemanticSourceArtifactError("partial protocol identity changed")
    journal = evidence.get("judge", {}).get("durable_journal")
    if not isinstance(journal, Mapping):
        raise PartialSemanticSourceArtifactError("semantic durable journal is missing")
    journal_path = Path(str(journal.get("path", "")))
    if (
        not journal_path.is_file()
        or _sha256_file(journal_path) != journal.get("sha256")
        or int(journal.get("completed_probes", -1))
        != int(evidence.get("observation_count", -2))
    ):
        raise PartialSemanticSourceArtifactError(
            "semantic durable journal identity or completion changed"
        )
    if len(source_bundle["sources"]) != 1:
        raise PartialSemanticSourceArtifactError(
            "partial semantic source requires exactly one source manifest"
        )
    source_manifest = source_bundle["sources"][0]
    records_by_id = {
        str(record["record_id"]): record for record in source_bundle["records"]
    }
    results_by_probe = {
        str(result["probe_id"]): result
        for result in source_bundle["probe_results"]
    }
    probes_by_id = {
        str(probe["probe_id"]): probe for probe in catalog["probes"]
    }
    observations = _validated_complete_observations(
        evidence=evidence,
        records_by_id=records_by_id,
        results_by_probe=results_by_probe,
        probes_by_id=probes_by_id,
    )
    declared = set(protocol["included_capabilities"])
    excluded = set(protocol["excluded_capabilities"])
    matrix = decision.get("capability_matrix", [])
    available = {
        str(row["capability"]) for row in matrix if row.get("available") is True
    }
    unavailable = {
        str(row["capability"]) for row in matrix if row.get("available") is False
    }
    if (
        declared != available
        or excluded != unavailable
        or declared & excluded
        or len(declared) != int(protocol["required_available_capabilities"])
    ):
        raise PartialSemanticSourceArtifactError(
            "declared partial capability set differs from the frozen decision"
        )

    selected_records = [
        record
        for record in source_bundle["records"]
        if str(record["capability"]) in declared
    ]
    selected_record_ids = {str(record["record_id"]) for record in selected_records}
    selected_probe_ids = {
        str(result["probe_id"])
        for result in source_bundle["probe_results"]
        if str(result["record_id"]) in selected_record_ids
    }
    semantic_results = []
    for probe_id in sorted(selected_probe_ids):
        observation = observations[probe_id]
        if observation.get("passed") is not True:
            # Search misses may remain in a survey vault; validation inventory
            # gates determine availability. Preserve their actual result.
            passed = False
        else:
            passed = True
        original_result = results_by_probe[probe_id]
        record = records_by_id[str(observation["record_id"])]
        judgment = observation["judgment"]
        score = sum(
            int(judgment[name])
            for name in (
                "linguistic_quality",
                "prompt_grounding",
                "task_correctness",
                "instruction_adherence",
            )
        ) / 16.0
        evaluator = {
            "kind": "independent_semantic_judge",
            "prompt_contract_sha256": observation["raw_prompt_sha256"],
            "semantic_protocol_sha256": evidence["protocol"]["sha256"],
            "semantic_evidence_sha256": evidence["evidence_sha256"],
            "judge_source_manifest_sha256": evidence["judge"][
                "source_manifest"
            ]["source_manifest_sha256"],
            "judge_observation_sha256": observation["observation_sha256"],
            "source_response_sha256": observation["source_response_sha256"],
            "original_probe_result_sha256": original_result[
                "probe_result_sha256"
            ],
        }
        semantic_results.append(
            build_probe_result(
                record=record,
                source_manifest_sha256=source_manifest[
                    "source_manifest_sha256"
                ],
                probe_id=probe_id,
                evaluator=evaluator,
                passed=passed,
                score=score,
                seed=0,
            )
        )

    thresholds = protocol["thresholds"]
    inventory = build_capability_inventory(
        source_manifest=source_manifest,
        records=selected_records,
        probe_results=semantic_results,
        minimum_distinct_probes=int(thresholds["minimum_distinct_probes"]),
        minimum_pass_rate=float(thresholds["minimum_pass_rate"]),
        minimum_wilson_lower_bound=float(
            thresholds["minimum_wilson_95_lower_bound"]
        ),
        qualification_splits=("validation",),
    )
    inventory_available = {
        str(entry["capability"])
        for entry in inventory["entries"]
        if entry["available"] is True
    }
    if inventory_available != declared:
        raise PartialSemanticSourceArtifactError(
            "rebuilt partial inventory differs from the declared passing set"
        )
    selection = build_inventory_survey_plan(inventory)
    unique_prompts: dict[str, int] = {}
    unique_outputs: dict[str, tuple[int, int]] = {}
    for record in selected_records:
        unique_prompts.setdefault(
            str(record["prompt_sha256"]), int(record["prompt_utf8_bytes"])
        )
        unique_outputs.setdefault(
            str(record["output_sha256"]),
            (int(record["output_utf8_bytes"]), int(record["teacher_tokens"])),
        )
    source_ledger = source_bundle["ledger"]
    ledger = {
        "schema_version": "abi-source-extraction-ledger/1",
        "status": "PARTIAL_SEMANTIC_SOURCE_SURVEY_NOT_LAYERCAKE_CERTIFIED",
        "raw_source_prompt_count": len(selected_records),
        "raw_source_prompt_bytes": sum(
            int(record["prompt_utf8_bytes"]) for record in selected_records
        ),
        "unique_prompt_utf8_bytes": sum(unique_prompts.values()),
        "teacher_generated_output_bytes": sum(
            int(record["output_utf8_bytes"]) for record in selected_records
        ),
        "duplicate_adjusted_teacher_output_bytes": sum(
            value[0] for value in unique_outputs.values()
        ),
        "teacher_tokens": sum(
            int(record["teacher_tokens"]) for record in selected_records
        ),
        "duplicate_adjusted_teacher_tokens": sum(
            value[1] for value in unique_outputs.values()
        ),
        "teacher_token_counter": "authoritative_generated_token_ids",
        "logits_stored_count": 0,
        "logits_stored_bytes": 0,
        "hidden_activations_stored_count": 0,
        "hidden_activations_stored_bytes": 0,
        "frozen_source_parameters_copied": 0,
        "frozen_source_parameter_bytes_copied": 0,
        "source_parameter_count_read": int(source_manifest["parameter_count"]),
        "source_weight_bytes_read": int(source_manifest["weight_bytes"]),
        "final_imported_substrate_parameters": 0,
        "bridge_parameters_trained": 0,
        "one_time_source_extraction_seconds": float(
            source_ledger["one_time_source_extraction_seconds"]
        ),
        "source_model_inference_seconds": float(
            source_ledger["source_model_inference_seconds"]
        ),
        "per_host_layercake_certification_seconds": None,
        "final_deployed_footprint_bytes": None,
        "final_cpu_inference_seconds": None,
        "artifact_disk_footprint_bytes": "recorded_in_receipt_sidecar",
        "source_extraction_devices": list(
            source_ledger["source_extraction_devices"]
        ),
        "source_inference_runtimes": [
            dict(amendment["reconstructed_runtime"])
        ],
        "source_runtime_evidence": {
            "audit_file_sha256": _sha256_file(source_runtime_audit_path),
            "audit_evidence_sha256": source_runtime_audit["evidence_sha256"],
            "protocol_sha256": amendment["source_runtime_audit"][
                "source_protocol_sha256"
            ],
            "reconstruction_boundary": amendment["reconstruction_boundary"],
        },
        "external_hardware_used": True,
        "external_hardware_description": "NVIDIA GeForce RTX 3080 Laptop GPU",
        "source_manifest_sha256": [source_manifest["source_manifest_sha256"]],
        "input_extraction_archives": [
            {
                "archive_sha256": source_bundle["verification"]["archive_sha256"],
                "manifest_sha256": source_bundle["verification"]["manifest_sha256"],
            }
        ],
        "semantic_qualification": {
            "evidence_file_sha256": _sha256_file(semantic_evidence_path),
            "evidence_sha256": evidence["evidence_sha256"],
            "judge_source_manifest_sha256": evidence["judge"][
                "source_manifest"
            ]["source_manifest_sha256"],
            "judge_generated_tokens": evidence["judge"]["generated_tokens"],
            "judge_load_seconds": evidence["judge"]["load_seconds"],
            "judge_inference_seconds": evidence["judge"]["inference_seconds"],
            "judge_runtime": evidence["judge"]["runtime"],
            "judge_durable_journal": evidence["judge"]["durable_journal"],
            "full_capability_result": decision["status"],
            "included_capabilities": sorted(declared),
            "excluded_capabilities": sorted(excluded),
        },
        "claim_boundary": (
            "This vault preserves only the thirteen capabilities that passed "
            "the complete V62 semantic survey. The failed grammar inventory is "
            "excluded. It remains source material and does not certify LayerCake."
        ),
    }
    bundle = build_extraction_bundle(
        output_path,
        source_manifests=[source_manifest],
        records=selected_records,
        probe_results=semantic_results,
        inventories=[inventory],
        selection=selection,
        budgets=_budgets(
            selected_records,
            ordering_seed=f"{catalog['catalog_id']}:partial-semantic-vault",
        ),
        ledger=ledger,
        artifact_role="source_capability_survey_vault",
    )
    verification = verify_extraction_bundle(output_path)
    receipt: dict[str, Any] = {
        "format": "abi-partial-semantic-source-vault-receipt/1",
        **bundle,
        "verified": verification["verified"],
        "source_bundle_sha256": source_bundle["verification"]["archive_sha256"],
        "semantic_evidence_sha256": evidence["evidence_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "available_entry_count": inventory["available_entry_count"],
        "included_capabilities": sorted(declared),
        "excluded_capabilities": sorted(excluded),
        "records": len(selected_records),
        "layercake_invoked": False,
        "layercake_training_authorized": False,
        "abi_transfer_proven": False,
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--source-runtime-audit", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--semantic-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build_partial_semantic_source_vault(
        protocol_path=Path(args.protocol),
        amendment_path=Path(args.amendment),
        decision_path=Path(args.decision),
        source_bundle_path=Path(args.source_bundle),
        source_runtime_audit_path=Path(args.source_runtime_audit),
        catalog_path=Path(args.catalog),
        semantic_evidence_path=Path(args.semantic_evidence),
        output_path=Path(args.output),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
