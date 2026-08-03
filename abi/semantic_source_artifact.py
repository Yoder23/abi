"""Build a derived ABI survey vault from frozen semantic-judge evidence."""

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
from .hf_extraction import load_probe_catalog
from .layercake_host import _canonical_json_bytes, _sha256_file


class SemanticSourceArtifactError(RuntimeError):
    """Raised when semantic evidence cannot produce an exact derived vault."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validated_observations(
    *,
    evidence: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
    results_by_probe: Mapping[str, Mapping[str, Any]],
    probes_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if evidence.get("format") != "abi-independent-semantic-source-qualification/1":
        raise SemanticSourceArtifactError("unsupported semantic evidence format")
    if evidence.get("status") != "PASS" or evidence.get("mode") != "full":
        raise SemanticSourceArtifactError(
            "semantic evidence must be a passing full qualification"
        )
    if evidence.get("evidence_sha256") != _canonical_sha(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_sha256"
        }
    ):
        raise SemanticSourceArtifactError("semantic evidence self-hash is stale")
    judge_manifest = evidence.get("judge", {}).get("source_manifest")
    if not isinstance(judge_manifest, Mapping):
        raise SemanticSourceArtifactError("semantic judge manifest is missing")
    validate_source_model_manifest(judge_manifest)
    runtime = evidence.get("judge", {}).get("runtime", {})
    if (
        runtime.get("device") != "cuda"
        or runtime.get("weight_execution_precision") != "bitsandbytes_int8"
        or runtime.get("cpu_offload_enabled") is not False
    ):
        raise SemanticSourceArtifactError("semantic judge runtime changed")
    observations: dict[str, Mapping[str, Any]] = {}
    seen_records: set[str] = set()
    for raw in evidence.get("observations", []):
        if not isinstance(raw, Mapping):
            raise SemanticSourceArtifactError("invalid semantic observation")
        observation = dict(raw)
        if observation.get("observation_sha256") != _canonical_sha(
            {
                key: value
                for key, value in observation.items()
                if key != "observation_sha256"
            }
        ):
            raise SemanticSourceArtifactError("semantic observation hash is stale")
        probe_id = str(observation.get("probe_id"))
        record_id = str(observation.get("record_id"))
        if probe_id in observations or record_id in seen_records:
            raise SemanticSourceArtifactError(
                "duplicate semantic probe or record identity"
            )
        result = results_by_probe.get(probe_id)
        record = records_by_id.get(record_id)
        probe = probes_by_id.get(probe_id)
        if result is None or record is None or probe is None:
            raise SemanticSourceArtifactError(
                "semantic observation lacks frozen source evidence"
            )
        if str(result["record_id"]) != record_id:
            raise SemanticSourceArtifactError(
                "semantic observation record binding changed"
            )
        raw_prompt_hash = hashlib.sha256(
            str(probe["prompt"]).encode("utf-8")
        ).hexdigest()
        if (
            observation.get("raw_prompt_sha256") != raw_prompt_hash
            or observation.get("source_response_sha256")
            != record["output_sha256"]
            or observation.get("capability") != probe["capability"]
            or observation.get("split") != probe["split"]
        ):
            raise SemanticSourceArtifactError(
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
            raise SemanticSourceArtifactError(
                "semantic observation runtime evidence is incomplete"
            )
        observations[probe_id] = observation
        seen_records.add(record_id)
    if set(observations) != set(probes_by_id):
        raise SemanticSourceArtifactError(
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
        raise SemanticSourceArtifactError(
            "semantic evidence aggregate accounting is stale"
        )
    return observations


def build_semantic_source_vault(
    *,
    protocol_path: Path,
    source_bundle_path: Path,
    catalog_path: Path,
    semantic_evidence_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or output_path.with_name(
        output_path.name + ".receipt.json"
    ).exists():
        raise SemanticSourceArtifactError(
            f"derived semantic source vault is immutable: {output_path}"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_bundle = read_extraction_bundle(source_bundle_path)
    catalog = load_probe_catalog(catalog_path)
    evidence = json.loads(semantic_evidence_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format")
        != "abi-english-semantic-source-artifact-protocol/1"
        or protocol["source_bundle"]["sha256"]
        != source_bundle["verification"]["archive_sha256"]
        or protocol["catalog"]["sha256"] != _sha256_file(catalog_path)
        or protocol["semantic_evidence"]["path"]
        != semantic_evidence_path.as_posix()
        or evidence.get("protocol", {}).get("sha256")
        != protocol["semantic_evidence"]["protocol_sha256"]
        or evidence.get("judge", {})
        .get("source_manifest", {})
        .get("source_manifest_sha256")
        != protocol["semantic_evidence"]["judge_source_manifest_sha256"]
    ):
        raise SemanticSourceArtifactError(
            "derived semantic artifact protocol identity changed"
        )
    journal = evidence.get("judge", {}).get("durable_journal")
    if not isinstance(journal, Mapping):
        raise SemanticSourceArtifactError("semantic durable journal is missing")
    journal_path = Path(str(journal.get("path", "")))
    if (
        not journal_path.is_file()
        or _sha256_file(journal_path) != journal.get("sha256")
        or int(journal.get("completed_probes", -1))
        != int(evidence.get("observation_count", -2))
    ):
        raise SemanticSourceArtifactError(
            "semantic durable journal identity or completion changed"
        )
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
    observations = _validated_observations(
        evidence=evidence,
        records_by_id=records_by_id,
        results_by_probe=results_by_probe,
        probes_by_id=probes_by_id,
    )
    source_manifest = source_bundle["sources"][0]
    semantic_results = []
    for probe_id in sorted(observations):
        observation = observations[probe_id]
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
                passed=bool(observation["passed"]),
                score=score,
                seed=0,
            )
        )
    thresholds = protocol["thresholds"]
    inventory = build_capability_inventory(
        source_manifest=source_manifest,
        records=source_bundle["records"],
        probe_results=semantic_results,
        minimum_distinct_probes=int(thresholds["minimum_distinct_probes"]),
        minimum_pass_rate=float(thresholds["minimum_pass_rate"]),
        minimum_wilson_lower_bound=float(
            thresholds["minimum_wilson_95_lower_bound"]
        ),
        qualification_splits=("validation",),
    )
    if (
        inventory["available_entry_count"]
        != int(protocol["required_capabilities"])
    ):
        raise SemanticSourceArtifactError(
            "semantic evidence did not qualify every required capability"
        )
    selection = build_inventory_survey_plan(inventory)
    ledger = {
        **dict(source_bundle["ledger"]),
        "status": "SEMANTICALLY_REQUALIFIED_SOURCE_SURVEY_NOT_LAYERCAKE_CERTIFIED",
        "input_extraction_archives": [
            {
                "archive_sha256": source_bundle["verification"][
                    "archive_sha256"
                ],
                "manifest_sha256": source_bundle["verification"][
                    "manifest_sha256"
                ],
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
        },
        "claim_boundary": (
            "This ledger preserves the original source extraction accounting "
            "and adds an independent semantic-qualification cost. It remains "
            "a teacher-material survey vault and does not certify LayerCake."
        ),
    }
    result = build_extraction_bundle(
        output_path,
        source_manifests=source_bundle["sources"],
        records=source_bundle["records"],
        probe_results=semantic_results,
        inventories=[inventory],
        selection=selection,
        budgets=source_bundle["budgets"],
        ledger=ledger,
        artifact_role="source_capability_survey_vault",
    )
    verification = verify_extraction_bundle(output_path)
    receipt = {
        "format": "abi-derived-semantic-source-vault-receipt/1",
        **result,
        "verified": verification["verified"],
        "source_bundle_sha256": source_bundle["verification"]["archive_sha256"],
        "semantic_evidence_sha256": evidence["evidence_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "available_entry_count": inventory["available_entry_count"],
        "layercake_invoked": False,
        "layercake_training_authorized": False,
        "abi_transfer_proven": False,
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--semantic-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build_semantic_source_vault(
        protocol_path=Path(args.protocol).resolve(),
        source_bundle_path=Path(args.source_bundle).resolve(),
        catalog_path=Path(args.catalog).resolve(),
        semantic_evidence_path=Path(args.semantic_evidence),
        output_path=Path(args.output),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
