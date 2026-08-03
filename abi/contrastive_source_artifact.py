"""Build a grammar-only ABI survey vault from contrastive source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .capability_pipeline import (
    build_capability_inventory,
    build_extraction_bundle,
    build_inventory_survey_plan,
    build_nested_teacher_budgets,
    build_probe_result,
    read_extraction_bundle,
    verify_extraction_bundle,
)
from .capability_segregation import build_segregated_extraction_record
from .contrastive_grammar_qualification import (
    FORMAT as CONTRASTIVE_FORMAT,
    _pair_from_probe,
)
from .hf_extraction import load_probe_catalog


class ContrastiveSourceArtifactError(RuntimeError):
    """Raised when contrastive evidence cannot create an exact source vault."""


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


def _validated_observations(
    *, evidence: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if evidence.get("format") != CONTRASTIVE_FORMAT or evidence.get("status") != "PASS":
        raise ContrastiveSourceArtifactError(
            "contrastive evidence must be a passing qualification"
        )
    if evidence.get("evidence_sha256") != _canonical_sha(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    ):
        raise ContrastiveSourceArtifactError("contrastive evidence self-hash is stale")
    probes_by_id = {
        str(probe["probe_id"]): probe for probe in catalog.get("probes", [])
    }
    observations: dict[str, Mapping[str, Any]] = {}
    for raw in evidence.get("observations", []):
        if not isinstance(raw, Mapping):
            raise ContrastiveSourceArtifactError("invalid contrastive observation")
        observation = dict(raw)
        if observation.get("observation_sha256") != _canonical_sha(
            {
                key: value
                for key, value in observation.items()
                if key != "observation_sha256"
            }
        ):
            raise ContrastiveSourceArtifactError(
                "contrastive observation self-hash is stale"
            )
        probe_id = str(observation.get("probe_id", ""))
        probe = probes_by_id.get(probe_id)
        if probe is None or probe_id in observations:
            raise ContrastiveSourceArtifactError(
                "contrastive observation identity is missing or duplicated"
            )
        wrong, correct = _pair_from_probe(probe)
        if (
            observation.get("wrong_sentence") != wrong
            or observation.get("correct_sentence") != correct
            or observation.get("split") != probe["split"]
            or observation.get("prompt_contract_sha256")
            != probe["evaluator"]["prompt_contract_sha256"]
            or observation.get("passed") is not True
        ):
            raise ContrastiveSourceArtifactError(
                "contrastive observation changed its frozen catalog binding"
            )
        for order, expected_label in (("ab", "B"), ("ba", "A")):
            row = observation.get(order)
            if (
                not isinstance(row, Mapping)
                or row.get("correct_label") != expected_label
                or not math.isfinite(float(row.get("margin", float("nan"))))
                or float(row["margin"]) <= 0.0
            ):
                raise ContrastiveSourceArtifactError(
                    "contrastive observation lacks a positive counterbalanced margin"
                )
        observations[probe_id] = observation
    if set(observations) != set(probes_by_id):
        raise ContrastiveSourceArtifactError(
            "contrastive evidence does not cover the exact catalog"
        )
    if int(evidence.get("summary", {}).get("passes", -1)) != len(observations):
        raise ContrastiveSourceArtifactError(
            "contrastive aggregate pass count is stale"
        )
    return observations


def _budgets(
    records: Sequence[Mapping[str, Any]], *, ordering_seed: str
) -> list[dict[str, Any]]:
    budgets: list[dict[str, Any]] = []
    for split in ("search", "validation"):
        rows = [row for row in records if row["split"] == split]
        if not rows:
            continue
        total = sum(int(row["teacher_tokens"]) for row in rows)
        candidates = sorted({max(1, total // 4), max(1, total // 2), total})
        budgets.extend(
            build_nested_teacher_budgets(
                rows,
                requested_teacher_token_budgets=candidates,
                split=split,
                ordering_seed=ordering_seed,
            )
        )
    return budgets


def _load_source_token_counter(
    *, model_id: str, revision: str, trust_remote_code: bool
) -> Callable[[str], int]:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ContrastiveSourceArtifactError(
            "transformers and huggingface_hub are required"
        ) from exc
    snapshot = snapshot_download(
        repo_id=model_id, revision=revision, local_files_only=True
    )
    if Path(snapshot).name != revision:
        raise ContrastiveSourceArtifactError("source tokenizer revision changed")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )

    def count(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ContrastiveSourceArtifactError(
                "selected contrastive output tokenized to zero source tokens"
            )
        return len(ids)

    return count


def build_contrastive_source_vault(
    *,
    protocol_path: Path,
    source_bundle_path: Path,
    catalog_path: Path,
    evidence_path: Path,
    output_path: Path,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise ContrastiveSourceArtifactError(
            f"contrastive source vault is immutable: {output_path}"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_bundle = read_extraction_bundle(source_bundle_path)
    catalog = load_probe_catalog(catalog_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "abi-english-contrastive-source-artifact-protocol/1":
        raise ContrastiveSourceArtifactError("unsupported artifact protocol")
    if (
        protocol["source_bundle"]["sha256"]
        != source_bundle["verification"]["archive_sha256"]
        or protocol["catalog"]["sha256"] != _sha256_file(catalog_path)
        or protocol["contrastive_evidence"]["file_sha256"]
        != _sha256_file(evidence_path)
        or protocol["contrastive_evidence"]["evidence_sha256"]
        != evidence.get("evidence_sha256")
    ):
        raise ContrastiveSourceArtifactError("artifact protocol identity changed")
    if len(source_bundle["sources"]) != 1:
        raise ContrastiveSourceArtifactError(
            "contrastive artifact requires exactly one frozen source manifest"
        )
    source_manifest = source_bundle["sources"][0]
    if (
        evidence.get("source", {}).get("source_manifest_sha256")
        != source_manifest["source_manifest_sha256"]
        or protocol["source_bundle"]["source_manifest_sha256"]
        != source_manifest["source_manifest_sha256"]
    ):
        raise ContrastiveSourceArtifactError("source manifest binding changed")
    observations = _validated_observations(evidence=evidence, catalog=catalog)
    if token_counter is None:
        token_counter = _load_source_token_counter(
            model_id=source_manifest["model_id"],
            revision=source_manifest["revision"],
            trust_remote_code=bool(source_manifest["trust_remote_code"]),
        )

    records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    probes_by_id = {str(probe["probe_id"]): probe for probe in catalog["probes"]}
    for probe_id in sorted(observations):
        observation = observations[probe_id]
        probe = probes_by_id[probe_id]
        output = str(observation["correct_sentence"])
        record = build_segregated_extraction_record(
            destination_scope="english_core",
            capability="grammar",
            domain="domain_independent",
            provenance=(
                f"contrastive:{evidence['evidence_sha256']}:"
                f"{observation['observation_sha256']}"
            ),
            split=str(probe["split"]),
            source_model=source_manifest["model_id"],
            source_model_revision=source_manifest["revision"],
            prompt=str(probe["prompt"]),
            output=output,
            teacher_tokens=int(token_counter(output)),
            teacher_token_counter=(
                "authoritative_source_tokenizer_posthoc_on_contrastive_selection"
            ),
            knowledge_class=str(probe["knowledge_class"]),
            content_basis=str(probe["content_basis"]),
            domain_labels=list(probe["domain_labels"]),
            domain_claims=list(probe["domain_claims"]),
            label_method=str(probe["label_method"]),
            label_evidence_sha256=str(probe["label_evidence_sha256"]),
            output_introduces_unsupplied_facts=bool(
                probe["output_introduces_unsupplied_facts"]
            ),
        )
        evaluator = {
            "kind": "counterbalanced_source_preference",
            "prompt_contract_sha256": observation["prompt_contract_sha256"],
            "contrastive_evidence_sha256": evidence["evidence_sha256"],
            "contrastive_observation_sha256": observation["observation_sha256"],
            "source_manifest_sha256": source_manifest[
                "source_manifest_sha256"
            ],
            "selected_output_sha256": record["output_sha256"],
            "ab_margin": observation["ab"]["margin"],
            "ba_margin": observation["ba"]["margin"],
            "teacher_generated_output": False,
        }
        minimum_margin = float(observation["minimum_counterbalanced_margin"])
        score = 1.0 / (1.0 + math.exp(-minimum_margin))
        result = build_probe_result(
            record=record,
            source_manifest_sha256=source_manifest["source_manifest_sha256"],
            probe_id=probe_id,
            evaluator=evaluator,
            passed=True,
            score=score,
            seed=int(probe["seed"]),
        )
        records.append(record)
        results.append(result)

    thresholds = protocol["thresholds"]
    inventory = build_capability_inventory(
        source_manifest=source_manifest,
        records=records,
        probe_results=results,
        minimum_distinct_probes=int(thresholds["minimum_distinct_probes"]),
        minimum_pass_rate=float(thresholds["minimum_pass_rate"]),
        minimum_wilson_lower_bound=float(
            thresholds["minimum_wilson_95_lower_bound"]
        ),
        qualification_splits=("validation",),
    )
    if inventory["available_entry_count"] != 1:
        raise ContrastiveSourceArtifactError(
            "contrastive grammar inventory is not available"
        )
    selection = build_inventory_survey_plan(inventory)
    accounting = evidence["imported_information_accounting"]
    unique_outputs: dict[str, tuple[int, int]] = {}
    for record in records:
        unique_outputs.setdefault(
            str(record["output_sha256"]),
            (int(record["output_utf8_bytes"]), int(record["teacher_tokens"])),
        )
    ledger = {
        "schema_version": "abi-source-extraction-ledger/1",
        "status": "CONTRASTIVE_SOURCE_SURVEY_NOT_LAYERCAKE_CERTIFIED",
        "raw_source_prompt_count": int(accounting["raw_source_prompt_count"]),
        "raw_source_prompt_bytes": int(accounting["raw_source_prompt_bytes"]),
        "unique_prompt_utf8_bytes": int(
            accounting["unique_source_prompt_utf8_bytes"]
        ),
        "teacher_generated_output_bytes": 0,
        "duplicate_adjusted_teacher_output_bytes": 0,
        "contrastive_selected_output_bytes": sum(
            int(record["output_utf8_bytes"]) for record in records
        ),
        "duplicate_adjusted_contrastive_selected_output_bytes": sum(
            value[0] for value in unique_outputs.values()
        ),
        "teacher_tokens": sum(int(record["teacher_tokens"]) for record in records),
        "duplicate_adjusted_teacher_tokens": sum(
            value[1] for value in unique_outputs.values()
        ),
        "teacher_token_counter": (
            "authoritative_source_tokenizer_posthoc_on_contrastive_selection"
        ),
        "logits_stored_count": int(
            accounting["selected_log_probabilities_stored"]
        ),
        "logits_stored_bytes": int(
            accounting["selected_log_probability_storage_bytes_if_float64"]
        ),
        "ephemeral_full_logit_elements_materialized": int(
            accounting["ephemeral_full_logit_elements_materialized"]
        ),
        "hidden_activations_stored_count": 0,
        "hidden_activations_stored_bytes": 0,
        "frozen_source_parameters_copied": 0,
        "frozen_source_parameter_bytes_copied": 0,
        "source_parameter_count_read": int(source_manifest["parameter_count"]),
        "source_weight_bytes_read": int(source_manifest["weight_bytes"]),
        "final_imported_substrate_parameters": 0,
        "bridge_parameters_trained": 0,
        "one_time_source_extraction_seconds": float(
            accounting["one_time_source_load_seconds"]
        )
        + float(accounting["source_model_inference_seconds"]),
        "source_model_inference_seconds": float(
            accounting["source_model_inference_seconds"]
        ),
        "per_host_layercake_certification_seconds": None,
        "final_deployed_footprint_bytes": None,
        "final_cpu_inference_seconds": None,
        "artifact_disk_footprint_bytes": "recorded_in_receipt_sidecar",
        "source_extraction_devices": ["cuda"],
        "source_inference_runtimes": [dict(evidence["source"]["runtime"])],
        "external_hardware_used": True,
        "external_hardware_description": str(evidence["source"]["hardware"]),
        "source_manifest_sha256": [source_manifest["source_manifest_sha256"]],
        "input_extraction_archives": [
            {
                "archive_sha256": source_bundle["verification"]["archive_sha256"],
                "manifest_sha256": source_bundle["verification"]["manifest_sha256"],
            }
        ],
        "contrastive_qualification": {
            "evidence_file_sha256": _sha256_file(evidence_path),
            "evidence_sha256": evidence["evidence_sha256"],
            "method": evidence["method"],
            "checks": evidence["checks"],
            "summary": evidence["summary"],
            "completion_tokens_scored": accounting[
                "completion_tokens_scored"
            ],
            "source_input_and_completion_tokens_evaluated": accounting[
                "source_input_and_completion_tokens_evaluated"
            ],
        },
        "claim_boundary": (
            "This vault contains source-selected contrastive grammar material, "
            "not teacher-generated text. It is not installable and does not "
            "certify LayerCake transfer or execution."
        ),
    }
    bundle = build_extraction_bundle(
        output_path,
        source_manifests=[source_manifest],
        records=records,
        probe_results=results,
        inventories=[inventory],
        selection=selection,
        budgets=_budgets(
            records,
            ordering_seed=f"{catalog['catalog_id']}:contrastive-source-vault",
        ),
        ledger=ledger,
        artifact_role="source_capability_survey_vault",
    )
    verification = verify_extraction_bundle(output_path)
    receipt: dict[str, Any] = {
        "format": "abi-contrastive-source-vault-receipt/1",
        **bundle,
        "verified": verification["verified"],
        "source_bundle_sha256": source_bundle["verification"]["archive_sha256"],
        "contrastive_evidence_sha256": evidence["evidence_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "available_entry_count": inventory["available_entry_count"],
        "records": len(records),
        "search_records": sum(row["split"] == "search" for row in records),
        "validation_records": sum(
            row["split"] == "validation" for row in records
        ),
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
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--contrastive-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    receipt = build_contrastive_source_vault(
        protocol_path=Path(args.protocol),
        source_bundle_path=Path(args.source_bundle),
        catalog_path=Path(args.catalog),
        evidence_path=Path(args.contrastive_evidence),
        output_path=Path(args.output),
    )
    receipt["artifact_construction_seconds_console_only"] = (
        time.perf_counter() - started
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
