"""Command line for ABI capability survey, selection, composition, and verify.

Examples
--------
Survey one locally cached source model:

    python -m abi.moonshot survey \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --license Apache-2.0 \
      --catalog catalogs/development_capability_probes_v1.json \
      --output results/abi_moonshot/qwen-survey.abix \
      --development

Compose a user-selected artifact from any number of verified surveys:

    python -m abi.moonshot compose \
      --input results/abi_moonshot/qwen-survey.abix \
      --english --domains python,mathematics \
      --domain-ontology catalogs/domain_ontology_v1.json \
      --output results/abi_moonshot/qwen-english-python-math.abix \
      --development
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .capability_pipeline import (
    CapabilityPipelineError,
    SEGREGATED_TRAINING_ARTIFACT_ROLE,
    build_capability_inventory,
    build_extraction_bundle,
    build_inventory_survey_plan,
    build_nested_teacher_budgets,
    build_user_selection_plan,
    read_extraction_bundle,
    records_for_selection,
    verify_extraction_bundle,
)
from .capability_segregation import (
    CapabilitySegregationError,
    build_core_domain_segregation_manifest,
    validate_domain_ontology,
)
from .hf_extraction import HuggingFaceCausalSource, load_probe_catalog, run_probe_catalog


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _domains(value: str) -> list[str]:
    return sorted({item.strip() for item in value.split(",") if item.strip()})


def _splits(value: str) -> list[str]:
    selected = sorted({item.strip() for item in value.split(",") if item.strip()})
    allowed = {"search", "validation", "final_test"}
    if not selected or not set(selected).issubset(allowed):
        raise argparse.ArgumentTypeError(
            "splits must be a comma-separated subset of search,validation,final_test"
        )
    return selected


def _capabilities(value: str) -> list[str]:
    selected = sorted({item.strip() for item in value.split(",") if item.strip()})
    if not selected:
        raise argparse.ArgumentTypeError(
            "capabilities must be a non-empty comma-separated list"
        )
    return selected


def _dedupe(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row[key])
        if identity in output and output[identity] != dict(row):
            raise CapabilityPipelineError(f"conflicting duplicate {key}: {identity}")
        output[identity] = dict(row)
    return [output[identity] for identity in sorted(output)]


def _selected_probe_results(
    probe_results: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    record_ids = {str(record["record_id"]) for record in records}
    selected = [
        dict(result)
        for result in probe_results
        if str(result["record_id"]) in record_ids
    ]
    if not selected:
        raise CapabilityPipelineError("selected records have no probe results")
    return _dedupe(selected, "probe_result_sha256")


def _catalog_id(record: Mapping[str, Any]) -> str:
    provenance = str(record.get("provenance", ""))
    if ":" not in provenance:
        raise CapabilityPipelineError(
            "record provenance does not bind a probe catalog"
        )
    return provenance.rsplit(":", 1)[0]


def _records_for_exact_inventory_selection(
    *,
    records: Sequence[Mapping[str, Any]],
    probe_results: Sequence[Mapping[str, Any]],
    inventories: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind chosen capabilities to their exact inventory/catalog evidence."""

    records_by_id = {str(record["record_id"]): record for record in records}
    results_by_hash = {
        str(result["probe_result_sha256"]): result for result in probe_results
    }
    inventories_by_hash = {
        str(inventory["inventory_sha256"]): inventory
        for inventory in inventories
    }
    source_hash_by_identity = {
        (str(source["model_id"]), str(source["revision"])): str(
            source["source_manifest_sha256"]
        )
        for source in sources
    }
    selected_catalogs: dict[tuple[str, str, str, str], set[str]] = {}
    for item in selection["selected_items"]:
        inventory = inventories_by_hash.get(str(item["inventory_sha256"]))
        if inventory is None:
            raise CapabilityPipelineError(
                "selection references an unavailable inventory"
            )
        matching_entries = [
            entry
            for entry in inventory["entries"]
            if entry["destination_scope"] == item["destination_scope"]
            and entry["domain"] == item["domain"]
            and entry["capability"] == item["capability"]
        ]
        if len(matching_entries) != 1:
            raise CapabilityPipelineError(
                "selected inventory has ambiguous capability evidence"
            )
        evidence_records: list[Mapping[str, Any]] = []
        for result_hash in matching_entries[0]["probe_result_sha256"]:
            result = results_by_hash.get(str(result_hash))
            if result is None:
                raise CapabilityPipelineError(
                    "selected inventory evidence is absent from inputs"
                )
            record = records_by_id.get(str(result["record_id"]))
            if record is None:
                raise CapabilityPipelineError(
                    "selected inventory record is absent from inputs"
                )
            evidence_records.append(record)
        catalog_ids = {_catalog_id(record) for record in evidence_records}
        if len(catalog_ids) != 1:
            raise CapabilityPipelineError(
                "selected capability evidence spans multiple catalogs"
            )
        key = (
            str(item["destination_scope"]),
            str(item["domain"]),
            str(item["capability"]),
            str(item["source_manifest_sha256"]),
        )
        selected_catalogs[key] = catalog_ids

    selected: list[dict[str, Any]] = []
    for record in records:
        source_hash = source_hash_by_identity.get(
            (str(record["source_model"]), str(record["source_model_revision"]))
        )
        key = (
            str(record["destination_scope"]),
            str(record["domain"]),
            str(record["capability"]),
            str(source_hash),
        )
        if _catalog_id(record) in selected_catalogs.get(key, set()):
            selected.append(dict(record))
    if not selected:
        raise CapabilityPipelineError("exact inventory selection has no records")
    return _dedupe(selected, "record_id")


def _automatic_budgets(
    records: Sequence[Mapping[str, Any]], *, ordering_seed: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("search", "validation"):
        split_rows = [record for record in records if record["split"] == split]
        if not split_rows:
            continue
        total = sum(int(record["teacher_tokens"]) for record in split_rows)
        strata: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for record in split_rows:
            strata[
                (
                    str(record["destination_scope"]),
                    str(record["domain"]),
                    str(record["capability"]),
                )
            ].append(record)
        minimum_complete_round = 0
        for rows in strata.values():
            first = min(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{ordering_seed}:{row['record_id']}".encode("utf-8")
                ).hexdigest(),
            )
            minimum_complete_round += int(first["teacher_tokens"])
        candidates = sorted(
            {
                minimum_complete_round,
                max(minimum_complete_round, total // 4),
                max(minimum_complete_round, total // 2),
                total,
            }
        )
        output.extend(
            build_nested_teacher_budgets(
                split_rows,
                requested_teacher_token_budgets=candidates,
                split=split,
                ordering_seed=ordering_seed,
            )
        )
    if not output:
        raise CapabilityPipelineError(
            "selection has no search/validation records for budget construction"
        )
    return output


def _extraction_ledger(
    sources: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    extraction_seconds: float,
    source_inference_seconds: float,
    source_extraction_devices: Sequence[str] = (),
    input_archives: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    unique_prompts: dict[str, int] = {}
    unique_outputs: dict[str, tuple[int, int]] = {}
    for record in records:
        unique_prompts.setdefault(
            str(record["prompt_sha256"]), int(record["prompt_utf8_bytes"])
        )
        unique_outputs.setdefault(
            str(record["output_sha256"]),
            (int(record["output_utf8_bytes"]), int(record["teacher_tokens"])),
        )
    return {
        "schema_version": "abi-source-extraction-ledger/1",
        "status": "EXTRACTION_ONLY_NOT_LAYERCAKE_CERTIFIED",
        "raw_source_prompt_count": len(records),
        "raw_source_prompt_bytes": sum(
            int(record["prompt_utf8_bytes"]) for record in records
        ),
        "unique_prompt_utf8_bytes": sum(unique_prompts.values()),
        "teacher_generated_output_bytes": sum(
            int(record["output_utf8_bytes"]) for record in records
        ),
        "duplicate_adjusted_teacher_output_bytes": sum(
            value[0] for value in unique_outputs.values()
        ),
        "teacher_tokens": sum(int(record["teacher_tokens"]) for record in records),
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
        "source_parameter_count_read": sum(
            int(source["parameter_count"]) for source in sources
        ),
        "source_weight_bytes_read": sum(
            int(source["weight_bytes"]) for source in sources
        ),
        "final_imported_substrate_parameters": 0,
        "bridge_parameters_trained": 0,
        "one_time_source_extraction_seconds": round(float(extraction_seconds), 6),
        "source_model_inference_seconds": round(float(source_inference_seconds), 6),
        "per_host_layercake_certification_seconds": None,
        "final_deployed_footprint_bytes": None,
        "final_cpu_inference_seconds": None,
        "artifact_disk_footprint_bytes": "recorded_in_receipt_sidecar",
        "source_extraction_devices": sorted(set(source_extraction_devices)),
        "external_hardware_used": False,
        "external_hardware_description": "",
        "source_manifest_sha256": sorted(
            str(source["source_manifest_sha256"]) for source in sources
        ),
        "input_extraction_archives": [dict(value) for value in input_archives],
        "claim_boundary": (
            "This ledger covers source survey/extraction only. It records zero "
            "imported substrate or bridge parameters and cannot certify a "
            "LayerCake core or cake."
        ),
    }


def _receipt_path(output: Path) -> Path:
    return output.with_name(output.name + ".receipt.json")


def _survey(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    catalog = load_probe_catalog(args.catalog)
    selected_split_names = set(args.splits)
    selected_catalog = {
        **catalog,
        "probes": [
            probe
            for probe in catalog["probes"]
            if probe["split"] in selected_split_names
            and (
                args.capabilities is None
                or probe["capability"] in set(args.capabilities)
            )
        ],
    }
    if not selected_catalog["probes"]:
        raise CapabilityPipelineError(
            "catalog has no probes in the requested survey splits"
        )
    source = HuggingFaceCausalSource(
        args.model,
        revision=args.revision,
        license_id=args.license,
        device=args.device,
        local_files_only=not args.allow_network,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=not args.no_chat_template,
    )
    inference_started = time.perf_counter()
    records, probe_results = run_probe_catalog(
        source, selected_catalog, batch_size=args.batch_size
    )
    source_inference_seconds = time.perf_counter() - inference_started
    inventory = build_capability_inventory(
        source_manifest=source.source_manifest,
        records=records,
        probe_results=probe_results,
        minimum_distinct_probes=args.minimum_distinct_probes,
        minimum_pass_rate=args.minimum_pass_rate,
        minimum_wilson_lower_bound=args.minimum_wilson_lower_bound,
    )
    selection = build_inventory_survey_plan(inventory)
    selected_records = records_for_selection(records, selection)
    selected_results = _selected_probe_results(probe_results, selected_records)
    budgets = _automatic_budgets(
        selected_records, ordering_seed=f"{catalog['catalog_id']}:survey"
    )
    elapsed = time.perf_counter() - started
    ledger = _extraction_ledger(
        [source.source_manifest],
        selected_records,
        extraction_seconds=elapsed,
        source_inference_seconds=source_inference_seconds,
        source_extraction_devices=[source.device],
    )
    output = Path(args.output)
    bundle = build_extraction_bundle(
        output,
        source_manifests=[source.source_manifest],
        records=selected_records,
        probe_results=selected_results,
        inventories=[inventory],
        selection=selection,
        budgets=budgets,
        ledger=ledger,
        artifact_role="source_capability_survey_vault",
    )
    receipt = {
        "schema_version": "abi-extraction-receipt/1",
        **bundle,
        "command": "survey",
        "catalog_id": catalog["catalog_id"],
        "source_manifest_sha256": source.source_manifest["source_manifest_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "available_entry_count": inventory["available_entry_count"],
        "survey_splits": sorted(selected_split_names),
        "survey_capabilities": args.capabilities,
        "development_selection": args.development,
        "layercake_import_or_certification_performed": False,
    }
    _write_json(_receipt_path(output), receipt)
    return receipt


def _compose(args: argparse.Namespace) -> dict[str, Any]:
    if args.domain_ontology is None:
        raise CapabilityPipelineError(
            "new LayerCake training compositions require --domain-ontology; "
            "historical unsegregated bundles remain verifiable but cannot be "
            "used to create a successor training artifact"
        )
    ontology_path = Path(args.domain_ontology)
    try:
        domain_ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityPipelineError(
            f"cannot read domain ontology: {ontology_path}"
        ) from exc
    try:
        validate_domain_ontology(domain_ontology)
    except CapabilitySegregationError as exc:
        raise CapabilityPipelineError(
            f"invalid domain ontology: {exc}"
        ) from exc

    loaded = [read_extraction_bundle(path) for path in args.input]
    sources = _dedupe(
        [source for bundle in loaded for source in bundle["sources"]],
        "source_manifest_sha256",
    )
    records = _dedupe(
        [record for bundle in loaded for record in bundle["records"]], "record_id"
    )
    probe_results = _dedupe(
        [result for bundle in loaded for result in bundle["probe_results"]],
        "probe_result_sha256",
    )
    inventories = _dedupe(
        [inventory for bundle in loaded for inventory in bundle["inventories"]],
        "inventory_sha256",
    )
    initial_selection = build_user_selection_plan(
        inventories,
        include_english_core=args.english,
        domains=_domains(args.domains),
        source_policy=args.source_policy,
        allow_unverified_development_selection=args.development,
    )
    selected_evidence_records = _records_for_exact_inventory_selection(
        records=records,
        probe_results=probe_results,
        inventories=inventories,
        sources=sources,
        selection=initial_selection,
    )
    selected_evidence_results = _selected_probe_results(
        probe_results, selected_evidence_records
    )
    selected_source_hashes = set(initial_selection["selected_source_manifest_sha256"])
    selected_sources = [
        source
        for source in sources
        if source["source_manifest_sha256"] in selected_source_hashes
    ]

    records_by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    results_by_record = {
        str(result["record_id"]): result for result in selected_evidence_results
    }
    for record in selected_evidence_records:
        records_by_source[
            (str(record["source_model"]), str(record["source_model_revision"]))
        ].append(record)
    rebuilt_inventories: list[dict[str, Any]] = []
    original_by_source = {
        inventory["source_manifest_sha256"]: inventory for inventory in inventories
    }
    for source in selected_sources:
        source_records = records_by_source[
            (str(source["model_id"]), str(source["revision"]))
        ]
        source_results = [
            results_by_record[str(record["record_id"])] for record in source_records
        ]
        thresholds = original_by_source[source["source_manifest_sha256"]]["thresholds"]
        rebuilt_inventories.append(
            build_capability_inventory(
                source_manifest=source,
                records=source_records,
                probe_results=source_results,
                minimum_distinct_probes=int(thresholds["minimum_distinct_probes"]),
                minimum_pass_rate=float(thresholds["minimum_pass_rate"]),
                minimum_wilson_lower_bound=float(
                    thresholds["minimum_wilson_95_lower_bound"]
                ),
                qualification_splits=thresholds["qualification_splits"],
            )
        )
    selection = build_user_selection_plan(
        rebuilt_inventories,
        include_english_core=args.english,
        domains=_domains(args.domains),
        source_policy=args.source_policy,
        allow_unverified_development_selection=args.development,
    )
    selected_records = records_for_selection(
        selected_evidence_records, selection, split="search"
    )
    if not selected_records:
        raise CapabilityPipelineError(
            "composition has no search records for LayerCake training; "
            "validation and final-test outputs are never training material"
        )
    selected_results = _selected_probe_results(probe_results, selected_records)
    passing_record_ids = {
        str(result["record_id"])
        for result in selected_results
        if result["passed"] is True
    }
    selected_records = [
        record
        for record in selected_records
        if str(record["record_id"]) in passing_record_ids
    ]
    selected_results = [
        result
        for result in selected_results
        if str(result["record_id"]) in passing_record_ids
    ]
    selected_item_keys = {
        (
            str(item["destination_scope"]),
            str(item["domain"]),
            str(item["capability"]),
            str(item["source_model"]),
            str(item["source_model_revision"]),
        )
        for item in selection["selected_items"]
    }
    observed_item_keys = {
        (
            str(record["destination_scope"]),
            str(record["domain"]),
            str(record["capability"]),
            str(record["source_model"]),
            str(record["source_model_revision"]),
        )
        for record in selected_records
    }
    if not selected_item_keys.issubset(observed_item_keys):
        missing = sorted(selected_item_keys - observed_item_keys)
        raise CapabilityPipelineError(
            f"selected capabilities lack passing search material: {missing}"
        )
    budgets = _automatic_budgets(
        selected_records, ordering_seed=f"{selection['selection_sha256']}:compose"
    )
    input_archives = [
        {
            "archive_sha256": bundle["verification"]["archive_sha256"],
            "manifest_sha256": bundle["verification"]["manifest_sha256"],
        }
        for bundle in loaded
    ]
    ledger = _extraction_ledger(
        selected_sources,
        selected_records,
        extraction_seconds=0,
        source_inference_seconds=0,
        source_extraction_devices=[],
        input_archives=input_archives,
    )
    try:
        segregation_manifest = build_core_domain_segregation_manifest(
            selected_records,
            domain_ontology=domain_ontology,
        )
    except CapabilitySegregationError as exc:
        raise CapabilityPipelineError(
            f"core/domain segregation failed: {exc}"
        ) from exc
    output = Path(args.output)
    bundle_result = build_extraction_bundle(
        output,
        source_manifests=selected_sources,
        records=selected_records,
        probe_results=selected_results,
        inventories=rebuilt_inventories,
        selection=selection,
        budgets=budgets,
        ledger=ledger,
        artifact_role=SEGREGATED_TRAINING_ARTIFACT_ROLE,
        domain_ontology=domain_ontology,
        segregation_manifest=segregation_manifest,
    )
    receipt = {
        "schema_version": "abi-extraction-receipt/1",
        **bundle_result,
        "command": "compose",
        "input_archives": input_archives,
        "requested_english_core": args.english,
        "requested_domains": _domains(args.domains),
        "source_policy": args.source_policy,
        "development_selection": args.development,
        "artifact_role": SEGREGATED_TRAINING_ARTIFACT_ROLE,
        "domain_ontology_sha256": domain_ontology["ontology_sha256"],
        "core_domain_segregation_sha256": segregation_manifest[
            "segregation_sha256"
        ],
        "domain_segregation_verified": True,
        "absolute_zero_world_knowledge_claimed": False,
        "layercake_import_or_certification_performed": False,
    }
    _write_json(_receipt_path(output), receipt)
    return receipt


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    return verify_extraction_bundle(args.path)


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    source = HuggingFaceCausalSource(
        args.model,
        revision=args.revision,
        license_id=args.license,
        device=args.device,
        local_files_only=not args.allow_network,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=not args.no_chat_template,
    )
    if args.output:
        _write_json(Path(args.output), source.source_manifest)
    return source.source_manifest


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--license", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m abi.moonshot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    _source_arguments(inspect_parser)
    inspect_parser.add_argument("--output")
    inspect_parser.set_defaults(handler=_inspect)

    survey_parser = subparsers.add_parser("survey")
    _source_arguments(survey_parser)
    survey_parser.add_argument("--catalog", required=True)
    survey_parser.add_argument("--output", required=True)
    survey_parser.add_argument("--minimum-distinct-probes", type=int, default=20)
    survey_parser.add_argument("--minimum-pass-rate", type=float, default=0.90)
    survey_parser.add_argument("--minimum-wilson-lower-bound", type=float, default=0.75)
    survey_parser.add_argument("--batch-size", type=int, default=8)
    survey_parser.add_argument(
        "--splits",
        type=_splits,
        default=_splits("search,validation"),
        help=(
            "comma-separated survey splits; defaults to search,validation so "
            "final_test stays sealed until the candidate is locked"
        ),
    )
    survey_parser.add_argument(
        "--capabilities",
        type=_capabilities,
        help="optional comma-separated capability subset for a bounded survey",
    )
    survey_parser.add_argument("--development", action="store_true")
    survey_parser.set_defaults(handler=_survey)

    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("--input", action="append", required=True)
    compose_parser.add_argument("--output", required=True)
    compose_parser.add_argument("--english", action="store_true")
    compose_parser.add_argument("--domains", default="")
    compose_parser.add_argument(
        "--domain-ontology",
        required=True,
        help=(
            "validated JSON ontology used to fail closed on English/domain "
            "segregation"
        ),
    )
    compose_parser.add_argument(
        "--source-policy",
        choices=("best_evidence", "all_qualified_sources"),
        default="best_evidence",
    )
    compose_parser.add_argument("--development", action="store_true")
    compose_parser.set_defaults(handler=_compose)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("path")
    verify_parser.set_defaults(handler=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except CapabilityPipelineError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
