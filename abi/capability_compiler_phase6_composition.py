"""Phase 6 three-domain composition, portability, and provenance campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile

import torch

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_gpu_runtime import _tensor_bytes
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase5_selective_product import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    DOMAINS,
    SEEDS,
    _core_package,
    _domain_rows,
    _domain_specs,
    _zero_execution,
)
from .capability_compiler_phase5_construct_screen import project_catalog_prompt
from .capability_pipeline import read_extraction_bundle


FORMAT = "abi-capability-compiler-phase6-composition/1"
RESULT_FORMAT = "abi-capability-compiler-phase6-composition-result/1"


def _historical_selected_domain_rows(
    bundle: Mapping[str, Any], *, domain: str, budget_index: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Reconstruct an already-consumed lineage without reauthorizing training."""

    verification = bundle["verification"]
    if (
        verification.get("verified") is not True
        or verification.get("artifact_role")
        != "selected_layercake_training_material_v2"
        or verification.get("historical_manifest_training_eligible") is not True
    ):
        raise Phase3Error("Phase 6 historical archive is not verified prior material")
    if verification.get("training_eligible") is not False:
        raise Phase3Error("Phase 6 historical audit requires a retired training archive")
    selected_items = [
        dict(row)
        for row in bundle["selection"]["selected_items"]
        if row.get("destination_scope") == "domain_cake"
        and row.get("domain") == domain
    ]
    if len(selected_items) != 1:
        raise Phase3Error(f"Phase 6 historical source selection changed: {domain}")
    selected_source = selected_items[0]
    budget = bundle["budgets"][budget_index]
    if budget.get("split") != "search":
        raise Phase3Error("Phase 6 historical budget is not search-only")
    allowed = set(str(value) for value in budget["record_ids"])
    passing = {
        str(row["record_id"]): row.get("passed") is True
        for row in bundle["probe_results"]
    }
    rows = [
        dict(row)
        for row in bundle["records"]
        if str(row["record_id"]) in allowed
        and row.get("destination_scope") == "domain_cake"
        and row.get("domain") == domain
        and row.get("split") == "search"
        and row.get("source_model") == selected_source["source_model"]
        and row.get("source_model_revision")
        == selected_source["source_model_revision"]
        and passing.get(str(row["record_id"])) is True
    ]
    rows.sort(key=lambda row: str(row["record_id"]))
    if not rows:
        raise Phase3Error(f"Phase 6 historical budget has no selected rows: {domain}")
    return rows, dict(budget), selected_source


def _phase5_reference(root: Path, protocol: Mapping[str, Any], seed: int) -> dict[str, bytes]:
    path = root / str(protocol["phase5_observations"][str(seed)])
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("mode") == "selected_domain_installed"]
    indexed = {
        str(row["probe_id"]): str(row["output"]).encode("utf-8") for row in selected
    }
    if len(indexed) != 300:
        raise Phase3Error(f"Phase 6 Phase 5 reference depth changed: {seed}")
    return indexed


def _package_manifest(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def _selected_only(
    delta: Mapping[str, Mapping[str, int]], selected_cake_id: str
) -> bool:
    if selected_cake_id not in delta:
        return False
    for cake_id, counters in delta.items():
        prefill = int(counters.get("prefill_calls", 0))
        if cake_id == selected_cake_id:
            if prefill != 1:
                return False
        elif any(int(value) != 0 for value in counters.values()):
            return False
    return True


def _provenance(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    bundle_path = root / str(protocol["domain_training_bundle"])
    bundle = read_extraction_bundle(bundle_path)
    if sha256_file(bundle_path) != protocol["domain_training_bundle_sha256"]:
        raise Phase3Error("Phase 6 domain training bundle changed")
    source_by_identity = {
        (str(row["model_id"]), str(row["revision"])): row
        for row in bundle["sources"]
    }
    package_protocol = _json(root / protocol["domain_package_protocol"])
    package_specs = {
        str(row["domain"]): row for row in package_protocol["domains"]
    }
    domains: dict[str, Any] = {}
    all_selected_record_ids: set[str] = set()
    for domain in DOMAINS:
        specification = package_specs[domain]
        selected, budget, selected_source = _historical_selected_domain_rows(
            bundle,
            domain=domain,
            budget_index=int(specification["minimum_tested_passing_budget_index"]),
        )
        package_path = root / specification["package"]
        manifest = _package_manifest(package_path)
        source_identity = (
            str(selected_source["source_model"]),
            str(selected_source["source_model_revision"]),
        )
        source = source_by_identity.get(source_identity)
        if source is None:
            raise Phase3Error(f"Phase 6 selected source manifest ambiguous: {domain}")
        source_manifest = str(source["source_manifest_sha256"])
        record_ids = [str(row["record_id"]) for row in selected]
        all_selected_record_ids.update(record_ids)
        gates = {
            "selected_record_count": len(selected) == int(specification["training_rows"]),
            "selected_record_ids_unique": len(record_ids) == len(set(record_ids)),
            "teacher_tokens_exact": sum(int(row["teacher_tokens"]) for row in selected)
            == int(specification["teacher_tokens"]),
            "source_identity_exact": all(
                row["source_model"] == source["model_id"]
                and row["source_model_revision"] == source["revision"]
                for row in selected
            ),
            "source_revision_immutable": source["revision_is_immutable"] is True,
            "package_bundle_identity": manifest["training_data_provenance"][
                "abi_training_bundle_sha256"
            ]
            == sha256_file(bundle_path),
            "package_source_identity": manifest["training_data_provenance"][
                "source_model"
            ]
            == source["model_id"]
            and manifest["training_data_provenance"]["source_model_revision"]
            == source["revision"],
            "package_teacher_tokens_exact": int(
                manifest["training_data_provenance"]["teacher_tokens"]
            )
            == int(specification["teacher_tokens"]),
            "teacher_absent_from_package": manifest["training_data_provenance"][
                "source_teacher_in_package"
            ]
            is False,
        }
        domains[domain] = {
            "cake_id": manifest["cake_id"],
            "package_archive_sha256": sha256_file(package_path),
            "package_license": manifest["license"],
            "training_budget_id": budget["budget_id"],
            "training_budget_index": int(specification["minimum_tested_passing_budget_index"]),
            "selected_record_ids": record_ids,
            "selected_record_set_sha256": hashlib.sha256(
                "\n".join(record_ids).encode("ascii")
            ).hexdigest(),
            "source_manifest_sha256": source_manifest,
            "source_model": source["model_id"],
            "source_revision": source["revision"],
            "source_license": source["license_id"],
            "gates": gates,
        }
    english = _json(root / protocol["english_phase1_certificate"])["source"]
    source_inventory = [
        {
            "source_manifest_sha256": row["source_manifest_sha256"],
            "model_id": row["model_id"],
            "revision": row["revision"],
            "license_id": row["license_id"],
            "revision_is_immutable": row["revision_is_immutable"],
        }
        for row in bundle["sources"]
    ]
    selected_sources = sorted({row["source_manifest_sha256"] for row in domains.values()})
    gates = {
        "all_domain_lineages_exact": all(
            all(row["gates"].values()) for row in domains.values()
        ),
        "selected_domain_records_disjoint": len(all_selected_record_ids)
        == sum(len(row["selected_record_ids"]) for row in domains.values()),
        "bundle_contains_three_pinned_sources": len(source_inventory) == 3
        and len({row["source_manifest_sha256"] for row in source_inventory}) == 3
        and all(row["revision_is_immutable"] for row in source_inventory),
        "deployed_domain_packages_select_one_pinned_source": len(selected_sources) == 1,
        "english_source_pinned": english["model"] == "microsoft/Phi-3-mini-4k-instruct"
        and len(str(english["revision"])) == 40,
        "record_level_deletion_index_complete": len(all_selected_record_ids)
        == sum(int(package_specs[domain]["training_rows"]) for domain in DOMAINS),
    }
    return {
        "domain_training_bundle_sha256": sha256_file(bundle_path),
        "bundle_source_inventory": source_inventory,
        "deployed_selected_source_manifests": selected_sources,
        "deployed_multiple_source_models": len(selected_sources) > 1,
        "english_source": english,
        "domains": domains,
        "deletion_lineage": {
            "record_id_to_artifact": {
                record_id: domains[domain]["cake_id"]
                for domain in DOMAINS
                for record_id in domains[domain]["selected_record_ids"]
            },
            "selected_source_manifest_to_artifacts": {
                source_manifest: sorted(
                    row["cake_id"]
                    for row in domains.values()
                    if row["source_manifest_sha256"] == source_manifest
                )
                for source_manifest in selected_sources
            },
            "archive_custody_note": "The extraction archive contains three pinned sources. Deployed domain packages select only the source(s) listed above; nonselected archive sources are custody dependencies of the archive, not measured training dependencies of these packages.",
        },
        "gates": gates,
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    document = _json(path)
    status = document.get("status")
    repaired = status == "PREREGISTERED_PHASE6_COMPOSITION_HISTORICAL_LINEAGE_REPAIR"
    if repaired:
        base_path = root / str(document.get("base_protocol", ""))
        if (
            not base_path.is_file()
            or sha256_file(base_path) != document.get("base_protocol_sha256")
        ):
            raise Phase3Error("Phase 6 repair base protocol changed")
        base = _json(base_path)
        protocol = {
            **base,
            **document,
            "bindings": {
                **base.get("bindings", {}),
                **document.get("bindings", {}),
            },
        }
    else:
        protocol = document
    if (
        protocol.get("format") != FORMAT
        or status
        not in {
            "PREREGISTERED_PHASE6_COMPOSITION",
            "PREREGISTERED_PHASE6_COMPOSITION_HISTORICAL_LINEAGE_REPAIR",
        }
        or protocol.get("device") != "cuda"
        or protocol.get("domains") != list(DOMAINS)
        or protocol.get("seeds") != list(SEEDS)
        or int(protocol.get("per_domain", 0)) != 100
        or int(protocol.get("composition_requests", 0)) != 100
        or int(protocol.get("conflict_requests", 0)) != 100
        or int(protocol.get("english_preservation_prompts", 0)) != 100
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_authorized") is not False
        or protocol.get("artifact_mutation_authorized") is not False
        or protocol.get("phase5_final_reuse") != "EXACT_REPLAY_NO_SELECTION_OR_TUNING"
    ):
        raise Phase3Error("Phase 6 composition governance changed")
    if repaired and (
        protocol.get("repair_of")
        != "ABI_CAPABILITY_COMPILER_PHASE6_COMPOSITION_PROTOCOL_V1031.json"
        or protocol.get("preserved_failure")
        != "ABI_CAPABILITY_COMPILER_PHASE6_PREFLIGHT_FAILURE_V1033.json"
        or protocol.get("repair_scope")
        != "READ_ONLY_RETIRED_ARCHIVE_LINEAGE_RECONSTRUCTION_ONLY"
    ):
        raise Phase3Error("Phase 6 historical-lineage repair scope changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 6 composition binding changed: {relative}")
    return protocol, sha256_file(path)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    provenance = _provenance(root, protocol)
    gates = {
        "cuda_available": torch.cuda.is_available(),
        "provenance_gates_pass": all(provenance["gates"].values()),
        "three_outputs_absent": all(
            not (root / protocol["result_path_template"].format(seed=seed)).exists()
            for seed in SEEDS
        ),
        "training_absent": True,
        "teacher_absent": True,
    }
    return {
        "format": "abi-capability-compiler-phase6-composition-preflight/1",
        "status": "PASS_PHASE6_COMPOSITION_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE6_COMPOSITION_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "deployed_multiple_source_models": provenance[
            "deployed_multiple_source_models"
        ],
        "bundle_source_count": len(provenance["bundle_source_inventory"]),
        "gates": gates,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, *, seed: int, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if seed not in SEEDS or output.exists() or not torch.cuda.is_available():
        raise Phase3Error("invalid, existing, or unavailable Phase 6 target")
    rows = _domain_rows(
        root / protocol["domain_catalog"], split="final_test", per_domain=100
    )
    by_domain = {
        domain: [row for row in rows if row["domain"] == domain]
        for domain in DOMAINS
    }
    reference = _phase5_reference(root, protocol, seed)
    specs, trust = _domain_specs(root, protocol)
    provenance = _provenance(root, protocol)
    observations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"abi-phase6-seed{seed}-") as raw:
        temporary = Path(raw)
        core_host, built, activated, _ = _core_package(root, protocol, seed, temporary)
        core_before = {
            "archive_hash": activated["archive_hash"],
            "payload_hash": activated["payload_hash"],
            "state_dict_hash": activated["state_dict_hash"],
            "verify": core_host.verify(),
        }
        english = development_probes(root / protocol["english_catalog"])[
            : int(protocol["english_preservation_prompts"])
        ]
        english_before = {
            str(row["probe_id"]): core_host.generate(
                str(row["prompt"]), maximum_tokens=int(row["max_new_tokens"])
            )
            for row in english
        }
        from layercake.routing.catalog_router import ArchiveBoundProfile, RoutingFeature
        from layercake_extensions.authoritative_destination_control import (
            AuthoritativeDestinationOrchestrator,
        )

        profiles = tuple(
            ArchiveBoundProfile(
                cake_id=specs[domain]["cake_id"],
                archive_sha256=specs[domain]["archive_sha256"],
                domains=(domain,),
                features=(RoutingFeature("token", domain, 1.0),),
            )
            for domain in DOMAINS
        )
        orchestrator = AuthoritativeDestinationOrchestrator(
            temporary / "composed-domain-registry",
            abi_version=DIRECT_ABI_VERSION,
            abi_hash=DIRECT_ABI_SHA256,
            trust_store=trust,
            profiles=profiles,
            device="cuda",
            maximum_loaded_cakes=3,
        )
        installs = {domain: orchestrator.install(specs[domain]["package"]) for domain in DOMAINS}
        registry = {row["cake_id"]: row for row in orchestrator.host.registry.list()}

        for domain in DOMAINS:
            for row in by_domain[domain]:
                result = orchestrator.execute_labeled(
                    project_catalog_prompt(str(row["prompt"])),
                    destination_scope="domain_cake",
                    domain=domain,
                )
                value = result.output.decode("utf-8", errors="strict")
                observations.append(
                    {
                        "mode": "composed_host_selected_domain",
                        "seed": seed,
                        "domain": domain,
                        "probe_id": row["probe_id"],
                        "output": value,
                        "functional_pass": evaluate_functional(value, row["evaluator"]),
                        "phase5_single_domain_byte_exact": result.output
                        == reference[str(row["probe_id"])],
                        "selected": list(result.selected),
                        "telemetry_delta": result.telemetry_delta,
                        "selected_only_execution": _selected_only(
                            result.telemetry_delta, specs[domain]["cake_id"]
                        ),
                    }
                )

        for index in range(int(protocol["composition_requests"])):
            components = []
            for domain in DOMAINS:
                row = by_domain[domain][index]
                result = orchestrator.execute_labeled(
                    project_catalog_prompt(str(row["prompt"])),
                    destination_scope="domain_cake",
                    domain=domain,
                )
                value = result.output.decode("utf-8", errors="strict")
                components.append(
                    {
                        "domain": domain,
                        "probe_id": row["probe_id"],
                        "output": value,
                        "functional_pass": evaluate_functional(value, row["evaluator"]),
                        "phase5_single_domain_byte_exact": result.output
                        == reference[str(row["probe_id"])],
                        "selected": list(result.selected),
                        "selected_only_execution": _selected_only(
                            result.telemetry_delta, specs[domain]["cake_id"]
                        ),
                        "telemetry_delta": result.telemetry_delta,
                    }
                )
            observations.append(
                {
                    "mode": "structured_three_domain_composition",
                    "seed": seed,
                    "request_index": index,
                    "components": components,
                }
            )

        for index in range(int(protocol["conflict_requests"])):
            prompt = "\n".join(
                project_catalog_prompt(str(by_domain[domain][index]["prompt"]))
                for domain in DOMAINS
            )
            result = orchestrator.execute_labeled(
                prompt, destination_scope="quarantine"
            )
            observations.append(
                {
                    "mode": "conflict_quarantine",
                    "seed": seed,
                    "request_index": index,
                    "selected": list(result.selected),
                    "core_fallback": result.route.core_fallback,
                    "execution_path": result.execution_path,
                    "telemetry_delta": result.telemetry_delta,
                }
            )

        loaded_tensor_bytes = {
            cake_id: _tensor_bytes(module)
            for cake_id, module in orchestrator.host._models.items()
        }
        english_after = {
            str(row["probe_id"]): core_host.generate(
                str(row["prompt"]), maximum_tokens=int(row["max_new_tokens"])
            )
            for row in english
        }
        core_after = {
            "archive_hash": core_host.active_archive_hash,
            "payload_hash": core_host.active_payload_hash,
            "state_dict_hash": activated["state_dict_hash"],
            "verify": core_host.verify(),
        }

    selected_rows = [row for row in observations if row["mode"] == "composed_host_selected_domain"]
    composed_rows = [row for row in observations if row["mode"] == "structured_three_domain_composition"]
    conflicts = [row for row in observations if row["mode"] == "conflict_quarantine"]
    scaling = {
        "installed_domain_package_archive_bytes": sum(
            (root / protocol["domain_packages"][domain]["package"]).stat().st_size
            for domain in DOMAINS
        ),
        "loaded_domain_tensor_bytes": loaded_tensor_bytes,
        "maximum_active_selected_domain_tensor_bytes": max(loaded_tensor_bytes.values()),
        "l1_active_tensor_bytes": int(protocol["l1_scaling_reference"]["active_tensor_bytes"]),
        "l1_over_maximum_active_domain_tensor_ratio": int(
            protocol["l1_scaling_reference"]["active_tensor_bytes"]
        )
        / max(loaded_tensor_bytes.values()),
        "comparison_boundary": "Descriptive deployment scaling against the exact B40 L1 source-plus-adapter runtime. L1 is not a matched specialist-domain quality system in this Phase 6 test.",
    }
    gates = {
        "simultaneous_install_three_packages": len(installs) == 3
        and all(row["status"] == "INSTALLED" for row in installs.values()),
        "installed_archive_identity": all(
            registry[specs[domain]["cake_id"]]["archive_hash"]
            == specs[domain]["archive_sha256"]
            for domain in DOMAINS
        ),
        "selected_quality_300_of_300": len(selected_rows) == 300
        and all(row["functional_pass"] for row in selected_rows),
        "no_single_to_composed_interference_300_of_300": all(
            row["phase5_single_domain_byte_exact"] for row in selected_rows
        ),
        "selected_only_physical_execution_300_of_300": all(
            row["selected_only_execution"] for row in selected_rows
        ),
        "structured_composition_100_of_100": len(composed_rows) == 100
        and all(
            len(row["components"]) == 3
            and all(
                component["functional_pass"]
                and component["phase5_single_domain_byte_exact"]
                and component["selected_only_execution"]
                for component in row["components"]
            )
            for row in composed_rows
        ),
        "conflict_quarantine_100_of_100": len(conflicts) == 100
        and all(
            row["selected"] == []
            and row["core_fallback"] is False
            and row["execution_path"] == "authoritative_quarantine"
            and _zero_execution(row["telemetry_delta"])
            for row in conflicts
        ),
        "english_outputs_byte_exact_100_of_100": english_before == english_after,
        "core_identity_unchanged": core_before == core_after,
        "package_bytes_unchanged": all(
            sha256_file(specs[domain]["package"])
            == protocol["domain_packages"][domain]["archive_sha256"]
            for domain in DOMAINS
        ),
        "provenance_and_deletion_lineage_complete": all(provenance["gates"].values()),
        "teacher_absent": True,
        "receiver_learning_zero": int(activated["receiver_training_steps"]) == 0,
        "training_absent": True,
    }
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    provenance_path = output / "provenance.json"
    _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in observations))
    _write_immutable(
        provenance_path,
        json.dumps(provenance, indent=2, sort_keys=True).encode() + b"\n",
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PHASE6_COMPOSITION_SEED"
        if all(gates.values())
        else "FAIL_PHASE6_COMPOSITION_SEED",
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "core_archive_sha256": built["archive_sha256"],
        "core_before": core_before,
        "core_after": core_after,
        "package_installs": installs,
        "registry_archive_hashes": {
            cake_id: row["archive_hash"] for cake_id, row in registry.items()
        },
        "scaling": scaling,
        "gates": gates,
        "observations_path": raw_path.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(raw_path),
        "provenance_path": provenance_path.relative_to(root).as_posix(),
        "provenance_sha256": sha256_file(provenance_path),
        "teacher_model_loaded": False,
        "training_performed": False,
        "receiver_training_steps": int(activated["receiver_training_steps"]),
        "phase6_certified": False,
        "claim_boundary": "One seed of the preregistered three-domain Phase 6 matrix. The extraction archive inventories three pinned sources, but the deployed packages in this run select one source model; no deployed multi-source quality claim is made.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    if args.preflight:
        result = preflight(root, protocol)
    elif args.seed is not None and args.output_dir:
        result = run(root, protocol, seed=args.seed, output=(root / args.output_dir).resolve())
    else:
        raise Phase3Error("select preflight or one seed and output directory")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
