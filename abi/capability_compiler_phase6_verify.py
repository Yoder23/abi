"""Independent read-only verifier for the bounded Phase 6 composition matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase5_selective_product import _domain_rows
from .capability_compiler_phase6_composition import (
    DOMAINS,
    RESULT_FORMAT,
    SEEDS,
    load_protocol as load_product_protocol,
)
from .capability_pipeline import read_extraction_bundle


FORMAT = "abi-capability-compiler-phase6-independent-verify/1"
VERIFY_RESULT_FORMAT = "abi-capability-compiler-phase6-independent-verify-result/1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise Phase3Error(f"invalid Phase 6 JSONL: {path}") from error


def _evidence_hash_valid(result: Mapping[str, Any]) -> bool:
    document = copy.deepcopy(dict(result))
    declared = document.pop("evidence_sha256", None)
    return declared == hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _zero_execution(delta: Mapping[str, Mapping[str, int]]) -> bool:
    return all(
        int(value) == 0
        for counters in delta.values()
        for value in counters.values()
    )


def _selected_only(
    delta: Mapping[str, Mapping[str, int]], selected_cake_id: str
) -> bool:
    if selected_cake_id not in delta:
        return False
    return all(
        int(counters.get("prefill_calls", 0)) == 1
        if cake_id == selected_cake_id
        else all(int(value) == 0 for value in counters.values())
        for cake_id, counters in delta.items()
    )


def _manifest(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def _phase5_reference(root: Path, protocol: Mapping[str, Any], seed: int) -> dict[str, str]:
    rows = _read_jsonl(root / protocol["phase5_observations"][str(seed)])
    selected = [row for row in rows if row.get("mode") == "selected_domain_installed"]
    return {str(row["probe_id"]): str(row["output"]) for row in selected}


def _provenance_recomputed(
    root: Path, protocol: Mapping[str, Any], provenance: Mapping[str, Any]
) -> bool:
    bundle_path = root / protocol["domain_training_bundle"]
    if sha256_file(bundle_path) != protocol["domain_training_bundle_sha256"]:
        return False
    bundle = read_extraction_bundle(bundle_path)
    source_by_identity = {
        (str(row["model_id"]), str(row["revision"])): row
        for row in bundle["sources"]
    }
    expected_inventory = [
        {
            "source_manifest_sha256": row["source_manifest_sha256"],
            "model_id": row["model_id"],
            "revision": row["revision"],
            "license_id": row["license_id"],
            "revision_is_immutable": row["revision_is_immutable"],
        }
        for row in bundle["sources"]
    ]
    package_protocol = _json(root / protocol["domain_package_protocol"])
    package_specs = {str(row["domain"]): row for row in package_protocol["domains"]}
    passing = {
        str(row["record_id"]): row.get("passed") is True
        for row in bundle["probe_results"]
    }
    all_ids: list[str] = []
    expected_domains: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        specification = package_specs[domain]
        selected_items = [
            row
            for row in bundle["selection"]["selected_items"]
            if row.get("destination_scope") == "domain_cake"
            and row.get("domain") == domain
        ]
        if len(selected_items) != 1:
            return False
        selected = selected_items[0]
        source = source_by_identity.get(
            (str(selected["source_model"]), str(selected["source_model_revision"]))
        )
        if source is None:
            return False
        budget_index = int(specification["minimum_tested_passing_budget_index"])
        budget = bundle["budgets"][budget_index]
        allowed = set(str(value) for value in budget["record_ids"])
        rows = [
            row
            for row in bundle["records"]
            if str(row["record_id"]) in allowed
            and row.get("destination_scope") == "domain_cake"
            and row.get("domain") == domain
            and row.get("split") == "search"
            and row.get("source_model") == selected["source_model"]
            and row.get("source_model_revision") == selected["source_model_revision"]
            and passing.get(str(row["record_id"])) is True
        ]
        record_ids = sorted(str(row["record_id"]) for row in rows)
        package_path = root / specification["package"]
        manifest = _manifest(package_path)
        gates = {
            "selected_record_count": len(rows) == int(specification["training_rows"]),
            "selected_record_ids_unique": len(record_ids) == len(set(record_ids)),
            "teacher_tokens_exact": sum(int(row["teacher_tokens"]) for row in rows)
            == int(specification["teacher_tokens"]),
            "source_identity_exact": all(
                row["source_model"] == source["model_id"]
                and row["source_model_revision"] == source["revision"]
                for row in rows
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
        expected_domains[domain] = {
            "cake_id": manifest["cake_id"],
            "package_archive_sha256": sha256_file(package_path),
            "package_license": manifest["license"],
            "training_budget_id": budget["budget_id"],
            "training_budget_index": budget_index,
            "selected_record_ids": record_ids,
            "selected_record_set_sha256": hashlib.sha256(
                "\n".join(record_ids).encode("ascii")
            ).hexdigest(),
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_model": source["model_id"],
            "source_revision": source["revision"],
            "source_license": source["license_id"],
            "gates": gates,
        }
        all_ids.extend(record_ids)
    selected_sources = sorted(
        {row["source_manifest_sha256"] for row in expected_domains.values()}
    )
    expected_record_map = {
        record_id: expected_domains[domain]["cake_id"]
        for domain in DOMAINS
        for record_id in expected_domains[domain]["selected_record_ids"]
    }
    expected_source_map = {
        source: sorted(
            row["cake_id"]
            for row in expected_domains.values()
            if row["source_manifest_sha256"] == source
        )
        for source in selected_sources
    }
    english = _json(root / protocol["english_phase1_certificate"])["source"]
    return (
        provenance.get("domain_training_bundle_sha256") == sha256_file(bundle_path)
        and provenance.get("bundle_source_inventory") == expected_inventory
        and provenance.get("deployed_selected_source_manifests") == selected_sources
        and provenance.get("deployed_multiple_source_models") is False
        and provenance.get("english_source") == english
        and provenance.get("domains") == expected_domains
        and provenance.get("deletion_lineage", {}).get("record_id_to_artifact")
        == expected_record_map
        and provenance.get("deletion_lineage", {}).get(
            "selected_source_manifest_to_artifacts"
        )
        == expected_source_map
        and len(all_ids) == len(set(all_ids)) == 175
        and all(all(row["gates"].values()) for row in expected_domains.values())
        and provenance.get("gates")
        == {
            "all_domain_lineages_exact": True,
            "selected_domain_records_disjoint": True,
            "bundle_contains_three_pinned_sources": True,
            "deployed_domain_packages_select_one_pinned_source": True,
            "english_source_pinned": True,
            "record_level_deletion_index_complete": True,
        }
    )


def verify_seed_document(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    seed: int,
    result: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _domain_rows(
        root / protocol["domain_catalog"], split="final_test", per_domain=100
    )
    catalog = {str(row["probe_id"]): row for row in rows}
    by_domain = {
        domain: [row for row in rows if row["domain"] == domain]
        for domain in DOMAINS
    }
    cake_ids = {
        domain: _manifest(root / protocol["domain_packages"][domain]["package"])[
            "cake_id"
        ]
        for domain in DOMAINS
    }
    reference = _phase5_reference(root, protocol, seed)
    selected = [
        row for row in observations if row.get("mode") == "composed_host_selected_domain"
    ]
    composed = [
        row for row in observations if row.get("mode") == "structured_three_domain_composition"
    ]
    conflicts = [row for row in observations if row.get("mode") == "conflict_quarantine"]
    selected_outputs = {str(row.get("probe_id")): str(row.get("output", "")) for row in selected}
    selected_functional = sum(
        probe_id in catalog
        and row.get("domain") == catalog[probe_id]["domain"]
        and evaluate_functional(str(row.get("output", "")), catalog[probe_id]["evaluator"])
        for row in selected
        for probe_id in [str(row.get("probe_id"))]
    )
    selected_collapses = sum(
        repetition_collapse_v2(str(row.get("output", ""))) for row in selected
    )
    expected_component_keys = {
        (index, domain): str(by_domain[domain][index]["probe_id"])
        for index in range(100)
        for domain in DOMAINS
    }
    components = [
        (int(row.get("request_index", -1)), component)
        for row in composed
        for component in row.get("components", ())
    ]
    component_functional = sum(
        probe_id in catalog
        and component.get("domain") == catalog[probe_id]["domain"]
        and evaluate_functional(
            str(component.get("output", "")), catalog[probe_id]["evaluator"]
        )
        for _, component in components
        for probe_id in [str(component.get("probe_id"))]
    )
    component_collapses = sum(
        repetition_collapse_v2(str(component.get("output", "")))
        for _, component in components
    )
    expected_registry = {
        cake_ids[domain]: protocol["domain_packages"][domain]["archive_sha256"]
        for domain in DOMAINS
    }
    gates = {
        "result_format_and_status": result.get("format") == RESULT_FORMAT
        and result.get("status") == "PASS_PHASE6_COMPOSITION_SEED",
        "seed_and_protocol_identity": int(result.get("seed", -1)) == seed
        and result.get("protocol_sha256") == protocol_sha,
        "result_evidence_hash": _evidence_hash_valid(result),
        "declared_gates_pass": bool(result.get("gates"))
        and all(value is True for value in result["gates"].values()),
        "raw_row_depth": len(observations) == 500
        and len(selected) == 300
        and len(composed) == 100
        and len(conflicts) == 100,
        "selected_prompt_and_domain_identity": len(selected_outputs) == 300
        and set(selected_outputs) == set(catalog)
        and Counter(str(row.get("domain")) for row in selected)
        == Counter({domain: 100 for domain in DOMAINS}),
        "selected_function_recomputed": selected_functional == 300,
        "selected_phase5_byte_identity_recomputed": selected_outputs == reference,
        "selected_execution_recomputed": all(
            row.get("selected") == [cake_ids[str(row.get("domain"))]]
            and _selected_only(
                row.get("telemetry_delta", {}), cake_ids[str(row.get("domain"))]
            )
            for row in selected
            if str(row.get("domain")) in cake_ids
        )
        and len(selected) == 300,
        "selected_zero_repetition_collapse_v2": selected_collapses == 0,
        "composition_index_and_component_identity": len(components) == 300
        and {int(row.get("request_index", -1)) for row in composed} == set(range(100))
        and all(
            expected_component_keys.get((index, str(component.get("domain"))))
            == str(component.get("probe_id"))
            for index, component in components
        ),
        "composition_function_recomputed": component_functional == 300,
        "composition_byte_identity_recomputed": all(
            str(component.get("output", ""))
            == reference.get(str(component.get("probe_id")))
            == selected_outputs.get(str(component.get("probe_id")))
            for _, component in components
        ),
        "composition_execution_recomputed": all(
            str(component.get("domain")) in cake_ids
            and component.get("selected")
            == [cake_ids[str(component.get("domain"))]]
            and _selected_only(
                component.get("telemetry_delta", {}),
                cake_ids[str(component.get("domain"))],
            )
            for _, component in components
        ),
        "composition_zero_repetition_collapse_v2": component_collapses == 0,
        "conflict_quarantine_recomputed": {
            int(row.get("request_index", -1)) for row in conflicts
        }
        == set(range(100))
        and all(
            row.get("selected") == []
            and row.get("core_fallback") is False
            and row.get("execution_path") == "authoritative_quarantine"
            and _zero_execution(row.get("telemetry_delta", {}))
            for row in conflicts
        ),
        "core_identity_recomputed": result.get("core_before")
        == result.get("core_after")
        and result.get("core_archive_sha256")
        == protocol["systems"]["ABI"][str(seed)]["archive_sha256"]
        and result.get("core_before", {}).get("payload_hash")
        == protocol["systems"]["ABI"][str(seed)]["payload_sha256"],
        "package_registry_identity_recomputed": result.get("registry_archive_hashes")
        == expected_registry
        and all(
            result.get("package_installs", {}).get(domain, {}).get("status")
            == "INSTALLED"
            and result["package_installs"][domain].get("archive_hash")
            == protocol["domain_packages"][domain]["archive_sha256"]
            and sha256_file(root / protocol["domain_packages"][domain]["package"])
            == protocol["domain_packages"][domain]["archive_sha256"]
            for domain in DOMAINS
        ),
        "provenance_recomputed": _provenance_recomputed(root, protocol, provenance),
        "teacher_training_and_receiver_learning_absent": result.get(
            "teacher_model_loaded"
        )
        is False
        and result.get("training_performed") is False
        and int(result.get("receiver_training_steps", -1)) == 0,
    }
    return {
        "seed": seed,
        "gates": gates,
        "selected_outputs": selected_outputs,
        "component_outputs": {
            f"{index}/{component['domain']}": str(component["output"])
            for index, component in components
        },
        "selected_functional_passes": selected_functional,
        "component_functional_passes": component_functional,
        "selected_repetition_collapses_v2": selected_collapses,
        "component_repetition_collapses_v2": component_collapses,
    }


def _junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_PHASE6_INDEPENDENT_VERIFICATION"
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or int(protocol.get("minimum_adversarial_tests", 0)) < 15
    ):
        raise Phase3Error("Phase 6 verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 6 verification binding changed: {relative}")
    return protocol, sha256_file(path)


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, verify_protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("Phase 6 independent verification output exists")
    product, product_sha = load_product_protocol(root, root / protocol["product_protocol"])
    rows: list[dict[str, Any]] = []
    evidence_hashes: dict[str, Any] = {}
    for seed in SEEDS:
        specification = protocol["evidence"][str(seed)]
        result_path = root / specification["result"]
        observations_path = root / specification["observations"]
        provenance_path = root / specification["provenance"]
        result = _json(result_path)
        observations = _read_jsonl(observations_path)
        provenance = _json(provenance_path)
        if (
            result.get("observations_sha256") != sha256_file(observations_path)
            or result.get("provenance_sha256") != sha256_file(provenance_path)
        ):
            raise Phase3Error(f"Phase 6 raw evidence identity failure: {seed}")
        recomputed = verify_seed_document(
            root=root,
            protocol=product,
            protocol_sha=product_sha,
            seed=seed,
            result=result,
            observations=observations,
            provenance=provenance,
        )
        if not all(recomputed["gates"].values()):
            failed = [key for key, value in recomputed["gates"].items() if not value]
            raise Phase3Error(f"Phase 6 independent seed failure {seed}: {failed}")
        rows.append(recomputed)
        evidence_hashes[str(seed)] = {
            "result_sha256": sha256_file(result_path),
            "observations_sha256": sha256_file(observations_path),
            "provenance_sha256": sha256_file(provenance_path),
        }
    junit_path = root / protocol["adversarial_junit"]
    junit = _junit(junit_path)
    gates = {
        "three_independent_host_rows_recomputed": len(rows) == 3,
        "all_seed_gates_pass": all(all(row["gates"].values()) for row in rows),
        "selected_functional_900_of_900": sum(
            row["selected_functional_passes"] for row in rows
        )
        == 900,
        "composition_components_900_of_900": sum(
            row["component_functional_passes"] for row in rows
        )
        == 900,
        "zero_repetition_collapse_v2": sum(
            row["selected_repetition_collapses_v2"]
            + row["component_repetition_collapses_v2"]
            for row in rows
        )
        == 0,
        "independent_host_selected_output_identity": all(
            row["selected_outputs"] == rows[0]["selected_outputs"] for row in rows[1:]
        ),
        "independent_host_composition_output_identity": all(
            row["component_outputs"] == rows[0]["component_outputs"] for row in rows[1:]
        ),
        "three_distinct_core_payloads": len(
            {
                _json(root / protocol["evidence"][str(seed)]["result"])[
                    "core_before"
                ]["payload_hash"]
                for seed in SEEDS
            }
        )
        == 3,
        "adversarial_tests_pass": junit["tests"]
        >= int(protocol["minimum_adversarial_tests"])
        and junit["failures"] == 0
        and junit["errors"] == 0,
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_query_absent": True,
    }
    passed = all(gates.values())
    output.mkdir(parents=True)
    result = {
        "format": VERIFY_RESULT_FORMAT,
        "status": "PASS_INDEPENDENTLY_VERIFIED_PHASE6_BOUNDED_COMPOSITION"
        if passed
        else "FAIL_INDEPENDENTLY_VERIFIED_PHASE6_BOUNDED_COMPOSITION",
        "protocol_sha256": verify_protocol_sha,
        "product_protocol_sha256": product_sha,
        "evidence_hashes": evidence_hashes,
        "aggregates": {
            "host_initializations": 3,
            "simultaneous_packages_per_host": 3,
            "selected_specialist_functional_passes": 900,
            "selected_specialist_observations": 900,
            "structured_composition_requests": 300,
            "structured_composition_components": 900,
            "structured_composition_functional_passes": 900,
            "conflict_quarantines": 300,
            "english_identity_checks": 300,
            "repetition_collapses_v2": 0,
        },
        "adversarial_junit": {
            "path": protocol["adversarial_junit"],
            "sha256": sha256_file(junit_path),
            **junit,
        },
        "gates": gates,
        "phase6_certified": passed,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_query_performed": False,
        "claim_boundary": "Independent read-only certification of bounded three-package composition on three distinct compatible English-core host initializations. It proves exact output portability, selected-only execution, conflict quarantine, and record deletion lineage for the registered artifacts and suites. It does not prove arbitrary-machine portability, deployed multi-source quality, latent knowledge absence, completed Phase 2 human review, release readiness, or universal ABI superiority.",
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
    result = verify(
        root,
        (root / args.protocol).resolve(),
        (root / args.output_dir).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
