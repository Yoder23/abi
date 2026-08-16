"""Independent read-only verifier for the Phase 5 selective product matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase5_selective_product import (
    DOMAINS,
    FORMAT as PRODUCT_FORMAT,
    RESULT_FORMAT,
    SEEDS,
    SYSTEMS,
    _domain_identity,
    _domain_rows,
    _explicit_product_abstention,
    _zero_execution,
    load_protocol as load_product_protocol,
)
from .capability_compiler_phase5_construct_screen import is_explicit_abstention


FORMAT = "abi-capability-compiler-phase5-independent-verify/1"
RESULT_FORMAT_V = "abi-capability-compiler-phase5-independent-verify-result/1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise Phase3Error(f"invalid Phase 5 JSONL: {path}") from error


def _evidence_hash_valid(result: Mapping[str, Any]) -> bool:
    document = copy.deepcopy(dict(result))
    declared = document.pop("evidence_sha256", None)
    return declared == hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _catalog_index(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _domain_rows(
        root / protocol["domain_catalog"],
        split=str(protocol["catalog_split"]),
        per_domain=int(protocol["per_domain"]),
    )
    return {str(row["probe_id"]): row for row in rows}


def _cake_ids(root: Path, protocol: Mapping[str, Any]) -> dict[str, str]:
    return {
        domain: _domain_identity(root / protocol["domain_packages"][domain]["package"])[0]
        for domain in DOMAINS
    }


def verify_result_document(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    product_protocol_sha: str,
    system: str,
    seed: int,
    result: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    catalog = _catalog_index(root, protocol)
    expected_protocol_sha = product_protocol_sha
    if system == "ABI":
        expected_protocol_sha = str(protocol["base_protocol_sha256"])
    common = {
        "result_format": result.get("format") == RESULT_FORMAT,
        "system_identity": result.get("system") == system,
        "seed_identity": int(result.get("seed", -1)) == seed,
        "protocol_identity": result.get("protocol_sha256") == expected_protocol_sha,
        "result_evidence_hash": _evidence_hash_valid(result),
        "declared_gates_pass": bool(result.get("gates"))
        and all(bool(value) for value in result["gates"].values()),
        "training_absent": result.get("training_performed") is False,
    }
    if system == "ABI":
        grouped = {
            mode: [row for row in observations if row.get("mode") == mode]
            for mode in (
                "domain_missing_before_install",
                "selected_domain_installed",
                "label_spoof",
                "cross_domain_uninstalled",
                "quarantine",
            )
        }
        selected = grouped["selected_domain_installed"]
        missing = grouped["domain_missing_before_install"]
        spoof = grouped["label_spoof"]
        cross = grouped["cross_domain_uninstalled"]
        quarantine = grouped["quarantine"]
        cakes = _cake_ids(root, protocol)
        selected_passes = sum(
            str(row.get("probe_id")) in catalog
            and row.get("domain") == catalog[str(row["probe_id"])]["domain"]
            and evaluate_functional(
                str(row.get("output", "")),
                catalog[str(row["probe_id"])]["evaluator"],
            )
            for row in selected
        )
        missing_abstentions = sum(
            _explicit_product_abstention(str(row.get("output", "")))
            for row in missing
        )
        gates = {
            **common,
            "row_mode_depth": len(observations) == 780
            and {mode: len(rows) for mode, rows in grouped.items()}
            == {
                "domain_missing_before_install": 300,
                "selected_domain_installed": 300,
                "label_spoof": 60,
                "cross_domain_uninstalled": 60,
                "quarantine": 60,
            },
            "selected_probe_identity": len(selected) == 300
            and len({str(row["probe_id"]) for row in selected}) == 300,
            "selected_function_recomputed": selected_passes == 300
            and int(result.get("selected_functional_passes", -1)) == selected_passes,
            "missing_abstention_recomputed": missing_abstentions == 300
            and int(result.get("missing_explicit_abstentions", -1))
            == missing_abstentions,
            "missing_zero_execution_recomputed": all(
                row.get("core_fallback") is False
                and row.get("selected") == []
                and _zero_execution(row.get("telemetry_delta", {}))
                for row in missing
            ),
            "spoof_outer_selection_recomputed": all(
                row.get("selected") == [cakes[str(row.get("domain"))]]
                and row.get("core_fallback") is False
                and sum(
                    int(values.get("prefill_calls", 0))
                    for values in row.get("telemetry_delta", {}).values()
                )
                == 1
                for row in spoof
            ),
            "cross_domain_zero_execution_recomputed": all(
                row.get("selected") == []
                and row.get("core_fallback") is False
                and row.get("execution_path") == "authoritative_domain_missing"
                and _zero_execution(row.get("telemetry_delta", {}))
                for row in cross
            ),
            "quarantine_zero_execution_recomputed": all(
                row.get("selected") == []
                and row.get("core_fallback") is False
                and row.get("execution_path") == "authoritative_quarantine"
                and _zero_execution(row.get("telemetry_delta", {}))
                for row in quarantine
            ),
            "core_identity_recomputed": result.get("core_before")
            == result.get("core_after")
            and result.get("core_archive_sha256")
            == protocol["systems"]["ABI"][str(seed)]["archive_sha256"],
            "lifecycle_recomputed": len(result.get("lifecycle", ())) == 3
            and {row.get("domain") for row in result["lifecycle"]} == set(DOMAINS)
            and all(
                row.get("missing_before_install_100_of_100") is True
                and row.get("install_status") == "INSTALLED"
                and row.get("verify_status") == "PASS"
                and row.get("remove_status") == "REMOVED"
                and row.get("missing_after_remove_100_of_100") is True
                and row.get("reinstall_status") == "INSTALLED"
                and row.get("restored_outputs_byte_exact_100_of_100") is True
                and row.get("archive_unchanged") is True
                for row in result["lifecycle"]
            ),
            "core_handler_zero": int(result.get("core_handler_calls_for_specialist_rows", -1)) == 0,
            "teacher_absent": result.get("teacher_model_loaded") is False,
            "receiver_learning_zero": int(result.get("receiver_training_steps", -1)) == 0,
        }
        return {
            "system": system,
            "seed": seed,
            "gates": gates,
            "specialist_functional_passes": selected_passes,
            "explicit_abstentions": missing_abstentions,
        }

    if len(observations) != 300:
        functional = -1
        abstentions = -1
        domain_summary: dict[str, Any] = {}
    else:
        functional = sum(
            str(row.get("probe_id")) in catalog
            and evaluate_functional(
                str(row.get("output", "")),
                catalog[str(row["probe_id"])]["evaluator"],
            )
            for row in observations
        )
        abstentions = sum(
            is_explicit_abstention(str(row.get("output", "")))
            for row in observations
        )
        domain_summary = {
            domain: {
                "observations": sum(row.get("domain") == domain for row in observations),
                "functional_passes": sum(
                    row.get("domain") == domain
                    and evaluate_functional(
                        str(row.get("output", "")),
                        catalog[str(row["probe_id"])]["evaluator"],
                    )
                    for row in observations
                ),
                "explicit_abstentions": sum(
                    row.get("domain") == domain
                    and is_explicit_abstention(str(row.get("output", "")))
                    for row in observations
                ),
            }
            for domain in DOMAINS
        }
    gates = {
        **common,
        "row_depth_and_identity": len(observations) == 300
        and len({str(row.get("probe_id")) for row in observations}) == 300
        and set(str(row.get("probe_id")) for row in observations) == set(catalog),
        "functional_recomputed": functional
        == int(result.get("specialist_functional_passes", -2)),
        "abstention_recomputed": abstentions
        == int(result.get("explicit_abstentions", -2)),
        "per_domain_recomputed": domain_summary == result.get("per_domain"),
        "source_base_boundary": result.get("source_base_present_at_inference")
        is (system == "L1"),
        "teacher_query_absent": result.get("teacher_query_performed") is False,
    }
    return {
        "system": system,
        "seed": seed,
        "gates": gates,
        "specialist_functional_passes": functional,
        "explicit_abstentions": abstentions,
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
        or protocol.get("status") != "PREREGISTERED_PHASE5_INDEPENDENT_VERIFICATION"
        or protocol.get("systems") != list(SYSTEMS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or int(protocol.get("minimum_adversarial_tests", 0)) < 12
    ):
        raise Phase3Error("Phase 5 verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 5 verification binding changed: {relative}")
    return protocol, sha256_file(path)


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("Phase 5 independent verification output exists")
    product_path = root / protocol["product_protocol"]
    product, product_sha = load_product_protocol(root, product_path)
    rows: list[dict[str, Any]] = []
    evidence_hashes: dict[str, dict[str, str]] = {}
    for system in SYSTEMS:
        for seed in SEEDS:
            spec = protocol["evidence"][system][str(seed)]
            result_path = root / spec["result"]
            observations_path = root / spec["observations"]
            result = _json(result_path)
            observations = _read_jsonl(observations_path)
            recomputed = verify_result_document(
                root=root,
                protocol=product,
                product_protocol_sha=product_sha,
                system=system,
                seed=seed,
                result=result,
                observations=observations,
            )
            if not all(recomputed["gates"].values()):
                failed = [key for key, value in recomputed["gates"].items() if not value]
                raise Phase3Error(
                    f"Phase 5 independent row failure {system}/{seed}: {failed}"
                )
            rows.append(recomputed)
            evidence_hashes[f"{system}/{seed}"] = {
                "result_sha256": sha256_file(result_path),
                "observations_sha256": sha256_file(observations_path),
            }
    junit_path = root / protocol["adversarial_junit"]
    junit = _junit(junit_path)
    by_system = {
        system: [row for row in rows if row["system"] == system]
        for system in SYSTEMS
    }
    gates = {
        "nine_rows_independently_recomputed": len(rows) == 9,
        "all_row_gates_pass": all(all(row["gates"].values()) for row in rows),
        "abi_selected_recovery_all_seed": [
            row["specialist_functional_passes"] for row in by_system["ABI"]
        ]
        == [300, 300, 300],
        "abi_missing_abstention_all_seed": [
            row["explicit_abstentions"] for row in by_system["ABI"]
        ]
        == [300, 300, 300],
        "l1_residual_control_complete": len(by_system["L1"]) == 3,
        "d0_residual_control_complete": len(by_system["D0"]) == 3,
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
        "format": RESULT_FORMAT_V,
        "status": "PASS_INDEPENDENTLY_VERIFIED_PHASE5_BOUNDED_EXCLUSION"
        if passed
        else "FAIL_INDEPENDENTLY_VERIFIED_PHASE5_BOUNDED_EXCLUSION",
        "protocol_sha256": protocol_sha,
        "product_protocol_sha256": product_sha,
        "evidence_hashes": evidence_hashes,
        "aggregates": {
            system: {
                "specialist_functional_passes_per_seed": [
                    row["specialist_functional_passes"] for row in by_system[system]
                ],
                "explicit_abstentions_per_seed": [
                    row["explicit_abstentions"] for row in by_system[system]
                ],
            }
            for system in SYSTEMS
        },
        "adversarial_junit": {
            "path": protocol["adversarial_junit"],
            "sha256": sha256_file(junit_path),
            **junit,
        },
        "gates": gates,
        "phase5_certified": passed,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_query_performed": False,
        "claim_boundary": "Independent read-only recomputation of the bounded three-domain Phase 5 matrix. Behavioral exclusion is certified at the immutable authoritative-label control plane; latent semantic purity, exhaustive domain discovery, completed human review, Phase 6, release, and universal ABI superiority remain unproved.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
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
