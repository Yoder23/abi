"""Open and evaluate the preregistered ABI Moonshot final test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from .hf_extraction import (
    HuggingFaceCausalSource,
    evaluate_output,
    load_probe_catalog,
    run_probe_catalog,
)
from .layercake_host import strip_source_chat_template
from .layercake_host_runtime import CAPABILITY_TO_ROUTE
from .layercake_product_host import (
    LayerCakeProductHost,
    ProductHostError,
    verify_domain_package,
)


PROTOCOL_FORMATS = {
    "abi-layercake-moonshot-final-test-protocol/1",
    "abi-layercake-moonshot-final-test-protocol/2",
}
SOURCE_EVIDENCE_FORMAT = "abi-source-final-test-evidence/1"
FINAL_EVIDENCE_FORMAT = "abi-layercake-moonshot-final-test-evidence/1"
FINAL_EVIDENCE_FORMAT_V2 = "abi-layercake-moonshot-final-test-evidence/2"
EXPECTED_LAYERCAKE_COMMIT = "04cf2927a16fba686cd640e18a78708e5658bbda"


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductHostError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProductHostError(f"JSON must be an object: {path}")
    return value


def _claim_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    claimed = payload.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha(payload):
        raise ProductHostError("evidence claim hash mismatch")
    return claimed


def _repository_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProductHostError("final-test path escapes ABI repository") from exc
    if not path.is_file():
        raise ProductHostError(f"final-test prerequisite is missing: {relative}")
    return path


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "clean": not bool(status.strip()),
        "porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    root = path.parent
    protocol = _read(path)
    if (
        protocol.get("format") not in PROTOCOL_FORMATS
        or protocol.get("status")
        != (
            "PREREGISTERED_AFTER_ALL_PREREQUISITES_"
            "BEFORE_FINAL_TEST_SOURCE_INFERENCE"
        )
        or protocol.get("final_test_open_authorized") is not True
    ):
        raise ProductHostError("final-test protocol is invalid")
    catalog = protocol["catalog"]
    catalog_path = _repository_file(root, catalog["path"])
    if _sha256_file(catalog_path) != catalog["sha256"]:
        raise ProductHostError("final-test catalog changed")
    for specification in protocol["prerequisites"].values():
        prerequisite = _repository_file(root, specification["path"])
        if _sha256_file(prerequisite) != specification["file_sha256"]:
            raise ProductHostError("pre-final prerequisite changed")
        if prerequisite.suffix == ".json":
            evidence = _read(prerequisite)
            if (
                evidence.get("status") != "PASS"
                or _claim_hash(evidence) != specification["evidence_sha256"]
                or evidence.get("final_test_accessed") is not False
            ):
                raise ProductHostError("pre-final evidence gate failed")
    selected: set[str] = set()
    for source in protocol["sources"]:
        capabilities = set(source["capabilities"])
        if selected & capabilities:
            raise ProductHostError("final-test capability source is ambiguous")
        selected.update(capabilities)
    expected = set(CAPABILITY_TO_ROUTE) | {
        "periodic_table",
        "independence_days",
        "python_generation",
    }
    if selected != expected:
        raise ProductHostError("final-test capability mapping is incomplete")
    return root, protocol


def run_source_final_test(
    *,
    protocol_path: str | Path,
    source_id: str,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    matches = [
        row for row in protocol["sources"] if row["id"] == source_id
    ]
    if len(matches) != 1:
        raise ProductHostError("final-test source identifier is invalid")
    specification = matches[0]
    output_path = (root / specification["output"]).resolve()
    if output_path.exists():
        raise ProductHostError(
            f"source final-test evidence is immutable: {output_path}"
        )
    catalog_path = _repository_file(root, protocol["catalog"]["path"])
    catalog = load_probe_catalog(catalog_path)
    capabilities = set(specification["capabilities"])
    selected_catalog = {
        **catalog,
        "probes": [
            probe
            for probe in catalog["probes"]
            if probe["split"] == "final_test"
            and probe["capability"] in capabilities
        ],
    }
    if len(selected_catalog["probes"]) != 100 * len(capabilities):
        raise ProductHostError("source final-test probe depth is incomplete")
    started = time.perf_counter()
    source = HuggingFaceCausalSource(
        specification["model"],
        revision=specification["revision"],
        license_id=specification["license"],
        device=specification["device"],
        local_files_only=True,
        trust_remote_code=False,
        use_chat_template=True,
    )
    if (
        source.source_manifest["source_manifest_sha256"]
        != specification["source_manifest_sha256"]
        or source.source_manifest["revision"] != specification["revision"]
    ):
        raise ProductHostError("frozen source identity changed")
    inference_started = time.perf_counter()
    records, results = run_probe_catalog(
        source,
        selected_catalog,
        batch_size=int(specification["batch_size"]),
    )
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, Any] = {}
    for capability in sorted(capabilities):
        selected_results = [
            row for row in results if row["capability"] == capability
        ]
        metrics[capability] = {
            "observations": len(selected_results),
            "passes": sum(bool(row["passed"]) for row in selected_results),
            "pass_rate": (
                sum(bool(row["passed"]) for row in selected_results)
                / len(selected_results)
            ),
        }
    evidence: dict[str, Any] = {
        "format": SOURCE_EVIDENCE_FORMAT,
        "status": "COMPLETE",
        "artifact_role": "final_test_evaluation_only",
        "admissible_for_training": False,
        "admissible_for_budget_or_checkpoint_selection": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "split": "final_test",
        },
        "source_id": source_id,
        "source_manifest": source.source_manifest,
        "capabilities": sorted(capabilities),
        "capability_metrics": metrics,
        "observation_count": len(records),
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
        "final_test_accessed": True,
        "candidate_loaded_in_source_process": False,
        "claim_boundary": (
            "Frozen-source final-test evidence only. The payload is forbidden "
            "for training, selection, budgets, or repair."
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


def _final_rows(
    root: Path,
    protocol: Mapping[str, Any],
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_path = _repository_file(root, protocol["catalog"]["path"])
    catalog = load_probe_catalog(catalog_path)
    probes = {
        f"{catalog['catalog_id']}:{probe['probe_id']}": probe
        for probe in catalog["probes"]
        if probe["split"] == "final_test"
    }
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for specification in protocol["sources"]:
        source_path = _repository_file(root, specification["output"])
        source = _read(source_path)
        if (
            source.get("format") != SOURCE_EVIDENCE_FORMAT
            or source.get("status") != "COMPLETE"
            or source.get("artifact_role") != "final_test_evaluation_only"
            or source.get("admissible_for_training") is not False
            or source.get(
                "admissible_for_budget_or_checkpoint_selection"
            )
            is not False
            or source.get("final_test_accessed") is not True
            or source.get("candidate_loaded_in_source_process") is not False
            or source.get("protocol", {}).get("sha256")
            != _sha256_file(protocol_path)
            or source.get("source_manifest", {}).get(
                "source_manifest_sha256"
            )
            != specification["source_manifest_sha256"]
        ):
            raise ProductHostError("source final-test evidence is invalid")
        _claim_hash(source)
        results = {
            row["record_id"]: row for row in source["probe_results"]
        }
        for record in source["records"]:
            result = results.get(record["record_id"])
            probe = probes.get(record["provenance"])
            if (
                result is None
                or probe is None
                or record["split"] != "final_test"
                or probe["evaluator"] != result["evaluator"]
                or record["capability"]
                not in specification["capabilities"]
            ):
                raise ProductHostError("source final-test row is unbound")
            rows.append(
                {
                    "source_id": specification["id"],
                    "record_id": record["record_id"],
                    "probe_id": result["probe_id"],
                    "provenance": record["provenance"],
                    "destination_scope": record["destination_scope"],
                    "domain": record["domain"],
                    "capability": record["capability"],
                    "prompt": strip_source_chat_template(record["prompt"]),
                    "evaluator": result["evaluator"],
                    "max_new_tokens": int(probe["max_new_tokens"]),
                    "source_output_sha256": record["output_sha256"],
                    "source_passed": bool(result["passed"]),
                    "source_score": float(result["score"]),
                }
            )
        sources.append(
            {
                "source_id": specification["id"],
                "path": str(source_path),
                "file_sha256": _sha256_file(source_path),
                "evidence_sha256": source["evidence_sha256"],
                "source_manifest_sha256": specification[
                    "source_manifest_sha256"
                ],
                "observation_count": source["observation_count"],
                "accounting": source["accounting"],
            }
        )
    if len(rows) != 1700 or len({row["record_id"] for row in rows}) != 1700:
        raise ProductHostError("final-test source rows are incomplete or duplicate")
    return sorted(rows, key=lambda row: row["record_id"]), sources


def evaluate_locked_candidate(
    *,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    root, protocol = _load_protocol(protocol_path)
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ProductHostError(f"final evidence is immutable: {output_path}")
    rows, sources = _final_rows(root, protocol, protocol_path)
    candidate = protocol["locked_candidate"]
    artifact = (root / candidate["english_artifact"]).resolve()
    metadata = _read(artifact / "metadata.json")
    if (
        metadata["host"]["deployment_manifest_sha256"]
        != candidate["host_manifest_sha256"]
        or metadata["runtime"]["graph_sha256"]
        != candidate["runtime_graph_sha256"]
        or (
            "metadata_file_sha256" in candidate
            and _sha256_file(artifact / "metadata.json")
            != candidate["metadata_file_sha256"]
        )
        or (
            "symbolic_surface_sha256" in candidate
            and metadata["symbolic_surface"]["sha256"]
            != candidate["symbolic_surface_sha256"]
        )
        or _sha256_file(Path(__file__).with_name("layercake_host_runtime.py"))
        != candidate["runtime_runner_sha256"]
        or (
            "symbolic_runtime_sha256" in candidate
            and _sha256_file(Path(__file__).with_name("symbolic_runtime.py"))
            != candidate["symbolic_runtime_sha256"]
        )
        or (
            "product_host_sha256" in candidate
            and _sha256_file(Path(__file__).with_name("layercake_product_host.py"))
            != candidate["product_host_sha256"]
        )
        or (
            "domain_worker_sha256" in candidate
            and _sha256_file(Path(__file__).with_name("layercake_domain_worker.py"))
            != candidate["domain_worker_sha256"]
        )
        or (
            "final_test_runner_sha256" in candidate
            and _sha256_file(Path(__file__))
            != candidate["final_test_runner_sha256"]
        )
    ):
        raise ProductHostError("locked final candidate changed")
    layercake_root = (
        root
        / json.loads(
            (
                root / "COMBINED_LAYERCAKE_HOST_CERTIFICATION_PROTOCOL.json"
            ).read_text(encoding="utf-8")
        )["sealed_layercake"]["relative_root"]
    ).resolve()
    before = _git_state(layercake_root)
    if before["commit"] != EXPECTED_LAYERCAKE_COMMIT or not before["clean"]:
        raise ProductHostError("sealed LayerCake is not pristine")

    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    package_map = candidate["packages"]
    with tempfile.TemporaryDirectory(prefix="abi-final-host-") as registry:
        with LayerCakeProductHost(
            english_artifact=artifact,
            layercake_root=layercake_root,
            registry_root=registry,
            threads=16,
        ) as host:
            installs = {}
            for domain, package in package_map.items():
                verified = verify_domain_package(
                    root / package["path"],
                    root / package["public_key"],
                )
                if verified["archive_sha256"] != package["archive_sha256"]:
                    raise ProductHostError("final package identity changed")
                installs[domain] = host.install(
                    root / package["path"],
                    root / package["public_key"],
                )
            for index, row in enumerate(rows, start=1):
                request_started = time.perf_counter()
                if row["destination_scope"] == "english_core":
                    result = host.generate(
                        row["prompt"],
                        max_new_tokens=row["max_new_tokens"],
                    )
                    expected_route = CAPABILITY_TO_ROUTE[row["capability"]]
                    route = result.evidence["route"]
                    route_correct = route == expected_route
                    inactive_calls = 0
                else:
                    result = host.generate(
                        row["prompt"],
                        cake_id=package_map[row["domain"]]["cake_id"],
                        max_new_tokens=96,
                        domain_device="cpu",
                    )
                    expected_route = None
                    route = None
                    route_correct = True
                    inactive_calls = sum(
                        sum(values.values())
                        for cake_id, values in result.evidence[
                            "telemetry_delta"
                        ].items()
                        if cake_id != package_map[row["domain"]]["cake_id"]
                    )
                passed, score = evaluate_output(
                    result.output, row["evaluator"]
                )
                try:
                    result.output.encode("utf-8").decode("utf-8")
                    invalid_utf8 = False
                except UnicodeError:
                    invalid_utf8 = True
                observations.append(
                    {
                        **row,
                        "layercake_output": result.output,
                        "layercake_output_sha256": result.output_sha256,
                        "layercake_passed": passed,
                        "layercake_score": score,
                        "route": route,
                        "expected_route": expected_route,
                        "route_correct": route_correct,
                        "inactive_domain_execution_calls": inactive_calls,
                        "invalid_utf8": invalid_utf8,
                        "latency_seconds": time.perf_counter()
                        - request_started,
                    }
                )
                if index % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "evaluated": index,
                                "total": len(rows),
                                "layercake_passes": sum(
                                    item["layercake_passed"]
                                    for item in observations
                                ),
                            }
                        ),
                        flush=True,
                    )
    after = _git_state(layercake_root)
    capability_metrics: dict[str, Any] = {}
    for capability in sorted(
        {row["capability"] for row in observations}
    ):
        selected = [
            row for row in observations if row["capability"] == capability
        ]
        source_passes = sum(row["source_passed"] for row in selected)
        layercake_passes = sum(row["layercake_passed"] for row in selected)
        regressions = sum(
            row["source_passed"] and not row["layercake_passed"]
            for row in selected
        )
        capability_metrics[capability] = {
            "destination_scope": selected[0]["destination_scope"],
            "domain": selected[0]["domain"],
            "observations": len(selected),
            "source_passes": source_passes,
            "layercake_passes": layercake_passes,
            "source_passing_regressions": regressions,
            "route_accuracy": (
                sum(row["route_correct"] for row in selected) / len(selected)
            ),
            "inactive_domain_execution_calls": sum(
                row["inactive_domain_execution_calls"] for row in selected
            ),
            "invalid_utf8_outputs": sum(
                row["invalid_utf8"] for row in selected
            ),
            "status": (
                "PASS"
                if (
                    len(selected) == 100
                    and layercake_passes == 100
                    and layercake_passes >= source_passes
                    and regressions == 0
                    and all(row["route_correct"] for row in selected)
                    and not any(
                        row["inactive_domain_execution_calls"]
                        for row in selected
                    )
                    and not any(row["invalid_utf8"] for row in selected)
                )
                else "FAIL"
            ),
        }
    gates = {
        "observation_count_exact": len(observations) == 1700,
        "capability_count_exact": len(capability_metrics) == 17,
        "all_capabilities_pass": all(
            row["status"] == "PASS"
            for row in capability_metrics.values()
        ),
        "teacher_absent_at_candidate_inference": True,
        "source_transformer_blocks_retained_zero": True,
        "source_processes_separate_from_candidate_process": True,
        "sealed_layercake_unchanged": after == before,
    }
    evidence: dict[str, Any] = {
        "format": (
            FINAL_EVIDENCE_FORMAT_V2
            if protocol["format"].endswith("/2")
            else FINAL_EVIDENCE_FORMAT
        ),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "locked_candidate": candidate,
        "source_final_test_evidence": sources,
        "installed_packages": {
            domain: {
                key: value
                for key, value in install.items()
                if key not in {"package_path", "public_key_path"}
            }
            for domain, install in installs.items()
        },
        "observation_count": len(observations),
        "capability_metrics": capability_metrics,
        "layercake_passes": sum(
            row["layercake_passed"] for row in observations
        ),
        "source_passes": sum(row["source_passed"] for row in observations),
        "source_passing_regressions": sum(
            row["source_passed"] and not row["layercake_passed"]
            for row in observations
        ),
        "invalid_utf8_outputs": sum(
            row["invalid_utf8"] for row in observations
        ),
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
        "gates": gates,
        "sealed_layercake": {
            "root": str(layercake_root),
            "before": before,
            "after": after,
            "unchanged": after == before,
        },
        "teacher_present_at_candidate_inference": False,
        "source_transformer_blocks_retained": 0,
        "final_test_accessed": True,
        "mathematics_branch_status": "CLOSED_FAILED_NOT_PROMOTED",
        "claim_boundary": protocol["claim_boundary"],
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--protocol", required=True)
    source.add_argument("--source", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source":
        result = run_source_final_test(
            protocol_path=args.protocol,
            source_id=args.source,
        )
    else:
        result = evaluate_locked_candidate(
            protocol_path=args.protocol,
            output_path=args.output,
        )
    display = {
        key: result[key]
        for key in (
            "status",
            "evidence_sha256",
            "observation_count",
            "capability_metrics",
            "layercake_passes",
            "source_passes",
            "source_passing_regressions",
            "gates",
        )
        if key in result
    }
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
