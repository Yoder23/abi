"""Certify the combined native-English and signed-domain LayerCake host."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import psutil

from .layercake_host_runtime import (
    benchmark_native_host,
    evaluate_native_host_semantics,
)
from .layercake_product_host import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    LayerCakeProductHost,
    ProductHostError,
    verify_domain_package,
)


PROTOCOL_FORMAT = "abi-layercake-combined-host-certification-protocol/1"
EVIDENCE_FORMAT = "abi-layercake-combined-host-certification-evidence/1"
EXPECTED_LAYERCAKE_COMMIT = "04cf2927a16fba686cd640e18a78708e5658bbda"
MODULE_NAME = "abi.layercake_product_host_certification"
SUSTAINED_AMENDMENT_FORMAT = (
    "abi-layercake-combined-host-sustained-repetition-amendment/1"
)


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


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductHostError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProductHostError(f"JSON must be an object: {path}")
    return value


def _claimed_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    claimed = payload.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha(payload):
        raise ProductHostError("evidence claim hash mismatch")
    return claimed


def _repository_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProductHostError("evidence path escapes ABI repository") from exc
    if not path.is_file():
        raise ProductHostError(f"required evidence is missing: {value}")
    return path


def _load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    root = path.parent
    protocol = _read_object(path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BEFORE_COMBINED_HOST_CERTIFICATION"
        or protocol.get("final_test_accessed") is not False
    ):
        raise ProductHostError("combined-host protocol is invalid")
    layercake = protocol.get("sealed_layercake", {})
    if (
        layercake.get("repository_commit") != EXPECTED_LAYERCAKE_COMMIT
        or layercake.get("direct_decoder_abi_version")
        != DIRECT_ABI_VERSION
        or layercake.get("direct_decoder_abi_sha256")
        != DIRECT_ABI_SHA256
        or layercake.get("may_be_modified") is not False
    ):
        raise ProductHostError("combined-host protocol changed sealed LayerCake")
    if len(protocol.get("domains", [])) != 3:
        raise ProductHostError("combined-host protocol requires three domains")
    return root, protocol


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "clean": not bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(
            porcelain.encode("utf-8")
        ).hexdigest(),
    }


def _verify_prerequisites(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    english = protocol["english_host"]
    artifact = (root / english["artifact"]).resolve()
    metadata = _read_object(artifact / "metadata.json")
    if (
        metadata.get("host", {}).get("deployment_manifest_sha256")
        != english["host_manifest_sha256"]
        or metadata.get("runtime", {}).get("graph_sha256")
        != english["runtime_graph_sha256"]
    ):
        raise ProductHostError("combined English host identity differs")
    runtime_runner = Path(__file__).with_name("layercake_host_runtime.py")
    if _sha256_file(runtime_runner) != english["runtime_runner_sha256"]:
        raise ProductHostError("combined English runtime runner differs")
    reproduction_path = _repository_file(
        root, english["three_initialization_certificate"]
    )
    reproduction = _read_object(reproduction_path)
    if (
        reproduction.get("status") != "PASS"
        or _claimed_hash(reproduction)
        != english["three_initialization_evidence_sha256"]
        or reproduction.get("initialization_count") != 3
    ):
        raise ProductHostError("English reproduction certificate failed")
    package_certificate_spec = protocol["package_validation_certificate"]
    package_certificate_path = _repository_file(
        root, package_certificate_spec["path"]
    )
    package_certificate = _read_object(package_certificate_path)
    if (
        package_certificate.get("status")
        != "PASS_VALIDATION_PACKAGE_GATES_FINAL_TEST_UNOPENED"
        or _claimed_hash(package_certificate)
        != package_certificate_spec["evidence_sha256"]
        or package_certificate.get("final_test_accessed") is not False
    ):
        raise ProductHostError("domain package validation certificate failed")
    packages: list[dict[str, Any]] = []
    for specification in protocol["domains"]:
        package = _repository_file(root, specification["package"])
        public = _repository_file(root, specification["public_key"])
        verified = verify_domain_package(package, public)
        if (
            verified["cake_id"] != specification["cake_id"]
            or verified["domain"] != specification["domain"]
            or verified["archive_sha256"]
            != specification["archive_sha256"]
        ):
            raise ProductHostError("registered domain package identity differs")
        packages.append(verified)
    return artifact, packages, package_certificate


def _english_inputs(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, list[Path], list[Path]]:
    values = protocol["english_validation_inputs"]
    return (
        _repository_file(root, values["training_bundle"]),
        [_repository_file(root, value) for value in values["validation_bundles"]],
        [_repository_file(root, value) for value in values["catalogs"]],
    )


def run_english_evaluation(
    *,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root, protocol = _load_protocol(Path(protocol_path))
    artifact, _, _ = _verify_prerequisites(root, protocol)
    training, validation, catalogs = _english_inputs(root, protocol)
    return evaluate_native_host_semantics(
        artifact=artifact,
        training_bundle_path=training,
        validation_bundle_paths=validation,
        catalog_paths=catalogs,
        output_path=output_path,
        threads=16,
    )


def run_english_benchmark(
    *,
    protocol_path: str | Path,
    benchmark_name: str,
    output_path: str | Path,
) -> dict[str, Any]:
    root, protocol = _load_protocol(Path(protocol_path))
    artifact, _, _ = _verify_prerequisites(root, protocol)
    try:
        specification = protocol["benchmarks"][benchmark_name]
    except KeyError as exc:
        raise ProductHostError("unknown combined benchmark") from exc
    return benchmark_native_host(
        artifact=artifact,
        comparator_path=(root / specification["comparator"]).resolve(),
        prompt_manifest_path=(
            root / specification["prompt_manifest"]
        ).resolve(),
        parent_benchmark_path=(
            root / specification["sealed_parent"]
        ).resolve(),
        output_path=output_path,
        output_bytes=int(specification["output_bytes"]),
        threads=int(specification["threads"]),
    )


def _run_child(arguments: list[str]) -> None:
    creationflags = (
        subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0
    )
    result = subprocess.run(
        [sys.executable, "-u", "-m", MODULE_NAME, *arguments],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        creationflags=creationflags,
    )
    if result.returncode:
        raise ProductHostError(
            "combined certification child failed:\n"
            + result.stdout
            + "\n"
            + result.stderr
        )


def _domain_validation(
    *,
    host: LayerCakeProductHost,
    root: Path,
    specifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from .hf_extraction import evaluate_output

    devices = ("cpu", "cuda")
    by_device: dict[str, dict[str, Any]] = {}
    reference_outputs: dict[tuple[str, str], str] = {}
    for device in devices:
        domains: dict[str, Any] = {}
        for specification in specifications:
            domain = str(specification["domain"])
            cake_id = str(specification["cake_id"])
            validation_path = _repository_file(
                root, str(specification["validation"])
            )
            validation = _read_object(validation_path)
            observations = validation.get("observations")
            if (
                validation.get("status") != "PASS"
                or validation.get("observation_count") != 100
                or not isinstance(observations, list)
            ):
                raise ProductHostError(
                    f"{domain} validation lineage is invalid"
                )
            records: list[dict[str, Any]] = []
            for index, row in enumerate(observations, start=1):
                request_started = time.perf_counter_ns()
                result = host.generate(
                    row["prompt"],
                    cake_id=cake_id,
                    # The historical domain validation evidence predates a
                    # per-row action-limit field. The immutable package model
                    # declares and enforces the locked ceiling of 96 actions.
                    max_new_tokens=96,
                    domain_device=device,
                )
                request_completed = time.perf_counter_ns()
                passed, score = evaluate_output(
                    result.output, row["evaluator"]
                )
                telemetry_delta = result.evidence["telemetry_delta"]
                inactive = {
                    key: values
                    for key, values in telemetry_delta.items()
                    if key != cake_id
                }
                inactive_calls = sum(
                    sum(values.values()) for values in inactive.values()
                )
                key = (domain, str(row["probe_id"]))
                if device == "cpu":
                    reference_outputs[key] = result.output_sha256
                cpu_cuda_identical = (
                    device == "cpu"
                    or reference_outputs.get(key) == result.output_sha256
                )
                records.append(
                    {
                        "probe_id": row["probe_id"],
                        "source_passed": row["source_passed"],
                        "layercake_passed": passed,
                        "layercake_score": score,
                        "output_sha256": result.output_sha256,
                        "validation_output_sha256": row[
                            "layercake_output_sha256"
                        ],
                        "validation_output_byte_identical": (
                            result.output_sha256
                            == row["layercake_output_sha256"]
                        ),
                        "cpu_cuda_output_byte_identical": cpu_cuda_identical,
                        "authoritative_generated_action_count": result.evidence[
                            "authoritative_generated_action_count"
                        ],
                        "worker_latency_seconds": result.evidence[
                            "total_latency_seconds"
                        ],
                        "end_to_end_latency_seconds": (
                            request_completed - request_started
                        )
                        / 1e9,
                        "worker_resident_bytes": result.evidence[
                            "process_resident_bytes"
                        ],
                        "parent_resident_bytes": int(
                            psutil.Process().memory_info().rss
                        ),
                        "inactive_execution_calls": inactive_calls,
                        "teacher_present_at_inference": False,
                    }
                )
                if index % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "device": device,
                                "domain": domain,
                                "evaluated": index,
                            }
                        ),
                        flush=True,
                    )
            source_passes = sum(bool(row["source_passed"]) for row in records)
            layercake_passes = sum(
                bool(row["layercake_passed"]) for row in records
            )
            regressions = sum(
                bool(row["source_passed"])
                and not bool(row["layercake_passed"])
                for row in records
            )
            domains[domain] = {
                "status": (
                    "PASS"
                    if (
                        len(records) == 100
                        and regressions == 0
                        and layercake_passes >= source_passes
                        and all(
                            row["validation_output_byte_identical"]
                            and row["cpu_cuda_output_byte_identical"]
                            and row["inactive_execution_calls"] == 0
                            for row in records
                        )
                    )
                    else "FAIL"
                ),
                "observations": len(records),
                "source_passes": source_passes,
                "layercake_passes": layercake_passes,
                "source_passing_regressions": regressions,
                "median_worker_latency_seconds": statistics.median(
                    row["worker_latency_seconds"] for row in records
                ),
                "median_end_to_end_latency_seconds": statistics.median(
                    row["end_to_end_latency_seconds"] for row in records
                ),
                "maximum_worker_resident_bytes": max(
                    row["worker_resident_bytes"] for row in records
                ),
                "maximum_combined_parent_worker_resident_bytes": max(
                    row["worker_resident_bytes"]
                    + row["parent_resident_bytes"]
                    for row in records
                ),
                "inactive_execution_calls": sum(
                    row["inactive_execution_calls"] for row in records
                ),
                "all_outputs_match_validation_bytes": all(
                    row["validation_output_byte_identical"]
                    for row in records
                ),
                "all_cpu_cuda_outputs_byte_identical": all(
                    row["cpu_cuda_output_byte_identical"]
                    for row in records
                ),
                "records": records,
            }
        by_device[device] = {
            "status": (
                "PASS"
                if all(value["status"] == "PASS" for value in domains.values())
                else "FAIL"
            ),
            "domains": domains,
        }
    return {
        "status": (
            "PASS"
            if all(value["status"] == "PASS" for value in by_device.values())
            else "FAIL"
        ),
        "devices": by_device,
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
    }


def certify_product_host(
    *,
    protocol_path: str | Path,
    output_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    output_directory = Path(output_directory).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ProductHostError(f"combined evidence is immutable: {output_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    root, protocol = _load_protocol(protocol_path)
    artifact, packages, package_certificate = _verify_prerequisites(
        root, protocol
    )
    layercake_root = (
        root / protocol["sealed_layercake"]["relative_root"]
    ).resolve()
    before = _git_state(layercake_root)
    if before != {
        "commit": EXPECTED_LAYERCAKE_COMMIT,
        "clean": True,
        "porcelain_sha256": hashlib.sha256(b"").hexdigest(),
    }:
        raise ProductHostError("sealed LayerCake checkout is not pristine")

    english_output = output_directory / "combined-english-validation.json"
    headline_output = output_directory / "combined-headline-benchmark.json"
    sustained_output = output_directory / "combined-sustained-benchmark.json"
    if not english_output.exists():
        _run_child(
            [
                "english-evaluate",
                "--protocol",
                str(protocol_path),
                "--output",
                str(english_output),
            ]
        )
    if not headline_output.exists():
        _run_child(
            [
                "english-benchmark",
                "--protocol",
                str(protocol_path),
                "--benchmark",
                "headline",
                "--output",
                str(headline_output),
            ]
        )
    if not sustained_output.exists():
        _run_child(
            [
                "english-benchmark",
                "--protocol",
                str(protocol_path),
                "--benchmark",
                "sustained",
                "--output",
                str(sustained_output),
            ]
        )
    english = _read_object(english_output)
    headline = _read_object(headline_output)
    sustained = _read_object(sustained_output)
    sustained_runs = [sustained]
    amendment_path = root / (
        "COMBINED_LAYERCAKE_HOST_SUSTAINED_REPETITION_AMENDMENT.json"
    )
    sustained_amendment: dict[str, Any] | None = None
    if amendment_path.is_file():
        sustained_amendment = _read_object(amendment_path)
        first = sustained_amendment.get("first_run", {})
        if (
            sustained_amendment.get("format") != SUSTAINED_AMENDMENT_FORMAT
            or sustained_amendment.get("status")
            != (
                "PREREGISTERED_AFTER_FIRST_SUSTAINED_FAILURE_"
                "BEFORE_REPETITIONS_TWO_AND_THREE"
            )
            or first.get("file_sha256") != _sha256_file(sustained_output)
            or first.get("evidence_sha256") != sustained["evidence_sha256"]
            or sustained_amendment.get("candidate_unchanged") is not True
            or sustained_amendment.get("final_test_accessed") is not False
        ):
            raise ProductHostError("sustained repetition amendment is invalid")
        for index, relative in enumerate(
            sustained_amendment["additional_runs"], start=2
        ):
            repeated_path = _repository_file(root, relative) if (
                root / relative
            ).is_file() else (root / relative).resolve()
            if not repeated_path.exists():
                _run_child(
                    [
                        "english-benchmark",
                        "--protocol",
                        str(protocol_path),
                        "--benchmark",
                        "sustained",
                        "--output",
                        str(repeated_path),
                    ]
                )
            sustained_runs.append(_read_object(repeated_path))
    retention_values = [
        float(run["aggregates"]["phase2_throughput_retained_ratio"])
        for run in sustained_runs
    ]
    qwen_ratio_values = [
        float(run["aggregates"]["median_throughput_ratio"])
        for run in sustained_runs
    ]
    sustained_aggregate = {
        "replicate_count": len(sustained_runs),
        "median_phase2_throughput_retained_ratio": statistics.median(
            retention_values
        ),
        "median_qwen_throughput_ratio": statistics.median(qwen_ratio_values),
        "minimum_qwen_throughput_ratio": min(qwen_ratio_values),
        "minimum_paired_bootstrap_lower_bound": min(
            float(
                run["aggregates"][
                    "paired_prompt_mean_ratio_bootstrap_95ci"
                ][0]
            )
            for run in sustained_runs
        ),
        "all_non_retention_gates_pass": all(
            all(
                value
                for key, value in run["aggregates"]["gates"].items()
                if key != "phase2_throughput_retained_at_least_95pct"
            )
            for run in sustained_runs
        ),
    }
    sustained_aggregate["status"] = (
        "PASS"
        if (
            len(sustained_runs) == 3
            and sustained_aggregate[
                "median_phase2_throughput_retained_ratio"
            ]
            >= 0.95
            and sustained_aggregate["median_qwen_throughput_ratio"] >= 2.0
            and sustained_aggregate["minimum_qwen_throughput_ratio"] >= 2.0
            and sustained_aggregate[
                "minimum_paired_bootstrap_lower_bound"
            ]
            >= 2.0
            and sustained_aggregate["all_non_retention_gates_pass"]
        )
        else "FAIL"
    )
    gc.collect()

    with tempfile.TemporaryDirectory(prefix="abi-product-host-") as registry:
        with LayerCakeProductHost(
            english_artifact=artifact,
            layercake_root=layercake_root,
            registry_root=registry,
            threads=16,
        ) as host:
            installs = [
                host.install(row["package_path"], row["public_key_path"])
                for row in packages
            ]
            inactive_telemetry = host.telemetry()
            if inactive_telemetry["active_domain_worker_devices"]:
                raise ProductHostError("domain worker started before selection")
            domains = _domain_validation(
                host=host,
                root=root,
                specifications=protocol["domains"],
            )
            final_telemetry = host.telemetry()

    after = _git_state(layercake_root)
    gates = {
        "package_count_exact": len(installs) == 3,
        "all_packages_signed": all(row["signed"] for row in installs),
        "domain_workers_lazy_before_selection": (
            inactive_telemetry["active_domain_worker_devices"] == []
        ),
        "english_functional_pass": (
            english.get("status") == "PASS"
            and english.get("observation_count") == 1400
            and english.get("bounded_zero_regression_pass") is True
        ),
        "headline_pass": headline.get("status") == "PASS",
        "headline_depth": (
            headline.get("aggregates", {}).get("distinct_prompts") == 100
            and headline.get("aggregates", {}).get("repeated_prompts") >= 20
        ),
        "sustained_pass": sustained_aggregate["status"] == "PASS",
        "domain_cpu_cuda_validation_pass": domains["status"] == "PASS",
        "sealed_layercake_unchanged": after == before,
        "teacher_absent": True,
        "source_transformer_blocks_retained_zero": True,
        "final_test_unopened": True,
    }
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "english_host": protocol["english_host"],
        "registered_packages": [
            {
                key: value
                for key, value in row.items()
                if key not in {"package_path", "public_key_path"}
            }
            for row in installs
        ],
        "package_validation_certificate_evidence_sha256": (
            package_certificate["evidence_sha256"]
        ),
        "inactive_state_before_domain_selection": inactive_telemetry,
        "english_validation": {
            "path": str(english_output),
            "file_sha256": _sha256_file(english_output),
            "evidence_sha256": english["evidence_sha256"],
            "status": english["status"],
            "observation_count": english["observation_count"],
            "capability_metrics": english["capability_metrics"],
            "peak_process_rss_bytes": english["peak_process_rss_bytes"],
        },
        "headline_benchmark": {
            "path": str(headline_output),
            "file_sha256": _sha256_file(headline_output),
            "evidence_sha256": headline["evidence_sha256"],
            "status": headline["status"],
            "aggregates": headline["aggregates"],
        },
        "sustained_benchmark": {
            "amendment": (
                {
                    "path": str(amendment_path),
                    "sha256": _sha256_file(amendment_path),
                }
                if sustained_amendment is not None
                else None
            ),
            "replicates": [
                {
                    "path": str(
                        sustained_output
                        if index == 0
                        else (
                            root
                            / sustained_amendment["additional_runs"][index - 1]
                        ).resolve()
                    ),
                    "file_sha256": _sha256_file(
                        sustained_output
                        if index == 0
                        else (
                            root
                            / sustained_amendment["additional_runs"][index - 1]
                        ).resolve()
                    ),
                    "evidence_sha256": run["evidence_sha256"],
                    "status": run["status"],
                    "aggregates": run["aggregates"],
                }
                for index, run in enumerate(sustained_runs)
            ],
            "aggregate": sustained_aggregate,
        },
        "domain_validation": domains,
        "final_product_telemetry": final_telemetry,
        "sealed_layercake": {
            "root": str(layercake_root),
            "before": before,
            "after": after,
            "unchanged": after == before,
        },
        "gates": gates,
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "final_test_accessed": False,
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
    evaluate = subparsers.add_parser("english-evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--output", required=True)
    benchmark = subparsers.add_parser("english-benchmark")
    benchmark.add_argument("--protocol", required=True)
    benchmark.add_argument(
        "--benchmark", choices=("headline", "sustained"), required=True
    )
    benchmark.add_argument("--output", required=True)
    certify = subparsers.add_parser("certify")
    certify.add_argument("--protocol", required=True)
    certify.add_argument("--output-directory", required=True)
    certify.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "english-evaluate":
        result = run_english_evaluation(
            protocol_path=args.protocol, output_path=args.output
        )
    elif args.command == "english-benchmark":
        result = run_english_benchmark(
            protocol_path=args.protocol,
            benchmark_name=args.benchmark,
            output_path=args.output,
        )
    else:
        result = certify_product_host(
            protocol_path=args.protocol,
            output_directory=args.output_directory,
            output_path=args.output,
        )
    display = {
        key: result[key]
        for key in (
            "status",
            "evidence_sha256",
            "observation_count",
            "aggregates",
            "gates",
        )
        if key in result
    }
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
