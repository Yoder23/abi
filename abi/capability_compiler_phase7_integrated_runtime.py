"""Phase 7 exact-product CPU/GPU runtime confirmation with three packages resident."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import psutil
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_cpu_runtime import PeakMonitor
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_b20_v25_physical_screen import _api
from .capability_compiler_phase4_b40_v25_product_conformance import _package
from .capability_compiler_phase4_b50_gpu_runtime import (
    _candidate_request,
    _runtime_metrics,
    _tensor_bytes,
    runtime_schedule,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase5_selective_product import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    DOMAINS,
    _domain_rows,
    _domain_specs,
)
from .capability_compiler_phase5_construct_screen import project_catalog_prompt


FORMAT = "abi-capability-compiler-phase7-integrated-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase7-integrated-runtime-result/1"
SEED = 104729


def _selected_only(
    delta: Mapping[str, Mapping[str, int]], selected_cake_id: str
) -> bool:
    if selected_cake_id not in delta:
        return False
    for cake_id, counters in delta.items():
        if cake_id == selected_cake_id:
            if int(counters.get("prefill_calls", 0)) != 1:
                return False
        elif any(int(value) != 0 for value in counters.values()):
            return False
    return True


def _domain_schedule(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = {
        domain: [dict(row) for row in rows if row["domain"] == domain]
        for domain in DOMAINS
    }
    distinct = [
        grouped[DOMAINS[index % len(DOMAINS)]][index // len(DOMAINS)]
        for index in range(100)
    ]
    scheduled = [*distinct, *distinct[:20]]
    if len({str(row["probe_id"]) for row in distinct}) != 100:
        raise Phase3Error("Phase 7 domain runtime schedule is not distinct")
    return distinct, scheduled


def _reference(path: Path, *, mode: str | None = None) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    if mode is not None:
        rows = [row for row in rows if row.get("mode") == mode]
    return {str(row["probe_id"]): str(row["output"]) for row in rows}


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    document = _json(path)
    status = document.get("status")
    repaired = status == "PREREGISTERED_PHASE7_CPU_RSS_THREE_REPLICATION_CONFIRMATION"
    if repaired:
        base_path = root / str(document.get("base_protocol", ""))
        if (
            not base_path.is_file()
            or sha256_file(base_path) != document.get("base_protocol_sha256")
        ):
            raise Phase3Error("Phase 7 RSS replication base protocol changed")
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
            "PREREGISTERED_EXACT_PHASE7_INTEGRATED_RUNTIME",
            "PREREGISTERED_PHASE7_CPU_RSS_THREE_REPLICATION_CONFIRMATION",
        }
        or int(protocol.get("seed", -1)) != SEED
        or protocol.get("devices") != ["cpu", "cuda"]
        or int(protocol.get("core_distinct_prompts", 0)) != 100
        or int(protocol.get("core_repeated_observations", 0)) != 20
        or int(protocol.get("domain_distinct_prompts", 0)) != 100
        or int(protocol.get("domain_repeated_observations", 0)) != 20
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_authorized") is not False
        or protocol.get("artifact_mutation_authorized") is not False
    ):
        raise Phase3Error("Phase 7 integrated runtime governance changed")
    if repaired and (
        protocol.get("repair_of")
        != "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_PROTOCOL_V1040.json"
        or protocol.get("preserved_failure")
        != "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_RESULT_V1043.json"
        or protocol.get("failed_attribution")
        != "ABI_CAPABILITY_COMPILER_PHASE7_RSS_PROFILE_RESULT_V1046.json"
        or protocol.get("repair_scope")
        != "UNCHANGED_CPU_RUNTIME_THREE_FRESH_PROCESS_REPLICATIONS_ONLY"
        or protocol.get("replication_ids") != ["r1", "r2", "r3"]
    ):
        raise Phase3Error("Phase 7 RSS replication scope changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 7 integrated runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    core_runtime = _json(root / protocol["phase4_runtime_verified_result"])
    phase6 = _json(root / protocol["phase6_seed_result"])
    repaired = (
        protocol.get("status")
        == "PREREGISTERED_PHASE7_CPU_RSS_THREE_REPLICATION_CONFIRMATION"
    )
    result_targets = (
        [
            root
            / protocol["result_path_template"].format(
                device="cpu", replication=replication
            )
            for replication in protocol["replication_ids"]
        ]
        if repaired
        else [
            root / protocol["result_path_template"].format(device=device)
            for device in ("cpu", "cuda")
        ]
    )
    gates = {
        "cuda_available": torch.cuda.is_available(),
        "exact_core_archive_lineage": core_runtime["candidate"]["archive_sha256"]
        == phase6["core_archive_sha256"]
        == protocol["product"]["core_archive_sha256"],
        "exact_core_payload_lineage": core_runtime["candidate"]["tensor_payload_hash"]
        == phase6["core_before"]["payload_hash"]
        == protocol["product"]["core_payload_sha256"],
        "package_lineage_exact": phase6["registry_archive_hashes"]
        == {
            protocol["product"]["packages"][domain]["cake_id"]: protocol["product"][
                "packages"
            ][domain]["archive_sha256"]
            for domain in DOMAINS
        },
        "registered_outputs_absent": not any(path.exists() for path in result_targets),
        "training_absent": True,
        "teacher_absent": True,
    }
    return {
        "format": "abi-capability-compiler-phase7-integrated-runtime-preflight/1",
        "status": "PASS_PHASE7_INTEGRATED_RUNTIME_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE7_INTEGRATED_RUNTIME_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
    }


def _build_core_archive(
    root: Path,
    protocol: Mapping[str, Any],
    temporary: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, Path]:
    core_protocol = _json(root / protocol["core_protocol"])
    for relative, expected in core_protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 7 inherited core binding changed: {relative}")
    specification = next(
        row for row in core_protocol["systems"] if int(row["seed"]) == SEED
    )
    api = _api((root / core_protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(core_protocol["research_signing_seed_hex"])
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    archive = temporary / "phase7-final-english-core.cake"
    built = _package(
        root, core_protocol, specification, archive, api, private, public
    )
    if (
        built["archive_sha256"] != protocol["product"]["core_archive_sha256"]
        or built["tensor_payload_hash"] != protocol["product"]["core_payload_sha256"]
    ):
        raise Phase3Error("Phase 7 exact core reconstruction changed")
    return core_protocol, api, built, public, archive


def _core_identity(host: Any, activated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "archive_hash": host.active_archive_hash,
        "payload_hash": host.active_payload_hash,
        "state_dict_hash": activated["state_dict_hash"],
        "verify": host.verify(),
    }


def _domain_request(
    orchestrator: Any,
    specification: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = orchestrator.execute_labeled(
        project_catalog_prompt(str(row["prompt"])),
        destination_scope="domain_cake",
        domain=str(row["domain"]),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - started
    output = result.output.decode("utf-8", errors="strict")
    raw = output.encode("utf-8")
    return {
        "mode": "domain_runtime",
        "probe_id": str(row["probe_id"]),
        "domain": str(row["domain"]),
        "output": output,
        "output_utf8_bytes": len(raw),
        "output_characters": len(output),
        "time_to_first_output_seconds": total,
        "total_seconds": total,
        "bytes_per_second": len(raw) / total if total else 0.0,
        "characters_per_second": len(output) / total if total else 0.0,
        "selected": list(result.selected),
        "execution_path": result.execution_path,
        "telemetry_delta": result.telemetry_delta,
        "selected_only_execution": _selected_only(
            result.telemetry_delta, str(specification["cake_id"])
        ),
    }


@torch.inference_mode()
def run(
    root: Path,
    protocol_path: Path,
    *,
    device: str,
    output: Path,
    replication: str | None = None,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    repaired = (
        protocol.get("status")
        == "PREREGISTERED_PHASE7_CPU_RSS_THREE_REPLICATION_CONFIRMATION"
    )
    if repaired:
        if device != "cpu" or replication not in protocol["replication_ids"]:
            raise Phase3Error("invalid Phase 7 RSS replication target")
    elif replication is not None:
        raise Phase3Error("base Phase 7 runtime has no replication identity")
    expected_output = (
        root
        / protocol["result_path_template"].format(
            device=device, replication=replication or ""
        )
    ).resolve()
    if device not in {"cpu", "cuda"} or output.exists() or output != expected_output:
        raise Phase3Error("invalid or existing Phase 7 runtime target")
    if device == "cuda" and not torch.cuda.is_available():
        raise Phase3Error("Phase 7 CUDA runtime unavailable")
    if device == "cpu":
        torch.set_num_threads(int(protocol["cpu_control"]["torch_threads"]))
        torch.set_num_interop_threads(
            int(protocol["cpu_control"]["torch_interop_threads"])
        )
    base_runtime = _json(root / protocol["base_runtime_protocol"])
    _, core_schedule = runtime_schedule(root, base_runtime)
    if len(core_schedule) != 120:
        raise Phase3Error("Phase 7 core schedule depth changed")
    core_reference = _reference(root / protocol["core_quality_reference"])
    domain_rows = _domain_rows(
        root / protocol["domain_catalog"], split="final_test", per_domain=100
    )
    _, domain_schedule = _domain_schedule(domain_rows)
    domain_reference = _reference(
        root / protocol["phase6_observations"],
        mode="composed_host_selected_domain",
    )
    process = psutil.Process()
    observations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"abi-phase7-{device}-") as raw:
        temporary = Path(raw)
        package_started = time.perf_counter()
        _, api, built, public, archive = _build_core_archive(
            root, protocol, temporary
        )
        package_seconds = time.perf_counter() - package_started
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rss_baseline = process.memory_info().rss
        monitor = PeakMonitor(lambda: process.memory_info().rss)
        monitor.__enter__()
        try:
            load_started = time.perf_counter()
            core_host = api["ClarificationRouteAllocationBoundedCoreHost"](
                temporary / "core-registry",
                trust_store={built["signer"]: public},
                device=device,
            )
            activated = core_host.activate(archive)
            core_before = _core_identity(core_host, activated)
            specs, trust = _domain_specs(root, protocol)
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
                temporary / "domain-registry",
                abi_version=DIRECT_ABI_VERSION,
                abi_hash=DIRECT_ABI_SHA256,
                trust_store=trust,
                profiles=profiles,
                device=device,
                maximum_loaded_cakes=3,
            )
            installs = {
                domain: orchestrator.install(specs[domain]["package"])
                for domain in DOMAINS
            }
            load_seconds = time.perf_counter() - load_started
            cold = _candidate_request(core_host, core_schedule[0])
            cold["mode"] = "single_cold_core_request"
            cold["single_cold_request"] = True
            cold["model_and_package_registry_load_seconds"] = load_seconds
            cold["time_to_first_output_from_cold_start_seconds"] = (
                load_seconds + float(cold["time_to_first_output_seconds"])
            )
            cold["total_from_cold_start_seconds"] = (
                load_seconds + float(cold["total_seconds"])
            )
            observations.append(cold)
            first_by_domain = {
                domain: next(row for row in domain_rows if row["domain"] == domain)
                for domain in DOMAINS
            }
            for domain in DOMAINS:
                warm = _domain_request(
                    orchestrator, specs[domain], first_by_domain[domain], device=device
                )
                warm["mode"] = "domain_residency_load"
                observations.append(warm)
            for probe in core_schedule[:3]:
                _candidate_request(core_host, probe)
            core_rows = []
            for probe in core_schedule:
                row = _candidate_request(core_host, probe)
                row["mode"] = "core_runtime"
                row["output_byte_exact"] = (
                    row["output"] == core_reference[str(row["probe_id"])]
                )
                core_rows.append(row)
                observations.append(row)
            domain_rows_timed = []
            for probe in domain_schedule:
                row = _domain_request(
                    orchestrator, specs[str(probe["domain"])], probe, device=device
                )
                row["output_byte_exact"] = (
                    row["output"] == domain_reference[str(row["probe_id"])]
                )
                domain_rows_timed.append(row)
                observations.append(row)
            core_after = _core_identity(core_host, activated)
            loaded_domain_tensor_bytes = {
                cake_id: _tensor_bytes(module)
                for cake_id, module in orchestrator.host._models.items()
            }
            core_active_tensor_bytes = sum(
                _tensor_bytes(module)
                for module in (core_host.model, core_host.router, core_host.residual)
            )
        finally:
            monitor.__exit__(None, None, None)
        peak_rss_delta = max(0, int(monitor.peak) - int(rss_baseline))
        peak_cuda = (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        )
    core_metrics = _runtime_metrics(core_rows)
    domain_metrics = _runtime_metrics(domain_rows_timed)
    total_active_tensor_bytes = core_active_tensor_bytes + sum(
        loaded_domain_tensor_bytes.values()
    )
    baseline = protocol["baselines"][device]
    throughput_ratio = core_metrics["median_bytes_per_second"] / float(
        baseline["median_bytes_per_second"]
    )
    retention = core_metrics["median_bytes_per_second"] / float(
        protocol["phase4_same_core_runtime"][device]["median_bytes_per_second"]
    )
    gates = {
        "same_exact_core_archive_and_payload": built["archive_sha256"]
        == protocol["product"]["core_archive_sha256"]
        and built["tensor_payload_hash"]
        == protocol["product"]["core_payload_sha256"],
        "three_exact_packages_installed": len(installs) == 3
        and all(row["status"] == "INSTALLED" for row in installs.values())
        and all(
            sha256_file(specs[domain]["package"])
            == protocol["product"]["packages"][domain]["archive_sha256"]
            for domain in DOMAINS
        ),
        "core_runtime_depth_and_output_identity": len(core_rows) == 120
        and len({row["probe_id"] for row in core_rows}) == 100
        and all(row["output_byte_exact"] for row in core_rows),
        "domain_runtime_depth_and_output_identity": len(domain_rows_timed) == 120
        and len({row["probe_id"] for row in domain_rows_timed}) == 100
        and all(row["output_byte_exact"] for row in domain_rows_timed),
        "domain_selected_only_physical_execution": all(
            row["selected_only_execution"] for row in domain_rows_timed
        ),
        "zero_repetition_collapse_v2": not any(
            repetition_collapse_v2(str(row["output"]))
            for row in [*core_rows, *domain_rows_timed]
        ),
        "core_throughput_retention_at_least_95_percent": retention
        >= float(protocol["gates"]["phase4_core_throughput_retention_minimum"]),
        "transformer_relative_throughput_at_least_2x": throughput_ratio
        >= float(protocol["gates"]["throughput_ratio_minimum"]),
        "cold_ttft_no_worse_than_baseline": cold[
            "time_to_first_output_from_cold_start_seconds"
        ]
        <= float(baseline["cold_ttft_seconds"]),
        "genuine_single_cold_request": cold["single_cold_request"] is True,
        "integrated_active_tensor_bytes_lower": total_active_tensor_bytes
        < int(baseline["active_tensor_bytes"]),
        "integrated_process_rss_lower": peak_rss_delta
        < int(baseline["peak_process_rss_bytes"]),
        "gpu_allocation_lower_if_applicable": device == "cpu"
        or peak_cuda < int(baseline["peak_cuda_allocated_bytes"]),
        "persistent_incremental_state": all(
            row["capability"] in {"coherence", "format_control"}
            or row["execution"].get("persistent_state_created") is True
            for row in core_rows
        ),
        "p95_supported_p99_not_promoted": core_metrics["p95_supported"] is True
        and core_metrics["p99_supported"] is False
        and domain_metrics["p95_supported"] is True
        and domain_metrics["p99_supported"] is False,
        "core_identity_unchanged": core_before == core_after,
        "teacher_absent": True,
        "training_absent": True,
        "receiver_learning_zero": int(activated["receiver_training_steps"]) == 0,
    }
    output.mkdir(parents=True)
    observations_path = output / "observations.jsonl"
    _write_immutable(
        observations_path,
        b"".join(canonical_json_bytes(row) for row in observations),
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PHASE7_INTEGRATED_RUNTIME"
        if all(gates.values())
        else "FAIL_PHASE7_INTEGRATED_RUNTIME",
        "protocol_sha256": protocol_sha,
        "device": device,
        "replication": replication,
        "seed": SEED,
        "product": protocol["product"],
        "package_build_seconds_one_time": package_seconds,
        "cold": cold,
        "core_metrics": core_metrics,
        "domain_metrics": domain_metrics,
        "comparisons": {
            "core_throughput_retention_vs_phase4": retention,
            "median_bytes_per_second_ratio_vs_baseline": throughput_ratio,
        },
        "memory": {
            "rss_baseline_bytes": rss_baseline,
            "peak_process_rss_delta_bytes": peak_rss_delta,
            "peak_cuda_allocated_bytes": peak_cuda,
            "core_active_tensor_bytes": core_active_tensor_bytes,
            "loaded_domain_tensor_bytes": loaded_domain_tensor_bytes,
            "total_integrated_active_tensor_bytes": total_active_tensor_bytes,
        },
        "core_before": core_before,
        "core_after": core_after,
        "package_installs": installs,
        "gates": gates,
        "observations_path": observations_path.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(observations_path),
        "teacher_model_loaded": False,
        "training_performed": False,
        "receiver_training_steps": int(activated["receiver_training_steps"]),
        "phase7_certified": False,
        "claim_boundary": "One preregistered same-product integrated runtime on one declared device. It confirms the exact Phase 4 core while all three exact Phase 6 packages are installed and resident; it does not by itself certify Phase 7, human preference, arbitrary hardware, or release.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--replication")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    if args.preflight:
        result = preflight(root, protocol)
    elif args.device and args.output_dir:
        result = run(
            root,
            protocol,
            device=args.device,
            output=(root / args.output_dir).resolve(),
            replication=args.replication,
        )
    else:
        raise Phase3Error("select preflight or one device and output directory")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
