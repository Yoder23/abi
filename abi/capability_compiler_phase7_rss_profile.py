"""Stage-resolved CPU RSS diagnostic for the failed Phase 7 integrated run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

import psutil
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_cpu_runtime import PeakMonitor
from .capability_compiler_phase4_b50_gpu_runtime import _candidate_request, runtime_schedule
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase5_selective_product import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    DOMAINS,
    _domain_rows,
    _domain_specs,
)
from .capability_compiler_phase7_integrated_runtime import (
    _build_core_archive,
    _domain_request,
    _domain_schedule,
    _reference,
)


FORMAT = "abi-capability-compiler-phase7-rss-profile/1"
RESULT_FORMAT = "abi-capability-compiler-phase7-rss-profile-result/1"


def _profile_stage(
    process: psutil.Process,
    baseline: int,
    name: str,
    operation: Callable[[], Any],
) -> tuple[dict[str, Any], Any]:
    before = process.memory_info().rss
    with PeakMonitor(lambda: process.memory_info().rss) as monitor:
        value = operation()
    after = process.memory_info().rss
    gc.collect()
    after_gc = process.memory_info().rss
    return (
        {
            "stage": name,
            "rss_before_bytes": before,
            "peak_rss_bytes": int(monitor.peak),
            "rss_after_bytes": after,
            "rss_after_gc_bytes": after_gc,
            "peak_delta_from_runtime_baseline_bytes": max(
                0, int(monitor.peak) - baseline
            ),
        },
        value,
    )


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_PHASE7_CPU_RSS_STAGE_PROFILE"
        or protocol.get("device") != "cpu"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_authorized") is not False
        or protocol.get("artifact_mutation_authorized") is not False
        or protocol.get("known_prompt_replay_only") is not True
    ):
        raise Phase3Error("Phase 7 RSS diagnostic governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 7 RSS diagnostic binding changed: {relative}")
    return protocol, sha256_file(path)


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("Phase 7 RSS diagnostic output exists")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime_protocol = _json(root / protocol["integrated_runtime_protocol"])
    base_runtime = _json(root / runtime_protocol["base_runtime_protocol"])
    _, core_schedule = runtime_schedule(root, base_runtime)
    domain_rows = _domain_rows(
        root / runtime_protocol["domain_catalog"], split="final_test", per_domain=100
    )
    _, domain_schedule = _domain_schedule(domain_rows)
    core_reference = _reference(root / runtime_protocol["core_quality_reference"])
    domain_reference = _reference(
        root / runtime_protocol["phase6_observations"],
        mode="composed_host_selected_domain",
    )
    process = psutil.Process()
    stages: list[dict[str, Any]] = []
    output_identity = {"cold": False, "core": 0, "domain_load": 0, "domain": 0}
    with tempfile.TemporaryDirectory(prefix="abi-phase7-rss-profile-") as raw:
        temporary = Path(raw)
        _, api, built, public, archive = _build_core_archive(
            root, runtime_protocol, temporary
        )
        gc.collect()
        runtime_baseline = process.memory_info().rss
        holders: dict[str, Any] = {}

        def load_core_and_install():
            core = api["ClarificationRouteAllocationBoundedCoreHost"](
                temporary / "core-registry",
                trust_store={built["signer"]: public},
                device="cpu",
            )
            activation = core.activate(archive)
            specs, trust = _domain_specs(root, runtime_protocol)
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
                device="cpu",
                maximum_loaded_cakes=3,
            )
            installs = {
                domain: orchestrator.install(specs[domain]["package"])
                for domain in DOMAINS
            }
            holders.update(
                core=core,
                activation=activation,
                specs=specs,
                orchestrator=orchestrator,
                installs=installs,
            )

        stage, _ = _profile_stage(
            process,
            runtime_baseline,
            "core_activation_and_package_install_without_domain_tensor_load",
            load_core_and_install,
        )
        stages.append(stage)
        core = holders["core"]
        orchestrator = holders["orchestrator"]
        specs = holders["specs"]

        def cold_core():
            row = _candidate_request(core, core_schedule[0])
            output_identity["cold"] = (
                row["output"] == core_reference[str(row["probe_id"])]
            )

        stage, _ = _profile_stage(
            process, runtime_baseline, "single_cold_core_request", cold_core
        )
        stages.append(stage)

        for domain in DOMAINS:
            probe = next(row for row in domain_rows if row["domain"] == domain)

            def load_domain(probe=probe, domain=domain):
                row = _domain_request(
                    orchestrator, specs[domain], probe, device="cpu"
                )
                output_identity["domain_load"] += int(
                    row["output"] == domain_reference[str(row["probe_id"])]
                )

            stage, _ = _profile_stage(
                process,
                runtime_baseline,
                f"first_{domain}_decoder_request",
                load_domain,
            )
            stages.append(stage)

        def complete_core_schedule():
            for probe in core_schedule[:3]:
                _candidate_request(core, probe)
            for probe in core_schedule:
                row = _candidate_request(core, probe)
                output_identity["core"] += int(
                    row["output"] == core_reference[str(row["probe_id"])]
                )

        stage, _ = _profile_stage(
            process,
            runtime_baseline,
            "complete_120_observation_core_runtime",
            complete_core_schedule,
        )
        stages.append(stage)

        def complete_domain_schedule():
            for probe in domain_schedule:
                row = _domain_request(
                    orchestrator,
                    specs[str(probe["domain"])],
                    probe,
                    device="cpu",
                )
                output_identity["domain"] += int(
                    row["output"] == domain_reference[str(row["probe_id"])]
                )

        stage, _ = _profile_stage(
            process,
            runtime_baseline,
            "complete_120_observation_domain_runtime",
            complete_domain_schedule,
        )
        stages.append(stage)
    largest = max(stages, key=lambda row: row["peak_delta_from_runtime_baseline_bytes"])
    gates = {
        "stage_order_exact": [row["stage"] for row in stages]
        == [
            "core_activation_and_package_install_without_domain_tensor_load",
            "single_cold_core_request",
            "first_chemistry_decoder_request",
            "first_civics_decoder_request",
            "first_python_decoder_request",
            "complete_120_observation_core_runtime",
            "complete_120_observation_domain_runtime",
        ],
        "all_replayed_outputs_exact": output_identity
        == {"cold": True, "core": 120, "domain_load": 3, "domain": 120},
        "failure_peak_reproduced_within_10_percent": largest[
            "peak_delta_from_runtime_baseline_bytes"
        ]
        >= 0.9 * int(protocol["failed_cpu_peak_rss_delta_bytes"]),
        "training_absent": True,
        "teacher_absent": True,
    }
    output.mkdir(parents=True)
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PHASE7_CPU_RSS_STAGE_PROFILE"
        if all(gates.values())
        else "FAIL_PHASE7_CPU_RSS_STAGE_PROFILE",
        "protocol_sha256": protocol_sha,
        "runtime_baseline_rss_bytes": runtime_baseline,
        "stages": stages,
        "largest_peak_stage": largest,
        "output_identity": output_identity,
        "gates": gates,
        "training_performed": False,
        "teacher_model_loaded": False,
        "model_artifact_mutated": False,
        "claim_boundary": "Stage attribution on the exact failed CPU product using only already-opened prompts. It authorizes no gate reinterpretation or Phase 7 certificate.",
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
    result = run(
        root,
        (root / args.protocol).resolve(),
        (root / args.output_dir).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
