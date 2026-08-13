"""Screen one exact v19 package on locked ordinary and prospective suites."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import psutil
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-v19-same-artifact-screen/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return values


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_V19_SAME_ARTIFACT_CPU_GPU_SCREEN"
        or protocol.get("devices") != ["cpu", "cuda"]
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v19 same-artifact screen governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v19 same-artifact binding changed: {relative}")
    return protocol, sha256_file(path)


def compare_ordinary(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "probe_id": actual["probe_id"] == expected["probe_id"],
        "capability": actual["capability"] == expected["capability"],
        "output": actual["output"] == expected["output"],
        "original_output": actual["original_output"] == expected["original_output"],
        "output_token_ids": actual["output_token_ids"] == expected["output_token_ids"],
        "automatic_capability_route": actual["automatic_capability_route"] == expected["automatic_capability_route"],
        "control_residual_route": actual["control_residual_route"] == expected["control_residual_route"],
        "task_route": actual["task_route"] == expected["task_route"],
        "guard_terminated": actual["guard_terminated"] == expected["guard_terminated"],
    }


def _host_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake_extensions.route_isolated_prompt_span_core_v19 import PromptSpanRouteIsolatedShallowSparseCoreHost
    return PromptSpanRouteIsolatedShallowSparseCoreHost


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("preregistered CUDA device is unavailable")
    reference = _rows(root / protocol["ordinary_reference_outputs"])
    probes = development_probes(root / protocol["development_catalog"])
    prospective = _rows(root / protocol["prospective_suite"])
    metadata = _json(root / protocol["package_metadata"])
    gates = {
        "ordinary_depth": len(reference) == len(probes) == 1400,
        "ordinary_identity_order": [row["probe_id"] for row in reference] == [row["probe_id"] for row in probes],
        "prospective_depth": len(prospective) == 400,
        "prospective_distinct": len({row["ir_record_id"] for row in prospective}) == 400,
        "package_bound": metadata["package"]["sha256"] == protocol["bindings"][protocol["package"]],
        "payload_bound": metadata["package"]["tensor_payload_hash"] == protocol["tensor_payload_hash"],
        "interface_bound": metadata["interface"] == protocol["interface"],
        "cuda_available": True,
        "training_prohibited": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "status": "PASS_V19_SAME_ARTIFACT_PREFLIGHT" if all(gates.values()) else "FAIL_V19_SAME_ARTIFACT_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "ordinary_observations": len(probes),
        "prospective_observations": len(prospective),
        "package_sha256": protocol["bindings"][protocol["package"]],
        "tensor_payload_hash": protocol["tensor_payload_hash"],
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_present": False,
        "final_test_accessed": False,
    }


def _ordinary(host: Any, probes: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for probe, expected in zip(probes, reference):
        state = host.prefill(str(probe["prompt"]))
        for _ in range(int(probe["max_new_tokens"])):
            if host.decode_step(state) is None:
                break
        output = host.realize(state).decode("utf-8", errors="strict")
        actual = {
            "probe_id": str(probe["probe_id"]),
            "capability": str(probe["canonical_capability"]),
            "output": output,
            "original_output": host.model_tokenizer.decode(state["generated_ids"]),
            "output_token_ids": host.model_tokenizer.encode(output),
            "automatic_capability_route": str(state["capability"]),
            "control_residual_route": None if int(state["weak_route"]) < 0 else int(state["weak_route"]),
            "task_route": int(state["task_route"].item()),
            "guard_terminated": bool(state["terminated_by_guard"]),
        }
        matches = compare_ordinary(actual, expected)
        evidence.append({**actual, "matches": matches, "all_exact": all(matches.values())})
    return evidence


def _prospective(host: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for row in rows:
        output = host.generate(
            str(row["normalized_generation_prompt"]),
            maximum_tokens=int(row["generation_max_new_tokens"]),
        ).decode("utf-8", errors="strict")
        pointer = dict(host.last_pointer_execution or {})
        pointer.pop("wall_seconds", None)
        evidence.append({
            "record_id": row["ir_record_id"],
            "namespace": row["namespace"],
            "family": row["family"],
            "output": output,
            "functional_pass_v1": evaluate_functional(output, row["functional_evaluator"]),
            "repetition_collapse_v2": repetition_collapse_v2(output),
            "pointer": pointer,
        })
    return evidence


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v19 screen output exists: {output}")
    check = preflight(root, protocol_path)
    if not check["status"].startswith("PASS"):
        raise Phase3Error("v19 same-artifact preflight failed")
    package = root / protocol["package"]
    public = (root / protocol["public_key"]).read_bytes()
    metadata = _json(root / protocol["package_metadata"])
    reference = _rows(root / protocol["ordinary_reference_outputs"])
    probes = development_probes(root / protocol["development_catalog"])
    prospective = _rows(root / protocol["prospective_suite"])
    Host = _host_api((root / protocol["layercake_root"]).resolve())
    process = psutil.Process()
    output.mkdir(parents=True)
    executions: dict[str, Any] = {}
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    for device in protocol["devices"]:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        peak_rss = process.memory_info().rss
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"abi-phase4-v19-{device}-") as raw:
            host = Host(Path(raw), trust_store={metadata["public_key"]["key_id"]: public}, device=device)
            activation = host.activate(package)
            ordinary = _ordinary(host, probes, reference)
            peak_rss = max(peak_rss, process.memory_info().rss)
            prospective_rows = _prospective(host, prospective)
            peak_rss = max(peak_rss, process.memory_info().rss)
            verification = host.verify()
        elapsed = time.perf_counter() - started
        raw_rows[device] = ordinary + prospective_rows
        ordinary_path = output / f"{device}_ordinary_outputs.jsonl"
        prospective_path = output / f"{device}_prospective_outputs.jsonl"
        _write_immutable(ordinary_path, b"".join(canonical_json_bytes(row) for row in ordinary))
        _write_immutable(prospective_path, b"".join(canonical_json_bytes(row) for row in prospective_rows))
        passes = sum(row["functional_pass_v1"] for row in prospective_rows)
        interval = wilson(passes, len(prospective_rows))
        by_namespace = {
            name: {"passes": sum(row["functional_pass_v1"] for row in prospective_rows if row["namespace"] == name), "observations": sum(row["namespace"] == name for row in prospective_rows)}
            for name in sorted({row["namespace"] for row in prospective_rows})
        }
        by_family = {
            str(family): {"passes": sum(row["functional_pass_v1"] for row in prospective_rows if row["family"] == family), "observations": sum(row["family"] == family for row in prospective_rows)}
            for family in range(4)
        }
        executions[device] = {
            "archive_sha256": activation["archive_hash"],
            "payload_hash": activation["payload_hash"],
            "ordinary_exact_rows": sum(row["all_exact"] for row in ordinary),
            "ordinary_observations": len(ordinary),
            "ordinary_outputs_sha256": sha256_file(ordinary_path),
            "prospective_functional_passes": passes,
            "prospective_wilson": interval,
            "prospective_by_namespace": by_namespace,
            "prospective_by_family": by_family,
            "prospective_collapses": sum(row["repetition_collapse_v2"] for row in prospective_rows),
            "pointer_rows": sum(bool(row["pointer"]) for row in prospective_rows),
            "six_candidate_rows": sum(row["pointer"].get("candidate_count") == 6 for row in prospective_rows),
            "evaluator_blind_rows": sum(row["pointer"].get("evaluator_used") is False for row in prospective_rows),
            "one_scoring_forward_rows": sum(row["pointer"].get("candidate_scoring_forward_passes") == 1 for row in prospective_rows),
            "one_active_route_rows": sum(row["pointer"].get("active_residual_routes") == 1 for row in prospective_rows),
            "prospective_outputs_sha256": sha256_file(prospective_path),
            "receiver_training_steps": activation["receiver_training_steps"],
            "receiver_calibration_runs": activation["receiver_calibration_runs"],
            "package_verification": verification["status"],
            "wall_seconds_descriptive_only": elapsed,
            "peak_process_rss_bytes_descriptive_only": int(peak_rss),
            "peak_cuda_allocated_bytes_descriptive_only": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        }
        del host
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    cpu_ordinary = raw_rows["cpu"][:1400]
    gpu_ordinary = raw_rows["cuda"][:1400]
    cpu_prospective = raw_rows["cpu"][1400:]
    gpu_prospective = raw_rows["cuda"][1400:]
    thresholds = protocol["prospective_thresholds"]
    gates = {
        "same_signed_package": len({value["archive_sha256"] for value in executions.values()}) == 1,
        "payload_preserved": all(value["payload_hash"] == protocol["tensor_payload_hash"] for value in executions.values()),
        "ordinary_exact_cpu": executions["cpu"]["ordinary_exact_rows"] == 1400,
        "ordinary_exact_cuda": executions["cuda"]["ordinary_exact_rows"] == 1400,
        "ordinary_cpu_cuda_identity": all(left["output"] == right["output"] for left, right in zip(cpu_ordinary, gpu_ordinary)),
        "prospective_cpu_cuda_identity": all(left["output"] == right["output"] for left, right in zip(cpu_prospective, gpu_prospective)),
        "prospective_point": all(value["prospective_wilson"]["point"] >= thresholds["point"] for value in executions.values()),
        "prospective_lower_95": all(value["prospective_wilson"]["lower_95"] >= thresholds["lower_95"] for value in executions.values()),
        "prospective_per_namespace": all(min(group["passes"] / group["observations"] for group in value["prospective_by_namespace"].values()) >= thresholds["per_stratum_point"] for value in executions.values()),
        "prospective_per_family": all(min(group["passes"] / group["observations"] for group in value["prospective_by_family"].values()) >= thresholds["per_stratum_point"] for value in executions.values()),
        "zero_collapse": all(value["prospective_collapses"] == 0 for value in executions.values()),
        "pointer_all_rows": all(value["pointer_rows"] == value["six_candidate_rows"] == value["evaluator_blind_rows"] == value["one_scoring_forward_rows"] == value["one_active_route_rows"] == 400 for value in executions.values()),
        "receiver_learning_zero": all(value["receiver_training_steps"] == value["receiver_calibration_runs"] == 0 for value in executions.values()),
        "package_verification": all(value["package_verification"] == "PASS" for value in executions.values()),
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-v19-same-artifact-screen-result/1",
        "status": "PASS_V19_SAME_ARTIFACT_SCREEN" if all(gates.values()) else "FAIL_V19_SAME_ARTIFACT_SCREEN",
        "protocol_sha256": protocol_sha,
        "package_sha256": sha256_file(package),
        "tensor_payload_hash": protocol["tensor_payload_hash"],
        "executions": executions,
        "gates": gates,
        "hardware": {"machine": platform.node(), "cpu_threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "teacher_present": False,
        "training_performed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "performance_certified": False,
        "phase4_certified": False,
        "claim_boundary": "Same-artifact ordinary conformance and frozen prospective quality screen only. Timing and memory fields are descriptive, not a repeated performance certificate; stable frontier, matched baselines, final test, Phase 4, and superiority remain unproven.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("run")
    command.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
