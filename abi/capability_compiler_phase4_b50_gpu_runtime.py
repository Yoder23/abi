"""Measure same-checkpoint CUDA runtime for verified exact-B50 systems."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import psutil
import torch

from . import capability_compiler_phase2_lora as lora
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    CompactTransformerLM,
    canonical_json_bytes,
    install_lora,
    load_catalog,
    load_lora,
    sha256_file,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase2_runtime import _adapter_states
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_cpu_runtime import PeakMonitor
from .capability_compiler_phase4_b50_baselines import (
    load_exact_records,
    train_exact_b50_router,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase4_v22_b50_rescreen import _api, _package


FORMAT = "abi-capability-compiler-phase4-b50-gpu-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-gpu-runtime-result/1"
SYSTEMS = ("ABI", "L0", "L1", "D0", "D1", "D2")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase3Error("runtime quantile requires observations")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _runtime_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Phase3Error("runtime metrics require observations")
    p95_supported = len(rows) >= 100
    p99_supported = len(rows) >= 1000
    return {
        "observations": len(rows),
        "median_bytes_per_second": statistics.median(
            float(row["bytes_per_second"]) for row in rows
        ),
        "median_characters_per_second": statistics.median(
            float(row["characters_per_second"]) for row in rows
        ),
        "median_time_to_first_output_seconds": statistics.median(
            float(row["time_to_first_output_seconds"]) for row in rows
        ),
        "median_total_seconds": statistics.median(
            float(row["total_seconds"]) for row in rows
        ),
        "p95_supported": p95_supported,
        "p95_time_to_first_output_seconds": _quantile(
            [float(row["time_to_first_output_seconds"]) for row in rows], 0.95
        )
        if p95_supported
        else None,
        "p95_total_seconds": _quantile(
            [float(row["total_seconds"]) for row in rows], 0.95
        )
        if p95_supported
        else None,
        "p05_bytes_per_second": _quantile(
            [float(row["bytes_per_second"]) for row in rows], 0.05
        )
        if p95_supported
        else None,
        "p05_characters_per_second": _quantile(
            [float(row["characters_per_second"]) for row in rows], 0.05
        )
        if p95_supported
        else None,
        "p99_supported": p99_supported,
        "p99_time_to_first_output_seconds": _quantile(
            [float(row["time_to_first_output_seconds"]) for row in rows], 0.99
        )
        if p99_supported
        else None,
        "p99_total_seconds": _quantile(
            [float(row["total_seconds"]) for row in rows], 0.99
        )
        if p99_supported
        else None,
    }


def _round_robin(groups: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    maximum = max((len(group) for group in groups), default=0)
    return [
        dict(groups[group_index][row_index])
        for row_index in range(maximum)
        for group_index in range(len(groups))
        if row_index < len(groups[group_index])
    ]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    runtime = protocol.get("runtime", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SAME_CHECKPOINT_B50_CUDA_RUNTIME"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("source_base_loading_for_lora_authorized") is not True
        or protocol.get("final_test_access") != "PROHIBITED"
        or int(runtime.get("distinct_prompts", 0)) < 100
        or int(runtime.get("repeated_observations", 0)) < 20
        or int(runtime.get("p95_minimum_observations", 0)) != 100
        or int(runtime.get("p99_minimum_observations", 0)) != 1000
    ):
        raise Phase3Error("matched B50 CUDA runtime governance changed")
    if set(protocol.get("systems", {})) != set(SYSTEMS):
        raise Phase3Error("matched B50 CUDA runtime system matrix changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"matched B50 CUDA runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def runtime_schedule(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog = load_catalog(root / str(protocol["development_catalog"]))
    grouped = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        capability = str(probe.get("canonical_capability"))
        if probe.get("split") == "validation" and capability in grouped:
            grouped[capability].append(probe)
    per_capability = []
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != 100:
            raise Phase3Error("matched B50 CUDA runtime catalog depth changed")
        per_capability.append(rows)
    distinct_count = int(protocol["runtime"]["distinct_prompts"])
    quotient, remainder = divmod(distinct_count, len(CAPABILITIES))
    selected_groups = [
        rows[: quotient + int(capability_index < remainder)]
        for capability_index, rows in enumerate(per_capability)
    ]
    # Interleave capabilities. Grouped ordering would materially undercount
    # prompt-routed adapter activation cost for L0/L1 and would concentrate the
    # repeated observations in only the alphabetically first capabilities.
    distinct = _round_robin(selected_groups)
    repeats = int(protocol["runtime"]["repeated_observations"])
    scheduled = [*distinct, *[distinct[index % len(distinct)] for index in range(repeats)]]
    digest = hashlib.sha256(
        "\n".join(str(row["probe_id"]) for row in scheduled).encode("utf-8")
    ).hexdigest()
    if (
        len(distinct) != distinct_count
        or len({str(row["probe_id"]) for row in distinct}) != distinct_count
        or digest != protocol["runtime"]["schedule_sha256"]
    ):
        raise Phase3Error("matched B50 CUDA runtime schedule changed")
    return distinct, scheduled


def _reference(path: Path, expected_sha256: str) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise Phase3Error(f"matched B50 CUDA quality reference changed: {path}")
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    indexed = {str(row["probe_id"]): row for row in rows}
    if len(indexed) != 1400:
        raise Phase3Error("matched B50 CUDA quality reference depth changed")
    return indexed


def _tensor_bytes(module: torch.nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in module.parameters()) + sum(
        value.numel() * value.element_size() for value in module.buffers()
    )


def _observation(
    *,
    probe: Mapping[str, Any],
    output: str,
    output_token_ids: Sequence[int],
    retokenized_output_token_ids: Sequence[int] | None = None,
    first_seconds: float,
    total_seconds: float,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = output.encode("utf-8")
    authoritative_ids = (
        [int(value) for value in retokenized_output_token_ids]
        if retokenized_output_token_ids is not None
        else [int(value) for value in output_token_ids]
    )
    return {
        "probe_id": str(probe["probe_id"]),
        "capability": str(probe["canonical_capability"]),
        "output": output,
        "output_token_ids": [int(value) for value in output_token_ids],
        "retokenized_output_token_ids": authoritative_ids,
        "output_utf8_bytes": len(raw),
        "output_characters": len(output),
        "authoritative_output_tokens": len(authoritative_ids),
        "token_accounting": "completed_response_retokenization",
        "time_to_first_output_seconds": first_seconds,
        "total_seconds": total_seconds,
        "bytes_per_second": len(raw) / total_seconds if total_seconds else 0.0,
        "characters_per_second": len(output) / total_seconds if total_seconds else 0.0,
        "execution": dict(execution or {}),
    }


def _candidate_request(host: Any, probe: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    device_type = next(host.model.parameters()).device.type

    def synchronize() -> None:
        if device_type == "cuda":
            torch.cuda.synchronize()

    capability = str(probe["canonical_capability"])
    prompt = str(probe["prompt"])
    maximum = int(probe["max_new_tokens"])
    first = None
    execution: dict[str, Any] = {}
    if capability in {"coherence", "format_control"}:
        output = host.generate(prompt, maximum_tokens=maximum).decode("utf-8", errors="strict")
        total = time.perf_counter() - started
        first = total
        pointer = dict(host.last_pointer_execution or {})
        pointer.pop("wall_seconds", None)
        format_record = dict(host.last_format_execution or {})
        format_record.pop("wall_seconds", None)
        execution = {"pointer": pointer, "format": format_record}
    else:
        state = host.prefill(prompt)
        for _ in range(maximum):
            token = host.decode_step(state)
            if token is None:
                break
            if first is None:
                synchronize()
                first = time.perf_counter() - started
        output = host.realize(state).decode("utf-8", errors="strict")
        synchronize()
        total = time.perf_counter() - started
        execution = {
            "guard_terminated": bool(state["terminated_by_guard"]),
            "active_residual_routes": 0 if int(state["weak_route"]) < 0 else 1,
            "persistent_state_created": state["past_key_values"] is not None,
            "routed_capability": str(state["capability"]),
            "route_correct": str(state["capability"]) == capability,
        }
    token_ids = [int(value) for value in host.model_tokenizer.encode(output)]
    return _observation(
        probe=probe,
        output=output,
        output_token_ids=token_ids,
        first_seconds=float(first if first is not None else total),
        total_seconds=total,
        execution=execution,
    )


def _build_candidate(
    root: Path,
    protocol: Mapping[str, Any],
    temporary: Path,
) -> tuple[Path, dict[str, Any], bytes, str, Mapping[str, Any], Mapping[str, Any]]:
    source = _json(root / str(protocol["candidate_screen_protocol"]))
    spec = next(
        (row for row in source["systems"] if int(row["seed"]) == int(protocol["systems"]["ABI"]["seed"])),
        None,
    )
    if spec is None:
        raise Phase3Error("matched B50 CUDA candidate seed changed")
    api = _api((root / str(source["layercake_root"])).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(source["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    package_path = temporary / "candidate.cake"
    built = _package(root, source, spec, package_path, api, private, public_pem)
    expected = protocol["systems"]["ABI"]
    if (
        built["archive_sha256"] != expected["archive_sha256"]
        or built["tensor_payload_hash"] != expected["tensor_payload_hash"]
        or built["archive_bytes"] != int(expected["archive_bytes"])
    ):
        raise Phase3Error("matched B50 CUDA candidate package changed")
    signer = api["key_id"](public_pem)
    return package_path, built, public_pem, signer, api, source


def _load_baseline(
    root: Path,
    protocol: Mapping[str, Any],
    system: str,
) -> dict[str, Any]:
    spec = protocol["systems"][system]
    checkpoint = (root / str(spec["checkpoint_path"])).resolve()
    if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise Phase3Error(f"matched B50 CUDA checkpoint changed: {system}")
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    adapters = None
    centroids = None
    current_route = {"value": None}
    if system in {"D0", "D1", "D2"}:
        from safetensors.torch import load_file

        model = CompactTransformerLM().to("cuda")
        model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
        model.eval()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")
        model.eval()
        model.config.use_cache = True
        rank = int(spec["rank"])
        install_lora(model, rank=rank, alpha=2.0 * rank, dropout=0.05)
        adapters = _adapter_states(checkpoint)
        source = _json(root / str(protocol["source_headline_protocol"]))
        records = load_exact_records(root / str(source["records_archive"]))
        centroids = train_exact_b50_router(
            [
                dict(row, normalized_acquisition_prompt=row["normalized_generation_prompt"])
                for row in records
            ]
        )
        # install_lora adds fresh Dropout modules after the source model's
        # initial eval() call. Reassert evaluation mode so runtime is exactly
        # the deterministic quality-evaluation lineage.
        model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "adapters": adapters,
        "centroids": centroids,
        "current_route": current_route,
    }


def _baseline_request(
    runtime: Mapping[str, Any],
    system: str,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    started = time.perf_counter()
    capability = str(probe["canonical_capability"])
    routed = capability
    if system == "L1":
        routed = lora.route_prompt(str(probe["prompt"]), runtime["centroids"])
    if system in {"L0", "L1"} and runtime["current_route"]["value"] != routed:
        load_lora(model, runtime["adapters"][routed])
        runtime["current_route"]["value"] = routed
    prompt_ids = lora._render_prompt(tokenizer, str(probe["prompt"]))
    maximum = int(probe["max_new_tokens"])
    eos = int(tokenizer.eos_token_id)
    output_ids: list[int] = []
    first = None
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if system in {"D0", "D1", "D2"}:
            sequence = list(prompt_ids)
            for _ in range(maximum):
                inputs = torch.tensor([sequence[-768:]], dtype=torch.long, device="cuda")
                token = int(model(inputs)[0, -1].argmax().item())
                if first is None:
                    torch.cuda.synchronize()
                    first = time.perf_counter() - started
                if token == eos:
                    break
                sequence.append(token)
                output_ids.append(token)
        else:
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            result = model(input_ids=inputs, use_cache=True)
            token = int(result.logits[0, -1].argmax().item())
            torch.cuda.synchronize()
            first = time.perf_counter() - started
            cache = result.past_key_values
            for step in range(maximum):
                if token == eos:
                    break
                output_ids.append(token)
                if step + 1 >= maximum:
                    break
                result = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = result.past_key_values
                token = int(result.logits[0, -1].argmax().item())
    torch.cuda.synchronize()
    total = time.perf_counter() - started
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    authoritative_ids = [
        int(value)
        for value in tokenizer(output, add_special_tokens=False).input_ids
    ]
    row = _observation(
        probe=probe,
        output=output,
        output_token_ids=output_ids,
        retokenized_output_token_ids=authoritative_ids,
        first_seconds=float(first if first is not None else total),
        total_seconds=total,
        execution={"routed_capability": routed, "route_correct": routed == capability},
    )
    return row


def _identity(rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(
        str(row["output"]) == str(reference[str(row["probe_id"])]["output"])
        and list(row["output_token_ids"])
        == [int(value) for value in reference[str(row["probe_id"])]["output_token_ids"]]
        for row in rows
    )


@torch.inference_mode()
def run(
    root: Path,
    protocol_path: Path,
    *,
    system: str,
    output: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if system not in SYSTEMS or output.exists():
        raise Phase3Error("invalid or existing matched B50 CUDA runtime target")
    if not torch.cuda.is_available():
        raise Phase3Error("CUDA unavailable for matched B50 runtime")
    distinct, scheduled = runtime_schedule(root, protocol)
    spec = protocol["systems"][system]
    reference = _reference(
        root / str(spec["quality_reference_outputs"]),
        str(spec["quality_reference_sha256"]),
    )
    process = psutil.Process()
    process_initial_rss = process.memory_info().rss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    package_build_seconds = 0.0
    temporary = tempfile.TemporaryDirectory(prefix=f"abi-b50-gpu-{system.lower()}-")
    temporary_path = Path(temporary.name)
    package_peak_rss_delta = 0

    if system == "ABI":
        package_started = time.perf_counter()
        with PeakMonitor(lambda: process.memory_info().rss) as package_monitor:
            package_path, package, public_pem, signer, api, _ = _build_candidate(
                root, protocol, temporary_path
            )
        package_build_seconds = time.perf_counter() - package_started
        package_peak_rss_delta = max(0, package_monitor.peak - process_initial_rss)
        gc.collect()
        runtime_idle_rss = process.memory_info().rss
        load_monitor = PeakMonitor(lambda: process.memory_info().rss)
        load_monitor.__enter__()
        cold_started = time.perf_counter()
        host = api["Host"](
            temporary_path / "registry",
            trust_store={signer: public_pem},
            device="cuda",
        )
        activation = host.activate(package_path)
        model_load_seconds = time.perf_counter() - cold_started
        cold = _candidate_request(host, distinct[0])
        cold["single_cold_request"] = True
        cold["cold_definition"] = (
            "model-residency cold; exactly one generation request after host/model "
            "load; operating-system filesystem cache was not purged"
        )
        cold["model_load_seconds"] = model_load_seconds
        cold["time_to_first_output_from_cold_start_seconds"] = (
            model_load_seconds + cold["time_to_first_output_seconds"]
        )
        cold["total_from_cold_start_seconds"] = time.perf_counter() - cold_started
        load_monitor.__exit__(None, None, None)
        with PeakMonitor(lambda: process.memory_info().rss) as monitor:
            for probe in distinct[: int(protocol["runtime"]["warmup_observations"])]:
                _candidate_request(host, probe)
            rows = [_candidate_request(host, probe) for probe in scheduled]
        active_tensor_bytes = sum(
            _tensor_bytes(module) for module in (host.model, host.router, host.residual)
        )
        execution_summary = {
            "archive_sha256": activation["archive_hash"],
            "tensor_payload_hash": activation["payload_hash"],
            "receiver_training_steps": activation["receiver_training_steps"],
            "receiver_calibration_runs": activation["receiver_calibration_runs"],
            "package_build_seconds_one_time_not_in_request": package_build_seconds,
            "package_build_peak_rss_delta_bytes_one_time": package_peak_rss_delta,
            "pointer_rows": sum(bool(row["execution"].get("pointer")) for row in rows),
            "format_rows": sum(bool(row["execution"].get("format")) for row in rows),
            "ordinary_persistent_state_rows": sum(
                row["execution"].get("persistent_state_created") is True for row in rows
            ),
        }
    else:
        runtime_idle_rss = process.memory_info().rss
        load_monitor = PeakMonitor(lambda: process.memory_info().rss)
        load_monitor.__enter__()
        cold_started = time.perf_counter()
        runtime = _load_baseline(root, protocol, system)
        model_load_seconds = time.perf_counter() - cold_started
        cold = _baseline_request(runtime, system, distinct[0])
        cold["single_cold_request"] = True
        cold["cold_definition"] = (
            "model-residency cold; exactly one generation request after host/model "
            "load; operating-system filesystem cache was not purged"
        )
        cold["model_load_seconds"] = model_load_seconds
        cold["time_to_first_output_from_cold_start_seconds"] = (
            model_load_seconds + cold["time_to_first_output_seconds"]
        )
        cold["total_from_cold_start_seconds"] = time.perf_counter() - cold_started
        load_monitor.__exit__(None, None, None)
        with PeakMonitor(lambda: process.memory_info().rss) as monitor:
            for probe in distinct[: int(protocol["runtime"]["warmup_observations"])]:
                _baseline_request(runtime, system, probe)
            rows = [_baseline_request(runtime, system, probe) for probe in scheduled]
        active_tensor_bytes = _tensor_bytes(runtime["model"])
        execution_summary = {
            "route_correct": sum(
                row["execution"].get("route_correct") is True for row in rows
            )
            if system in {"L0", "L1"}
            else None,
            "source_base_present_at_inference": system in {"L0", "L1"},
        }
    peak_rss_delta = max(
        0, max(load_monitor.peak, monitor.peak) - runtime_idle_rss
    )
    peak_cuda = int(torch.cuda.max_memory_allocated())
    identity = _identity(rows, reference)
    metrics = _runtime_metrics(rows)
    expected_pointer = sum(
        str(probe["canonical_capability"]) == "coherence" for probe in scheduled
    )
    expected_format = sum(
        str(probe["canonical_capability"]) == "format_control" for probe in scheduled
    )
    physical_execution = True
    route_execution = True
    if system == "ABI":
        physical_execution = (
            execution_summary["pointer_rows"] == expected_pointer
            and execution_summary["format_rows"] == expected_format
            and execution_summary["ordinary_persistent_state_rows"]
            == len(rows) - expected_pointer - expected_format
            and all(
                row["execution"]["pointer"].get("candidate_count") == 6
                and row["execution"]["pointer"].get(
                    "candidate_scoring_forward_passes"
                )
                == 1
                and row["execution"]["pointer"].get("active_residual_routes") == 1
                and row["execution"]["pointer"].get(
                    "persistent_prompt_state_reused"
                )
                is True
                and row["execution"]["pointer"].get("evaluator_used") is False
                for row in rows
                if row["capability"] == "coherence"
            )
            and all(
                row["execution"]["format"].get("deterministic_transducer") is True
                and row["execution"]["format"].get("prompt_prefill_forward_passes")
                == 1
                and row["execution"]["format"].get(
                    "candidate_scoring_forward_passes"
                )
                == 0
                and row["execution"]["format"].get("decode_forward_passes") == 0
                and row["execution"]["format"].get("active_residual_routes") == 0
                for row in rows
                if row["capability"] == "format_control"
            )
        )
        route_execution = all(
            row["execution"].get("route_correct") is True
            for row in rows
            if row["capability"] not in {"coherence", "format_control"}
        )
    elif system in {"L0", "L1"}:
        route_execution = execution_summary["route_correct"] == len(rows)
    gates = {
        "quality_output_identity": identity == len(rows),
        "depth": len(rows) == 120 and len({row["probe_id"] for row in rows}) == 100,
        "authoritative_token_accounting": all(
            row["token_accounting"] == "completed_response_retokenization"
            and row["authoritative_output_tokens"]
            == len(row["retokenized_output_token_ids"])
            for row in rows
        ),
        "p95_supported": bool(metrics["p95_supported"]),
        "p99_not_promoted": not bool(metrics["p99_supported"]),
        "single_cold_request": cold["single_cold_request"] is True
        and cold["model_load_seconds"] > 0,
        "physical_sparse_execution": physical_execution,
        "route_execution": route_execution,
        "teacher_query_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    output.mkdir(parents=True)
    observations = output / "observations.jsonl"
    _write_immutable(
        observations,
        b"".join(canonical_json_bytes(row) for row in rows),
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_SAME_CHECKPOINT_B50_CUDA_RUNTIME"
        if all(gates.values())
        else "FAIL_SAME_CHECKPOINT_B50_CUDA_RUNTIME",
        "protocol_sha256": protocol_sha,
        "system": system,
        "seed": int(spec["seed"]),
        "metrics": metrics,
        "cold": cold,
        "active_tensor_bytes": active_tensor_bytes,
        "peak_process_rss_delta_bytes": peak_rss_delta,
        "runtime_rss_baseline_bytes": runtime_idle_rss,
        "package_build_peak_rss_delta_bytes_one_time": package_peak_rss_delta,
        "peak_cuda_allocated_bytes": peak_cuda,
        "memory_accounting": {
            "active_tensor_bytes": "parameters and buffers resident in the executing modules",
            "runtime_peak_rss_delta": "peak process RSS during load/cold/warm runtime minus post-packaging pre-load process RSS",
            "package_build_peak_rss_delta": "separately reported one-time deterministic packaging peak; excluded from inference RSS",
        },
        "quality_output_identities": identity,
        "execution": execution_summary,
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "prompt_depth": {"distinct": 100, "repeated": 20, "observations": 120},
        "gates": gates,
        "observations_path": observations.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(observations),
        "training_performed": False,
        "teacher_query_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One same-checkpoint CUDA runtime measurement with quality-output identity. No CPU comparison, cross-system composition, final test, Phase 4, or ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    temporary.cleanup()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(
        root,
        (root / args.protocol).resolve(),
        system=args.system,
        output=(root / args.output_dir).resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "system": result["system"],
                "metrics": result["metrics"],
                "identity": result["quality_output_identities"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
