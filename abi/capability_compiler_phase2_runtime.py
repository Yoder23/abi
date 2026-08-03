"""Measure Phase 2 baseline TTFT, throughput, and active memory."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import psutil
import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    CompactTransformerLM,
    Phase2Error,
    canonical_json_bytes,
    install_lora,
    load_catalog,
    load_lora,
    sha256_file,
)
from .capability_compiler_phase2_lora import route_prompt, train_router
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot


SYSTEMS = ("T0", "L0", "L1", "D0", "D1", "D2")


def _probes(root: Path, count: int) -> list[dict[str, Any]]:
    catalog = load_catalog(root / "catalogs/capability_compiler_phase1_frozen_v1.json")
    rows = sorted(
        (
            row
            for row in catalog["probes"]
            if row.get("split") == "validation" and row.get("canonical_capability") in CAPABILITIES
        ),
        key=lambda row: (str(row["canonical_capability"]), str(row["probe_id"])),
    )
    if len(rows) != 1_400 or not 1 <= count <= len(rows):
        raise Phase2Error("runtime probe suite depth changed")
    grouped = {
        capability: [row for row in rows if row["canonical_capability"] == capability]
        for capability in CAPABILITIES
    }
    interleaved = [
        grouped[capability][index]
        for index in range(100)
        for capability in CAPABILITIES
    ]
    return interleaved[:count]


def _load_candidate(path: Path | None, *, root: Path, system: str) -> dict[str, Any]:
    if system == "T0":
        if path is not None:
            raise Phase2Error("T0 does not accept a candidate manifest")
        return {"system": "T0"}
    if path is None:
        raise Phase2Error("trained baseline requires a frozen candidate manifest")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "FROZEN_BEFORE_RUNTIME" or value.get("system") != system:
        raise Phase2Error("candidate was not frozen for this runtime system")
    checkpoint = (root / str(value["checkpoint_path"])).resolve()
    if root.resolve() not in checkpoint.parents or sha256_file(checkpoint) != value["checkpoint_sha256"]:
        raise Phase2Error("runtime checkpoint identity changed")
    value["_checkpoint"] = checkpoint
    return value


def _adapter_states(checkpoint: Path) -> dict[str, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    combined = load_file(checkpoint, device="cpu")
    output = {capability: {} for capability in CAPABILITIES}
    for name, value in combined.items():
        capability, separator, adapter_name = name.partition(".")
        if not separator or capability not in output:
            raise Phase2Error("invalid combined adapter checkpoint")
        output[capability][adapter_name] = value
    if any(not state for state in output.values()):
        raise Phase2Error("adapter checkpoint capability coverage changed")
    return output


def _load_runtime(root: Path, system: str, candidate: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    centroids = None
    adapters = None
    if system in {"D0", "D1", "D2"}:
        from safetensors.torch import load_file

        model = CompactTransformerLM().to("cuda")
        state = load_file(candidate["_checkpoint"], device="cpu")
        model.load_state_dict(state, strict=True)
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
        if system in {"L0", "L1"}:
            rank = int(candidate["rank"])
            allowed = {"L0": {16, 64}, "L1": {8, 32}}
            if rank not in allowed[system]:
                raise Phase2Error("runtime LoRA rank changed")
            install_lora(model, rank=rank, alpha=2.0 * rank, dropout=0.05)
            adapters = _adapter_states(candidate["_checkpoint"])
            from .capability_compiler_phase2_common import load_phase1_records

            records = load_phase1_records(root / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir")
            centroids = train_router(records)
    return model, tokenizer, adapters, centroids


def _request(
    *,
    model: Any,
    tokenizer: Any,
    system: str,
    probe: Mapping[str, Any],
    adapters: Mapping[str, Mapping[str, torch.Tensor]] | None,
    centroids: Mapping[str, Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    capability = str(probe["canonical_capability"])
    routed = capability
    if system == "L1":
        assert centroids is not None
        routed = route_prompt(str(probe["prompt"]), centroids)
    if system in {"L0", "L1"}:
        assert adapters is not None
        load_lora(model, adapters[routed])
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(probe["prompt"])}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = [int(value) for value in tokenizer(rendered, add_special_tokens=False).input_ids]
    output_ids: list[int] = []
    eos = int(tokenizer.eos_token_id)
    maximum = int(probe["max_new_tokens"])
    first_output_seconds = None
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if system in {"D0", "D1", "D2"}:
            sequence = list(prompt_ids)
            for _ in range(maximum):
                inputs = torch.tensor([sequence[-768:]], dtype=torch.long, device="cuda")
                token = int(model(inputs)[0, -1].argmax().item())
                if first_output_seconds is None:
                    torch.cuda.synchronize()
                    first_output_seconds = time.perf_counter() - started
                if token == eos:
                    break
                sequence.append(token)
                output_ids.append(token)
        else:
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            result = model(input_ids=inputs, use_cache=True)
            token = int(result.logits[0, -1].argmax().item())
            torch.cuda.synchronize()
            first_output_seconds = time.perf_counter() - started
            cache = result.past_key_values
            for _ in range(maximum):
                if token == eos:
                    break
                output_ids.append(token)
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
    return {
        "probe_id": probe["probe_id"],
        "capability": capability,
        "routed_capability": routed,
        "input_tokens": len(prompt_ids),
        "output_tokens": len(output_ids),
        "output_utf8_bytes": len(output.encode("utf-8")),
        "output_characters": len(output),
        "time_to_first_output_seconds": float(first_output_seconds),
        "total_seconds": total,
        "bytes_per_second": len(output.encode("utf-8")) / total if total else 0.0,
        "characters_per_second": len(output) / total if total else 0.0,
        "tokens_per_second_diagnostic": len(output_ids) / total if total else 0.0,
    }


def benchmark(
    *,
    root: Path,
    system: str,
    candidate_manifest: Path | None,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise Phase2Error("immutable runtime evidence already exists")
    if system not in SYSTEMS or mode not in {"cold", "warm"}:
        raise Phase2Error("invalid runtime configuration")
    candidate = _load_candidate(candidate_manifest, root=root, system=system)
    cold_started = time.perf_counter()
    model, tokenizer, adapters, centroids = _load_runtime(root, system, candidate)
    model_load_seconds = time.perf_counter() - cold_started
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    observations: list[dict[str, Any]] = []
    probes = _probes(root, 23 if mode == "warm" else 1)
    if mode == "warm":
        for probe in probes[:3]:
            _request(model=model, tokenizer=tokenizer, system=system, probe=probe, adapters=adapters, centroids=centroids)
        for probe in probes[3:]:
            observations.append(_request(model=model, tokenizer=tokenizer, system=system, probe=probe, adapters=adapters, centroids=centroids))
    else:
        observation = _request(model=model, tokenizer=tokenizer, system=system, probe=probes[0], adapters=adapters, centroids=centroids)
        observation["first_output_from_cold_start_seconds"] = model_load_seconds + observation["time_to_first_output_seconds"]
        observation["total_from_cold_start_seconds"] = model_load_seconds + observation["total_seconds"]
        observations.append(observation)
    result = {
        "format": "abi-capability-compiler-phase2-runtime/1",
        "status": "PASS",
        "system": system,
        "mode": mode,
        "model_load_seconds": model_load_seconds,
        "observations": observations,
        "observation_count": len(observations),
        "median_time_to_first_output_seconds": statistics.median(row["time_to_first_output_seconds"] for row in observations),
        "median_total_seconds": statistics.median(row["total_seconds"] for row in observations),
        "median_bytes_per_second": statistics.median(row["bytes_per_second"] for row in observations),
        "median_characters_per_second": statistics.median(row["characters_per_second"] for row in observations),
        "median_tokens_per_second_diagnostic": statistics.median(row["tokens_per_second_diagnostic"] for row in observations),
        "p95_supported": len(observations) >= 100,
        "p99_supported": len(observations) >= 1_000,
        "peak_process_rss_bytes": int(process.memory_info().rss),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "candidate_manifest_path": candidate_manifest.as_posix() if candidate_manifest else None,
        "candidate_manifest_sha256": sha256_file(candidate_manifest) if candidate_manifest else None,
        "final_prompts_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--mode", choices=("cold", "warm"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = benchmark(
        root=Path.cwd().resolve(),
        system=args.system,
        candidate_manifest=Path(args.candidate_manifest).resolve() if args.candidate_manifest else None,
        mode=args.mode,
        output=Path(args.output).resolve(),
    )
    print(json.dumps({key: result[key] for key in ("status", "system", "mode", "observation_count", "median_bytes_per_second")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
