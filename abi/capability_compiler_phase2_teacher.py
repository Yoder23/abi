"""Generate the development-only frozen-teacher Phase 2 reference."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    evaluate_functional,
    load_catalog,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_prepare import (
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    _tokenizer,
    _verified_snapshot,
)


def development_probes(catalog_path: Path) -> list[dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    grouped = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        capability = probe.get("canonical_capability")
        if probe.get("split") == "validation" and capability in grouped:
            grouped[str(capability)].append(probe)
    selected: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != 100:
            raise Phase2Error("teacher development suite depth changed")
        selected.extend(rows)
    return selected


def generate_teacher_reference(*, root: Path, output_dir: Path) -> dict[str, Any]:
    receipt_path = output_dir / "receipt.json"
    outputs_path = output_dir / "development_outputs.jsonl"
    if receipt_path.exists() or outputs_path.exists():
        raise Phase2Error("immutable teacher reference already exists")
    run_started = time.perf_counter()
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    from transformers import AutoModelForCausalLM

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 3_821_079_552:
        raise Phase2Error("teacher parameter count changed")
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    inference_seconds = 0.0
    rows: list[dict[str, Any]] = []
    for probe in development_probes(root / "catalogs/capability_compiler_phase1_frozen_v1.json"):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(probe["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = [int(value) for value in tokenizer(rendered, add_special_tokens=False).input_ids]
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                do_sample=False,
                max_new_tokens=int(probe["max_new_tokens"]),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.eos_token_id),
                use_cache=True,
            )[0, len(prompt_ids):]
        inference_seconds += time.perf_counter() - started
        token_ids = [int(value) for value in generated.tolist()]
        if token_ids and token_ids[-1] == int(tokenizer.eos_token_id):
            token_ids.pop()
        output = tokenizer.decode(token_ids, skip_special_tokens=True)
        rows.append({
            "probe_id": probe["probe_id"],
            "capability": probe["canonical_capability"],
            "output": output,
            "output_token_ids": token_ids,
            "functional_pass": evaluate_functional(output, probe["evaluator"]),
            "repetition_collapse": repetition_collapse(output),
        })
        peak_rss = max(peak_rss, process.memory_info().rss)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    receipt = {
        "format": "abi-capability-compiler-phase2-teacher-reference/1",
        "status": "PASS",
        "system": "T0",
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_parameters": parameter_count,
        "trainable_parameters": 0,
        "source_base_present_at_inference": True,
        "observations": len(rows),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "output_tokens": sum(len(row["output_token_ids"]) for row in rows),
        "output_bytes": sum(len(row["output"].encode("utf-8")) for row in rows),
        "source_load_seconds": load_seconds,
        "source_inference_seconds": inference_seconds,
        "total_wall_seconds_including_verification": time.perf_counter() - run_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": int(peak_rss),
        "outputs_path": outputs_path.relative_to(root).as_posix(),
        "outputs_sha256": sha256_file(outputs_path),
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "final_prompts_accessed": False,
        "abi_candidate_trained": False,
    }
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = generate_teacher_reference(root=Path.cwd().resolve(), output_dir=Path(args.output_dir).resolve())
    print(json.dumps({key: result[key] for key in ("status", "observations", "functional_passes", "repetition_collapses")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
