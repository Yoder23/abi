"""Prospectively score R81 with the frozen candidate-prior correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil
import torch

from abi.capability_compiler_phase2_common import sha256_file
from abi.capability_compiler_phase2_prepare import (
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    _tokenizer,
    _verified_snapshot,
)
from abi.hf_extraction import load_probe_catalog
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CATALOG_SHA256 = "0d2b24417bf1ea925b97747d54ac7053ac7a38f517966431e8befb8f893d2c54"
EXPECTED_PARAMETERS = 3_821_079_552
ROWS = 1_400
MINIMUM = 1_330
FAMILY_MINIMUM = 180


class SourceScoreError(RuntimeError):
    pass


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render(tokenizer: Any, prompt: str, candidate: str) -> tuple[str, list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    full_ids = tokenizer.encode(rendered + candidate, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SourceScoreError("candidate changed rendered prompt prefix")
    candidate_ids = full_ids[len(prompt_ids):]
    if not candidate_ids:
        raise SourceScoreError("candidate tokenization is empty")
    return rendered, full_ids, candidate_ids


def run(root: Path, catalog_path: Path, output: Path, batch_size: int) -> dict[str, Any]:
    if output.exists():
        raise SourceScoreError(f"immutable R81 source output exists: {output}")
    if sha256_file(catalog_path) != CATALOG_SHA256:
        raise SourceScoreError("R81 catalog identity changed")
    if not torch.cuda.is_available() or not 1 <= batch_size <= 4:
        raise SourceScoreError("R81 source scoring requires CUDA batch 1..4")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    if len(probes) != ROWS or any(row["split"] != "validation" for row in probes):
        raise SourceScoreError("R81 matrix changed")

    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from transformers import AutoModelForCausalLM

    run_started = time.perf_counter()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise SourceScoreError("source parameter count changed")

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    inference_seconds = 0.0
    scored_tokens = 0
    for offset in range(0, ROWS, batch_size):
        group = probes[offset : offset + batch_size]
        sequences: list[list[int]] = []
        metadata: list[dict[str, Any]] = []
        for probe in group:
            choices = list(probe.get("conditional_candidates", []))
            if len(choices) != 3 or len(set(choices)) != 3:
                raise SourceScoreError("conditional candidate contract changed")
            neutral_prompt = (
                f"Candidate codes in random order: {' | '.join(choices)}. "
                "There is no relation evidence in this control. Output one candidate code."
            )
            for condition, prompt in (("actual", str(probe["prompt"])), ("neutral", neutral_prompt)):
                for candidate in choices:
                    rendered, full_ids, candidate_ids = _render(tokenizer, prompt, candidate)
                    sequences.append(full_ids)
                    metadata.append({
                        "probe": probe,
                        "condition": condition,
                        "prompt": prompt,
                        "rendered": rendered,
                        "candidate": candidate,
                        "prompt_length": len(full_ids) - len(candidate_ids),
                        "candidate_ids": candidate_ids,
                    })
        maximum = max(map(len, sequences))
        input_ids = torch.full((len(sequences), maximum), int(tokenizer.pad_token_id), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        starts: list[int] = []
        for row_index, ids in enumerate(sequences):
            start = maximum - len(ids)
            input_ids[row_index, start:] = torch.tensor(ids, dtype=torch.long)
            attention[row_index, start:] = 1
            starts.append(start)
        input_ids, attention = input_ids.to("cuda"), attention.to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started

        scored: list[dict[str, Any]] = []
        for row_index, item in enumerate(metadata):
            positions = [
                starts[row_index] + item["prompt_length"] + local_index - 1
                for local_index in range(len(item["candidate_ids"]))
            ]
            values = [
                float(log_probs[row_index, position, token].item())
                for position, token in zip(positions, item["candidate_ids"], strict=True)
            ]
            if not values or any(not math.isfinite(value) for value in values):
                raise SourceScoreError("non-finite candidate score")
            scored_tokens += len(values)
            scored.append({"mean_log_probability": sum(values) / len(values), "token_count": len(values)})

        for group_index, probe in enumerate(group):
            base = group_index * 6
            choices = list(probe["conditional_candidates"])
            actual = scored[base : base + 3]
            neutral = scored[base + 3 : base + 6]
            corrected = [
                float(actual[index]["mean_log_probability"]) - float(neutral[index]["mean_log_probability"])
                for index in range(3)
            ]
            raw_values = [float(row["mean_log_probability"]) for row in actual]
            corrected_index = max(range(3), key=corrected.__getitem__)
            raw_index = max(range(3), key=raw_values.__getitem__)
            corrected_ordered = sorted(corrected, reverse=True)
            raw_ordered = sorted(raw_values, reverse=True)
            expected = str(probe["evaluator"]["value"])
            family = int(str(probe["probe_id"]).split("-")[2][1:])
            rows.append({
                "probe_id": probe["probe_id"],
                "catalog_index": offset + group_index,
                "premise_family": family,
                "destination_display_index": int(probe["destination_display_index"]),
                "prompt_sha256": _digest(str(probe["prompt"])),
                "neutral_prompt_sha256": _digest(metadata[base + 3]["prompt"]),
                "candidate_codes": choices,
                "actual_scores": actual,
                "neutral_scores": neutral,
                "prior_corrected_scores": corrected,
                "raw_selected_index": raw_index,
                "raw_selected_output": choices[raw_index],
                "raw_passed": choices[raw_index] == expected,
                "raw_tie": raw_ordered[0] == raw_ordered[1],
                "prior_corrected_selected_index": corrected_index,
                "prior_corrected_selected_output": choices[corrected_index],
                "prior_corrected_passed": choices[corrected_index] == expected,
                "prior_corrected_tie": corrected_ordered[0] == corrected_ordered[1],
                "prior_corrected_top_margin": corrected_ordered[0] - corrected_ordered[1],
                "expected_output_sha256": _digest(expected),
            })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if len(rows) % 200 == 0:
            print(json.dumps({
                "campaign": "r81",
                "scored": len(rows),
                "raw_passing": sum(row["raw_passed"] for row in rows),
                "prior_corrected_passing": sum(row["prior_corrected_passed"] for row in rows),
            }), flush=True)

    output.mkdir(parents=True)
    raw_path = output / "prior_corrected_scores.jsonl"
    write_jsonl_once(raw_path, rows)
    raw_family = Counter(str(row["premise_family"]) for row in rows if row["raw_passed"])
    corrected_family = Counter(str(row["premise_family"]) for row in rows if row["prior_corrected_passed"])
    display = Counter(str(row["destination_display_index"]) for row in rows if row["prior_corrected_passed"])
    raw_passing = sum(row["raw_passed"] for row in rows)
    corrected_passing = sum(row["prior_corrected_passed"] for row in rows)
    raw_ties = sum(row["raw_tie"] for row in rows)
    corrected_ties = sum(row["prior_corrected_tie"] for row in rows)
    gates = {
        "rows": len(rows) == ROWS,
        "minimum_prior_corrected_passing": corrected_passing >= MINIMUM,
        "prior_corrected_family_floor": all(corrected_family[str(index)] >= FAMILY_MINIMUM for index in range(7)),
        "all_destination_display_positions": set(display) == {"0", "1", "2"},
        "zero_prior_corrected_ties": corrected_ties == 0,
    }
    result = {
        "format": "abi-r81-prospective-prior-corrected-source-score/1",
        "campaign": "r81",
        "verdict": "PASS_R81_SOURCE" if all(gates.values()) else "FAIL_R81_SOURCE",
        "catalog_sha256": CATALOG_SHA256,
        "method": "actual_mean_log_probability_minus_same-list_no-relation-control_mean_log_probability",
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_parameters_read": parameters,
        "source_weights_copied": 0,
        "metrics": {
            "rows": len(rows),
            "raw_passing": raw_passing,
            "raw_pass_rate": raw_passing / len(rows),
            "raw_ties": raw_ties,
            "raw_family_passing": dict(sorted(raw_family.items())),
            "prior_corrected_passing": corrected_passing,
            "prior_corrected_pass_rate": corrected_passing / len(rows),
            "prior_corrected_ties": corrected_ties,
            "prior_corrected_family_passing": dict(sorted(corrected_family.items())),
            "prior_corrected_display_position_passing": dict(sorted(display.items())),
        },
        "accounting": {
            "scalar_candidate_scores_stored": len(rows) * 6,
            "candidate_score_storage_bytes_float64_equivalent": len(rows) * 6 * 8,
            "source_candidate_tokens_scored": scored_tokens,
            "full_vocabulary_logit_vectors_stored": 0,
            "hidden_activations_stored": 0,
            "source_load_seconds": load_seconds,
            "source_inference_seconds": inference_seconds,
            "total_wall_seconds": time.perf_counter() - run_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_process_rss_bytes": int(peak_rss),
        },
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gates": gates,
        "artifacts": {
            "prior_corrected_scores": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            },
        },
        "candidate_accessed": False,
        "layercake_loaded": False,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    run(Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
