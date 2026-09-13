"""Score R70 candidate completions directly under frozen Phi weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path

import psutil
import torch

from abi.capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from abi.capability_compiler_phase2_prepare import (
    SOURCE_MANIFEST_SHA256, SOURCE_MODEL, SOURCE_REVISION, _tokenizer, _verified_snapshot,
)
from abi.hf_extraction import load_probe_catalog
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CATALOG_SHA256 = "d4ec9697f1ea256a3ae93b4cfda4d845a55153ef3165b485ae63feed078c2ecc"
EXPECTED_PARAMETERS = 3_821_079_552
ROWS = 2_000
CHOICES = 3
INDEX = re.compile(r"r70-reasoning-search-(\d{4})$")


class ExtractionError(RuntimeError):
    pass


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_codes(probe_id: str) -> list[str]:
    match = INDEX.fullmatch(probe_id)
    if match is None:
        raise ExtractionError(f"invalid R70 probe ID: {probe_id}")
    suffix = match.group(1)
    return [f"A{suffix}", f"B{suffix}", f"C{suffix}"]


def run(root: Path, catalog_path: Path, output: Path, batch_size: int) -> dict:
    if output.exists():
        raise ExtractionError(f"immutable R71 output exists: {output}")
    if sha256_file(catalog_path) != CATALOG_SHA256:
        raise ExtractionError("R71 catalog changed")
    if not torch.cuda.is_available() or batch_size < 1 or batch_size > 8:
        raise ExtractionError("R71 requires CUDA and batch size 1..8")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    if len(probes) != ROWS or len({row["probe_id"] for row in probes}) != ROWS:
        raise ExtractionError("R71 matrix identity changed")

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
        raise ExtractionError("R71 source parameter count changed")
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    rows = []
    inference_seconds = 0.0
    scored_tokens = 0

    for offset in range(0, len(probes), batch_size):
        group = probes[offset : offset + batch_size]
        sequences = []
        meta = []
        for probe in group:
            prompt = str(probe["prompt"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            for candidate in _candidate_codes(str(probe["probe_id"])):
                full_ids = tokenizer.encode(rendered + candidate, add_special_tokens=False)
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise ExtractionError("candidate changed the rendered prompt token prefix")
                candidate_ids = full_ids[len(prompt_ids) :]
                if not candidate_ids:
                    raise ExtractionError("empty source candidate tokenization")
                sequences.append(full_ids)
                meta.append((probe, prompt, rendered, candidate, len(prompt_ids), candidate_ids))
        maximum = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), maximum), int(tokenizer.pad_token_id), dtype=torch.long,
        )
        attention = torch.zeros_like(input_ids)
        starts = []
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
        score_groups = []
        for row_index, (_, _, _, _, prompt_length, candidate_ids) in enumerate(meta):
            sequence_start = starts[row_index]
            positions = [sequence_start + prompt_length + i - 1 for i in range(len(candidate_ids))]
            values = [
                float(log_probs[row_index, position, token].item())
                for position, token in zip(positions, candidate_ids, strict=True)
            ]
            if not values or any(not math.isfinite(value) for value in values):
                raise ExtractionError("non-finite or empty candidate likelihood")
            scored_tokens += len(values)
            score_groups.append({"mean_log_probability": sum(values) / len(values), "token_count": len(values)})
        for local_index, probe in enumerate(group):
            base = local_index * CHOICES
            choices = _candidate_codes(str(probe["probe_id"]))
            scores = score_groups[base : base + CHOICES]
            values = [row["mean_log_probability"] for row in scores]
            selected_index = max(range(CHOICES), key=values.__getitem__)
            ordered = sorted(values, reverse=True)
            tie = ordered[0] == ordered[1]
            expected = str(probe["evaluator"]["values"][0])
            selected = choices[selected_index]
            prompt = str(probe["prompt"])
            rendered = meta[base][2]
            rows.append({
                "probe_id": probe["probe_id"], "catalog_index": offset + local_index,
                "prompt_sha256": _digest(prompt), "rendered_prompt_sha256": _digest(rendered),
                "premise_family": (offset + local_index) % 8,
                "instruction_family": ((offset + local_index) // 8) % 4,
                "candidate_codes": choices, "candidate_scores": scores,
                "selected_index": selected_index, "selected_output": selected,
                "selected_output_sha256": _digest(selected),
                "expected_output_sha256": _digest(expected),
                "passed": selected == expected, "tie": tie,
                "top_margin_mean_log_probability": ordered[0] - ordered[1],
            })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if len(rows) % 200 == 0 or len(rows) == ROWS:
            print(json.dumps({"scored": len(rows), "passing": sum(row["passed"] for row in rows)}), flush=True)

    output.mkdir(parents=True)
    raw_path = output / "choice_scores.jsonl"
    write_jsonl_once(raw_path, rows)
    by_premise = {str(i): sum(row["passed"] for row in rows if row["premise_family"] == i) / sum(row["premise_family"] == i for row in rows) for i in range(8)}
    by_instruction = {str(i): sum(row["passed"] for row in rows if row["instruction_family"] == i) / sum(row["instruction_family"] == i for row in rows) for i in range(4)}
    passing = sum(row["passed"] for row in rows)
    ties = sum(row["tie"] for row in rows)
    gates = {
        "rows": len(rows) == ROWS, "minimum_passing": passing >= 1_800,
        "premise_families": min(by_premise.values()) >= 0.85,
        "instruction_families": min(by_instruction.values()) >= 0.85,
        "zero_ties": ties == 0,
    }
    result = {
        "format": "abi-r71-conditional-choice-weight-interrogation/1",
        "verdict": "PASS_R71_AUTHORIZE_SEGREGATED_PACKAGING" if all(gates.values()) else "FAIL_R71_CONDITIONAL_CHOICE_EXTRACTION",
        "catalog_sha256": CATALOG_SHA256,
        "source_model": SOURCE_MODEL, "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_parameters_read": parameters, "source_weights_copied": 0,
        "metrics": {"rows": len(rows), "passing": passing, "pass_rate": passing / len(rows), "ties": ties, "by_premise_family": by_premise, "by_instruction_family": by_instruction},
        "accounting": {
            "candidate_scores_stored": ROWS * CHOICES,
            "candidate_score_storage_bytes_float64_equivalent": ROWS * CHOICES * 8,
            "source_candidate_tokens_scored": scored_tokens,
            "full_vocabulary_logit_vectors_stored": 0,
            "hidden_activations_stored": 0,
            "source_load_seconds": load_seconds,
            "source_inference_seconds": inference_seconds,
            "total_wall_seconds": time.perf_counter() - run_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_process_rss_bytes": int(peak_rss),
        },
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "gates": gates,
        "artifacts": {"choice_scores": {"path": raw_path.name, "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)}},
        "candidate_accessed": False, "layercake_loaded": False,
        "promotion_eligible": False, "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run(Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
