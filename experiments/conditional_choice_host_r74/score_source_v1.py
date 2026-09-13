"""Score frozen R75 answer candidates under the pinned Phi source weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

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


CATALOG_SHA256 = "e8cfc9199d05b796c7625803706a62b245b6537d2962924123fdf8d9c607f086"
EXPECTED_PARAMETERS = 3_821_079_552
ROWS = 700


class SourceScoreError(RuntimeError):
    pass


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _choices(probe: dict) -> list[str]:
    probe_id = str(probe["probe_id"])
    try:
        family = int(probe_id.split("-")[2][1:])
        local = int(probe_id.rsplit("-", 1)[1])
    except (ValueError, IndexError) as exc:
        raise SourceScoreError(f"invalid R75 probe ID: {probe_id}") from exc
    numeric = 50_000 + family * 100 + local
    return [f"D{numeric:05d}", f"E{numeric:05d}", f"F{numeric:05d}"]


def run(root: Path, catalog_path: Path, output: Path, batch_size: int) -> dict:
    if output.exists():
        raise SourceScoreError(f"immutable R75 source output exists: {output}")
    if sha256_file(catalog_path) != CATALOG_SHA256:
        raise SourceScoreError("R75 catalog identity changed")
    if not torch.cuda.is_available() or not 1 <= batch_size <= 8:
        raise SourceScoreError("R75 source scoring requires CUDA batch 1..8")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    if len(probes) != ROWS or any(row["split"] != "validation" for row in probes):
        raise SourceScoreError("R75 matrix changed")
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
    rows = []
    inference_seconds = 0.0
    scored_tokens = 0
    for offset in range(0, ROWS, batch_size):
        group = probes[offset : offset + batch_size]
        sequences, metadata = [], []
        for probe in group:
            prompt = str(probe["prompt"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            for candidate in _choices(probe):
                full_ids = tokenizer.encode(rendered + candidate, add_special_tokens=False)
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise SourceScoreError("candidate changed rendered prompt prefix")
                candidate_ids = full_ids[len(prompt_ids):]
                if not candidate_ids:
                    raise SourceScoreError("candidate tokenization is empty")
                sequences.append(full_ids)
                metadata.append((probe, prompt, rendered, candidate, len(prompt_ids), candidate_ids))
        maximum = max(map(len, sequences))
        input_ids = torch.full((len(sequences), maximum), int(tokenizer.pad_token_id), dtype=torch.long)
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
        scored = []
        for row_index, item in enumerate(metadata):
            prompt_length, candidate_ids = item[4], item[5]
            positions = [starts[row_index] + prompt_length + index - 1 for index in range(len(candidate_ids))]
            values = [float(log_probs[row_index, position, token].item()) for position, token in zip(positions, candidate_ids, strict=True)]
            if not values or any(not math.isfinite(value) for value in values):
                raise SourceScoreError("non-finite candidate score")
            scored_tokens += len(values)
            scored.append({"mean_log_probability": sum(values) / len(values), "token_count": len(values)})
        for group_index, probe in enumerate(group):
            base = group_index * 3
            choices = _choices(probe)
            candidate_scores = scored[base : base + 3]
            means = [row["mean_log_probability"] for row in candidate_scores]
            selected_index = max(range(3), key=means.__getitem__)
            ordered = sorted(means, reverse=True)
            expected = str(probe["evaluator"]["values"][0])
            selected = choices[selected_index]
            rows.append({
                "probe_id": probe["probe_id"],
                "catalog_index": offset + group_index,
                "premise_family": int(str(probe["probe_id"]).split("-")[2][1:]),
                "prompt_sha256": _digest(str(probe["prompt"])),
                "rendered_prompt_sha256": _digest(metadata[base][2]),
                "candidate_codes": choices,
                "candidate_scores": candidate_scores,
                "selected_index": selected_index,
                "selected_output": selected,
                "selected_output_sha256": _digest(selected),
                "expected_output_sha256": _digest(expected),
                "passed": selected == expected,
                "tie": ordered[0] == ordered[1],
                "top_margin_mean_log_probability": ordered[0] - ordered[1],
            })
        peak_rss = max(peak_rss, process.memory_info().rss)
        if len(rows) % 100 == 0:
            print(json.dumps({"scored": len(rows), "passing": sum(row["passed"] for row in rows)}), flush=True)
    output.mkdir(parents=True)
    raw_path = output / "choice_scores.jsonl"
    write_jsonl_once(raw_path, rows)
    by_family = Counter(str(row["premise_family"]) for row in rows if row["passed"])
    passing = sum(row["passed"] for row in rows)
    ties = sum(row["tie"] for row in rows)
    gates = {
        "rows": len(rows) == ROWS,
        "minimum_passing": passing >= 665,
        "family_floor": all(by_family[str(index)] >= 90 for index in range(7)),
        "zero_ties": ties == 0,
    }
    result = {
        "format": "abi-r75-source-conditional-choice-validation/1",
        "verdict": "PASS_R75_SOURCE" if all(gates.values()) else "FAIL_R75_SOURCE",
        "catalog_sha256": CATALOG_SHA256,
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_parameters_read": parameters,
        "source_weights_copied": 0,
        "metrics": {"rows": len(rows), "passing": passing, "pass_rate": passing / len(rows), "ties": ties, "family_passing": dict(sorted(by_family.items()))},
        "accounting": {
            "candidate_scores_stored": ROWS * 3,
            "candidate_score_storage_bytes_float64_equivalent": ROWS * 3 * 8,
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
        "artifacts": {"choice_scores": {"path": raw_path.name, "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}},
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
