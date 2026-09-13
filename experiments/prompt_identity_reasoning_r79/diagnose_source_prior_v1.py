"""Post-failure R80 diagnostic for candidate-surface prior correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from abi.capability_compiler_phase2_common import sha256_file
from abi.capability_compiler_phase2_prepare import (
    SOURCE_MANIFEST_SHA256, _tokenizer, _verified_snapshot,
)
from abi.hf_extraction import load_probe_catalog
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CATALOG_SHA256 = "5c2975520630fd6cc1b57e473c50ed89e43540fb5e810ade91fba70c9b98b894"
RESULT_SHA256 = "39ba2fe33f0dfb7f8f9a18092e562f0cdfb4b08e3dce1fa657b12cd0a8573653"
RAW_SHA256 = "53851417b06dd34ea4269a237bf59e30fafe80d1ded1aa92c31d1c6276fff5c3"
RESULT_EVIDENCE_SHA256 = "01c5ed371d5696c7a61bfdc585e2643982d54caeba673fd1f2e23fcea59ec078"
EXPECTED_PARAMETERS = 3_821_079_552
ROWS = 1_400


class DiagnosticError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != RAW_SHA256:
        raise DiagnosticError("frozen R80 raw evidence is absent or changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != ROWS or any(not isinstance(row, dict) for row in rows):
        raise DiagnosticError("R80 raw evidence is incomplete")
    return rows


def run(root: Path, catalog_path: Path, result_path: Path, raw_path: Path, output: Path, batch_size: int) -> dict[str, Any]:
    if output.exists():
        raise DiagnosticError(f"immutable diagnostic exists: {output}")
    if sha256_file(catalog_path) != CATALOG_SHA256 or sha256_file(result_path) != RESULT_SHA256:
        raise DiagnosticError("R80 catalog or result changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("verdict") != "FAIL_R80_SOURCE" or result.get("evidence_sha256") != RESULT_EVIDENCE_SHA256:
        raise DiagnosticError("diagnostic requires the exact R80 source failure")
    observations = _jsonl(raw_path)
    probes = list(load_probe_catalog(catalog_path)["probes"])
    by_id = {str(row["probe_id"]): row for row in observations}
    if len(probes) != ROWS or len(by_id) != ROWS or set(by_id) != {str(row["probe_id"]) for row in probes}:
        raise DiagnosticError("R80 diagnostic coverage changed")
    if not torch.cuda.is_available() or not 1 <= batch_size <= 8:
        raise DiagnosticError("R80 prior diagnostic requires CUDA batch 1..8")
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise DiagnosticError("source parameter count changed")
    rows, inference_seconds, scored_tokens = [], 0.0, 0
    started_total = time.perf_counter()
    for offset in range(0, ROWS, batch_size):
        group = probes[offset : offset + batch_size]
        sequences, metadata = [], []
        for probe in group:
            choices = list(probe["conditional_candidates"])
            neutral = (
                f"Candidate codes in random order: {' | '.join(choices)}. "
                "There is no relation evidence in this control. Output one candidate code."
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": neutral}], tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            for choice in choices:
                full_ids = tokenizer.encode(rendered + choice, add_special_tokens=False)
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise DiagnosticError("neutral candidate changed prompt token prefix")
                candidate_ids = full_ids[len(prompt_ids):]
                if not candidate_ids:
                    raise DiagnosticError("neutral candidate is empty")
                sequences.append(full_ids)
                metadata.append((probe, neutral, rendered, choice, len(prompt_ids), candidate_ids))
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
        neutral_scores = []
        for row_index, item in enumerate(metadata):
            prompt_length, candidate_ids = item[4], item[5]
            positions = [starts[row_index] + prompt_length + local - 1 for local in range(len(candidate_ids))]
            values = [float(log_probs[row_index, position, token].item()) for position, token in zip(positions, candidate_ids, strict=True)]
            if not values or any(not math.isfinite(value) for value in values):
                raise DiagnosticError("neutral source score is invalid")
            scored_tokens += len(values)
            neutral_scores.append({"mean_log_probability": sum(values) / len(values), "token_count": len(values)})
        for local, probe in enumerate(group):
            observation = by_id[str(probe["probe_id"])]
            actual = list(observation["candidate_scores"])
            neutral = neutral_scores[local * 3 : local * 3 + 3]
            corrected = [float(actual[index]["mean_log_probability"]) - float(neutral[index]["mean_log_probability"]) for index in range(3)]
            selected_index = max(range(3), key=corrected.__getitem__)
            selected = probe["conditional_candidates"][selected_index]
            expected = str(probe["evaluator"]["value"])
            rows.append({
                "probe_id": probe["probe_id"],
                "premise_family": int(str(probe["probe_id"]).split("-")[2][1:]),
                "prompt_sha256": hashlib.sha256(str(probe["prompt"]).encode()).hexdigest(),
                "neutral_prompt_sha256": hashlib.sha256(metadata[local * 3][1].encode()).hexdigest(),
                "candidate_codes": probe["conditional_candidates"],
                "actual_scores": actual,
                "neutral_scores": neutral,
                "prior_corrected_scores": corrected,
                "raw_passed": bool(observation["passed"]),
                "prior_corrected_selected_index": selected_index,
                "prior_corrected_selected_output": selected,
                "prior_corrected_passed": selected == expected,
            })
        if len(rows) % 200 == 0:
            print(json.dumps({"scored": len(rows), "raw_passing": sum(row["raw_passed"] for row in rows), "prior_corrected_passing": sum(row["prior_corrected_passed"] for row in rows)}), flush=True)
    output.mkdir(parents=True)
    evidence_path = output / "prior_corrected_scores.jsonl"
    write_jsonl_once(evidence_path, rows)
    raw_family = Counter(str(row["premise_family"]) for row in rows if row["raw_passed"])
    corrected_family = Counter(str(row["premise_family"]) for row in rows if row["prior_corrected_passed"])
    raw_passing = sum(row["raw_passed"] for row in rows)
    corrected_passing = sum(row["prior_corrected_passed"] for row in rows)
    fixed = sum(not row["raw_passed"] and row["prior_corrected_passed"] for row in rows)
    regressed = sum(row["raw_passed"] and not row["prior_corrected_passed"] for row in rows)
    output_result = {
        "format": "abi-r80-postfailure-candidate-prior-diagnostic/1",
        "verdict": (
            "PRIOR_CORRECTION_SUPPORTS_PROSPECTIVE_REPLICATION"
            if corrected_passing >= 1_330 and all(corrected_family[str(index)] >= 180 for index in range(7))
            else "PRIOR_CORRECTION_DOES_NOT_CLEAR_SOURCE_INTERFACE"
        ),
        "catalog_sha256": CATALOG_SHA256,
        "source_result_sha256": RESULT_SHA256,
        "source_raw_sha256": RAW_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "method": "actual_mean_log_probability_minus_same-list_no-relation-control_mean_log_probability",
        "metrics": {
            "rows": len(rows),
            "raw_passing": raw_passing,
            "raw_family_passing": dict(sorted(raw_family.items())),
            "prior_corrected_passing": corrected_passing,
            "prior_corrected_family_passing": dict(sorted(corrected_family.items())),
            "failures_fixed": fixed,
            "raw_passes_regressed": regressed,
            "neutral_candidate_tokens_scored": scored_tokens,
            "source_inference_seconds": inference_seconds,
            "wall_seconds": time.perf_counter() - started_total,
        },
        "candidate_accessed": False,
        "layercake_loaded": False,
        "post_failure_diagnostic": True,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "artifacts": {"prior_corrected_scores": {"path": evidence_path.name, "sha256": sha256_file(evidence_path), "bytes": evidence_path.stat().st_size}},
    }
    output_result["evidence_sha256"] = evidence_hash(output_result)
    write_json_once(output / "result.json", output_result)
    print(json.dumps(output_result, indent=2, sort_keys=True))
    return output_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--source-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    run(Path.cwd(), args.catalog.resolve(), args.source_result.resolve(), args.source_raw.resolve(), args.output.resolve(), args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
