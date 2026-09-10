"""Replace R21 free label generation with restricted next-token scoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.factual_semantic_r16.public_qualification import _load_source, _render_chat
from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

from .label_repair_binding import load_repair_config

SYSTEM = (
    "Classify the requested behavior using exactly one letter: A=prose composition, "
    "B=summarization, C=email drafting, D=bullet formatting, E=clarification request, "
    "F=safe abstention. Return only the letter."
)


@torch.inference_mode()
def _score(tokenizer: Any, model: Any, rendered: str, token_ids: list[int]) -> list[float]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    logits = model(**encoded, use_cache=False, return_dict=True).logits[0, -1, token_ids].float()
    return [float(value) for value in logits.softmax(dim=-1).cpu()]


def validate_repaired_source(
    root: Path, repair_config_path: Path, source_run: Path
) -> list[dict[str, Any]]:
    config = load_repair_config(root, repair_config_path)
    receipt = json_object(source_run / "receipt.json")
    rows_path = source_run / "source_observations.jsonl"
    try:
        rows = [
            json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 repaired source rows unreadable") from exc
    original_path = root / str(config["failed_source_rows"]["path"])
    try:
        original = [
            json.loads(line)
            for line in original_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 original source rows unreadable") from exc
    response_identity = [
        (row.get("record_id"), row.get("teacher_output"), row.get("teacher_output_sha256"))
        for row in rows
    ]
    original_identity = [
        (row.get("record_id"), row.get("teacher_output"), row.get("teacher_output_sha256"))
        for row in original
    ]
    if (
        receipt.get("format") != "abi-r21-public-source-acquisition/2"
        or receipt.get("verdict") != "PASS_SOURCE_LABEL_REPAIR"
        or receipt.get("repair_config_sha256") != sha256_file(repair_config_path)
        or receipt.get("evidence_sha256") != evidence_hash(receipt)
        or receipt.get("artifacts", {}).get("source_rows", {}).get("sha256")
        != sha256_file(rows_path)
        or len(rows) != 600
        or len(original) != 600
        or response_identity != original_identity
        or receipt.get("response_rows_reused_byte_exact") is not True
        or sum(row.get("teacher_label_exact") is True for row in rows)
        < int(config["minimum_exact_labels"])
        or any(row.get("teacher_label") not in config["candidate_letters"].values() for row in rows)
    ):
        raise R14Error("R21 repaired source evidence failed")
    return rows


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 label repair exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_repair_config(root, config_path)
    original_receipt = json_object(root / config["failed_source_receipt"]["path"])
    original_path = root / config["failed_source_rows"]["path"]
    original_rows = [
        json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(original_rows) != 600 or original_receipt.get("verdict") != "FAIL_SOURCE":
        raise R14Error("R21 failed source prerequisite changed")
    base = json_object(root / config["base_config"]["path"])
    if not torch.cuda.is_available():
        raise R14Error("R21 label repair requires registered CUDA source")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    load_started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(
        base["source"]["model_id"], base["source"]["revision"]
    )
    load_seconds = time.perf_counter() - load_started
    letters = list(config["candidate_letters"])
    token_ids = []
    for letter in letters:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise R14Error("R21 repair candidate is not one source token")
        token_ids.append(int(ids[0]))
    rows = []
    rendered_tokens = 0
    rendered_bytes = 0
    peak_rss = int(psutil.Process().memory_info().rss)
    for index, row in enumerate(original_rows, 1):
        rendered = _render_chat(tokenizer, SYSTEM, str(row["prompt"]))
        scores = _score(tokenizer, model, rendered, token_ids)
        selected_letter = letters[max(range(len(scores)), key=scores.__getitem__)]
        selected = config["candidate_letters"][selected_letter]
        repaired = dict(row)
        repaired["generated_teacher_label_original"] = row["teacher_label"]
        repaired["generated_teacher_label_raw_original"] = row["teacher_label_raw"]
        repaired["teacher_label"] = selected
        repaired["teacher_label_raw"] = selected_letter
        repaired["teacher_label_token_count"] = 1
        repaired["teacher_label_valid"] = True
        repaired["teacher_label_exact"] = selected == row["task"]
        repaired["label_candidate_letters"] = letters
        repaired["label_candidate_probabilities"] = scores
        repaired["label_selected_letter"] = selected_letter
        repaired["label_rendered_prompt_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
        rows.append(repaired)
        encoded = tokenizer.encode(rendered, add_special_tokens=False)
        rendered_tokens += len(encoded)
        rendered_bytes += len(rendered.encode())
        peak_rss = max(peak_rss, int(psutil.Process().memory_info().rss))
        if index == 1 or index % 25 == 0 or index == len(original_rows):
            print(
                json.dumps(
                    {
                        "rows": index,
                        "exact": sum(item["teacher_label_exact"] for item in rows),
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "source_observations.jsonl"
    write_jsonl_once(rows_path, rows)
    exact = sum(row["teacher_label_exact"] for row in rows)
    receipt = {
        "format": "abi-r21-public-source-acquisition/2",
        "verdict": "PASS_SOURCE_LABEL_REPAIR"
        if exact >= int(config["minimum_exact_labels"])
        else "FAIL_SOURCE_LABEL_REPAIR",
        "repair_config_sha256": sha256_file(config_path),
        "base_config_sha256": config["base_config"]["sha256"],
        "original_source_receipt_sha256": config["failed_source_receipt"]["sha256"],
        "original_source_rows_sha256": config["failed_source_rows"]["sha256"],
        "response_rows_reused_byte_exact": all(
            a["record_id"] == b["record_id"]
            and a["teacher_output"] == b["teacher_output"]
            and a["teacher_output_sha256"] == b["teacher_output_sha256"]
            for a, b in zip(original_rows, rows)
        ),
        "metrics": {"rows": len(rows), "exact_labels": exact, "valid_labels": len(rows)},
        "source": {
            "model_id": base["source"]["model_id"],
            "revision": base["source"]["revision"],
            "parameters": original_receipt["source"]["parameters"],
            "training_steps": 0,
            "device": "cuda",
            "present_at_student_training": False,
            "present_at_package_execution": False,
        },
        "artifacts": {"source_rows": {"path": rows_path.name, "sha256": sha256_file(rows_path)}},
        "information_accounting": {
            "teacher_responses_reused": 600,
            "teacher_response_tokens_reused": original_receipt["information_accounting"][
                "teacher_generated_tokens"
            ],
            "teacher_response_bytes_reused": original_receipt["information_accounting"][
                "teacher_output_bytes"
            ],
            "new_label_forward_passes": 600,
            "candidate_score_values_stored": 3600,
            "candidate_token_instances_scored": 3600,
            "rendered_label_prompt_tokens": rendered_tokens,
            "rendered_label_prompt_bytes": rendered_bytes,
            "new_teacher_generated_tokens": 0,
            "new_teacher_output_bytes": 0,
            "source_load_seconds": load_seconds,
            "label_scoring_seconds": time.perf_counter() - started - load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes_observed": peak_rss,
            "source_snapshot_files": len([path for path in snapshot.rglob("*") if path.is_file()]),
            "logits_stored": 3600,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
        },
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
