"""Acquire only R38's missing independence-day abstention stratum."""

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

from experiments.english_sufficiency_r31.acquire import MODEL_ID, REVISION, SYSTEMS
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import INSTRUCTIONS
from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


QUOTA = 8


def _candidate(ordinal: int) -> dict[str, Any]:
    index = 930_000 + ordinal * 3
    nonce = f"Virelon{index:05d}"
    instruction = INSTRUCTIONS["abstention"][ordinal % len(INSTRUCTIONS["abstention"])]
    details = [
        f"identifier={nonce}",
        f"request=the national independence day for {nonce}",
        "supplied_context=no answer or supporting source is provided",
    ]
    prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
    record_id = "r38-source-" + hashlib.sha256(f"r38|{ordinal}|{prompt}".encode()).hexdigest()[:20]
    return {"record_id": record_id, "split": "train", "oracle_task": "abstention", "instruction": instruction, "details": details, "nonce": nonce, "prompt": prompt, "source_ordinal": ordinal}


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R38 source output exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R38 source acquisition requires CUDA")
    output.mkdir(parents=True)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    load_seconds = time.perf_counter() - started
    accepted, attempts, quarantine = [], [], []
    rendered_tokens = generated_tokens = 0
    for ordinal in range(QUOTA):
        row = _candidate(ordinal)
        chosen = None
        for attempt_index, system in enumerate(SYSTEMS, 1):
            rendered = _render_chat(tokenizer, system, row["prompt"])
            call_started = time.perf_counter()
            answer, token_count = _generate(tokenizer, model, rendered, 192)
            call_seconds = time.perf_counter() - call_started
            answer = answer.strip().replace("\r\n", "\n")
            score = v7._score(row, answer, answer)
            attempt = {"record_id": row["record_id"], "source_ordinal": ordinal, "attempt": attempt_index, "system_sha256": hashlib.sha256(system.encode()).hexdigest(), "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "rendered_tokens": len(tokenizer.encode(rendered, add_special_tokens=False)), "output": answer, "output_sha256": hashlib.sha256(answer.encode()).hexdigest(), "generated_tokens": token_count, "call_seconds": call_seconds, "score": score}
            attempts.append(attempt)
            rendered_tokens += attempt["rendered_tokens"]
            generated_tokens += token_count
            peak_rss = max(peak_rss, process.memory_info().rss)
            if score["functional"]:
                chosen = attempt
                break
        if chosen is None:
            quarantine.append({**row, "attempts": len(SYSTEMS)})
        else:
            accepted.append({**row, "teacher_output": chosen["output"], "teacher_output_sha256": chosen["output_sha256"], "teacher_output_tokens": chosen["generated_tokens"], "accepted_attempt": chosen["attempt"]})
        print(json.dumps({"processed": ordinal + 1, "accepted": len(accepted), "teacher_calls": len(attempts), "seconds": time.perf_counter() - started}), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    accepted_path = output / "accepted_rows.jsonl"
    attempts_path = output / "attempts.jsonl"
    quarantine_path = output / "quarantine.jsonl"
    write_jsonl_once(accepted_path, accepted)
    write_jsonl_once(attempts_path, attempts)
    write_jsonl_once(quarantine_path, quarantine)
    passed = len(accepted) == QUOTA and not quarantine
    result = {
        "format": "abi-r38-targeted-abstention-source/1",
        "verdict": "PASS_TARGETED_SOURCE" if passed else "FAIL_TARGETED_SOURCE",
        "source": {"model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot), "device": "cuda"},
        "metrics": {"candidate_prompts": QUOTA, "accepted_rows": len(accepted), "teacher_calls": len(attempts), "quarantined_rows": len(quarantine), "accepted_by_attempt": {str(index): sum(row["accepted_attempt"] == index for row in accepted) for index in range(1, 4)}},
        "information_accounting": {"unique_prompt_bytes": sum(len(_candidate(index)["prompt"].encode()) for index in range(QUOTA)), "teacher_output_bytes_all_attempts": sum(len(row["output"].encode()) for row in attempts), "teacher_generated_tokens": generated_tokens, "rendered_prompt_tokens": rendered_tokens, "stored_logits": 0, "stored_hidden_activations": 0, "source_parameters_copied": 0, "source_load_seconds": load_seconds, "source_inference_seconds": sum(row["call_seconds"] for row in attempts), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss), "external_hardware_used": False},
        "artifacts": {name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for name, path in (("accepted_rows", accepted_path), ("attempts", attempts_path), ("quarantine", quarantine_path))},
        "claim_ceiling": "TARGETED_MISSING_SOURCE_STRATUM_ONLY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
