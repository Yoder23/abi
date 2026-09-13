"""Capture the prospective R60 source comparison from pinned local Phi-3."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch

from abi.capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from abi.capability_compiler_phase2_prepare import (
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    _tokenizer,
    _verified_snapshot,
)
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog


CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
EXPECTED_PARAMETERS = 3_821_079_552
EXPECTED_ROWS = 1_400


class SourceCaptureError(RuntimeError):
    pass


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows(catalog_path: Path) -> list[dict[str, Any]]:
    if sha256_file(catalog_path) != CATALOG_SHA256:
        raise SourceCaptureError("R60 catalog changed")
    catalog = load_probe_catalog(catalog_path)
    rows = [dict(row) for row in catalog["probes"]]
    counts = Counter(str(row["capability"]) for row in rows)
    if len(rows) != EXPECTED_ROWS or len(counts) != 14 or set(counts.values()) != {100}:
        raise SourceCaptureError("R60 prospective matrix changed")
    if len({str(row["probe_id"]) for row in rows}) != EXPECTED_ROWS:
        raise SourceCaptureError("R60 probe IDs are not unique")
    return rows


def _batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        item = dict(row)
        item["_catalog_index"] = index
        grouped[int(row["max_new_tokens"])].append(item)
    for maximum in sorted(grouped):
        values = grouped[maximum]
        for offset in range(0, len(values), batch_size):
            yield values[offset : offset + batch_size]


def capture(root: Path, catalog_path: Path, output: Path, batch_size: int) -> dict[str, Any]:
    if output.exists():
        raise SourceCaptureError(f"immutable source output exists: {output}")
    if batch_size < 1 or batch_size > 16:
        raise SourceCaptureError("batch size must be between 1 and 16")
    if not torch.cuda.is_available():
        raise SourceCaptureError("R60 source capture requires CUDA")
    probes = _rows(catalog_path)
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from transformers import AutoModelForCausalLM

    run_started = time.perf_counter()
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
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise SourceCaptureError("source parameter count changed")

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    generated_rows: list[dict[str, Any]] = []
    inference_seconds = 0.0
    batch_id = 0
    for batch in _batches(probes, batch_size):
        maximum = int(batch[0]["max_new_tokens"])
        if any(int(row["max_new_tokens"]) != maximum for row in batch):
            raise SourceCaptureError("mixed generation limits in source batch")
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(row["prompt"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            rendered, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids = encoded.input_ids.to("cuda")
        attention_mask = encoded.attention_mask.to("cuda")
        input_width = int(input_ids.shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=maximum,
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        inference_seconds += elapsed
        for row, rendered_prompt, sequence, mask in zip(
            batch, rendered, sequences, attention_mask, strict=True
        ):
            token_ids = [int(value) for value in sequence[input_width:].tolist()]
            if int(tokenizer.eos_token_id) in token_ids:
                token_ids = token_ids[: token_ids.index(int(tokenizer.eos_token_id))]
            while token_ids and token_ids[-1] == int(tokenizer.pad_token_id):
                token_ids.pop()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            passed, score = evaluate_output(text, row["evaluator"])
            prompt = str(row["prompt"])
            generated_rows.append(
                {
                    "catalog_index": int(row["_catalog_index"]),
                    "probe_id": str(row["probe_id"]),
                    "capability": str(row["capability"]),
                    "split": "prospective_r60",
                    "prompt": prompt,
                    "prompt_sha256": _digest(prompt),
                    "rendered_prompt_sha256": _digest(rendered_prompt),
                    "input_tokens": int(mask.sum().item()),
                    "max_new_tokens": maximum,
                    "output": text,
                    "output_sha256": _digest(text),
                    "output_token_ids": token_ids,
                    "teacher_tokens": len(token_ids),
                    "teacher_token_count_authoritative": True,
                    "evaluator": row["evaluator"],
                    "passed": bool(passed),
                    "score": float(score),
                    "collapse": _collapse_metrics(
                        token_ids, text, tokenizer.encode(prompt + "\n"), prompt
                    ),
                    "generation_batch_id": batch_id,
                    "batch_sequences": len(batch),
                    "batch_latency_seconds": elapsed,
                }
            )
        batch_id += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
        if len(generated_rows) % 100 == 0 or len(generated_rows) == EXPECTED_ROWS:
            print(
                json.dumps(
                    {
                        "captured": len(generated_rows),
                        "functional": sum(row["passed"] for row in generated_rows),
                        "collapses": sum(
                            row["collapse"]["collapse_detected"] for row in generated_rows
                        ),
                    }
                ),
                flush=True,
            )

    generated_rows.sort(key=lambda row: int(row["catalog_index"]))
    if [row["probe_id"] for row in generated_rows] != [row["probe_id"] for row in probes]:
        raise SourceCaptureError("source output order or coverage changed")
    output.mkdir(parents=True)
    rows_path = output / "source_outputs.jsonl"
    rows_path.write_bytes(b"".join(canonical_json_bytes(row) for row in generated_rows))
    config_path = snapshot / "config.json"
    receipt = {
        "format": "abi-r60-live-source-capture/1",
        "status": "COMPLETE_PROSPECTIVE_SOURCE_CAPTURE",
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_parameters": parameters,
        "source_config_sha256": sha256_file(config_path),
        "catalog_sha256": CATALOG_SHA256,
        "split": "prospective_r60",
        "observations": len(generated_rows),
        "capability_counts": dict(
            sorted(Counter(row["capability"] for row in generated_rows).items())
        ),
        "functional_passes": sum(row["passed"] for row in generated_rows),
        "repetition_collapses": sum(
            row["collapse"]["collapse_detected"] for row in generated_rows
        ),
        "teacher_tokens": sum(row["teacher_tokens"] for row in generated_rows),
        "output_utf8_bytes": sum(
            len(row["output"].encode("utf-8")) for row in generated_rows
        ),
        "generation": {
            "do_sample": False,
            "chat_template": "source_tokenizer_apply_chat_template_user_only",
            "generation_calls": batch_id,
            "sequences": len(generated_rows),
            "batch_size_maximum": batch_size,
            "dtype": "bfloat16",
            "attention_implementation": "eager",
        },
        "source_load_seconds": load_seconds,
        "source_inference_seconds": inference_seconds,
        "total_wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": int(peak_rss),
        "outputs": {
            "path": rows_path.name,
            "sha256": sha256_file(rows_path),
            "bytes": rows_path.stat().st_size,
        },
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "training_steps": 0,
        "candidate_accessed": False,
        "candidate_generations": 0,
        "source_teacher_present_in_final_layercake": False,
        "full_abi_moonshot": "OPEN",
    }
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    capture(Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size)


if __name__ == "__main__":
    main()

