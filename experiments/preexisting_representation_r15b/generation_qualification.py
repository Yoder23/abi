"""Public R15B source qualification through the teacher's generation path."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from .public_qualification import (
    MODEL_ID,
    MODEL_REVISION,
    QualificationError,
    build_rows,
    canonical_json_bytes,
    sha256_bytes,
    wilson_lower,
)

FINAL_PATTERN = re.compile(r"FINAL:\s*([0-7])(?:\D|$)", re.IGNORECASE)


def parse_final_digit(text: str) -> int | None:
    matches = FINAL_PATTERN.findall(text)
    return int(matches[-1]) if matches else None


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise calculator. Work through every nested operation "
                    "carefully, then end with FINAL: followed by exactly one digit."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def evaluate_generation(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
    model_id: str,
    model_revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(
        snapshot_download(model_id, revision=model_revision, local_files_only=True)
    ).resolve()
    if snapshot.name != model_revision:
        raise QualificationError("source snapshot revision changed")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    observations = []
    generated_tokens = 0
    started = time.perf_counter()
    for offset in range(0, len(rows), int(batch_size)):
        batch = rows[offset : offset + int(batch_size)]
        prompts = [_chat_prompt(tokenizer, str(row["prompt"])) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_width = int(encoded["input_ids"].shape[1])
        for index, row in enumerate(batch):
            token_slice = generated[index, input_width:]
            generated_tokens += int(token_slice.numel())
            completion = tokenizer.decode(token_slice, skip_special_tokens=True)
            prediction = parse_final_digit(completion)
            observations.append(
                {
                    "row_id": row["row_id"],
                    "depth": row["depth"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "parseable": prediction is not None,
                    "correct": prediction == int(row["answer"]),
                    "completion": completion,
                    "completion_sha256": sha256_bytes(completion.encode()),
                }
            )
    elapsed = time.perf_counter() - started
    correct = sum(int(row["correct"]) for row in observations)
    parseable = sum(int(row["parseable"]) for row in observations)
    by_depth = {}
    for depth in sorted({int(row["depth"]) for row in observations}):
        selected = [row for row in observations if row["depth"] == depth]
        selected_correct = sum(int(row["correct"]) for row in selected)
        by_depth[str(depth)] = {
            "rows": len(selected),
            "correct": selected_correct,
            "accuracy": selected_correct / len(selected),
            "wilson_95_lower": wilson_lower(selected_correct, len(selected)),
        }
    return observations, {
        "rows": len(observations),
        "correct": correct,
        "accuracy": correct / len(observations),
        "wilson_95_lower": wilson_lower(correct, len(observations)),
        "parseable": parseable,
        "parseable_fraction": parseable / len(observations),
        "by_depth": by_depth,
        "generated_tokens": generated_tokens,
        "wall_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed,
        "source_parameters_trainable": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-depth", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1515001)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--prompt-style",
        choices=("expression_reasoning", "indexed_steps"),
        default="expression_reasoning",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise QualificationError(f"immutable output already exists: {args.output}")
    rows = build_rows(
        rows_per_depth=args.rows_per_depth,
        seed=args.seed,
        prompt_style=args.prompt_style,
    )
    observations, summary = evaluate_generation(
        rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    receipt = {
        "format": "abi-r15b-public-generation-qualification/1",
        "status": "PUBLIC_DIAGNOSTIC_NOT_CERTIFICATION",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "source_training_steps": 0,
        "prompt_style": args.prompt_style,
        "capability_label": "mathematics/modular-arithmetic/compositional-affine-mod8",
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "row_generation": {
            "seed": args.seed,
            "rows_per_depth_requested": args.rows_per_depth,
            "rows_sha256": sha256_bytes(canonical_json_bytes(rows)),
        },
        "summary": summary,
        "observations": observations,
        "claim_ceiling": "SOURCE_QUALIFICATION_ONLY_NOT_EXTRACTION",
    }
    receipt["evidence_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
