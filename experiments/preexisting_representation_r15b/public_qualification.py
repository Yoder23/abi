"""Qualify an unmodified pretrained source for the R15B representation pivot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
MODEL_REVISION = "ba1cf1846d7df0a0591d6c00649f57e798519da8"
DEPTHS = (1, 2, 4, 8, 12)
OPERATIONS = (
    ("increment", 1, 1),
    ("triple", 3, 0),
    ("decrement", 1, 7),
)
OPERATION_TEXT = (
    "add 1 to the current value and reduce modulo 8",
    "multiply the current value by 3 and reduce modulo 8",
    "subtract 1 from the current value and reduce modulo 8",
)


class QualificationError(RuntimeError):
    """Raised when the public source screen cannot be recomputed safely."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def apply_program(start: int, program: tuple[int, ...]) -> int:
    value = int(start)
    for operator in program:
        _, multiplier, offset = OPERATIONS[operator]
        value = (multiplier * value + offset) % 8
    return value


def build_rows(*, rows_per_depth: int, seed: int) -> list[dict[str, Any]]:
    if rows_per_depth < 32:
        raise QualificationError("at least 32 public rows per depth are required")
    generator = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for depth in DEPTHS:
        possibilities = 8 * (3**depth)
        wanted = min(int(rows_per_depth), possibilities)
        attempts = 0
        while sum(row["depth"] == depth for row in rows) < wanted:
            attempts += 1
            if attempts > wanted * 1000:
                raise QualificationError("could not construct unique public rows")
            start = generator.randrange(8)
            program = tuple(generator.randrange(3) for _ in range(depth))
            key = f"{start}:" + ",".join(str(value) for value in program)
            if key in seen:
                continue
            seen.add(key)
            operation_text = "; then ".join(OPERATION_TEXT[index] for index in program)
            prompt = (
                "Work in integers modulo 8. Start with "
                f"{start}. Apply these operations in order: {operation_text}. "
                "What is the final value? Reply with exactly one digit from 0 to 7."
            )
            identity = {"depth": depth, "start": start, "program": list(program)}
            rows.append(
                {
                    **identity,
                    "row_id": sha256_bytes(canonical_json_bytes(identity)),
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "answer": apply_program(start, program),
                }
            )
    return rows


def wilson_lower(correct: int, total: int, *, z: float = 1.959963984540054) -> float:
    if total <= 0:
        raise QualificationError("Wilson interval requires observations")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    return (center - spread) / denominator


def _digit_ids(tokenizer: Any) -> list[int]:
    result = []
    for value in range(8):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise QualificationError(f"digit {value} is not a single source token")
        result.append(int(encoded[0]))
    if len(set(result)) != 8:
        raise QualificationError("canonical digit token IDs are not unique")
    return result


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a precise calculator. Follow the user's instruction exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def evaluate(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
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
        snapshot, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    digit_ids = _digit_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    started = time.perf_counter()
    observations: list[dict[str, Any]] = []
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
        output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        final = output.hidden_states[-1]
        # Causal batches are left-padded, so the final non-padding prompt token
        # is the final tensor position for every row.
        logits = output.logits[:, -1].float()
        residuals = final[:, -1].float()
        canonical = torch.softmax(
            logits.index_select(-1, torch.tensor(digit_ids, device="cuda")), dim=-1
        )
        predictions = canonical.argmax(dim=-1).cpu()
        for index, row in enumerate(batch):
            residual = residuals[index].cpu().contiguous()
            probabilities = canonical[index].cpu().contiguous()
            observations.append(
                {
                    "row_id": row["row_id"],
                    "depth": row["depth"],
                    "answer": row["answer"],
                    "prediction": int(predictions[index]),
                    "correct": int(predictions[index]) == int(row["answer"]),
                    "canonical_probabilities": [float(value) for value in probabilities],
                    "residual_dtype": str(residual.dtype),
                    "residual_shape": list(residual.shape),
                    "residual_sha256": sha256_bytes(residual.numpy().tobytes()),
                }
            )
    elapsed = time.perf_counter() - started
    correct = sum(int(row["correct"]) for row in observations)
    by_depth = {}
    for depth in DEPTHS:
        selected = [row for row in observations if row["depth"] == depth]
        selected_correct = sum(int(row["correct"]) for row in selected)
        by_depth[str(depth)] = {
            "rows": len(selected),
            "correct": selected_correct,
            "accuracy": selected_correct / len(selected),
            "wilson_95_lower": wilson_lower(selected_correct, len(selected)),
        }
    summary = {
        "rows": len(observations),
        "correct": correct,
        "accuracy": correct / len(observations),
        "wilson_95_lower": wilson_lower(correct, len(observations)),
        "by_depth": by_depth,
        "wall_seconds": elapsed,
        "rows_per_second": len(observations) / elapsed,
        "digit_token_ids": digit_ids,
        "source_parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return observations, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-depth", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1515001)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise QualificationError(f"immutable output already exists: {args.output}")
    rows = build_rows(rows_per_depth=args.rows_per_depth, seed=args.seed)
    observations, summary = evaluate(
        rows,
        batch_size=args.batch_size,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    receipt = {
        "format": "abi-r15b-public-source-qualification/1",
        "status": "PUBLIC_DIAGNOSTIC_NOT_CERTIFICATION",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "source_training_steps": 0,
        "capability_label": {
            "namespace": "mathematics/modular-arithmetic/compositional-affine-mod8",
            "operations": [item[0] for item in OPERATIONS],
        },
        "depths": list(DEPTHS),
        "behavior_space_at_max_depth": 8 * 3 ** max(DEPTHS),
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
