"""Small public diagnostic for source response formatting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .public_qualification import MODEL_ID, MODEL_REVISION, _chat_prompt, build_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument(
        "--prompt-style",
        choices=("instructions", "expression", "expression_reasoning", "indexed_steps"),
        default="instructions",
    )
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            args.model_id,
            revision=args.model_revision,
            local_files_only=True,
        )
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    rows = [
        row
        for row in build_rows(
            rows_per_depth=32,
            seed=1515001,
            prompt_style=args.prompt_style,
        )
        if row["depth"] == 1
    ]
    for row in rows[: args.rows]:
        prompt = _chat_prompt(tokenizer, str(row["prompt"]))
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=24,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(f"prompt={row['prompt']!r} expected={row['answer']} completion={completion!r}")


if __name__ == "__main__":
    main()
