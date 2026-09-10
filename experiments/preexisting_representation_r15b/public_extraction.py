"""Public R15B representation-to-R11 package extraction preflight."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.foreign_capability_r14.capability import AffineCapability, generate_rows
from experiments.native_isa_r11.core import (
    sha256_file,
    transition_accuracy,
    write_package_once,
)

from .generation_qualification import _chat_prompt, parse_final_digit
from .public_qualification import (
    OPERATIONS,
    QualificationError,
    _render_prompt,
    canonical_json_bytes,
    sha256_bytes,
)
from .representation import (
    RepresentationError,
    decode_labels,
    decode_transition,
    labels_to_operations,
    representation_logits,
)

FINAL_WITH_DIGIT = re.compile(r"FINAL:\s*([0-7])(?:\D|$)", re.IGNORECASE)
LABEL_NAMESPACE = "mathematics/modular-arithmetic/compositional-affine-mod8"


def _source_identity(model_id: str, revision: str) -> str:
    return sha256_bytes(canonical_json_bytes({"model_id": model_id, "revision": revision}))


def _anchor_rows() -> list[dict[str, Any]]:
    rows = []
    for operator in range(3):
        for start in (0, 1):
            prompt = _render_prompt(start, (operator,), style="expression_reasoning")
            _, multiplier, offset = OPERATIONS[operator]
            answer = (multiplier * start + offset) % 8
            identity = {"anonymous_slot": operator, "anchor": start}
            rows.append(
                {
                    **identity,
                    "row_id": sha256_bytes(canonical_json_bytes(identity)),
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "answer": answer,
                }
            )
    return rows


def _next_answer_representation(
    model: Any,
    tokenizer: Any,
    rendered_chat: str,
    completion: str,
    *,
    digit_ids: list[int],
) -> tuple[torch.Tensor, list[float], int, str]:
    matches = list(FINAL_WITH_DIGIT.finditer(completion))
    if not matches:
        raise QualificationError("source completion lacks a final answer marker")
    match = matches[-1]
    digit = int(match.group(1))
    prefix = rendered_chat + completion[: match.start(1)]
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    residual = output.hidden_states[-1][0, -1].float().cpu().contiguous()
    canonical = output.logits[0, -1, digit_ids].float().cpu().contiguous()
    predicted = int(canonical.argmax())
    if predicted != digit:
        raise QualificationError("retokenized pre-answer representation changed source answer")
    return residual, [float(value) for value in canonical.softmax(dim=-1)], digit, prefix


@torch.inference_mode()
def extract(
    *,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    digit_ids = []
    for value in range(8):
        encoded_digit = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded_digit) != 1:
            raise QualificationError("source digit token contract changed")
        digit_ids.append(int(encoded_digit[0]))
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output = model.get_output_embeddings()
    if not hasattr(output, "weight"):
        raise QualificationError("source output head is unavailable")
    output_rows = (
        output.weight.detach()
        .index_select(0, torch.tensor(digit_ids, device="cuda"))
        .float()
        .cpu()
        .contiguous()
    )

    residuals: list[torch.Tensor] = []
    observations = []
    generated_tokens = 0
    started = time.perf_counter()
    for row in _anchor_rows():
        rendered = _chat_prompt(tokenizer, str(row["prompt"]))
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        token_slice = generated[0, encoded["input_ids"].shape[1] :]
        generated_tokens += int(token_slice.numel())
        completion = tokenizer.decode(token_slice, skip_special_tokens=True)
        parsed = parse_final_digit(completion)
        residual, probabilities, predicted, prefix = _next_answer_representation(
            model,
            tokenizer,
            rendered,
            completion,
            digit_ids=digit_ids,
        )
        if parsed != predicted or predicted != int(row["answer"]):
            raise QualificationError("source failed an R15B public anchor")
        residuals.append(residual)
        observations.append(
            {
                "row_id": row["row_id"],
                "anonymous_slot": row["anonymous_slot"],
                "anchor": row["anchor"],
                "answer": row["answer"],
                "prediction": predicted,
                "canonical_probabilities": probabilities,
                "prompt_sha256": row["prompt_sha256"],
                "completion": completion,
                "completion_sha256": sha256_bytes(completion.encode()),
                "pre_answer_prefix_sha256": sha256_bytes(prefix.encode()),
                "residual_sha256": sha256_bytes(residual.numpy().tobytes()),
            }
        )
    elapsed = time.perf_counter() - started
    stacked = torch.stack(residuals).reshape(3, 2, -1).contiguous()
    direct_logits = representation_logits(stacked, output_rows)
    recorded = torch.tensor(
        [
            [row["canonical_probabilities"] for row in observations[index : index + 2]]
            for index in range(0, 6, 2)
        ]
    )
    reconstructed = direct_logits.softmax(dim=-1)
    max_error = float((recorded - reconstructed).abs().max())
    if max_error > 1e-5:
        raise QualificationError("anonymous representation projection is not exact")
    return (
        stacked,
        output_rows,
        observations,
        {
            "wall_seconds": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_second": generated_tokens / elapsed,
            "pre_answer_projection_max_abs_error": max_error,
            "source_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "source_trainable_parameters": 0,
            "digit_token_ids": digit_ids,
        },
    )


def _control_accuracy(
    residuals: torch.Tensor,
    output_rows: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    control: str,
) -> float:
    candidate_residuals = residuals.clone()
    candidate_rows = output_rows.clone()
    if control == "zero":
        candidate_residuals.zero_()
    elif control == "shuffle":
        candidate_residuals = candidate_residuals.flatten(0, 1)[[4, 1, 5, 0, 3, 2]].reshape_as(
            candidate_residuals
        )
    elif control == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1515007)
        candidate_residuals = torch.randn(
            candidate_residuals.shape, generator=generator, dtype=candidate_residuals.dtype
        )
    elif control == "head_shuffle":
        candidate_rows = candidate_rows[[3, 0, 7, 2, 6, 1, 5, 4]]
    else:
        raise QualificationError("unknown representation control")
    try:
        transition = decode_transition(candidate_residuals, candidate_rows)
    except (RepresentationError, RuntimeError):
        return 0.0
    return transition_accuracy(transition, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--model-revision", default="f2826a00ceef68f0f2b946d945ecc0477ce4450c")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        raise QualificationError(f"immutable output already exists: {args.output}")

    residuals, output_rows, observations, source = extract(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
    )
    labels = decode_labels(residuals, output_rows)
    operations = labels_to_operations(labels)
    expected = [(item[1], item[2]) for item in OPERATIONS]
    if operations != expected:
        raise QualificationError("public representation decoder recovered wrong operations")
    transition = decode_transition(residuals, output_rows)
    capability = AffineCapability(
        capability_id="r15b-public-preexisting-mod8-v1",
        operations=tuple(expected),  # type: ignore[arg-type]
        seed_commitment="0" * 64,
    )
    evaluation = generate_rows(
        capability,
        split="r15b_public_package_evaluation",
        rows=10_000,
        depths=[13, 14, 15, 16, 17, 18],
        seed=1515011,
    )
    package_accuracy = transition_accuracy(transition, evaluation)
    if package_accuracy != 1.0:
        raise QualificationError("public representation package is not exact")
    controls = {
        name: _control_accuracy(residuals, output_rows, evaluation, control=name)
        for name in ("zero", "shuffle", "random", "head_shuffle")
    }
    if max(controls.values()) > 0.30:
        raise QualificationError("public representation control exceeded chance gate")

    args.output.mkdir(parents=True)
    bundle_path = args.output / "anonymous_representation.safetensors"
    save_file(
        {"residuals": residuals, "output_rows": output_rows},
        str(bundle_path),
        metadata={
            "format": "abi-r15b-anonymous-pre-answer-representation/1",
            "source_revision": args.model_revision,
        },
    )
    identity = _source_identity(args.model_id, args.model_revision)
    package = write_package_once(
        args.output / "packages",
        transition,
        {"teacher_before_sha256": identity, "teacher_after_sha256": identity},
    )
    label_manifest = {
        "format": "abi-r15b-semantic-label-manifest/1",
        "package_sha256": package["sha256"],
        "namespace": LABEL_NAMESPACE,
        "operation_labels": [item[0] for item in OPERATIONS],
        "source_model_id": args.model_id,
        "source_revision": args.model_revision,
    }
    label_manifest["evidence_sha256"] = sha256_bytes(canonical_json_bytes(label_manifest))
    label_path = args.output / "semantic_label.json"
    label_path.write_bytes(canonical_json_bytes(label_manifest))
    receipt = {
        "format": "abi-r15b-public-representation-extraction/1",
        "status": "PUBLIC_PREFLIGHT_NOT_HELDOUT_CERTIFICATION",
        "source": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "training_steps": 0,
            **source,
        },
        "extractor_input": {
            "prompts": 0,
            "answers": 0,
            "behavior_rows": 0,
            "residual_shape": list(residuals.shape),
            "output_rows_shape": list(output_rows.shape),
            "bundle_sha256": sha256_file(bundle_path),
            "bundle_bytes": bundle_path.stat().st_size,
        },
        "anchors": observations,
        "decoded_labels": labels,
        "decoded_operations": operations,
        "package": package,
        "package_evaluation": {"rows": len(evaluation), "accuracy": package_accuracy},
        "controls": controls,
        "semantic_label": {
            **label_manifest,
            "file_sha256": sha256_file(label_path),
        },
        "teacher_present_at_package_execution": False,
        "recipient_optimizer_steps": 0,
        "claim_ceiling": "PUBLIC_PREEXISTING_REPRESENTATION_PREREQUISITE_ONLY",
    }
    receipt["evidence_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    (args.output / "receipt.json").write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
