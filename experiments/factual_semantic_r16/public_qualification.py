"""Run the R16 public factual source and segregation prerequisite."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.preexisting_representation_r15b.public_qualification import (
    canonical_json_bytes,
    sha256_bytes,
)

from .facts import (
    PUBLIC_FACTS,
    canonical_answer,
    namespace_from_question,
    question,
    render_multiple_choice,
)

CHOICE_PATTERN = re.compile(r"^\s*([A-D])(?:\s|[.)]|$)", re.IGNORECASE)


def _load_source(model_id: str, revision: str) -> tuple[Any, Any, Path]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(
        snapshot_download(model_id, revision=revision, local_files_only=True)
    ).resolve()
    if snapshot.name != revision:
        raise R14Error("R16 source revision changed")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, dtype=torch.float16
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, snapshot


def _render_chat(tokenizer: Any, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def _generate(tokenizer: Any, model: Any, rendered: str, max_new_tokens: int) -> tuple[str, int]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(tokens, skip_special_tokens=True), int(tokens.numel())


@torch.inference_mode()
def _prompt_end_representation(
    tokenizer: Any, model: Any, rendered: str, label_ids: list[int]
) -> tuple[torch.Tensor, list[float], int]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    output = model(
        **encoded, use_cache=False, output_hidden_states=True, return_dict=True
    )
    residual = output.hidden_states[-1][0, -1].float().cpu().contiguous()
    logits = output.logits[0, -1, label_ids].float().cpu().contiguous()
    return residual, [float(value) for value in logits.softmax(dim=-1)], int(logits.argmax())


def run(model_id: str, revision: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R16 public output exists: {output}")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(model_id, revision)
    label_ids = []
    for label in "ABCD":
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise R14Error("R16 choice label is not one source token")
        label_ids.append(int(encoded[0]))
    output_head = model.get_output_embeddings().weight.detach().index_select(
        0, torch.tensor(label_ids, device="cuda")
    ).float().cpu().contiguous()

    rows = []
    residuals = []
    generated_tokens = 0
    teacher_output_bytes = 0
    open_exact = 0
    choice_generated_exact = 0
    choice_projected_exact = 0
    semantic_exact = 0
    extracted: dict[str, dict[str, Any]] = {}

    open_system = "Answer the factual question with only the answer and no explanation."
    choice_system = (
        "Answer the multiple-choice factual question with exactly one uppercase letter "
        "(A, B, C, or D) and no other text."
    )
    for fact in PUBLIC_FACTS:
        open_question = question(fact, 0)
        open_rendered = _render_chat(tokenizer, open_system, open_question)
        completion, token_count = _generate(tokenizer, model, open_rendered, 12)
        generated_tokens += token_count
        teacher_output_bytes += len(completion.encode())
        open_ok = canonical_answer(completion) == canonical_answer(fact.value)
        open_exact += int(open_ok)
        rows.append(
            {
                "kind": "open",
                "fact_id": fact.fact_id,
                "namespace_oracle": fact.namespace,
                "question": open_question,
                "question_sha256": sha256_bytes(open_question.encode()),
                "completion": completion,
                "completion_sha256": sha256_bytes(completion.encode()),
                "oracle_value": fact.value,
                "exact": open_ok,
            }
        )
        selected_values = []
        for view in range(3):
            prompt, options, oracle_index = render_multiple_choice(fact, view)
            if any("correct" in line.casefold() for line in prompt.splitlines()[1:]):
                raise R14Error("R16 option text leaks a correctness marker")
            rendered = _render_chat(tokenizer, choice_system, prompt)
            residual, probabilities, projected = _prompt_end_representation(
                tokenizer, model, rendered, label_ids
            )
            completion, token_count = _generate(tokenizer, model, rendered, 3)
            generated_tokens += token_count
            teacher_output_bytes += len(completion.encode())
            match = CHOICE_PATTERN.match(completion)
            generated_index = ord(match.group(1).upper()) - ord("A") if match else -1
            generated_ok = generated_index == oracle_index
            projected_ok = projected == oracle_index
            namespace = namespace_from_question(prompt.splitlines()[0])
            namespace_ok = namespace == fact.namespace
            choice_generated_exact += int(generated_ok)
            choice_projected_exact += int(projected_ok)
            semantic_exact += int(namespace_ok)
            selected_values.append(options[projected])
            residuals.append(residual)
            rows.append(
                {
                    "kind": "multiple_choice",
                    "fact_id": fact.fact_id,
                    "view": view,
                    "namespace_oracle": fact.namespace,
                    "namespace_predicted": namespace,
                    "question": prompt.splitlines()[0],
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "options": list(options),
                    "oracle_index": oracle_index,
                    "projected_index": projected,
                    "generated_index": generated_index,
                    "canonical_probabilities": probabilities,
                    "completion": completion,
                    "completion_sha256": sha256_bytes(completion.encode()),
                    "residual_sha256": sha256_bytes(residual.numpy().tobytes()),
                    "generated_exact": generated_ok,
                    "projected_exact": projected_ok,
                    "namespace_exact": namespace_ok,
                }
            )
        agreement = len(set(selected_values)) == 1 and selected_values[0] == fact.value
        extracted[fact.fact_id] = {
            "namespace": fact.namespace,
            "relation": fact.relation,
            "entity": fact.entity,
            "value": selected_values[0] if agreement else None,
            "views_agree": agreement,
        }

    stacked = torch.stack(residuals).contiguous()
    bundle = output / "prompt_end_representations.safetensors"
    save_file(
        {"residuals": stacked, "output_rows": output_head},
        str(bundle),
        metadata={
            "format": "abi-r16-public-prompt-end-representations/1",
            "source_model": model_id,
            "source_revision": revision,
        },
    )
    rows_path = output / "observations.jsonl"
    write_jsonl_once(rows_path, rows)
    extracted_path = output / "extracted_facts.json"
    write_json_once(extracted_path, {"facts": extracted})
    elapsed = time.perf_counter() - started
    metrics = {
        "open_exact": open_exact,
        "open_total": len(PUBLIC_FACTS),
        "choice_generated_exact": choice_generated_exact,
        "choice_generated_total": len(PUBLIC_FACTS) * 3,
        "choice_projected_exact": choice_projected_exact,
        "choice_projected_total": len(PUBLIC_FACTS) * 3,
        "semantic_exact": semantic_exact,
        "semantic_total": len(PUBLIC_FACTS) * 3,
        "facts_extracted_exact": sum(int(item["views_agree"]) for item in extracted.values()),
        "facts_total": len(PUBLIC_FACTS),
    }
    verdict = "PASS" if all(
        metrics[key] == metrics[key.replace("_exact", "_total")]
        for key in (
            "open_exact",
            "choice_generated_exact",
            "choice_projected_exact",
            "semantic_exact",
            "facts_extracted_exact",
        )
    ) else "FAIL"
    result = {
        "format": "abi-r16-public-factual-qualification/1",
        "verdict": verdict,
        "claim_ceiling": "PUBLIC_SOURCE_PREREQUISITE_ONLY",
        "source": {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path_name": snapshot.name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "training_steps": 0,
        },
        "metrics": metrics,
        "information_accounting": {
            "raw_prompts": len(rows),
            "teacher_generated_tokens": generated_tokens,
            "teacher_output_bytes": teacher_output_bytes,
            "residual_values": int(stacked.numel()),
            "residual_bytes": int(stacked.numel() * stacked.element_size()),
            "output_head_values": int(output_head.numel()),
            "output_head_bytes": int(output_head.numel() * output_head.element_size()),
            "bundle_bytes": bundle.stat().st_size,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "artifacts": {
            "observations": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
            "extracted_facts": {"path": extracted_path.name, "sha256": sha256_file(extracted_path)},
            "representations": {"path": bundle.name, "sha256": sha256_file(bundle)},
        },
        "protocol_sha256": sha256_file(Path(__file__).with_name("PUBLIC_PROTOCOL.md")),
        "fact_registry_sha256": sha256_bytes(
            canonical_json_bytes([fact.__dict__ for fact in PUBLIC_FACTS])
        ),
        "unproven": [
            "held-out factual extraction",
            "autonomous open-world labeling",
            "teacher-free package execution",
            "LayerCake ingestion",
            "English transfer",
            "global minimality",
            "superiority to LoRA or distillation",
        ],
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument(
        "--revision", default="f2826a00ceef68f0f2b946d945ecc0477ce4450c"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_id, args.revision, args.output), indent=2))


if __name__ == "__main__":
    main()
