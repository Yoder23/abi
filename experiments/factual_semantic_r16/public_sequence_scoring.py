"""R16 public v2: answer-free generation and candidate-sequence scoring."""

from __future__ import annotations

import argparse
import gc
import json
import time
import unicodedata
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

from .facts import PUBLIC_FACTS, namespace_from_question, question
from .public_qualification import _generate, _load_source, _render_chat


def normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    unaccented = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(unaccented.strip().rstrip(".!\n").casefold().split())


@torch.inference_mode()
def _candidate_scores(
    tokenizer: Any,
    model: Any,
    rendered_prompt: str,
    candidates: tuple[str, ...],
) -> tuple[list[float], torch.Tensor]:
    prompt_ids = tokenizer(
        rendered_prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to("cuda")
    prompt_output = model(
        input_ids=prompt_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    residual = prompt_output.hidden_states[-1][0, -1].float().cpu().contiguous()
    scores = []
    for candidate in candidates:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not candidate_ids:
            raise R14Error("R16 candidate tokenization is empty")
        suffix = torch.tensor([candidate_ids], device="cuda", dtype=prompt_ids.dtype)
        combined = torch.cat((prompt_ids, suffix), dim=1)
        output = model(input_ids=combined, use_cache=False, return_dict=True)
        start = int(prompt_ids.shape[1]) - 1
        token_logits = output.logits[0, start : start + len(candidate_ids)].float()
        targets = suffix[0]
        token_scores = token_logits.log_softmax(dim=-1).gather(1, targets[:, None])[:, 0]
        scores.append(float(token_scores.mean().cpu()))
    return scores, residual


def _candidate_universe(relation: str) -> tuple[str, ...]:
    return tuple(sorted({fact.value for fact in PUBLIC_FACTS if fact.relation == relation}))


def _gates_pass(metrics: dict[str, int]) -> bool:
    pairs = (
        ("open_exact", "open_total"),
        ("candidate_scored_exact", "candidate_scored_total"),
        ("semantic_exact", "semantic_total"),
        ("facts_extracted_exact", "facts_total"),
    )
    return all(metrics[observed] == metrics[required] for observed, required in pairs)


def run(model_id: str, revision: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R16 public v2 output exists: {output}")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(model_id, revision)
    system = "Answer the factual question with only the answer and no explanation."
    rows = []
    residuals = []
    generated_tokens = 0
    teacher_output_bytes = 0
    open_exact = 0
    scored_exact = 0
    semantic_exact = 0
    extracted: dict[str, dict[str, Any]] = {}
    score_values_stored = 0

    for fact in PUBLIC_FACTS:
        candidates = _candidate_universe(fact.relation)
        selected_values = []
        for view in range(3):
            stem = question(fact, view)
            if any(normalized_text(value) in normalized_text(stem) for value in candidates):
                raise R14Error("R16 v2 answer-free prompt contains a candidate value")
            rendered = _render_chat(tokenizer, system, stem)
            completion, token_count = _generate(tokenizer, model, rendered, 12)
            scores, residual = _candidate_scores(tokenizer, model, rendered, candidates)
            predicted = max(range(len(scores)), key=scores.__getitem__)
            generated_ok = normalized_text(completion) == normalized_text(fact.value)
            scored_ok = candidates[predicted] == fact.value
            namespace = namespace_from_question(stem)
            namespace_ok = namespace == fact.namespace
            open_exact += int(generated_ok)
            scored_exact += int(scored_ok)
            semantic_exact += int(namespace_ok)
            generated_tokens += token_count
            teacher_output_bytes += len(completion.encode())
            score_values_stored += len(scores)
            selected_values.append(candidates[predicted])
            residuals.append(residual)
            rows.append(
                {
                    "fact_id": fact.fact_id,
                    "view": view,
                    "namespace_oracle": fact.namespace,
                    "namespace_predicted": namespace,
                    "question": stem,
                    "question_sha256": sha256_bytes(stem.encode()),
                    "completion": completion,
                    "completion_sha256": sha256_bytes(completion.encode()),
                    "oracle_value": fact.value,
                    "candidate_values": list(candidates),
                    "candidate_mean_log_probabilities": scores,
                    "candidate_prediction": candidates[predicted],
                    "residual_sha256": sha256_bytes(residual.numpy().tobytes()),
                    "generated_exact": generated_ok,
                    "scored_exact": scored_ok,
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
    bundle = output / "prompt_end_residuals.safetensors"
    save_file(
        {"residuals": stacked},
        str(bundle),
        metadata={
            "format": "abi-r16-public-answer-free-residuals/1",
            "source_model": model_id,
            "source_revision": revision,
        },
    )
    rows_path = output / "observations.jsonl"
    write_jsonl_once(rows_path, rows)
    extracted_path = output / "extracted_facts.json"
    write_json_once(extracted_path, {"facts": extracted})
    elapsed = time.perf_counter() - started
    total = len(PUBLIC_FACTS) * 3
    metrics = {
        "open_exact": open_exact,
        "open_total": total,
        "candidate_scored_exact": scored_exact,
        "candidate_scored_total": total,
        "semantic_exact": semantic_exact,
        "semantic_total": total,
        "facts_extracted_exact": sum(int(item["views_agree"]) for item in extracted.values()),
        "facts_total": len(PUBLIC_FACTS),
    }
    verdict = "PASS" if _gates_pass(metrics) else "FAIL"
    result = {
        "format": "abi-r16-public-factual-qualification/2",
        "verdict": verdict,
        "claim_ceiling": "PUBLIC_SOURCE_PREREQUISITE_ONLY",
        "source": {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path_name": snapshot.name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": 0,
            "training_steps": 0,
        },
        "metrics": metrics,
        "information_accounting": {
            "raw_answer_free_prompts": len(rows),
            "teacher_generated_tokens": generated_tokens,
            "teacher_output_bytes": teacher_output_bytes,
            "candidate_score_values": score_values_stored,
            "residual_values": int(stacked.numel()),
            "residual_bytes": int(stacked.numel() * stacked.element_size()),
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
