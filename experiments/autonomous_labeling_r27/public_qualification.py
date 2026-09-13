"""Run the disclosed R27 free-label source-interface qualification."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.factual_semantic_r16.public_sequence_scoring import normalized_text
from experiments.foreign_capability_r14.core import R14Error, evidence_hash, write_json_once, write_jsonl_once

from .facts import PUBLIC_FACTS, SYSTEM_PROMPT


MODEL_ID = "Qwen/Qwen2-7B-Instruct"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
ACCEPTED_LABELS = {
    "chemistry": {"chemistry"},
    "geography": {"geography"},
    "mathematics": {
        "mathematics", "arithmetic", "algebra", "geometry", "calculus",
        "statistics", "probability", "numbertheory",
    },
    "python": {"python"},
}


def _parse(text: str) -> tuple[str, str]:
    value = json.loads(text.strip())
    if not isinstance(value, dict) or set(value) != {"answer", "domain"}:
        raise ValueError("response is not the registered two-field object")
    answer, domain = value["answer"], value["domain"]
    if not isinstance(answer, str) or not isinstance(domain, str):
        raise ValueError("response fields must be strings")
    if not re.fullmatch(r"[a-z]+", domain):
        raise ValueError("domain is not one lowercase ASCII word")
    return answer.strip(), domain


def _answer_equivalent(candidate: str, expected: str) -> bool:
    """Normalize only unit suffixes and call punctuation with no semantic content."""
    left = normalized_text(candidate)
    right = normalized_text(expected)
    if left.endswith("()"):
        left = left[:-2].rstrip()
    if right.endswith("()"):
        right = right[:-2].rstrip()
    if left.endswith(" degrees") and right.replace(".", "", 1).isdigit():
        left = left[:-8].rstrip()
    if right.endswith(" degrees") and left.replace(".", "", 1).isdigit():
        right = right[:-8].rstrip()
    return left == right


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R27 public output exists: {output}")
    output.mkdir(parents=True)
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    generated_tokens = 0
    output_bytes = 0
    for fact in PUBLIC_FACTS:
        for view, question in enumerate(fact.extraction_questions):
            rendered = _render_chat(tokenizer, SYSTEM_PROMPT, question)
            completion, tokens = _generate(tokenizer, model, rendered, 32)
            try:
                answer, label = _parse(completion)
                parse_exact = True
            except (ValueError, json.JSONDecodeError):
                answer, label, parse_exact = "", "", False
            rows.append({
                "fact_id": fact.fact_id,
                "oracle_domain": fact.oracle_domain,
                "view": view,
                "question": question,
                "oracle_answer": fact.answer,
                "completion": completion,
                "parse_exact": parse_exact,
                "answer": answer,
                "answer_exact": _answer_equivalent(answer, fact.answer),
                "free_label": label,
                "label_exact": label in ACCEPTED_LABELS[fact.oracle_domain],
            })
            generated_tokens += tokens
            output_bytes += len(completion.encode("utf-8"))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    path = output / "source_rows.jsonl"
    write_jsonl_once(path, rows)
    metrics = {
        "rows": len(rows),
        "parse_exact": sum(row["parse_exact"] for row in rows),
        "answer_exact": sum(row["answer_exact"] for row in rows),
        "label_exact": sum(row["label_exact"] for row in rows),
        "facts_with_label_consensus": sum(
            len({row["free_label"] for row in rows if row["fact_id"] == fact.fact_id}) == 1
            for fact in PUBLIC_FACTS
        ),
        "oracle_domains_semantically_valid": sum(
            len({row["free_label"] for row in rows if row["oracle_domain"] == domain}) >= 1
            and all(
                row["free_label"] in ACCEPTED_LABELS[domain]
                for row in rows if row["oracle_domain"] == domain
            )
            for domain in ACCEPTED_LABELS
        ),
        "distinct_labels": len({row["free_label"] for row in rows}),
    }
    passed = metrics == {
        "rows": 24, "parse_exact": 24, "answer_exact": 24, "label_exact": 24,
        "facts_with_label_consensus": 8, "oracle_domains_semantically_valid": 4, "distinct_labels": 5,
    }
    result = {
        "format": "abi-r27-public-free-label-qualification/2",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "DISCLOSED_FREE_LABEL_SOURCE_INTERFACE_ONLY",
        "claim_ceiling": "NOT_HELDOUT_AUTONOMOUS_DISCOVERY_OR_LAYERCAKE_IMPORT",
        "source": {"model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot)},
        "metrics": metrics,
        "information_accounting": {
            "source_calls": len(rows), "generated_tokens": generated_tokens,
            "output_bytes": output_bytes, "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "rows": {"path": path.name, "bytes": path.stat().st_size},
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
