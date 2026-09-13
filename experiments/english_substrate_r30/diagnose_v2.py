"""One-shot teacher-native capability-set diagnosis for R30."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch

from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once
from .acquire import MODEL_ID, REVISION
from .protocol import rows


SYSTEM = (
    "You are diagnosing reusable language capabilities. Group behaviorally equivalent user "
    "instructions even when their wording differs. Ignore subjects, names, and numbers. Invent "
    "one abstract capability name per group; no label choices are provided. Return strict JSON "
    "only as {\"groups\":[{\"name\":\"lowercase name\",\"ids\":[\"opaque id\",...]},...]}."
)
SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,47}$")


def _catalog() -> tuple[list[dict[str, str]], dict[str, str]]:
    unique: dict[str, str] = {}
    oracle: dict[str, str] = {}
    for row in rows("train"):
        instruction = str(row["instruction"])
        opaque = hashlib.sha256(("r30-instruction|" + instruction).encode()).hexdigest()[:16]
        unique[opaque] = instruction
        oracle[opaque] = str(row["oracle_task"])
    catalog = [{"id": key, "instruction": value} for key, value in unique.items()]
    random.Random(30_002).shuffle(catalog)
    return catalog, oracle


def _strict_parse(raw: str, ids: set[str]) -> tuple[list[dict], dict[str, str]]:
    value = json.loads(raw)
    if set(value) != {"groups"} or not isinstance(value["groups"], list):
        raise ValueError("diagnosis root schema mismatch")
    mapping: dict[str, str] = {}
    groups = []
    slugs = set()
    for item in value["groups"]:
        if set(item) != {"name", "ids"} or not isinstance(item["name"], str) or not isinstance(item["ids"], list):
            raise ValueError("diagnosis group schema mismatch")
        slug = re.sub(r"[^a-z0-9]+", "-", item["name"].casefold()).strip("-")
        if not SLUG_RE.fullmatch(slug) or slug in slugs:
            raise ValueError("diagnosis capability slug invalid or duplicate")
        slugs.add(slug)
        if len(item["ids"]) != 3 or len(set(item["ids"])) != 3:
            raise ValueError("diagnosis group must contain three unique IDs")
        for opaque in item["ids"]:
            if opaque not in ids or opaque in mapping:
                raise ValueError("diagnosis ID missing, unknown, or duplicated")
            mapping[opaque] = slug
        groups.append({"name": item["name"], "slug": slug, "ids": item["ids"]})
    if len(groups) != 12 or set(mapping) != ids:
        raise ValueError("diagnosis must partition all instructions into twelve groups")
    return groups, mapping


def run(source: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 diagnosis exists: {output}")
    receipt = source / "receipt.json"
    source_rows = source / "source_rows.jsonl"
    if not receipt.is_file() or not source_rows.is_file():
        raise RuntimeError("R30 v1 source evidence missing")
    bound_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    if bound_receipt.get("format") != "abi-r30-public-source/1" or bound_receipt.get("verdict") != "FAIL_SOURCE":
        raise RuntimeError("R30 v2 must bind the preserved v1 source failure")
    if bound_receipt["artifacts"]["source_rows"]["sha256"] != sha256_file(source_rows):
        raise RuntimeError("R30 v1 source rows changed")
    catalog, oracle = _catalog()
    payload = json.dumps({"instructions": catalog}, sort_keys=True, separators=(",", ":"))
    if not torch.cuda.is_available():
        raise RuntimeError("R30 diagnosis requires the declared CUDA source")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    rendered = _render_chat(tokenizer, SYSTEM, payload)
    raw, tokens = _generate(tokenizer, model, rendered, 1800)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    output.mkdir(parents=True)
    raw_path = output / "diagnosis_raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    parse_error = None
    groups: list[dict] = []
    mapping: dict[str, str] = {}
    try:
        groups, mapping = _strict_parse(raw, {row["id"] for row in catalog})
    except (ValueError, json.JSONDecodeError) as error:
        parse_error = f"{type(error).__name__}: {error}"
    pure = 0
    details = []
    if mapping:
        for group in groups:
            tasks = [oracle[item] for item in group["ids"]]
            is_pure = len(set(tasks)) == 1
            pure += int(is_pure)
            details.append({**group, "oracle_tasks": tasks, "oracle_pure": is_pure})
    passed = parse_error is None and pure == 12
    result = {
        "format": "abi-r30-capability-set-diagnosis/2",
        "verdict": "PASS_DIAGNOSIS" if passed else "FAIL_DIAGNOSIS",
        "v1_source_receipt_sha256": sha256_file(receipt),
        "v1_source_rows_sha256": sha256_file(source_rows),
        "source": {"model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot), "teacher_training_steps": 0},
        "interface": {"label_choices_supplied": 0, "task_names_supplied": 0, "instructions": len(catalog), "opaque_ids": len(catalog)},
        "metrics": {"strict_parse": parse_error is None, "parse_error": parse_error, "groups": len(groups), "assigned_ids": len(mapping), "oracle_pure_groups": pure},
        "groups": details,
        "artifacts": {"raw": {"path": raw_path.name, "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}},
        "information_accounting": {"source_calls": 1, "rendered_prompt_bytes": len(rendered.encode()), "teacher_generated_tokens": tokens, "teacher_output_bytes": len(raw.encode()), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "source_parameters_copied": 0, "logits_stored": 0, "hidden_activations_stored": 0},
        "claim_ceiling": "PUBLIC_INSTRUCTION_SET_DIAGNOSIS_ONLY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
