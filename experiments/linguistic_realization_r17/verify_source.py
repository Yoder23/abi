"""Fail-closed recomputation of R17 public source prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.factual_semantic_r16.public_qualification import _render_chat
from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .frames import public_rows
from .frames_v2 import public_rows_v2
from .public_qualification import SYSTEM

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"


def _tokenizer() -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)).resolve()
    if snapshot.name != REVISION:
        raise R14Error("R17 source revision changed")
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error("R17 source rows are unavailable") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise R14Error("R17 source rows changed")
    return rows


def verify_source(run_dir: Path, interface: str) -> dict[str, Any]:
    if interface not in {"v1", "v2"}:
        raise R14Error("unknown R17 interface")
    receipt_path = run_dir / "receipt.json"
    rows_path = run_dir / "source_observations.jsonl"
    if not receipt_path.is_file() or not rows_path.is_file():
        raise R14Error("R17 required source evidence is missing")
    receipt = json_object(receipt_path)
    scientific = {key: value for key, value in receipt.items() if key != "evidence_sha256"}
    if receipt.get("evidence_sha256") != evidence_hash(scientific):
        raise R14Error("R17 receipt evidence hash changed")
    if (
        receipt.get("format") != "abi-r17-public-realization-qualification/1"
        or receipt.get("source", {}).get("model_id") != MODEL_ID
        or receipt.get("source", {}).get("revision") != REVISION
        or receipt.get("source", {}).get("training_steps") != 0
        or receipt.get("artifacts", {}).get("source_rows", {}).get("sha256")
        != sha256_file(rows_path)
    ):
        raise R14Error("R17 receipt identity changed")
    protocol_name = "PUBLIC_PROTOCOL.md" if interface == "v1" else "PUBLIC_PROTOCOL_V2.md"
    if interface == "v2" and (
        receipt.get("interface_revision") != "v2"
        or receipt.get("protocol_sha256") != sha256_file(Path(__file__).with_name(protocol_name))
    ):
        raise R14Error("R17 v2 protocol binding changed")
    builder = public_rows if interface == "v1" else public_rows_v2
    expected_rows = {
        row["record_id"]: row for split in ("extraction", "evaluation") for row in builder(split)
    }
    rows = _read_jsonl(rows_path)
    if len(rows) != 120 or len({row.get("record_id") for row in rows}) != 120:
        raise R14Error("R17 source row count or identity changed")
    tokenizer = _tokenizer()
    counters = {
        "raw_source_prompts": 0,
        "rendered_prompt_utf8_bytes": 0,
        "rendered_prompt_token_instances": 0,
        "teacher_generated_tokens": 0,
        "teacher_output_bytes": 0,
    }
    exact = {"extraction": 0, "evaluation": 0}
    totals = {"extraction": 0, "evaluation": 0}
    for row in rows:
        expected_row = expected_rows.get(row.get("record_id"))
        if expected_row is None:
            raise R14Error("R17 source row is not registered")
        for key in (
            "record_id",
            "split",
            "signature",
            "features",
            "slots",
            "prompt",
            "expected",
        ):
            if row.get(key) != expected_row[key]:
                raise R14Error(f"R17 registered source field changed: {key}")
        rendered = _render_chat(tokenizer, SYSTEM, str(row["prompt"]))
        encoded = tokenizer.encode(rendered, add_special_tokens=False)
        output = row.get("output")
        if (
            not isinstance(output, str)
            or row.get("rendered_prompt_sha256") != hashlib.sha256(rendered.encode()).hexdigest()
            or row.get("rendered_prompt_utf8_bytes") != len(rendered.encode())
            or row.get("input_token_count") != len(encoded)
            or row.get("output_sha256") != hashlib.sha256(output.encode()).hexdigest()
            or not isinstance(row.get("output_token_count"), int)
            or row["output_token_count"] <= 0
            or row.get("functional_exact") != (output == row["expected"])
        ):
            raise R14Error("R17 generated source evidence changed")
        split = str(row["split"])
        totals[split] += 1
        exact[split] += int(row["functional_exact"])
        counters["raw_source_prompts"] += 1
        counters["rendered_prompt_utf8_bytes"] += len(rendered.encode())
        counters["rendered_prompt_token_instances"] += len(encoded)
        counters["teacher_generated_tokens"] += int(row["output_token_count"])
        counters["teacher_output_bytes"] += len(output.encode())
    metrics = {
        "extraction_source_exact": exact["extraction"],
        "extraction_rows": totals["extraction"],
        "evaluation_source_exact": exact["evaluation"],
        "evaluation_rows": totals["evaluation"],
    }
    verdict = (
        "PASS"
        if exact == totals == {"extraction": 72, "evaluation": 48}
        else "FAIL_SOURCE_PREREQUISITE"
    )
    if (
        receipt.get("metrics") != metrics
        or receipt.get("information_accounting") != counters
        or receipt.get("verdict") != verdict
        or any(
            (run_dir / name).exists()
            for name in ("source_bundle.json", "evaluation.jsonl", "extraction")
        )
    ):
        raise R14Error("R17 source verdict or stopped-boundary changed")
    result = {
        "format": "abi-r17-public-source-strict-verification/1",
        "verdict": verdict,
        "interface": interface,
        "source_rows": len(rows),
        "metrics": metrics,
        "information_accounting": counters,
        "receipt_sha256": sha256_file(receipt_path),
        "source_rows_sha256": sha256_file(rows_path),
        "compiler_invoked": False,
        "layercake_invoked": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interface", choices=("v1", "v2"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_source(args.run_dir, args.interface)
    if args.output is not None:
        write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
