"""Build a direct-target curriculum for the same-topology capacity oracle.

This control intentionally removes ABI minimization and open-weight-source
requirements.  It uses the licensed reference answers already shipped with
the immutable Alpaca snapshot for only the prompt surfaces admitted by the
frozen natural-instruction catalog.  The result is oracle-only: it cannot be
composed into an ABI English artifact or credited as open-weight transfer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer

from .hf_extraction import load_probe_catalog
from .layercake_host import _canonical_json_bytes, _sha256_file
from .natural_instruction_catalog import _normalize, _prompt


FORMAT = "abi-same-topology-direct-target-oracle-curriculum/1"
PROVENANCE = "oracle-only-alpaca-licensed-reference-answer"


class OracleCurriculumError(RuntimeError):
    """Raised when oracle curriculum provenance or identity is incomplete."""


def _prompt_index(
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    index: dict[str, Mapping[str, Any]] = {}
    identical_duplicate_rows_collapsed = 0
    for row in source_rows:
        instruction = _normalize(str(row.get("instruction", "")))
        supplied_text = _normalize(str(row.get("input", "")))
        if not instruction or not supplied_text:
            continue
        prompt = _prompt(instruction, supplied_text)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest in index:
            prior_output = _normalize(str(index[digest].get("output", "")))
            current_output = _normalize(str(row.get("output", "")))
            if prior_output != current_output:
                raise OracleCurriculumError(
                    "duplicate source prompt has conflicting reference answers"
                )
            identical_duplicate_rows_collapsed += 1
            continue
        index[digest] = row
    return index, identical_duplicate_rows_collapsed


def build_oracle_curriculum(
    *,
    catalog: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map every frozen catalog prompt to its licensed direct reference."""

    if catalog.get("catalog_id") != "abi-natural-domain-filtered-instruction-search-v1":
        raise OracleCurriculumError("unexpected natural prompt catalog")
    index, identical_duplicate_rows_collapsed = _prompt_index(source_rows)
    curriculum: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    output_token_count = 0
    output_bytes = 0
    prompt_bytes = 0
    output_hashes: list[str] = []
    for probe in catalog.get("probes", []):
        natural_sha = str(probe.get("natural_prompt_sha256", ""))
        source = index.get(natural_sha)
        if source is None:
            raise OracleCurriculumError(
                f"catalog prompt has no exact source row: {probe.get('probe_id')}"
            )
        prompt = str(probe["prompt"])
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != natural_sha:
            raise OracleCurriculumError("catalog natural prompt hash drifted")
        response = _normalize(str(source.get("output", "")))
        if not response:
            raise OracleCurriculumError("source reference answer is empty")
        capability = str(probe["capability"])
        response_sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
        encoded = tokenizer.encode(response)
        token_ids = encoded.ids if hasattr(encoded, "ids") else list(encoded)
        row = {
            "id": f"same-topology-oracle:{probe['probe_id']}",
            "split": "train",
            "task": capability,
            "prompt": prompt,
            "prompt_sha256": natural_sha,
            "response": response,
            "response_sha256": response_sha,
            "teacher_tokens": len(token_ids),
            "provenance": PROVENANCE,
            "oracle_only": True,
            "production_eligible": False,
            "source_probe_id": str(probe["probe_id"]),
        }
        curriculum.append(row)
        counts[capability] += 1
        output_token_count += len(token_ids)
        output_bytes += len(response.encode("utf-8"))
        prompt_bytes += len(prompt.encode("utf-8"))
        output_hashes.append(response_sha)
    if len(curriculum) != len(catalog.get("probes", [])):
        raise OracleCurriculumError("not every catalog probe was retained")
    if len({row["prompt_sha256"] for row in curriculum}) != len(curriculum):
        raise OracleCurriculumError("oracle prompts are not unique")
    accounting = {
        "records": len(curriculum),
        "capability_counts": dict(sorted(counts.items())),
        "unique_prompt_utf8_bytes": prompt_bytes,
        "reference_output_utf8_bytes": output_bytes,
        "reference_output_layercake_tokens": output_token_count,
        "unique_reference_output_hashes": len(set(output_hashes)),
        "identical_duplicate_source_rows_collapsed": (
            identical_duplicate_rows_collapsed
        ),
        "reference_output_sha256_set_sha256": hashlib.sha256(
            "\n".join(sorted(output_hashes)).encode("ascii")
        ).hexdigest(),
    }
    return curriculum, accounting


def _write_immutable(path: Path, raw: str) -> None:
    if path.exists():
        raise OracleCurriculumError(f"oracle output is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    catalog_path = Path(args.catalog).resolve()
    source_path = Path(args.source_rows).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if _sha256_file(source_path) != args.source_sha256.lower():
        raise OracleCurriculumError("source snapshot hash mismatch")
    catalog = load_probe_catalog(catalog_path)
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise OracleCurriculumError("source snapshot must be a JSON list")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows, accounting = build_oracle_curriculum(
        catalog=catalog,
        source_rows=source_rows,
        tokenizer=tokenizer,
    )
    jsonl = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    _write_immutable(output_path, jsonl)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "ORACLE_ONLY_DIRECT_TARGET_CURRICULUM_NOT_ABI_ARTIFACT",
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "catalog_id": catalog["catalog_id"],
        },
        "source": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
            "dataset_id": "yahma/alpaca-cleaned",
            "revision": "12567cabf869d7c92e573c7c783905fc160e9639",
            "license": "cc-by-4.0",
            "reported_reference_generator": "text-davinci-003",
            "open_weight_teacher": False,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": _sha256_file(tokenizer_path),
            "role": "authoritative LayerCake token accounting",
        },
        "curriculum": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            **accounting,
        },
        "claim_boundary": (
            "This direct-target curriculum can test exact-topology LayerCake "
            "capacity only. Its non-open-weight reference generator, unqualified "
            "answer segregation, and lack of ABI minimization make it permanently "
            "ineligible for an ABI artifact or open-weight transfer claim."
        ),
        "domain_segregation_qualified": False,
        "final_test_records": 0,
        "production_eligible": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_immutable(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
