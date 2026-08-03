"""Build the fresh abstention-only Phase 1 V2 source supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_pipeline import canonical_json_bytes
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import PROBE_CATALOG_SCHEMA, load_probe_catalog, probe_label_evidence_sha256
from .natural_english_catalog import BUILDERS, _v3_probe


SUPPLEMENT_RECORDS = 400
DEFAULT_SEED = 2_729_031
EXTRA_ABSTENTION_CONSTRUCTIONS = (
    "cannot be known",
    "can't be known",
    "cannot be determined",
    "can't be determined",
    "cannot be predicted",
    "can't be predicted",
    "does not specify",
    "doesn't specify",
    "insufficient information",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_catalog() -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    wrappers = (
        lambda body, ref: f"Answer this self-contained uncertainty check. Reference {ref}.\n{body}",
        lambda body, ref: f"Use only the stated information for case {ref}.\n{body}",
        lambda body, ref: f"State clearly when the requested detail is unknowable. Item {ref}.\n{body}",
        lambda body, ref: f"Respond to the bounded request without inventing missing facts. Ticket {ref}.\n{body}",
    )
    for local_index in range(SUPPLEMENT_RECORDS):
        source_index = 700_000 + local_index
        family = local_index % 4
        body, evaluator, maximum = BUILDERS["abstention"](source_index, family)
        probe: dict[str, Any] = {
            "probe_id": f"phase1-v2-search-abstention-{local_index:04d}-v1",
            "destination_scope": "english_core",
            "capability": "abstention",
            "canonical_capability": "abstention",
            "domain": "domain_independent",
            "split": "search",
            "prompt": body,
            "max_new_tokens": max(int(maximum), 96),
            "temperature": 0,
            "seed": DEFAULT_SEED + source_index,
            "evaluator": evaluator,
            "record_schema": SEGREGATED_RECORD_SCHEMA,
            "knowledge_class": LINGUISTIC_FORM,
            "content_basis": "supplied_non_domain_context",
            "domain_labels": [],
            "domain_claims": [],
            "label_method": "preregistered_catalog",
            "output_introduces_unsupplied_facts": False,
            "phase1_template_family": f"search_v2:abstention:builder-{family}:wrapper-{family}",
        }
        probe = _v3_probe(probe)
        values = list(probe["evaluator"]["values"])
        probe["evaluator"] = {
            "kind": "contains_any",
            "values": values + [value for value in EXTRA_ABSTENTION_CONSTRUCTIONS if value not in values],
        }
        reference = f"P1S-ABS-{local_index:04d}"
        probe["prompt"] = wrappers[family](str(probe["prompt"]), reference)
        probe["max_new_tokens"] = 192
        probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
        probes.append(probe)
    prompt_hashes = [_sha256(row["prompt"].encode("utf-8")) for row in probes]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("V2 abstention supplement contains duplicate prompts")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "phase1_catalog_format": "abi-capability-compiler-phase1-abstention-supplement/1",
        "catalog_id": "abi-capability-compiler-phase1-abstention-v2",
        "status": "PREREGISTERED_FRESH_SEARCH_ONLY_AFTER_V1_FAILURE",
        "claim_boundary": (
            "This fresh abstention-only supplement tests a versioned evaluator "
            "repair. V1 failed outputs remain failed and are not reclassified."
        ),
        "generation": {
            "generator": "abi.capability_compiler_phase1_abstention_catalog",
            "seed": DEFAULT_SEED,
            "search_records": SUPPLEMENT_RECORDS,
            "validation_records": 0,
            "final_records": 0,
            "prior_prompt_overlap": 0,
            "repair_rounds": 0,
        },
        "observed_v1_constructions_added": list(EXTRA_ABSTENTION_CONSTRUCTIONS),
        "probes": probes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = build_catalog()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    load_probe_catalog(output)
    print(json.dumps({"output": str(output), "records": len(catalog["probes"]), "sha256": _sha256(output.read_bytes())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
