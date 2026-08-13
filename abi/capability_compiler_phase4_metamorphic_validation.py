"""Build one model-blind, evaluator-derived Phase 4 coherence validation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-metamorphic-validation-build/1"
SUITE_FORMAT = "abi-capability-compiler-phase4-metamorphic-coherence-suite/1"
NAMES = ("Mira", "Jon", "Asha", "Luis", "Nora", "Omar", "Priya", "Theo", "Uma", "Wei")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_MODEL_BLIND_METAMORPHIC_SUITE_BUILD"
        or protocol.get("system_inference_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_output_access") != "PROHIBITED"
        or protocol.get("training_authorized") is not False
    ):
        raise Phase3Error("metamorphic suite governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"metamorphic suite binding changed: {relative}")
    return protocol, sha256_file(path)


def _coherence_row(base: int, family: int, sample: int) -> dict[str, Any]:
    index = base + family * 10 + sample
    name = NAMES[index % len(NAMES)]
    code = f"N{index:06d}{name.upper()}"
    sequences = (
        (("PREP", "ACT", "DONE"), (f"[{code}-ACT] {name} entered the room", f"[{code}-DONE] the conversation ended", f"[{code}-PREP] {name} opened the door")),
        (("START", "MIDDLE", "END"), (f"[{code}-END] the parcel was sealed", f"[{code}-START] the box was opened", f"[{code}-MIDDLE] the note was placed inside")),
        (("FIRST", "NEXT", "LAST"), (f"[{code}-NEXT] {name} read the message", f"[{code}-LAST] {name} replied", f"[{code}-FIRST] the message arrived")),
        (("ONE", "TWO", "THREE"), (f"[{code}-THREE] the lights went out", f"[{code}-ONE] the last visitor left", f"[{code}-TWO] the door was locked")),
    )
    labels, events = sequences[family]
    body = "Put the event labels in logical order. Return the labels in order without commentary: " + "; ".join(events) + "."
    namespace = f"N{base // 1000:03d}"
    record_id = f"phase4-metamorphic-coherence-{namespace.lower()}-f{family}-s{sample:02d}"
    prompt = f"Independent nonce-ordering check {record_id}.\n{body}"
    return {
        "suite_format": SUITE_FORMAT,
        "ir_record_id": record_id,
        "capability": "coherence",
        "namespace": namespace,
        "family": family,
        "sample": sample,
        "normalized_generation_prompt": prompt,
        "generation_max_new_tokens": 192,
        "functional_evaluator": {"kind": "ordered_contains", "values": [f"{code}-{label}" for label in labels]},
        "training_eligible": False,
        "selection_eligible": True,
        "teacher_output_present": False,
    }


def build_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _coherence_row(int(base), family, sample)
        for base in protocol["namespace_bases"]
        for family in range(int(protocol["families"]))
        for sample in range(int(protocol["samples_per_family_namespace"]))
    ]
    if len({row["ir_record_id"] for row in rows}) != len(rows):
        raise Phase3Error("metamorphic record IDs are not unique")
    return rows


def _abicir_records(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        if "records.jsonl" not in archive.namelist():
            raise Phase3Error(f"governed ABI IR lacks records.jsonl: {path}")
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise Phase3Error(f"governed ABI IR contains a non-object row: {path}")
    return rows


def _governed_inventory(root: Path, protocol: Mapping[str, Any]) -> tuple[set[str], set[str], dict[str, int]]:
    prompts: set[str] = set()
    evaluators: set[str] = set()
    counts: dict[str, int] = {}

    def add(source: str, rows: Iterable[Mapping[str, Any]], prompt_keys: tuple[str, ...], evaluator_keys: tuple[str, ...]) -> None:
        count = 0
        for row in rows:
            count += 1
            for key in prompt_keys:
                value = row.get(key)
                if isinstance(value, str):
                    prompts.add(hashlib.sha256(value.encode("utf-8")).hexdigest())
            for key in evaluator_keys:
                value = row.get(key)
                if isinstance(value, dict):
                    evaluators.add(hashlib.sha256(canonical_json_bytes(value)).hexdigest())
        counts[source] = count

    phase1 = _json(root / protocol["governed_sources"]["phase1_catalog"])
    add("phase1_probes", phase1["probes"], ("prompt",), ("evaluator",))
    add("phase1_domain_isolation", phase1["domain_isolation_probes"], ("prompt",), ())
    add("phase1_adversarial", phase1["adversarial_probes"], ("prompt",), ())
    targeted = _json(root / protocol["governed_sources"]["targeted_catalog"])
    add("targeted_probes", targeted["probes"], ("prompt",), ("evaluator",))
    for label, relative in (("phase1_ir", protocol["governed_sources"]["phase1_ir"]), ("targeted_ir", protocol["governed_sources"]["targeted_ir"]), ("host_supervision", protocol["governed_sources"]["host_supervision"])):
        records = _abicir_records(root / relative)
        add(label, records, ("normalized_generation_prompt", "host_prompt", "prompt"), ("functional_evaluator", "evaluator"))
    return prompts, evaluators, counts


def build(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable metamorphic output already exists")
    rows = build_rows(protocol)
    expected = int(protocol["expected_records"])
    if len(rows) != expected:
        raise Phase3Error("metamorphic suite depth changed")
    governed_prompts, governed_evaluators, governed_counts = _governed_inventory(root, protocol)
    suite_prompts = [hashlib.sha256(row["normalized_generation_prompt"].encode("utf-8")).hexdigest() for row in rows]
    suite_evaluators = [hashlib.sha256(canonical_json_bytes(row["functional_evaluator"])).hexdigest() for row in rows]
    gates = {
        "expected_depth": len(rows) == expected,
        "ten_namespaces": len({row["namespace"] for row in rows}) == 10,
        "four_families": len({row["family"] for row in rows}) == 4,
        "unique_prompts": len(suite_prompts) == len(set(suite_prompts)),
        "unique_evaluators": len(suite_evaluators) == len(set(suite_evaluators)),
        "prompt_hash_disjoint": not (set(suite_prompts) & governed_prompts),
        "evaluator_hash_disjoint": not (set(suite_evaluators) & governed_evaluators),
        "no_training_eligible_rows": not any(row["training_eligible"] for row in rows),
        "no_teacher_outputs": not any(row["teacher_output_present"] for row in rows),
        "no_system_inference": True,
        "no_final_output_access": True,
    }
    output.mkdir(parents=True)
    suite = output / "suite.jsonl"
    _write_immutable(suite, b"".join(canonical_json_bytes(row) for row in rows))
    manifest = {
        "format": "abi-capability-compiler-phase4-metamorphic-validation-manifest/1",
        "status": "PASS_MODEL_BLIND_METAMORPHIC_SUITE_BUILD" if all(gates.values()) else "FAIL_METAMORPHIC_SUITE_BUILD",
        "protocol_sha256": protocol_sha,
        "records": len(rows),
        "records_per_namespace": {name: sum(row["namespace"] == name for row in rows) for name in sorted({row["namespace"] for row in rows})},
        "records_per_family": {str(family): sum(row["family"] == family for row in rows) for family in range(4)},
        "governed_inventory": governed_counts,
        "suite_sha256": sha256_file(suite),
        "gates": gates,
        "system_inference_performed": False,
        "teacher_model_loaded": False,
        "training_performed": False,
        "final_outputs_accessed": False,
        "claim_boundary": "Model-blind validation materialization only; no route comparison, candidate construction, promotion, final test, Phase 4 certificate, or superiority claim.",
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_immutable(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build(Path.cwd().resolve(), Path(args.protocol).resolve(), Path(args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
