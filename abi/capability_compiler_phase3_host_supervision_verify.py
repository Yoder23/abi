"""Independent hostile verifier for the V480 host-supervision artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_functional_v2 import NUMBER_WORDS, evaluate_functional_v2
from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase1_ir import normalize_text


FORMAT = "abi-capability-compiler-phase3-host-supervision-verifier/1"
WEAK = ("abstention", "coherence", "fluent_realization", "tone_control")
ENTRIES = ("accounting.json", "manifest.json", "records.jsonl")
BUILDER = re.compile(r"builder-(\d+)")
TARGETED_PREFIXES = (
    "Complete this new bounded English practice task.",
    "Use only the material supplied in search item",
    "Follow the exact requested wording and format for item",
    "Respond directly to this independent English search task",
    "Work only from the text below. Search case",
    "Give only the requested answer for new exercise",
    "Read the supplied context, then answer search ticket",
    "Handle this language task without outside facts. Search ref",
)
META_MARKERS = ("<prior_answer>", "<machine_requirements>", "Repair one answer")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_HOSTILE_ARTIFACT_VERIFICATION"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("host supervision verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"host supervision verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _builder(family: str) -> int:
    match = BUILDER.search(family)
    if match is None:
        raise Phase3Error("artifact family lacks builder identity")
    return int(match.group(1))


def independent_host_projection(prompt: str) -> str:
    first, separator, remainder = prompt.partition("\n")
    if not separator or not first.startswith(TARGETED_PREFIXES) or not remainder.strip():
        raise Phase3Error("source prompt is not a bounded targeted task")
    return remainder.strip()


def independent_surface_repair(output: str, evaluator: Mapping[str, Any], capability: str) -> tuple[str, tuple[str, ...]]:
    if evaluate_functional(output, evaluator):
        return output, ()
    if not evaluate_functional_v2(output, dict(evaluator), capability):
        raise Phase3Error("source output is not V2 valid")
    value = output
    steps: list[str] = []
    for word, digit in NUMBER_WORDS.items():
        updated = re.sub(rf"\b{re.escape(word)}\b", digit, value, flags=re.IGNORECASE)
        if updated != value:
            steps.append(f"number_word_{word}_to_{digit}")
            value = updated
    if capability == "abstention":
        for pattern in (r"\bcannot be known\b", r"\bcan not be known\b", r"\bcan't be known\b"):
            updated = re.sub(pattern, "cannot determine", value, flags=re.IGNORECASE)
            if updated != value:
                steps.append("abstention_cannot_be_known_to_cannot_determine")
                value = updated
    if not evaluate_functional(value, evaluator):
        raise Phase3Error("closed verifier normalization cannot recover V1 validity")
    return value, tuple(steps)


def _archive_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if tuple(sorted(names)) != ENTRIES or len(names) != len(set(names)):
            raise Phase3Error("artifact ZIP entry set changed")
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if info.flag_bits & 1 or pure.is_absolute() or ".." in pure.parts:
                raise Phase3Error("artifact ZIP contains unsafe metadata")
        return {name: archive.read(name) for name in names}


def _expected_selection(
    protocol_sha: str,
    depth: int,
    catalog: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    probes = {str(row["probe_id"]): row for row in catalog}
    by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_probe[str(row["probe_id"])].append(row)
    eligible: dict[tuple[str, int], list[tuple[str, str, str]]] = defaultdict(list)
    for probe_id, values in by_probe.items():
        probe = probes[probe_id]
        capability = str(values[0]["canonical_capability"])
        if capability not in WEAK:
            continue
        builder = _builder(str(probe["phase3_targeted_template_family"]))
        for attempt in sorted(values, key=lambda row: (int(row["attempt_index"]), str(row["attempt_sha256"]))):
            if attempt.get("finish_reason") != "eos_token":
                continue
            normalized, _ = normalize_text(str(attempt["output"]))
            try:
                independent_surface_repair(normalized, attempt["functional_evaluator"], capability)
            except Phase3Error:
                continue
            key = hashlib.sha256(f"{protocol_sha}:{probe_id}:{attempt['attempt_sha256']}".encode("ascii")).hexdigest()
            eligible[(capability, builder)].append((key, probe_id, str(attempt["attempt_sha256"])))
            break
    selected: set[tuple[str, str]] = set()
    for capability in WEAK:
        for builder in range(4):
            values = sorted(eligible[(capability, builder)])
            if len(values) < depth:
                raise Phase3Error("source no longer satisfies verifier selection depth")
            selected.update((probe_id, attempt_sha) for _, probe_id, attempt_sha in values[:depth])
    return selected


def verify_entries(
    entries: Mapping[str, bytes],
    *,
    build_protocol_sha: str,
    depth: int,
    catalog: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    if tuple(sorted(entries)) != ENTRIES:
        raise Phase3Error("artifact payload entry set changed")
    manifest = json.loads(entries["manifest.json"])
    accounting = json.loads(entries["accounting.json"])
    rows = [json.loads(line) for line in entries["records.jsonl"].splitlines() if line]
    if manifest.get("protocol_sha256") != build_protocol_sha or manifest.get("records") != len(rows):
        raise Phase3Error("artifact manifest identity changed")
    if hashlib.sha256(entries["records.jsonl"]).hexdigest() != manifest.get("records_jsonl_sha256"):
        raise Phase3Error("artifact record payload hash changed")
    if hashlib.sha256(entries["accounting.json"]).hexdigest() != manifest.get("accounting_sha256"):
        raise Phase3Error("artifact accounting hash changed")
    probes = {str(row["probe_id"]): row for row in catalog}
    attempt_by_sha = {str(row["attempt_sha256"]): row for row in attempts}
    expected_pairs = _expected_selection(build_protocol_sha, depth, catalog, attempts)
    actual_pairs: set[tuple[str, str]] = set()
    record_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    counts = Counter()
    surface = repairs = teacher_input = teacher_output = output_bytes = 0
    for row in rows:
        if row.get("format") != "abi-host-conformant-weak-supervision-record/1":
            raise Phase3Error("artifact record format changed")
        payload = dict(row)
        record_id = str(payload.pop("record_id"))
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != record_id or record_id in record_ids:
            raise Phase3Error("artifact record identity invalid")
        record_ids.add(record_id)
        probe_id = str(row["probe_id"])
        attempt_sha = str(row["source_attempt_sha256"])
        probe = probes.get(probe_id)
        attempt = attempt_by_sha.get(attempt_sha)
        if probe is None or attempt is None or str(attempt["probe_id"]) != probe_id:
            raise Phase3Error("artifact provenance link invalid")
        capability = str(row["capability"])
        builder = int(row["builder"])
        if capability not in WEAK or builder != _builder(str(probe["phase3_targeted_template_family"])):
            raise Phase3Error("artifact capability-family label invalid")
        normalized_prompt, prompt_steps = normalize_text(str(probe["prompt"]))
        host_prompt = independent_host_projection(normalized_prompt)
        if row["host_prompt"] != host_prompt or row["prompt_normalization_steps"] != prompt_steps:
            raise Phase3Error("artifact host projection invalid")
        if any(marker in host_prompt for marker in META_MARKERS):
            raise Phase3Error("teacher repair meta-prompt leaked into host input")
        if hashlib.sha256(host_prompt.encode()).hexdigest() != row["host_prompt_sha256"]:
            raise Phase3Error("artifact host prompt hash invalid")
        normalized_output, output_steps = normalize_text(str(attempt["output"]))
        expected_output, surface_steps = independent_surface_repair(normalized_output, attempt["functional_evaluator"], capability)
        if row["output"] != expected_output or row["output_normalization_steps"] != output_steps or row["surface_normalization_steps"] != list(surface_steps):
            raise Phase3Error("artifact output normalization invalid")
        if not evaluate_functional_v2(normalized_output, attempt["functional_evaluator"], capability) or not evaluate_functional(expected_output, attempt["functional_evaluator"]):
            raise Phase3Error("artifact functional validity invalid")
        if row["source_authoritative_generated_token_ids"] != attempt["authoritative_generated_token_ids"] or len(attempt["authoritative_generated_token_ids"]) != int(attempt["teacher_tokens"]):
            raise Phase3Error("artifact authoritative token accounting invalid")
        if not row.get("teacher_generation_prompt_excluded_from_host_input") or row.get("teacher_present_at_host_training_or_inference"):
            raise Phase3Error("artifact teacher isolation claim invalid")
        actual_pairs.add((probe_id, attempt_sha))
        prompt_hashes.add(str(row["host_prompt_sha256"]))
        counts[(capability, builder)] += 1
        surface += int(bool(surface_steps))
        repairs += int(str(attempt["kind"]) != "initial")
        teacher_input += int(attempt["teacher_input_tokens"])
        teacher_output += int(attempt["teacher_tokens"])
        output_bytes += len(expected_output.encode())
    expected_counts = Counter({(capability, builder): depth for capability in WEAK for builder in range(4)})
    if counts != expected_counts or actual_pairs != expected_pairs or len(rows) != depth * 16:
        raise Phase3Error("artifact deterministic selection or balance invalid")
    expected_accounting = {
        "format": "abi-host-conformant-weak-supervision-accounting/1",
        "records": len(rows),
        "unique_prompt_bytes": sum(len(str(row["host_prompt"]).encode()) for row in {str(row["host_prompt_sha256"]): row for row in rows}.values()),
        "teacher_output_bytes": output_bytes,
        "teacher_input_tokens": teacher_input,
        "teacher_output_tokens": teacher_output,
        "surface_normalized_records": surface,
        "repair_attempt_records": repairs,
        "stored_logits": 0,
        "stored_hidden_activations": 0,
        "copied_source_parameters": 0,
        "teacher_model_loaded_for_this_build": False,
    }
    if accounting != expected_accounting:
        raise Phase3Error("artifact information accounting invalid")
    return {"records": len(rows), "counts": {f"{capability}:{builder}": counts[(capability, builder)] for capability in WEAK for builder in range(4)}, "accounting": accounting, "record_ids": len(record_ids), "prompt_hashes": len(prompt_hashes)}


def _expect_rejection(entries: dict[str, bytes], **kwargs: Any) -> bool:
    try:
        verify_entries(entries, **kwargs)
    except (Phase3Error, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return True
    return False


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable host supervision verifier output exists: {output}")
    artifact = root / protocol["artifact"]["path"]
    entries = _archive_entries(artifact)
    catalog = _json(root / protocol["source"]["catalog"])["probes"]
    attempts = _jsonl(root / protocol["source"]["journal"])
    manifest = json.loads(entries["manifest.json"])
    if (
        manifest.get("source_catalog_sha256") != protocol["source"]["catalog_sha256"]
        or manifest.get("source_journal_sha256") != protocol["source"]["journal_sha256"]
    ):
        raise Phase3Error("artifact source identity changed")
    kwargs = {"build_protocol_sha": protocol["artifact"]["build_protocol_sha256"], "depth": int(protocol["artifact"]["records_per_capability_builder"]), "catalog": catalog, "attempts": attempts}
    verified = verify_entries(entries, **kwargs)
    attacks: dict[str, bool] = {}
    extra = dict(entries); extra["unexpected.bin"] = b"x"; attacks["extra_entry_rejected"] = _expect_rejection(extra, **kwargs)
    corrupt_manifest = dict(entries); manifest = json.loads(entries["manifest.json"]); manifest["records"] += 1; corrupt_manifest["manifest.json"] = json.dumps(manifest).encode(); attacks["manifest_mutation_rejected"] = _expect_rejection(corrupt_manifest, **kwargs)
    corrupt_record = dict(entries); rows = entries["records.jsonl"].splitlines(); first = json.loads(rows[0]); first["output"] += " tampered"; rows[0] = canonical_json_bytes(first); corrupt_record["records.jsonl"] = b"\n".join(rows) + b"\n"; attacks["record_mutation_rejected"] = _expect_rejection(corrupt_record, **kwargs)
    corrupt_accounting = dict(entries); accounting = json.loads(entries["accounting.json"]); accounting["teacher_output_tokens"] += 1; corrupt_accounting["accounting.json"] = json.dumps(accounting).encode(); attacks["accounting_mutation_rejected"] = _expect_rejection(corrupt_accounting, **kwargs)
    if not all(attacks.values()):
        raise Phase3Error("hostile mutation control escaped verifier")
    result = {
        "format": FORMAT,
        "status": "PASS_INDEPENDENT_HOSTILE_ARTIFACT_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "artifact_sha256": sha256_file(artifact),
        "verified": verified,
        "adversarial_attacks": attacks,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_HOST_SUPERVISION_VERIFY_PROTOCOL_V481.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_host_supervision/verify_v482/result.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
