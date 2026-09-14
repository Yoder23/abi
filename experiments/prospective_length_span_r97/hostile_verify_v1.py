"""Exercise R97 fail-closed behavior against physically altered shadow trees."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.prospective_length_span_r97.verify_evidence_v1 import VerificationError, verify


BASE_FILES = {
    "results/length_invariant_span_r96/candidate_v1/bridge.safetensors",
    "results/length_invariant_span_r96/candidate_v1/metadata.json",
    "results/length_invariant_span_r96/candidate_binding_v1.json",
    "results/role_invariant_choice_r76/candidate_v1/model.safetensors",
    "results/role_invariant_choice_r76/candidate_v1/metadata.json",
    "results/role_invariant_choice_r76/candidate_v1/merges.txt",
    "results/role_invariant_choice_r76/candidate_v1/special_tokens_map.json",
    "results/role_invariant_choice_r76/candidate_v1/tokenizer.json",
    "results/role_invariant_choice_r76/candidate_v1/tokenizer_config.json",
    "results/role_invariant_choice_r76/candidate_v1/vocab.json",
    "results/role_invariant_choice_r76/reasoning-role-invariant-search-v2.abix",
    "results/prospective_length_span_r97/candidate_binding_v1.json",
    "results/prospective_length_span_r97/screen_v1/result.json",
    "results/prospective_length_span_r97/screen_v1/evaluation.jsonl",
    "results/prospective_length_span_r97/source_v1/result.json",
    "results/prospective_length_span_r97/source_v1/prior_corrected_scores.jsonl",
    "catalogs/prospective_length_span_r97_v1.json",
}


CASES = (
    ("missing_result", "results/prospective_length_span_r97/screen_v1/result.json", "missing"),
    ("tampered_result_boolean", "results/prospective_length_span_r97/screen_v1/result.json", "json_boolean"),
    ("missing_raw_rows", "results/prospective_length_span_r97/screen_v1/evaluation.jsonl", "missing"),
    ("truncated_raw_rows", "results/prospective_length_span_r97/screen_v1/evaluation.jsonl", "truncate"),
    ("missing_source_result", "results/prospective_length_span_r97/source_v1/result.json", "missing"),
    ("tampered_source_evidence", "results/prospective_length_span_r97/source_v1/result.json", "json_evidence"),
    ("missing_source_raw", "results/prospective_length_span_r97/source_v1/prior_corrected_scores.jsonl", "missing"),
    ("tampered_binding", "results/prospective_length_span_r97/candidate_binding_v1.json", "append"),
    ("missing_candidate_checkpoint", "results/length_invariant_span_r96/candidate_v1/bridge.safetensors", "missing"),
    ("missing_tokenizer", "results/role_invariant_choice_r76/candidate_v1/tokenizer.json", "missing"),
    ("tampered_catalog", "catalogs/prospective_length_span_r97_v1.json", "append"),
    ("missing_bound_adapter", "experiments/length_invariant_span_r96/core_v1.py", "missing"),
)


def _all_files(root: Path) -> set[str]:
    files = set(BASE_FILES)
    binding = json.loads((root / "results/prospective_length_span_r97/candidate_binding_v1.json").read_text(encoding="utf-8"))
    files.update(str(item["path"]) for item in binding["files"].values())
    return files


def _shadow(root: Path, target: Path, altered: str, mode: str) -> None:
    for relative in _all_files(root):
        source = root / relative; destination = target / relative
        if relative == altered and mode == "missing":
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == altered:
            shutil.copy2(source, destination)
            if mode == "truncate":
                data = destination.read_bytes(); destination.write_bytes(data[: len(data) // 2])
            elif mode == "append":
                with destination.open("ab") as handle: handle.write(b"\nHOSTILE\n")
            elif mode.startswith("json_"):
                value = json.loads(destination.read_text(encoding="utf-8"))
                if mode == "json_boolean": value["gates"]["candidate_quality"] = False
                else: value["evidence_sha256"] = "0" * 64
                destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                raise ValueError(mode)
        else:
            os.link(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists(): parser.error("immutable hostile receipt exists")
    pristine = verify(root)
    observed = []
    with tempfile.TemporaryDirectory(prefix="abi-r97-hostile-") as temporary:
        base = Path(temporary)
        for index, (name, altered, mode) in enumerate(CASES):
            shadow = base / f"case-{index:02d}"; _shadow(root, shadow, altered, mode)
            try:
                verify(shadow)
            except (VerificationError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
                observed.append({"case": name, "verdict": "REJECTED", "error_type": type(error).__name__})
            else:
                raise RuntimeError(f"hostile case was accepted: {name}")
    receipt = {"format": "abi-r97-hostile-verifier-audit/1", "verdict": "PASS_R97_HOSTILE_AUDIT", "pristine_rows": pristine["rows"], "hostile_cases": len(CASES), "rejected_cases": len(observed), "accepted_cases": 0, "observations": observed, "full_abi_moonshot": "OPEN", "claim_boundary": "Fail-closed audit of bounded R97 evidence only."}
    receipt["evidence_sha256"] = evidence_hash(receipt); write_json_once(args.output, receipt); print(json.dumps(receipt, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
