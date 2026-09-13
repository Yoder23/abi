"""Live teacher-free validated cascade over three R31 package variants."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.english_substrate_r30 import package_v4 as base
from experiments.english_substrate_r30 import package_v7 as v7
from experiments.english_substrate_r30.protocol import TASKS
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once, write_jsonl_once
from .ladder import MAXIMUM_ACTIONS


VARIANTS = ("first24", "first48", "mdl24")


def _infer_task(prompt: str) -> str:
    lowered = prompt.casefold()
    if "option_a=" in lowered and "option_b=" in lowered:
        return "comparison"
    if "rule=every glim" in lowered:
        return "reasoning"
    if "supplied_context=" in lowered:
        return "abstention"
    if "missing=" in lowered:
        return "clarification"
    if "tone=" in lowered:
        return "tone"
    if "goal=" in lowered and "resource=" in lowered and "constraint=" in lowered:
        return "planning"
    if "first=" in lowered and "second=" in lowered and "third=" in lowered:
        return "bullets"
    if "recipient=" in lowered and "sender=" in lowered:
        return "email"
    if "message=" in lowered and "preference=" in lowered:
        return "conversation"
    instruction = lowered.splitlines()[0]
    if any(value in instruction for value in ("summar", "condense")):
        return "summary"
    if any(value in instruction for value in ("rewrite", "paraphrase", "improve")):
        return "rewrite"
    if any(value in instruction for value in ("prose", "sentence", "combine")):
        return "prose"
    raise RuntimeError("R31 runtime could not infer a registered prompt contract")


def _runtime_score(prompt: str, output: str) -> tuple[str, dict[str, Any]]:
    task = _infer_task(prompt)
    nonce_match = re.search(r"Virelon\d+", prompt)
    if nonce_match is None:
        raise RuntimeError("R31 prompt has no runtime identifier")
    pseudo = {"oracle_task": task, "prompt": prompt, "nonce": nonce_match.group(0), "details": prompt.splitlines()[2:]}
    return task, v7._score(pseudo, output, "")


def run(source: Path, diagnosis: Path, ladder_result: Path, mdl_result: Path, output: Path):
    if output.exists():
        raise RuntimeError(f"immutable R31 cascade exists: {output}")
    ladder_doc = json.loads(ladder_result.read_text(encoding="utf-8"))
    mdl_doc = json.loads(mdl_result.read_text(encoding="utf-8"))
    if ladder_doc.get("verdict") != "FAIL_PUBLIC_SUFFICIENCY_LADDER" or mdl_doc.get("verdict") != "FAIL_PUBLIC_MDL_SELECTION":
        raise RuntimeError("R31 cascade prerequisites changed")
    rows_path = source / "accepted_rows.jsonl"
    rows = [row for row in base._jsonl(rows_path) if row["split"] == "evaluation"]
    diagnosed = json.loads((diagnosis / "result.json").read_text(encoding="utf-8"))
    mapping = {opaque: group["name"] for group in diagnosed["groups"] for opaque in group["ids"]}
    router = base.CompactRouter.fit([(row["instruction"], mapping[base._instruction_id(row["instruction"])]) for row in base._jsonl(rows_path) if row["split"] == "train"])
    roots = {
        "first24": ladder_result.parent / "level-24",
        "first48": ladder_result.parent / "level-48",
        "mdl24": mdl_result.parent / "level-24",
    }
    expected_hashes = {
        "first24": {item["cluster"]: item["sha256"] for item in ladder_doc["levels"][0]["packages"]},
        "first48": {item["cluster"]: item["sha256"] for item in ladder_doc["levels"][1]["packages"]},
        "mdl24": {item["cluster"]: item["sha256"] for item in mdl_doc["level"]["packages"]},
    }
    api = base._layercake(Path(__file__).resolve().parents[2])
    public = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED).public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public)
    output.mkdir(parents=True)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    attempts, selected_rows = [], []
    with tempfile.TemporaryDirectory(prefix="r31-cascade-") as raw:
        hosts = {variant: api["DirectCakeHost"](Path(raw) / variant, abi_version=base.ABI_VERSION, abi_hash=base.ABI_SHA256, trust_store={signer: public}, device="cuda") for variant in VARIANTS}
        installed = {variant: {} for variant in VARIANTS}
        for variant in VARIANTS:
            for cluster, expected in expected_hashes[variant].items():
                package = roots[variant] / "packages" / f"{cluster}.cake"
                if sha256_file(package) != expected:
                    raise RuntimeError("R31 cascade package hash changed")
                record = hosts[variant].install(package)
                if record["archive_hash"] != expected:
                    raise RuntimeError("R31 cascade install hash changed")
                installed[variant][cluster] = record["cake_id"]
        for index, row in enumerate(rows, 1):
            predicted = router.predict(row["instruction"])
            expected = mapping[base._instruction_id(row["instruction"])]
            selected = None
            for variant in VARIANTS:
                generated = hosts[variant].generate(installed[variant][predicted], v7._normalized_prompt(row), maximum_actions=MAXIMUM_ACTIONS)
                text = generated.output.decode("utf-8", errors="strict")
                inferred, score = _runtime_score(row["prompt"], text)
                attempt = {"record_id": row["record_id"], "variant": variant, "cluster": predicted, "inferred_task": inferred, "output": text, "score": score}
                attempts.append(attempt)
                if score["functional"]:
                    selected = attempt
                    break
            if selected is None:
                selected = attempts[-1]
            selected_rows.append({"record_id": row["record_id"], "oracle_task": row["oracle_task"], "expected_cluster": expected, "predicted_cluster": predicted, "route_exact": predicted == expected, "inferred_task": selected["inferred_task"], "contract_exact": selected["inferred_task"] == row["oracle_task"], "selected_variant": selected["variant"], "attempt_count": sum(item["record_id"] == row["record_id"] for item in attempts), "output": selected["output"], "score": selected["score"]})
            peak_rss = max(peak_rss, process.memory_info().rss)
            if index % 36 == 0:
                print(json.dumps({"evaluated": index, "functional": sum(item["score"]["functional"] for item in selected_rows), "executions": len(attempts)}), flush=True)
        removals = []
        for variant in VARIANTS:
            for cluster, cake_id in installed[variant].items():
                hosts[variant].remove(cake_id)
                rejected = False
                try:
                    hosts[variant].generate(cake_id, v7._normalized_prompt(rows[0]), maximum_actions=4)
                except (KeyError, ValueError, FileNotFoundError):
                    rejected = True
                removals.append({"variant": variant, "cluster": cluster, "rejected": rejected})
    attempts_path = output / "attempts.jsonl"
    selected_path = output / "selected.jsonl"
    write_jsonl_once(attempts_path, attempts)
    write_jsonl_once(selected_path, selected_rows)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in selected_rows:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in selected_rows)
    passed = functional >= 132 and all(item["functional"] >= 11 for item in by_task.values()) and by_task["abstention"]["functional"] == 12 and sum(row["route_exact"] for row in selected_rows) == 144 and sum(row["contract_exact"] for row in selected_rows) == 144 and len(removals) == 36 and all(row["rejected"] for row in removals)
    result = {"format": "abi-r31-live-validated-cascade/3", "verdict": "PASS_PUBLIC_VALIDATED_CASCADE" if passed else "FAIL_PUBLIC_VALIDATED_CASCADE", "ladder_result_sha256": sha256_file(ladder_result), "mdl_result_sha256": sha256_file(mdl_result), "source_rows_sha256": sha256_file(rows_path), "diagnosis_sha256": sha256_file(diagnosis / "result.json"), "metrics": {"evaluation_rows": len(selected_rows), "functional": functional, "by_task": by_task, "route_exact": sum(row["route_exact"] for row in selected_rows), "contract_exact": sum(row["contract_exact"] for row in selected_rows), "live_executions": len(attempts), "mean_executions_per_prompt": len(attempts) / len(selected_rows), "selected_variants": {variant: sum(row["selected_variant"] == variant for row in selected_rows) for variant in VARIANTS}, "removal_rejections": sum(row["rejected"] for row in removals)}, "information_accounting": {"teacher_present": False, "teacher_reference_outputs_read_at_runtime": False, "source_parameters_copied": 0, "installed_packages": 36, "maximum_active_package_parameters": max(item["parameters"] for level in ladder_doc["levels"] for item in level["packages"]), "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "peak_cpu_rss_bytes": int(peak_rss)}, "artifacts": {"attempts": {"path": attempts_path.name, "sha256": sha256_file(attempts_path), "bytes": attempts_path.stat().st_size}, "selected": {"path": selected_path.name, "sha256": sha256_file(selected_path), "bytes": selected_path.stat().st_size}, "removals": removals}, "claim_ceiling": "PUBLIC_DISCLOSED_VALIDATED_CASCADE_REQUIRES_HIDDEN_REPLICATION", "full_abi_moonshot": "OPEN"}
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--ladder-result", type=Path, required=True)
    parser.add_argument("--mdl-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.diagnosis.resolve(), args.ladder_result.resolve(), args.mdl_result.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
