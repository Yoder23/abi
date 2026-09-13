"""Preregistered unseen-prompt replay of the frozen R31 package cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
from experiments.english_substrate_r30.protocol import INSTRUCTIONS, TASKS, _details
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from .cascade_v3 import VARIANTS, _runtime_score
from .ladder import MAXIMUM_ACTIONS


EXPECTED_PUBLIC_CASCADE_SHA256 = "9038c59287390d137665dfed2152893c61e7a721c0c668d07adcee2d1194ea39"
EXPECTED_LADDER_SHA256 = "4bee74452b1c4d6e6b7aacdbc2f896df2d75b3ba3ca12b88fab3f670b8f23cd9"
EXPECTED_MDL_SHA256 = "d47286b0f10a884896f614dfd9188fab13bd3dfd85b48ec1970f5247b8bdbd51"
HIDDEN_ORDINAL_START = 10_000
HIDDEN_ROWS_PER_TASK = 12
INSTRUCTION_SHUFFLE_SEED = 931_031
TASK_TO_CLUSTER = {
    "prose": "capability-781f838d49a2",
    "summary": "capability-6a05ac87d84b",
    "rewrite": "capability-9c14f5ea0d08",
    "email": "capability-663bb9820de9",
    "bullets": "capability-1804ef5e49f4",
    "tone": "capability-660267e0e217",
    "clarification": "capability-77540ff46ac4",
    "conversation": "capability-afba74547ae1",
    "planning": "capability-8f722f34058c",
    "comparison": "capability-ebc97d41c4c9",
    "reasoning": "capability-defbdf34a3d4",
    "abstention": "capability-68a9963f947e",
}


def _hidden_fixture() -> list[dict[str, str]]:
    rows = []
    for task_index, task in enumerate(TASKS):
        bank = list(INSTRUCTIONS[task])
        random.Random(INSTRUCTION_SHUFFLE_SEED + task_index).shuffle(bank)
        for repeat in range(HIDDEN_ROWS_PER_TASK):
            ordinal = HIDDEN_ORDINAL_START + repeat
            index = 800_000 + task_index * 10_000 + ordinal
            instruction = bank[repeat % len(bank)]
            details = _details(task, index)
            prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
            digest = hashlib.sha256(f"r31-v4|{task}|{ordinal}|{prompt}".encode()).hexdigest()[:20]
            rows.append(
                {
                    "record_id": f"r31-v4-{digest}",
                    "oracle_task": task,
                    "nonce": f"Virelon{index:05d}",
                    "prompt": prompt,
                }
            )
    return rows


def _checked_inputs(public_cascade: Path, ladder_result: Path, mdl_result: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = (
        (public_cascade, EXPECTED_PUBLIC_CASCADE_SHA256),
        (ladder_result, EXPECTED_LADDER_SHA256),
        (mdl_result, EXPECTED_MDL_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R31 v4 frozen prerequisite missing or changed: {path}")
    public_doc = json.loads(public_cascade.read_text(encoding="utf-8"))
    ladder_doc = json.loads(ladder_result.read_text(encoding="utf-8"))
    mdl_doc = json.loads(mdl_result.read_text(encoding="utf-8"))
    if public_doc.get("verdict") != "PASS_PUBLIC_VALIDATED_CASCADE":
        raise RuntimeError("R31 v4 public cascade prerequisite is not the frozen pass")
    if ladder_doc.get("verdict") != "FAIL_PUBLIC_SUFFICIENCY_LADDER" or mdl_doc.get("verdict") != "FAIL_PUBLIC_MDL_SELECTION":
        raise RuntimeError("R31 v4 component-result prerequisites changed")
    return ladder_doc, mdl_doc


def run(public_cascade: Path, ladder_result: Path, mdl_result: Path, output: Path, device: str) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R31 v4 output exists: {output}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R31 v4 requested CUDA but CUDA is unavailable")
    ladder_doc, mdl_doc = _checked_inputs(public_cascade, ladder_result, mdl_result)
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
    if set(TASK_TO_CLUSTER.values()) != set(expected_hashes["first24"]):
        raise RuntimeError("R31 v4 registered route map does not match package inventory")

    rows = _hidden_fixture()
    if len(rows) != 144 or len({row["record_id"] for row in rows}) != 144:
        raise RuntimeError("R31 v4 fixture cardinality changed")
    api = base._layercake(Path(__file__).resolve().parents[2])
    public = Ed25519PrivateKey.from_private_bytes(base.SIGNING_SEED).public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public)
    output.mkdir(parents=True)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    package_inventory: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="r31-v4-host-") as raw:
        hosts = {
            variant: api["DirectCakeHost"](
                Path(raw) / variant,
                abi_version=base.ABI_VERSION,
                abi_hash=base.ABI_SHA256,
                trust_store={signer: public},
                device=device,
            )
            for variant in VARIANTS
        }
        installed: dict[str, dict[str, str]] = {variant: {} for variant in VARIANTS}
        for variant in VARIANTS:
            for cluster, expected_hash in sorted(expected_hashes[variant].items()):
                package = roots[variant] / "packages" / f"{cluster}.cake"
                actual_hash = sha256_file(package) if package.is_file() else None
                if actual_hash != expected_hash:
                    raise RuntimeError(f"R31 v4 package missing or changed: {variant}/{cluster}")
                record = hosts[variant].install(package)
                if record["archive_hash"] != expected_hash:
                    raise RuntimeError("R31 v4 LayerCake install hash mismatch")
                installed[variant][cluster] = record["cake_id"]
                package_inventory.append(
                    {
                        "variant": variant,
                        "cluster": cluster,
                        "cake_id": record["cake_id"],
                        "sha256": actual_hash,
                        "bytes": package.stat().st_size,
                    }
                )

        for index, fixture in enumerate(rows, 1):
            prompt = fixture["prompt"]
            inferred_before = _runtime_score(prompt, "placeholder")[0]
            cluster = TASK_TO_CLUSTER[inferred_before]
            selected = None
            record_attempts = 0
            for variant in VARIANTS:
                generated = hosts[variant].generate(
                    installed[variant][cluster],
                    v7._normalized_prompt({"prompt": prompt, "nonce": fixture["nonce"]}),
                    maximum_actions=MAXIMUM_ACTIONS,
                )
                text = generated.output.decode("utf-8", errors="strict")
                inferred, score = _runtime_score(prompt, text)
                attempt = {
                    "record_id": fixture["record_id"],
                    "variant": variant,
                    "cluster": cluster,
                    "inferred_task": inferred,
                    "output": text,
                    "score": score,
                }
                attempts.append(attempt)
                record_attempts += 1
                if score["functional"]:
                    selected = attempt
                    break
            if selected is None:
                selected = attempt
            selected_rows.append(
                {
                    "record_id": fixture["record_id"],
                    "oracle_task": fixture["oracle_task"],
                    "predicted_cluster": cluster,
                    "route_exact": cluster == TASK_TO_CLUSTER[fixture["oracle_task"]],
                    "inferred_task": selected["inferred_task"],
                    "contract_exact": selected["inferred_task"] == fixture["oracle_task"],
                    "selected_variant": selected["variant"],
                    "attempt_count": record_attempts,
                    "output": selected["output"],
                    "score": selected["score"],
                }
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if index % 36 == 0:
                print(
                    json.dumps(
                        {
                            "evaluated": index,
                            "functional": sum(item["score"]["functional"] for item in selected_rows),
                            "executions": len(attempts),
                        }
                    ),
                    flush=True,
                )

        removals = []
        for variant in VARIANTS:
            for cluster, cake_id in sorted(installed[variant].items()):
                hosts[variant].remove(cake_id)
                rejected = False
                try:
                    hosts[variant].generate(cake_id, "INSTRUCTION: test", maximum_actions=4)
                except (KeyError, ValueError, FileNotFoundError):
                    rejected = True
                removals.append({"variant": variant, "cluster": cluster, "rejected": rejected})

    attempts_path = output / "attempts.jsonl"
    selected_path = output / "selected.jsonl"
    fixture_path = output / "fixture.jsonl"
    write_jsonl_once(attempts_path, attempts)
    write_jsonl_once(selected_path, selected_rows)
    write_jsonl_once(fixture_path, rows)
    by_task = {task: {"rows": 0, "functional": 0} for task in TASKS}
    for row in selected_rows:
        by_task[row["oracle_task"]]["rows"] += 1
        by_task[row["oracle_task"]]["functional"] += int(row["score"]["functional"])
    functional = sum(row["score"]["functional"] for row in selected_rows)
    route_exact = sum(row["route_exact"] for row in selected_rows)
    contract_exact = sum(row["contract_exact"] for row in selected_rows)
    transformers_loaded = "transformers" in sys.modules
    passed = (
        functional >= 132
        and all(item["functional"] >= 11 for item in by_task.values())
        and by_task["abstention"]["functional"] == 12
        and route_exact == 144
        and contract_exact == 144
        and len(package_inventory) == 36
        and len(removals) == 36
        and all(item["rejected"] for item in removals)
        and not transformers_loaded
    )
    result = {
        "format": "abi-r31-frozen-cascade-replication/4",
        "verdict": "PASS_FROZEN_CASCADE_REPLICATION" if passed else "FAIL_FROZEN_CASCADE_REPLICATION",
        "frozen_inputs": {
            "public_cascade_sha256": sha256_file(public_cascade),
            "ladder_result_sha256": sha256_file(ladder_result),
            "mdl_result_sha256": sha256_file(mdl_result),
            "variant_order": list(VARIANTS),
            "hidden_ordinal_start": HIDDEN_ORDINAL_START,
            "instruction_shuffle_seed": INSTRUCTION_SHUFFLE_SEED,
        },
        "metrics": {
            "evaluation_rows": len(selected_rows),
            "functional": functional,
            "by_task": by_task,
            "route_exact": route_exact,
            "contract_exact": contract_exact,
            "live_executions": len(attempts),
            "mean_executions_per_prompt": len(attempts) / len(selected_rows),
            "selected_variants": {variant: sum(row["selected_variant"] == variant for row in selected_rows) for variant in VARIANTS},
            "removal_rejections": sum(item["rejected"] for item in removals),
        },
        "isolation": {
            "teacher_model_loaded": False,
            "transformers_module_loaded": transformers_loaded,
            "source_corpus_input": False,
            "teacher_output_input": False,
            "diagnosis_artifact_input": False,
            "public_evaluation_input": False,
            "runtime_selection_inputs": ["prompt", "generated_output", "registered_contract_map"],
        },
        "information_accounting": {
            "source_parameters_copied": 0,
            "installed_packages": len(package_inventory),
            "installed_package_bytes": sum(item["bytes"] for item in package_inventory),
            "maximum_active_package_parameters": max(
                item["parameters"] for level in ladder_doc["levels"] for item in level["packages"]
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
            "peak_cpu_rss_bytes": int(peak_rss),
            "device": device,
        },
        "package_inventory": package_inventory,
        "artifacts": {
            name: {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in (("fixture", fixture_path), ("attempts", attempts_path), ("selected", selected_path))
        },
        "removals": removals,
        "claim_ceiling": "REGISTERED_SUPPLIED_CONTENT_REPLICATION_NOT_GENERAL_ENGLISH_OR_ABI_SUPERIORITY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-cascade", type=Path, required=True)
    parser.add_argument("--ladder-result", type=Path, required=True)
    parser.add_argument("--mdl-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = run(
        args.public_cascade.resolve(),
        args.ladder_result.resolve(),
        args.mdl_result.resolve(),
        args.output.resolve(),
        args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
