"""Development-only disclosed-split screen for the fixed R51-v1a endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

from abi.english_generalization_evaluation import _collapse_metrics, _source_by_probe
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.external_router_replication_r50 import run_v1a as r50
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CANDIDATE_SHA256 = "3a080529732deaf8a56b53572f7c5faf7d44584b5fab3494b5bf7f7acdba45b8"
METADATA_SHA256 = "0b61419a095efa60a3970823cb76bcc23efffd4d7b3032c3b450037aebc6572f"
ROUTER_SHA256 = r50.ROUTER_SHA256
BROAD_SHA256 = "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
ANCHOR_SHA256 = "f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820"
PARENT_COUNTS = {"validation": 1_277, "final_test": 1_261}


class ScreenError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ScreenError(f"required JSONL missing: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError(f"required JSONL unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ScreenError(f"required JSONL contains non-object: {path}")
    return rows


def _source(
    split: str,
    source_bundles: Sequence[Path],
    live_source_dir: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if (split == "final_test") != (live_source_dir is not None):
        raise ScreenError("final_test requires live source; validation requires bundles")
    if live_source_dir is not None:
        receipt = live_source_dir / "receipt.json"
        outputs = live_source_dir / "source_outputs.jsonl"
        if (
            _sha256_file(receipt) != r50.SOURCE_RECEIPT_SHA256
            or _sha256_file(outputs) != r50.SOURCE_OUTPUTS_SHA256
        ):
            raise ScreenError("R51 live source evidence changed")
        rows = _jsonl(outputs)
        return (
            {
                str(row["probe_id"]): {
                    "passed": bool(row["passed"]),
                    "score": float(row["score"]),
                    "output_sha256": str(row["output_sha256"]),
                }
                for row in rows
            },
            [{"receipt_sha256": r50.SOURCE_RECEIPT_SHA256, "outputs_sha256": r50.SOURCE_OUTPUTS_SHA256}],
        )
    if tuple(_sha256_file(path) for path in source_bundles) != r49.SOURCE_SHA256:
        raise ScreenError("R51 validation source bundles changed")
    source, identities = _source_by_probe(source_bundles, split=split)
    return source, identities


def run(
    candidate: Path,
    router_path: Path,
    catalog_path: Path,
    split: str,
    source_bundles: Sequence[Path],
    live_source_dir: Path | None,
    broad_bundle: Path,
    anchor_bundle: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R51 screen exists: {output}")
    if split not in PARENT_COUNTS:
        raise ScreenError("R51 screen permits validation or final_test only")
    frozen = (
        (candidate / "model.safetensors", CANDIDATE_SHA256),
        (candidate / "metadata.json", METADATA_SHA256),
        (router_path, ROUTER_SHA256),
        (catalog_path, r49.CATALOG_SHA256),
        (broad_bundle, BROAD_SHA256),
        (anchor_bundle, ANCHOR_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise ScreenError(f"R51 frozen input changed: {path}")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    stored_manifest = unsigned.pop("manifest_sha256", None)
    if not isinstance(stored_manifest, str) or _manifest_sha(unsigned) != stored_manifest:
        raise ScreenError("R51 candidate manifest digest changed")
    context = metadata.get("context_compatibility", {})
    training = metadata.get("training", {})
    imported = metadata.get("imported_artifact", {})
    anchor = metadata.get("broad_behavior_anchor", {})
    if (
        metadata.get("status") != "TRAINED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED"
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("seed") != 51_001
        or training.get("parent_logit_preservation_weight") != 0.5
        or imported.get("archive_sha256_after") != BROAD_SHA256
        or imported.get("selected_english_records") != 24_419
        or imported.get("selected_teacher_tokens") != 2_784_634
        or anchor.get("archive_sha256_after") != ANCHOR_SHA256
        or anchor.get("selected_english_records") != 1_438
        or context.get("excluded_record_count") != 2
        or context.get("retained_record_count") != 24_419
    ):
        raise ScreenError("R51 training contract changed")
    if not torch.cuda.is_available():
        raise ScreenError("R51 development screen requires CUDA")

    catalog = load_probe_catalog(catalog_path)
    probes = [dict(row) for row in catalog["probes"] if row["split"] == split]
    if len(probes) != 1_400:
        raise ScreenError("R51 disclosed screen depth changed")
    source, source_identities = _source(split, source_bundles, live_source_dir)
    selected_ids = {str(row["probe_id"]) for row in probes}
    missing = selected_ids - set(source)
    surplus = set(source) - selected_ids
    if missing:
        raise ScreenError("R51 source coverage changed")
    source_inventory = {
        "selected": len(selected_ids),
        "available": len(source),
        "missing": len(missing),
        "surplus_not_evaluated": len(surplus),
        "surplus_ids_sha256": hashlib.sha256(
            "\n".join(sorted(surplus)).encode("utf-8")
        ).hexdigest(),
    }
    router = torch.nn.Linear(r49.FEATURES, 10)
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    features, labels = r49._matrix(probes)
    router_score = r49._score(router, features, labels)
    if router_score["accuracy"] != 1.0 or router_score["rotated_correct"] != 0:
        raise ScreenError("R51 external router prerequisite failed")

    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"])
        route = int(router(r49._feature(prompt)).argmax())
        output_text, token_ids, latency, physical = r49._generate(
            model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device
        )
        passed, score = evaluate_output(output_text, probe["evaluator"])
        source_row = source[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"], "capability": probe["capability"],
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "evaluator": probe["evaluator"], "output": output_text,
            "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            "output_token_ids": token_ids, "functional_pass": bool(passed),
            "functional_score": float(score), "route": route,
            "expected_route": CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "route_correct": route == CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "maximum_cakes_called_per_model_invocation": max(map(len, physical)),
            "all_calls_selected_only": all(value == (route,) for value in physical),
            "latency_seconds": latency,
            "collapse": _collapse_metrics(token_ids, output_text, tokenizer.encode(prompt + "\n"), prompt),
            "generation_error": None, "source": source_row,
            "source_passing_regression": bool(source_row["passed"] and not passed),
        })
        if ordinal % 100 == 0:
            print(json.dumps({
                "split": split, "evaluated": ordinal,
                "functional": sum(row["functional_pass"] for row in rows),
                "source_functional": sum(row["source"]["passed"] for row in rows),
                "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
            }), flush=True)

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    by_capability = {
        capability: {
            "functional": sum(row["capability"] == capability and row["functional_pass"] for row in rows),
            "source_functional": sum(row["capability"] == capability and row["source"]["passed"] for row in rows),
            "regressions": sum(row["capability"] == capability and row["source_passing_regression"] for row in rows),
            "collapses": sum(row["capability"] == capability and row["collapse"]["collapse_detected"] for row in rows),
        }
        for capability in sorted(CAPABILITY_TO_ROUTE)
    }
    functional = sum(row["functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    metrics = {
        "rows": len(rows), "functional": functional,
        "parent_functional": PARENT_COUNTS[split], "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": r49._bootstrap(
            [row["functional_pass"] for row in rows],
            [row["source"]["passed"] for row in rows],
        ),
        "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
        "generation_errors": 0, "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(
            row["all_calls_selected_only"] and row["maximum_cakes_called_per_model_invocation"] == 1
            for row in rows
        ),
        "by_capability": by_capability,
        "generation_wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "matrix": len(rows) == 1_400, "functional": functional >= 1_260,
        "parent_nondegradation": functional >= PARENT_COUNTS[split],
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional,
        "source_retention": metrics["source_retention"] >= 0.94,
        "zero_collapse": metrics["collapses"] == 0, "zero_generation_error": True,
        "route_exact": metrics["route_correct"] == 1_400,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "artifacts_unchanged": all(_sha256_file(path) == digest for path, digest in frozen),
    }
    if not math.isfinite(metrics["generation_wall_seconds"]) or metrics["generation_wall_seconds"] <= 0:
        raise ScreenError("R51 screen wall time invalid")
    passed = all(gates.values())
    result = {
        "format": "abi-r51-broad-payload-development-screen/1",
        "verdict": "PASS_R51_DISCLOSED_SCREEN" if passed else "FAIL_R51_DISCLOSED_SCREEN",
        "split": split,
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_metadata_sha256": METADATA_SHA256,
        "router_sha256": ROUTER_SHA256,
        "source_identities": source_identities,
        "source_inventory": source_inventory,
        "router": router_score, "metrics": metrics, "gates": gates,
        "artifacts": {"evaluation": {"path": raw_path.name, "sha256": _sha256_file(raw_path), "bytes": raw_path.stat().st_size}},
        "training_steps_during_screen": 0, "teacher_calls_during_screen": 0,
        "teacher_present_at_inference": False, "symbolic_output_calls": 0,
        "planner_calls": 0, "prospective_promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(PARENT_COUNTS), required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", default=[])
    parser.add_argument("--live-source-dir", type=Path)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.candidate.resolve(), args.router.resolve(), args.catalog.resolve(), args.split,
        [path.resolve() for path in args.source_bundle],
        args.live_source_dir.resolve() if args.live_source_dir else None,
        args.broad_bundle.resolve(), args.anchor_bundle.resolve(),
        args.layercake_root.resolve(), args.output.resolve(),
    )


if __name__ == "__main__":
    main()
