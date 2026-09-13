"""Fail-closed recomputation and live neural replay of the R53 negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

from abi.english_generalization_evaluation import _collapse_metrics, _source_by_probe
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import _sha256_file
from experiments.broad_payload_reconstruction_r51 import screen_v1a as common
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.isolated_capability_cakes_r53 import fit_router, screen_v1


EXPECTED_RESULT_SHA256 = "2f446c8fc90ab1d2b916e5a1673b08eec9b4dd94ffc278f801af20b6af26b344"
EXPECTED_RAW_SHA256 = "e0227caded990712bb2d5f490c5bf920376ec5162dd1fe45387716e5d0e7a473"
SOURCE_SHA256 = r49.SOURCE_SHA256


class VerificationError(RuntimeError):
    """Raised if any R53 evidence cannot be independently reconstructed."""


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"required JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VerificationError(f"required JSONL absent: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"required JSONL unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise VerificationError(f"required JSONL contains non-object: {path}")
    return rows


def verify(
    run_dir: Path,
    candidate: Path,
    router_path: Path,
    catalog_path: Path,
    source_paths: Sequence[Path],
    broad_bundle: Path,
    anchor_bundle: Path,
    layercake_root: Path,
    live: bool,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    raw_path = run_dir / "evaluation.jsonl"
    if _sha256_file(result_path) != EXPECTED_RESULT_SHA256:
        raise VerificationError("R53 frozen result changed")
    if _sha256_file(raw_path) != EXPECTED_RAW_SHA256:
        raise VerificationError("R53 frozen raw rows changed")
    result = _json(result_path)
    unsigned = dict(result)
    stored_evidence = unsigned.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(unsigned) != stored_evidence:
        raise VerificationError("R53 evidence digest changed")
    if (
        result.get("format")
        != "abi-r53-isolated-capability-cakes-development-screen/1"
        or result.get("verdict") != "FAIL_R53_DISCLOSED_SCREEN"
        or result.get("split") != "validation"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R53 result scope changed")
    if tuple(_sha256_file(path) for path in source_paths) != SOURCE_SHA256:
        raise VerificationError("R53 source bundles changed")
    frozen = (
        (candidate / "model.safetensors", screen_v1.CANDIDATE_SHA256),
        (candidate / "metadata.json", screen_v1.METADATA_SHA256),
        (router_path, screen_v1.ROUTER_SHA256),
        (catalog_path, fit_router.CATALOG_SHA256),
        (broad_bundle, common.BROAD_SHA256),
        (anchor_bundle, common.ANCHOR_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise VerificationError(f"R53 frozen input changed: {path}")
    screen_v1._preflight(candidate)
    artifact = result.get("artifacts", {}).get("evaluation", {})
    if (
        artifact.get("path") != raw_path.name
        or artifact.get("sha256") != EXPECTED_RAW_SHA256
        or artifact.get("bytes") != raw_path.stat().st_size
    ):
        raise VerificationError("R53 raw-row receipt changed")

    catalog = load_probe_catalog(catalog_path)
    probes = [
        dict(row) for row in catalog["probes"] if row["split"] == "validation"
    ]
    source, source_identities = _source_by_probe(source_paths, split="validation")
    rows = _jsonl(raw_path)
    if (
        len(probes) != 1_400
        or len(rows) != 1_400
        or [row.get("probe_id") for row in rows]
        != [row["probe_id"] for row in probes]
        or ({str(row["probe_id"]) for row in probes} - set(source))
    ):
        raise VerificationError("R53 matrix identity changed")
    if source_identities != result.get("source_identities"):
        raise VerificationError("R53 source identities changed")

    router = torch.nn.Linear(r49.FEATURES, len(fit_router.CAPABILITY_TO_INDEX))
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    features, labels = fit_router.matrix(probes)
    router_score = fit_router.score(router, features, labels)
    if router_score != result.get("router"):
        raise VerificationError("R53 router predictions changed")

    if live and not torch.cuda.is_available():
        raise VerificationError("R53 live replay requires CUDA")
    device = torch.device("cuda" if live else "cpu")
    model, tokenizer, _ = load_layercake_core(
        candidate, layercake_root=layercake_root, device=device
    )
    model.eval()
    live_exact = 0
    for ordinal, (probe, row) in enumerate(zip(probes, rows, strict=True), 1):
        prompt = str(probe["prompt"])
        route = int(router(r49._feature(prompt)).argmax())
        output = row.get("output")
        token_ids = row.get("output_token_ids")
        if not isinstance(output, str) or not isinstance(token_ids, list):
            raise VerificationError(f"R53 output absent: {probe['probe_id']}")
        passed, score = evaluate_output(output, probe["evaluator"])
        collapse = _collapse_metrics(
            token_ids,
            output,
            tokenizer.encode(prompt + "\n"),
            prompt,
        )
        expected_route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
        source_row = source[str(probe["probe_id"])]
        if (
            row.get("capability") != probe["capability"]
            or row.get("prompt") != prompt
            or row.get("prompt_sha256")
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or row.get("evaluator") != probe["evaluator"]
            or row.get("output_sha256")
            != hashlib.sha256(output.encode("utf-8")).hexdigest()
            or row.get("functional_pass") is not bool(passed)
            or row.get("functional_score") != float(score)
            or row.get("route") != route
            or row.get("expected_route") != expected_route
            or row.get("route_correct") is not (route == expected_route)
            or row.get("collapse") != collapse
            or row.get("source") != source_row
            or row.get("source_passing_regression")
            is not bool(source_row["passed"] and not passed)
            or row.get("generation_error") is not None
            or not isinstance(row.get("latency_seconds"), (int, float))
            or not math.isfinite(row["latency_seconds"])
            or row["latency_seconds"] <= 0
        ):
            raise VerificationError(f"R53 raw row changed: {probe['probe_id']}")
        if live:
            live_output, live_tokens, _, physical = r49._generate(
                model,
                tokenizer,
                prompt,
                route,
                int(probe["max_new_tokens"]),
                device,
            )
            if (
                live_output != output
                or live_tokens != token_ids
                or not physical
                or max(map(len, physical)) != 1
                or not all(value == (route,) for value in physical)
            ):
                raise VerificationError(f"R53 live replay changed: {probe['probe_id']}")
            live_exact += 1
            if ordinal % 100 == 0:
                print(json.dumps({"live_replayed": ordinal, "exact": live_exact}), flush=True)

    by_capability = {
        capability: {
            "functional": sum(
                row["capability"] == capability and row["functional_pass"]
                for row in rows
            ),
            "source_functional": sum(
                row["capability"] == capability and row["source"]["passed"]
                for row in rows
            ),
            "regressions": sum(
                row["capability"] == capability
                and row["source_passing_regression"]
                for row in rows
            ),
            "collapses": sum(
                row["capability"] == capability
                and row["collapse"]["collapse_detected"]
                for row in rows
            ),
        }
        for capability in sorted(fit_router.CAPABILITY_TO_INDEX)
    }
    functional = sum(row["functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    metrics = {
        "rows": len(rows),
        "functional": functional,
        "parent_functional": screen_v1.PARENT_COUNTS["validation"],
        "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": r49._bootstrap(
            [row["functional_pass"] for row in rows],
            [row["source"]["passed"] for row in rows],
        ),
        "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
        "generation_errors": 0,
        "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(
            row["all_calls_selected_only"]
            and row["maximum_cakes_called_per_model_invocation"] == 1
            for row in rows
        ),
        "by_capability": by_capability,
        "generation_wall_seconds": result["metrics"]["generation_wall_seconds"],
    }
    if metrics != result.get("metrics"):
        raise VerificationError("R53 metrics differ from raw recomputation")
    gates = {
        "matrix": len(rows) == 1_400,
        "functional": functional >= 1_260,
        "parent_nondegradation": functional >= 1_277,
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional,
        "source_retention": metrics["source_retention"] >= 0.94,
        "zero_collapse": metrics["collapses"] == 0,
        "zero_generation_error": True,
        "route_exact": metrics["route_correct"] == 1_400,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "artifacts_unchanged": all(_sha256_file(path) == digest for path, digest in frozen),
    }
    if gates != result.get("gates"):
        raise VerificationError("R53 gate vector differs from raw recomputation")
    return {
        "format": "abi-r53-strict-negative-verification/1",
        "verdict": "PASS_STRICT_VERIFICATION_OF_FAILED_R53",
        "result_sha256": EXPECTED_RESULT_SHA256,
        "raw_sha256": EXPECTED_RAW_SHA256,
        "raw_rows_recomputed": len(rows),
        "live_rows_exact": live_exact,
        "functional": functional,
        "source_functional": source_functional,
        "source_retention": metrics["source_retention"],
        "failed_gates": [name for name, value in gates.items() if not value],
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", required=True)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    receipt = verify(
        args.run_dir.resolve(),
        args.candidate.resolve(),
        args.router.resolve(),
        args.catalog.resolve(),
        [path.resolve() for path in args.source_bundle],
        args.broad_bundle.resolve(),
        args.anchor_bundle.resolve(),
        args.layercake_root.resolve(),
        args.live,
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
