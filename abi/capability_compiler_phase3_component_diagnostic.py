"""Preregistered read-only component ablation of the sealed V6 B1 checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_analysis import stratified_bootstrap, wilson
from .capability_compiler_phase3_sequence_bridge import (
    _generate,
    load_candidate,
    load_protocol,
)


VARIANTS = ("R1", "R2", "R3")
EXPECTED_MUTATIONS = {
    "R1": tuple(f"task_cakes.{route}.up.weight" for route in range(6)),
    "R2": ("abi_sequence_bridge.route_embedding.weight",),
    "R3": (
        "abi_sequence_bridge.route_embedding.weight",
        *(f"task_cakes.{route}.up.weight" for route in range(6)),
    ),
}


class ComponentDiagnosticError(RuntimeError):
    """Raised when the diagnostic identity or scope changes."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComponentDiagnosticError(f"expected object: {path}")
    return value


def load_diagnostic_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != "abi-capability-compiler-phase3-component-diagnostic/1"
        or protocol.get("status") != "PREREGISTERED_DIAGNOSTIC_ONLY"
        or protocol.get("phase3_promotion_eligible") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("variants")
        != {
            "R0": "sealed B1 checkpoint and existing unmodified evaluation",
            "R1": "same B1 checkpoint with all six output cakes bypassed in memory",
            "R2": "same B1 checkpoint with route embedding bypassed in memory",
            "R3": "same B1 checkpoint with output cakes and route embedding bypassed in memory",
        }
    ):
        raise ComponentDiagnosticError("component-diagnostic governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise ComponentDiagnosticError(f"diagnostic binding changed: {relative}")
    return protocol, sha256_file(path)


def _apply_ablation(model: Any, variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ComponentDiagnosticError("unregistered diagnostic variant")
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        if variant in {"R1", "R3"}:
            for route in range(6):
                model.task_cakes[str(route)].up.weight.zero_()
        if variant in {"R2", "R3"}:
            model.abi_sequence_bridge.route_embedding.weight.zero_()
    after = model.state_dict()
    changed = sorted(name for name in before if not torch.equal(before[name], after[name].cpu()))
    if tuple(changed) != tuple(sorted(EXPECTED_MUTATIONS[variant])):
        raise ComponentDiagnosticError(f"{variant} in-memory mutation scope changed: {changed}")
    return changed


@torch.inference_mode()
def evaluate(
    *, root: Path, protocol_path: Path, variant: str, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    diagnostic, diagnostic_sha = load_diagnostic_protocol(root, protocol_path.resolve())
    v6_path = (root / diagnostic["v6_protocol"]["path"]).resolve()
    v6, v6_sha = load_protocol(root, v6_path)
    if v6_sha != diagnostic["v6_protocol"]["sha256"]:
        raise ComponentDiagnosticError("V6 protocol identity changed")
    candidate_dir = (root / diagnostic["checkpoint"]["path"]).resolve()
    metadata = _json(candidate_dir / "metadata.json")
    checkpoint = candidate_dir / "model.safetensors"
    checkpoint_sha = sha256_file(checkpoint)
    if (
        metadata.get("system") != "B1"
        or checkpoint_sha != diagnostic["checkpoint"]["sha256"]
        or checkpoint_sha != metadata.get("checkpoint", {}).get("sha256")
    ):
        raise ComponentDiagnosticError("sealed B1 checkpoint identity changed")
    if output_dir.exists():
        raise ComponentDiagnosticError(f"diagnostic output is immutable: {output_dir}")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise ComponentDiagnosticError("registered diagnostic GPU is unavailable")
    model, tokenizer = load_candidate(
        root=root, protocol=v6, candidate_dir=candidate_dir, device=device
    )
    changed = _apply_ablation(model, variant)
    probes = development_probes(root / diagnostic["development_catalog"])
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes):
        output, token_ids, route = _generate(
            model,
            tokenizer,
            str(probe["prompt"]),
            int(probe["max_new_tokens"]),
            device,
        )
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["canonical_capability"]),
                "output": output,
                "output_token_ids": token_ids,
                "authoritative_output_tokens": len(token_ids),
                "automatic_route": route,
                "functional_pass": evaluate_functional(output, probe["evaluator"]),
                "repetition_collapse": repetition_collapse(output),
            }
        )
        if (index + 1) % 100 == 0:
            print(json.dumps({"variant": variant, "evaluated": index + 1}), flush=True)
    output_dir.mkdir(parents=True)
    outputs_path = output_dir / "development_outputs.jsonl"
    outputs_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    grouped = {
        capability: [row for row in rows if row["capability"] == capability]
        for capability in CAPABILITIES
    }
    receipt = {
        "format": "abi-capability-compiler-phase3-component-diagnostic-evaluation/1",
        "status": "PASS_EXECUTION_DIAGNOSTIC_ONLY",
        "variant": variant,
        "diagnostic_protocol_sha256": diagnostic_sha,
        "v6_protocol_sha256": v6_sha,
        "checkpoint_sha256_before": checkpoint_sha,
        "checkpoint_sha256_after": sha256_file(checkpoint),
        "in_memory_changed_tensors": changed,
        "checkpoint_persisted": False,
        "observations": len(rows),
        "distinct_prompts": len({row["probe_id"] for row in rows}),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "per_capability": {
            capability: {
                "passes": sum(bool(row["functional_pass"]) for row in values),
                "observations": len(values),
                "collapses": sum(bool(row["repetition_collapse"]) for row in values),
            }
            for capability, values in grouped.items()
        },
        "automatic_route_counts": dict(sorted(Counter(row["automatic_route"] for row in rows).items())),
        "output_tokens": sum(len(row["output_token_ids"]) for row in rows),
        "output_bytes": sum(len(row["output"].encode("utf-8")) for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "outputs_path": outputs_path.relative_to(root).as_posix(),
        "outputs_sha256": sha256_file(outputs_path),
        "final_test_accessed": False,
        "promotion_eligible": False,
    }
    _write_immutable(output_dir / "receipt.json", canonical_json_bytes(receipt))
    return receipt


def analyze(*, root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnostic, diagnostic_sha = load_diagnostic_protocol(root, protocol_path.resolve())
    if output_path.exists():
        raise ComponentDiagnosticError(f"diagnostic decision is immutable: {output_path}")
    baseline_path = root / diagnostic["baseline_outputs"]
    baseline_receipt = _json(baseline_path.parent / "receipt.json")
    if sha256_file(baseline_path) != baseline_receipt.get("outputs_sha256"):
        raise ComponentDiagnosticError("R0 baseline output identity changed")
    raw: dict[str, dict[str, Any]] = {}
    systems: dict[str, Any] = {}
    for variant in ("R0", *VARIANTS):
        if variant == "R0":
            outputs_path = baseline_path
            receipt = baseline_receipt
            expected_checkpoint = diagnostic["checkpoint"]["sha256"]
        else:
            evaluation = root / diagnostic["output_root"] / variant
            outputs_path = evaluation / "development_outputs.jsonl"
            receipt = _json(evaluation / "receipt.json")
            expected_checkpoint = receipt.get("checkpoint_sha256_before")
            if (
                receipt.get("variant") != variant
                or receipt.get("diagnostic_protocol_sha256") != diagnostic_sha
                or receipt.get("in_memory_changed_tensors") != list(EXPECTED_MUTATIONS[variant])
                or receipt.get("checkpoint_persisted") is not False
                or receipt.get("promotion_eligible") is not False
            ):
                raise ComponentDiagnosticError(f"{variant} receipt identity changed")
        rows = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines() if line]
        if (
            sha256_file(outputs_path) != receipt.get("outputs_sha256")
            or expected_checkpoint != diagnostic["checkpoint"]["sha256"]
            or len(rows) != 1400
            or len({row["probe_id"] for row in rows}) != 1400
            or sum(bool(row["functional_pass"]) for row in rows) != receipt.get("functional_passes")
            or sum(bool(row["repetition_collapse"]) for row in rows) != receipt.get("repetition_collapses")
            or receipt.get("final_test_accessed") is not False
        ):
            raise ComponentDiagnosticError(f"{variant} evidence verification failed")
        raw[variant] = {str(row["probe_id"]): row for row in rows}
        systems[variant] = {
            "functional_passes": receipt["functional_passes"],
            "functional": wilson(receipt["functional_passes"], 1400),
            "repetition_collapses": receipt["repetition_collapses"],
            "output_tokens": receipt["output_tokens"],
            "wall_seconds": receipt["wall_seconds"],
            "outputs_sha256": receipt["outputs_sha256"],
        }
    ids = {variant: set(rows) for variant, rows in raw.items()}
    if len({frozenset(value) for value in ids.values()}) != 1:
        raise ComponentDiagnosticError("diagnostic prompt pairing changed")
    comparisons = {
        f"{variant}_minus_R0": stratified_bootstrap(
            raw[variant], raw["R0"], replicates=10000, seed=2718
        )
        for variant in VARIANTS
    }
    result = {
        "format": "abi-capability-compiler-phase3-component-diagnostic-decision/1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "protocol_sha256": diagnostic_sha,
        "systems": systems,
        "paired_bootstrap": comparisons,
        "training_performed": False,
        "checkpoint_persisted": False,
        "phase3_certified": False,
        "phase4_status": "LOCKED",
        "final_test_accessed": False,
        "claim_boundary": "Post-training component ablation can localize dependence but cannot promote a checkpoint or prove how a retrained architecture will behave.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_COMPONENT_DIAGNOSTIC_PROTOCOL_V8.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--variant", choices=VARIANTS, required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_component_diagnostic/decision_v1.json",
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.command == "evaluate":
        result = evaluate(
            root=root,
            protocol_path=(root / args.protocol).resolve(),
            variant=args.variant,
            output_dir=(root / args.output_dir).resolve(),
        )
    else:
        result = analyze(
            root=root,
            protocol_path=(root / args.protocol).resolve(),
            output_path=(root / args.output).resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
