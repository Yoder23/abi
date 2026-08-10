"""Read-only forced-route oracle for the failed qualified-transition control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_qualified_transition_control import (
    _load_parent,
    load_protocol as load_base_protocol,
)


FORMAT = "abi-capability-compiler-phase3-route-collision-oracle/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_FORCED_ROUTE_ATTRIBUTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_mutation_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("route-collision oracle governance changed")
    for name, expected in protocol["bindings"].items():
        candidate = Path(name) if Path(name).is_absolute() else root / name
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise Phase3Error(f"route-collision oracle binding changed: {name}")
    return protocol, sha256_file(path)


@torch.inference_mode()
def _forced_generate(model, tokenizer, prompt: str, maximum: int, route: int):
    device = model.transformer.wte.weight.device
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    selected_route = torch.tensor([route], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long, device=device),
        task_routes=selected_route,
        use_cache=True,
    )
    cache = result["past_key_values"]
    logits = result["logits"][:, -1]
    generated = []
    for _ in range(maximum):
        selected = logits.argmax(dim=-1)
        token = int(selected.item())
        if token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=selected_route, past_key_values=cache, use_cache=True)
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), generated


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable route-oracle output exists: {output}")
    base_path = (root / protocol["base_protocol"]["path"]).resolve()
    base, _ = load_base_protocol(root, base_path)
    candidate = (root / protocol["candidate"]["directory"]).resolve()
    metadata = _json(candidate / "metadata.json")
    if sha256_file(candidate / "model.safetensors") != protocol["candidate"]["checkpoint_sha256"]:
        raise Phase3Error("route-oracle candidate changed")
    _, model, tokenizer, _ = _load_parent(root, base, torch.device("cuda"))
    model.load_state_dict(load_file(str(candidate / "model.safetensors"), device="cuda"), strict=True)
    model.eval()
    probes = {str(row["probe_id"]): row for row in development_probes((root / protocol["development_catalog"]).resolve())}
    primary = [json.loads(line) for line in (root / protocol["primary_outputs"]["path"]).read_text(encoding="utf-8").splitlines()]
    problematic = [row for row in primary if not row["functional_pass"] or row["repetition_collapse"]]
    expected_count = int(protocol["population"]["problematic_records"])
    if len(problematic) != expected_count:
        raise Phase3Error("route-oracle problematic population changed")
    rows = []
    started = time.perf_counter()
    for index, original in enumerate(problematic):
        probe = probes[str(original["probe_id"])]
        alternatives = []
        for route in range(int(protocol["routes"]["existing_routes"])):
            value, tokens = _forced_generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), route)
            alternatives.append({
                "route": route,
                "output": value,
                "output_token_ids": tokens,
                "functional_pass": evaluate_functional(value, probe["evaluator"]),
                "repetition_collapse": repetition_collapse(value),
            })
        rows.append({
            "probe_id": str(original["probe_id"]),
            "capability": str(original["capability"]),
            "primary_route": int(original["automatic_route"]),
            "primary_functional_pass": bool(original["functional_pass"]),
            "primary_repetition_collapse": bool(original["repetition_collapse"]),
            "alternatives": alternatives,
        })
        if (index + 1) % 25 == 0:
            print(json.dumps({"problematic_records_evaluated": index + 1}), flush=True)
    output.mkdir(parents=True)
    raw = output / "forced_route_outputs.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    recovered = [row for row in rows if any(value["functional_pass"] and not value["repetition_collapse"] for value in row["alternatives"] if value["route"] != row["primary_route"])]
    per_capability = {}
    for capability in sorted({row["capability"] for row in rows}):
        values = [row for row in rows if row["capability"] == capability]
        fixed = [row for row in values if row in recovered]
        per_capability[capability] = {"problematic": len(values), "recovered_by_alternative_existing_route": len(fixed)}
    recovery_rate = len(recovered) / len(rows) if rows else 0.0
    gate = recovery_rate >= float(protocol["attribution_gate"]["alternative_route_recovery_rate_minimum"])
    result = {
        "format": FORMAT,
        "status": "PASS_ROUTE_COLLISION_ATTRIBUTION" if gate else "FAIL_ROUTE_COLLISION_NOT_SUFFICIENTLY_ATTRIBUTED",
        "protocol_sha256": protocol_sha,
        "problematic_records": len(rows),
        "forced_generations": len(rows) * int(protocol["routes"]["existing_routes"]),
        "recovered_by_alternative_existing_route": len(recovered),
        "alternative_route_recovery_rate": recovery_rate,
        "per_capability": per_capability,
        "attribution_gate": gate,
        "raw_outputs_sha256": sha256_file(raw),
        "wall_seconds": time.perf_counter() - started,
        "training_performed": False,
        "artifact_mutated": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Development-only forced-route oracle over already problematic records; diagnostic attribution only, not a routing, quality, runtime, or Phase 3 claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_COLLISION_ORACLE_PROTOCOL_V444.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_route_collision_oracle/oracle_v445")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
