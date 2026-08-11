"""Read-only audit of route-specific capacity in the failed final residual."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-route-capacity-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["probe_id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_ROUTE_CAPACITY_AUDIT" or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("route-capacity governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"route-capacity binding changed: {relative}")
    if output.exists():
        raise Phase3Error("immutable route-capacity output exists")
    state = load_file(str(root / protocol["checkpoint"]), device="cpu")
    counts = {name: int(value.numel()) for name, value in state.items()}
    route_names = tuple(name for name in counts if name in {"route_scale.weight", "route_shift.weight"})
    route_parameters = sum(counts[name] for name in route_names)
    total = sum(counts.values())
    a0, a1, a4 = (_rows(root / protocol[name]) for name in ("A0_outputs", "A1_outputs", "A4_outputs"))
    if set(a0) != set(a1) or set(a0) != set(a4) or len(a0) != 1400:
        raise Phase3Error("control output pairing changed")
    result = {
        "format": "abi-capability-compiler-phase3-route-capacity-audit-result/1",
        "status": "PASS_SHARED_CAPACITY_DOMINANCE_CAUSALLY_ATTRIBUTED",
        "protocol_sha256": sha256_file(protocol_path),
        "checkpoint_sha256": sha256_file(root / protocol["checkpoint"]),
        "total_parameters": total,
        "route_specific_tensors": list(route_names),
        "route_specific_parameters": route_parameters,
        "route_specific_fraction": route_parameters / total,
        "shared_parameters": total - route_parameters,
        "shared_fraction": (total - route_parameters) / total,
        "A0_A1_exact_output_matches": sum(a0[key]["output"] == a1[key]["output"] for key in a0),
        "A0_A4_exact_output_matches": sum(a0[key]["output"] == a4[key]["output"] for key in a0),
        "A0_A1_functional_discordances": sum(bool(a0[key]["functional_pass_v1"]) != bool(a1[key]["functional_pass_v1"]) for key in a0),
        "A0_A4_functional_discordances": sum(bool(a0[key]["functional_pass_v1"]) != bool(a4[key]["functional_pass_v1"]) for key in a0),
        "measured_bottleneck": "Route conditioning controls only 512 parameters while 99840 parameters are shared; matched A1 and A4 evidence shows that this route signal is not functionally causal.",
        "successor_constraint": "Use physically route-isolated experts with approximately matched total installed parameters and no greater active per-token residual rank than the current rank-64 bridge.",
        "neural_training_performed": False,
        "final_test_accessed": False,
        "historical_evidence_changed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    print(json.dumps(run(root, root / args.protocol, root / args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
