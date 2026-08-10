"""Recover exact source-neuron identities encoded in the failed layer-2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_routed_v15_progressive_extract as progressive
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v15-layer2-neuron-identity-audit/1"


def _pair_hash(gate: torch.Tensor, up: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(gate.contiguous().numpy().tobytes())
    digest.update(up.contiguous().numpy().tobytes())
    return digest.hexdigest()


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_EXACT_NEURON_IDENTITY_RECOVERY"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("neuron identity governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"neuron identity binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True)
    extraction, extraction_sha = progressive._load_protocol(
        root, root / protocol["extraction_protocol"]
    )
    if extraction_sha != protocol["extraction_protocol_sha256"]:
        raise Phase3Error("extraction protocol identity changed")
    base = json.loads((root / extraction["base_protocol"]).read_text(encoding="utf-8"))
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(torch.device("cuda")).eval()
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(
        base["source"]["parameter_count"]
    ):
        raise Phase3Error("loaded source parameter count changed")
    gate_up = teacher.model.layers[2].mlp.gate_up_proj.weight.detach().to(torch.float16).cpu()
    source_neurons = gate_up.shape[0] // 2
    source_pairs: dict[str, list[int]] = {}
    for index in range(source_neurons):
        value = _pair_hash(gate_up[index], gate_up[source_neurons + index])
        source_pairs.setdefault(value, []).append(index)
    failed = load_file(str(root / protocol["failed_checkpoint"]["path"]), device="cpu")
    stored = failed["layers.2.sparse_gate_up_projection.weight"].to(torch.float16)
    sparse_width = stored.shape[0] // 2
    recovered = []
    unmatched = []
    ambiguous = []
    for row in range(sparse_width):
        value = _pair_hash(stored[row], stored[sparse_width + row])
        candidates = source_pairs.get(value, [])
        exact = [
            index for index in candidates
            if torch.equal(stored[row], gate_up[index])
            and torch.equal(stored[sparse_width + row], gate_up[source_neurons + index])
        ]
        if not exact:
            unmatched.append(row)
        elif len(exact) != 1:
            ambiguous.append({"stored_row": row, "source_indices": exact})
        else:
            recovered.append(exact[0])
    unique = len(set(recovered)) == len(recovered)
    passed = not unmatched and not ambiguous and unique and len(recovered) == sparse_width
    identity_payload = {
        "format": "abi-capability-compiler-source-neuron-identity/1",
        "source_layer": 2,
        "canonical_dtype": "float16",
        "ordered_source_neuron_indices": recovered,
    }
    identity_bytes = json.dumps(identity_payload, indent=2, sort_keys=True).encode() + b"\n"
    identity_path = output / "source_neuron_identity.json"
    _write_immutable(identity_path, identity_bytes)
    result = {
        "format": FORMAT,
        "status": "PASS_EXACT_NEURON_IDENTITIES_RECOVERED" if passed else "FAIL_EXACT_NEURON_IDENTITY",
        "protocol_sha256": sha256_file(protocol_path),
        "extraction_protocol_sha256": extraction_sha,
        "source_neurons": source_neurons,
        "stored_sparse_width": sparse_width,
        "exact_matches": len(recovered),
        "unique_recovered_indices": len(set(recovered)),
        "unmatched_stored_rows": unmatched,
        "ambiguous_stored_rows": ambiguous,
        "identity": {
            "path": identity_path.name,
            "sha256": sha256_file(identity_path),
            "ordered_indices_sha256": hashlib.sha256(
                json.dumps(recovered, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "passed": passed,
        "training_performed": False,
        "checkpoint_written": False,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only exact fp16 source-neuron identity recovery only; no repaired layer, artifact, English quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(
        output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_LAYER2_NEURON_IDENTITY_PROTOCOL_V315.json",
    )
    parser.add_argument(
        "--output", default="results/abi_capability_compiler_phase3_routed_v15/layer2_identity_v316"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
