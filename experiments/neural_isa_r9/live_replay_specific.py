"""Live-replay a frozen R9 Gate A backend and compare every stored observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    module_sha256,
    sha256_file,
)

from .backend import PackageConditionedGRUBackend
from .run_specific_diagnostic import (
    _condition_packages,
    _evaluate,
    _json,
    _resolve,
)
from .verify_specific_diagnostic import R9VerificationError, _jsonl, verify


class R9ReplayError(RuntimeError):
    """Raised when a frozen backend does not reproduce its raw observations."""


def _compare_rows(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]], label: str
) -> None:
    if len(expected) != len(actual):
        raise R9ReplayError(f"{label} replay depth changed")
    for index, (stored, live) in enumerate(zip(expected, actual)):
        identity = ("row_id", "capability_id", "prompt_sha256", "depth", "flavor", "condition")
        if any(stored.get(key) != live.get(key) for key in identity):
            raise R9ReplayError(f"{label} replay identity changed at row {index}")
        if stored.get("prediction_token_id") != live.get("prediction_token_id"):
            raise R9ReplayError(f"{label} live prediction changed at row {index}")
        for key in ("canonical_output_probabilities", "teacher_canonical_probabilities"):
            left = stored.get(key)
            right = live.get(key)
            if (
                not isinstance(left, list)
                or not isinstance(right, list)
                or len(left) != len(right)
                or max(abs(float(a) - float(b)) for a, b in zip(left, right)) > 1e-6
            ):
                raise R9ReplayError(f"{label} live probabilities changed at row {index}")
        if abs(float(stored["teacher_recipient_tv"]) - float(live["teacher_recipient_tv"])) > 1e-6:
            raise R9ReplayError(f"{label} live distribution distance changed at row {index}")


def replay(config_path: Path, run_dir: Path) -> dict[str, Any]:
    verified = verify(config_path, run_dir)
    config = _json(config_path)
    root = Path(__file__).resolve().parents[2]
    receipt = _json(run_dir / "receipt.json")
    reference = config["r8_reference"]
    r8_config = _json(_resolve(root, str(reference["config"])))
    latent_path = _resolve(root, str(reference["canonical_latents"]))
    if sha256_file(latent_path) != reference["canonical_latents_sha256"]:
        raise R9ReplayError("canonical package changed before replay")
    tensors = load_file(str(latent_path), device="cpu")
    gate = config["gate_a"]
    settings = gate["backend"]
    state_layers = tuple(str(value) for value in settings.get("recipient_state_layers", ["final"]))
    split = r8_config["splits"]
    capabilities = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    capability_index = int(gate["development_capability_index"])
    capability = capabilities[capability_index]
    after = tensors["development_after"][capability_index].float()
    before = tensors["before"].float()
    wrong = tensors["development_after"][int(gate["wrong_capability_index"])].float()
    evaluation_rows = generate_rows(
        capability,
        split="r9_specific_evaluation",
        rows=int(gate["evaluation_rows"]),
        depths=gate["evaluation_depths"],
        seed=int(gate["seed"]) + 2,
    )
    training_rows = generate_rows(
        capability,
        split="r9_specific_train",
        rows=int(gate["train_rows"]),
        depths=gate["train_depths"],
        seed=int(gate["seed"]) + 1,
    )
    host = FrozenNeuralHost(SPECS[str(gate["host"])], device="cuda")
    backend = PackageConditionedGRUBackend(
        host.hidden_width * len(state_layers), hidden_width=int(settings["hidden_width"])
    ).to(host.device)
    backend_path = run_dir / str(receipt["backend"]["path"])
    backend.load_state_dict(load_file(str(backend_path), device=str(host.device)), strict=True)
    backend.eval()
    for parameter in backend.parameters():
        parameter.requires_grad_(False)
    if module_sha256(backend) != receipt["backend"]["state_sha256_after_evaluation"]:
        raise R9ReplayError("loaded backend state changed")
    conditions = _condition_packages(after, before, wrong, capability.capability_id)
    live_evaluation = _evaluate(
        host,
        backend,
        evaluation_rows,
        conditions,
        after=after,
        batch_size=int(settings["batch_size"]),
        residual_scale=float(settings["residual_scale"]),
        state_layers=state_layers,
    )
    stored_evaluation = _jsonl(run_dir / str(receipt["observations"]["path"]))
    _compare_rows(stored_evaluation, live_evaluation, "evaluation")
    live_training = _evaluate(
        host,
        backend,
        training_rows,
        {"AFTER": after},
        after=after,
        batch_size=int(settings["batch_size"]),
        residual_scale=float(settings["residual_scale"]),
        state_layers=state_layers,
    )
    stored_training = _jsonl(run_dir / str(receipt["training_observations"]["path"]))
    _compare_rows(stored_training, live_training, "training")
    host.verify_frozen()
    if module_sha256(backend) != receipt["backend"]["state_sha256_after_evaluation"]:
        raise R9ReplayError("backend changed during live replay")
    result = {
        "format": "abi-neural-isa-r9-live-replay/1",
        "status": "PASS_EXACT_LIVE_REPLAY",
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(run_dir / "receipt.json"),
        "backend_sha256": sha256_file(backend_path),
        "observations_sha256": sha256_file(run_dir / str(receipt["observations"]["path"])),
        "training_observations_sha256": sha256_file(
            run_dir / str(receipt["training_observations"]["path"])
        ),
        "evaluation_rows_replayed": len(live_evaluation),
        "training_rows_replayed": len(live_training),
        "verified_result_evidence_sha256": verified["evidence_sha256"],
        "trusted_scientific_booleans_consumed": 0,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable output exists: {output}"}, indent=2))
        return 2
    try:
        value = replay(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except (OSError, ValueError, R9VerificationError, R9ReplayError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
