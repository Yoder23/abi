"""Public-only R11 host feasibility check; never certification evidence."""

from __future__ import annotations

import gc
import json

import torch

from experiments.copy_paste_r10.run import R10FrozenNeuralHost
from experiments.copy_paste_r10.runtime import canonical_prediction
from experiments.native_transfer_r8.capability_generator import (
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import SPECS

from .core import (
    NativeResidualISAHost,
    R11Error,
    RecurrentTransitionNeuralISA,
    sha256_bytes,
    train_teacher_transition,
    transition_accuracy,
)

HOST_SCALES = {"source": 10.0, "pythia": 200.0, "qwen2": 100.0, "t5": 2.0}


def main() -> int:
    capability = public_capabilities(90210, split="development", count=1)[0]
    training = generate_rows(
        capability,
        split="r11_public_preflight_training",
        rows=288,
        depths=[1, 2, 3],
        seed=4101,
    )
    evaluation = generate_rows(
        capability,
        split="r11_public_preflight_evaluation",
        rows=128,
        depths=[4, 5, 6, 7],
        seed=4103,
    )
    transition, training_receipt = train_teacher_transition(
        training,
        steps=300,
        learning_rate=0.1,
        seed=4105,
        device="cuda",
    )
    if transition_accuracy(transition, evaluation) != 1.0:
        raise R11Error("public teacher preflight did not generalize exactly")
    executor = RecurrentTransitionNeuralISA().to("cuda").eval()
    results = {}
    for host_key, scale in HOST_SCALES.items():
        raw_host = R10FrozenNeuralHost(SPECS[host_key], device="cuda")
        weight = raw_host.model.get_output_embeddings().weight.detach().float()
        indices = torch.tensor(raw_host.target_token_ids, device=weight.device)
        codec = weight.index_select(0, indices)
        codec = codec / codec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        codec = codec.cpu().float().contiguous()
        codec_sha256 = sha256_bytes(codec.numpy().tobytes())
        del raw_host, weight, indices
        gc.collect()
        torch.cuda.empty_cache()
        host = NativeResidualISAHost(
            host_key,
            codec=codec,
            codec_sha256=codec_sha256,
            scale=scale,
            device="cuda",
        )
        correct = 0
        for offset in range(0, len(evaluation), 32):
            batch = evaluation[offset : offset + 32]
            prompts = [str(row["prompt"]) for row in batch]
            distribution = executor(transition.to("cuda"), prompts)
            predictions = host.logits(prompts, distribution).argmax(dim=-1).cpu().tolist()
            canonical = [
                canonical_prediction(token_id, host.target_token_ids) for token_id in predictions
            ]
            correct += sum(
                int(prediction == row["answer"]) for prediction, row in zip(canonical, batch)
            )
        accuracy = correct / len(evaluation)
        host.verify_frozen()
        results[host_key] = {"accuracy": accuracy, "scale": scale}
        print(json.dumps({host_key: results[host_key]}, sort_keys=True), flush=True)
        del host, codec
        gc.collect()
        torch.cuda.empty_cache()
    if any(item["accuracy"] != 1.0 for item in results.values()):
        raise R11Error("one or more public native hosts were not exact")
    print(
        json.dumps(
            {
                "status": "PUBLIC_PREFLIGHT_ONLY_NOT_CERTIFICATION",
                "teacher_training": training_receipt,
                "hosts": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
