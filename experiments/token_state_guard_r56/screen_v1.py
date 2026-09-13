"""Screen frozen R55 with the preregistered response-only token-state guard."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from abi.layercake_host import (
    _select_next_token,
    _truncate_novel_lexical_repetition,
)
from experiments.broad_payload_reconstruction_r51 import screen_v1a as common
from experiments.deep_sparse_adapters_r55 import screen_v1 as r55
from experiments.external_router_cake_r49.run_v1 import R49Error
from experiments.isolated_capability_cakes_r53 import fit_router


CANDIDATE_SHA256 = r55.CANDIDATE_SHA256
METADATA_SHA256 = r55.METADATA_SHA256
ROUTER_SHA256 = r55.ROUTER_SHA256
PARENT_COUNTS = r55.PARENT_COUNTS
NO_REPEAT_NGRAM_SIZE = 4


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    route: int,
    maximum: int,
    device: torch.device,
):
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + maximum > model.config.max_tokens:
        raise R49Error("R56 validation prompt exceeds context")
    route_tensor = torch.tensor([route], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
        task_routes=route_tensor,
        use_cache=True,
    )
    physical = [tuple(model.last_cake_calls)]
    generated: list[int] = []
    state = {
        "past_key_values": result["past_key_values"],
        "next_logits": result["logits"][:, -1],
    }
    for _ in range(maximum):
        selected = _select_next_token(
            state["next_logits"][0],
            generated=generated,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        ).to(device)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        result = model(
            selected[:, None],
            task_routes=route_tensor,
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        physical.append(tuple(model.last_cake_calls))
        state = {
            "past_key_values": result["past_key_values"],
            "next_logits": result["logits"][:, -1],
        }
    elapsed = time.perf_counter() - started
    output = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output = _truncate_novel_lexical_repetition(output, prompt, threshold=1)
    generated = tokenizer.encode(output)
    return output, generated, elapsed, physical


def _preflight(candidate: Path) -> None:
    r55._preflight(candidate)


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
    candidate = args.candidate.resolve()
    _preflight(candidate)
    common.run(
        candidate,
        args.router.resolve(),
        args.catalog.resolve(),
        args.split,
        [path.resolve() for path in args.source_bundle],
        args.live_source_dir.resolve() if args.live_source_dir else None,
        args.broad_bundle.resolve(),
        args.anchor_bundle.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
        candidate_sha256=CANDIDATE_SHA256,
        metadata_sha256=METADATA_SHA256,
        parent_counts=PARENT_COUNTS,
        result_format="abi-r56-token-state-guard-development-screen/1",
        campaign_name="R56",
        expected_training_seed=55_001,
        router_sha256=ROUTER_SHA256,
        capability_to_route=fit_router.CAPABILITY_TO_INDEX,
        generation_fn=generate,
        extra_result_fields={
            "runtime_delta": {
                "algorithm": "greedy_response_only_no_repeat_ngram",
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                "prompt_ngrams_in_history": False,
                "weights_changed": False,
            }
        },
    )


if __name__ == "__main__":
    main()
