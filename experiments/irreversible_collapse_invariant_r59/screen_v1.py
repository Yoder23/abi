"""Screen R55 under an exact irreversible collapse-transition invariant."""

from __future__ import annotations

import argparse
from collections import Counter
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from abi.layercake_host import _truncate_novel_lexical_repetition
from experiments.broad_payload_reconstruction_r51 import screen_v1a as common
from experiments.deep_sparse_adapters_r55 import screen_v1 as r55
from experiments.external_router_cake_r49.run_v1 import R49Error
from experiments.isolated_capability_cakes_r53 import fit_router


MAXIMUM_IDENTICAL_TOKEN_RUN = 6
REPEATED_NOVEL_LEXICAL_FOURGRAMS = 4
CANDIDATE_SHA256 = r55.CANDIDATE_SHA256
METADATA_SHA256 = r55.METADATA_SHA256
ROUTER_SHA256 = r55.ROUTER_SHA256
PARENT_COUNTS = r55.PARENT_COUNTS


def _completed_output_words(output: str) -> list[str]:
    matches = list(re.finditer(r"[\w']+", output.casefold()))
    if matches and matches[-1].end() == len(output):
        matches = matches[:-1]
    return [match.group() for match in matches]


def _irreversible_collapse_reason(
    token_ids: Sequence[int],
    output: str,
    prompt: str,
) -> str | None:
    maximum_run = 0
    current_run = 0
    previous = None
    for token in token_ids:
        if token == previous:
            current_run += 1
        else:
            previous = token
            current_run = 1
        maximum_run = max(maximum_run, current_run)
    if maximum_run >= MAXIMUM_IDENTICAL_TOKEN_RUN:
        return "maximum_identical_token_run"

    # A decoded prefix may end midway through a BPE-composed word. That last
    # lexical item is not irreversible until a delimiter closes it.
    output_words = _completed_output_words(output)
    prompt_words = re.findall(r"[\w']+", prompt.casefold())
    prompt_fourgrams = {
        tuple(prompt_words[index : index + 4])
        for index in range(max(0, len(prompt_words) - 3))
    }
    output_fourgrams = [
        tuple(output_words[index : index + 4])
        for index in range(max(0, len(output_words) - 3))
    ]
    repeated = sum(
        count - 1
        for value, count in Counter(output_fourgrams).items()
        if count > 1 and value not in prompt_fourgrams
    )
    if repeated >= REPEATED_NOVEL_LEXICAL_FOURGRAMS:
        return "repeated_novel_lexical_fourgrams"
    return None


def _maximum_identical_token_run(token_ids: Sequence[int]) -> int:
    maximum_run = 0
    current_run = 0
    previous = None
    for token in token_ids:
        if token == previous:
            current_run += 1
        else:
            previous = token
            current_run = 1
        maximum_run = max(maximum_run, current_run)
    return maximum_run


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    route: int,
    maximum: int,
    device: torch.device,
    *,
    telemetry: dict[str, Any],
):
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + maximum > model.config.max_tokens:
        raise R49Error("R59 validation prompt exceeds context")
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
        selected = state["next_logits"][0].argmax(dim=-1).reshape(1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            telemetry["language_model_eos_stops"] += 1
            break
        candidate = generated + [token]
        telemetry["candidate_transitions_checked"] += 1
        if _maximum_identical_token_run(candidate) >= MAXIMUM_IDENTICAL_TOKEN_RUN:
            telemetry["rejected_boundary_tokens"] += 1
            telemetry["rejections_by_condition"][
                "maximum_identical_token_run"
            ] += 1
            break
        generated.append(token)
        telemetry["accepted_tokens"] += 1
        result = model(
            selected[:, None],
            task_routes=route_tensor,
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        physical.append(tuple(model.last_cake_calls))
        if tuple(model.last_cake_calls) != (route,):
            raise R49Error("R59 model executed another capability route")
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
    baseline_output = _truncate_novel_lexical_repetition(
        output,
        prompt,
        threshold=1,
    )
    if baseline_output != output:
        telemetry["inherited_r55_lexical_truncations"] += 1
        output = baseline_output
        generated = tokenizer.encode(output)
    return output, generated, elapsed, physical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(r55.PARENT_COUNTS), required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", default=[])
    parser.add_argument("--live-source-dir", type=Path)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    r55._preflight(candidate)
    telemetry: dict[str, Any] = {
        "algorithm": "r55_greedy_plus_sixth_identical_token_circuit_breaker",
        "maximum_identical_token_run_boundary": MAXIMUM_IDENTICAL_TOKEN_RUN,
        "repeated_novel_lexical_fourgram_boundary": REPEATED_NOVEL_LEXICAL_FOURGRAMS,
        "candidate_transitions_checked": 0,
        "accepted_tokens": 0,
        "rejected_boundary_tokens": 0,
        "rejections_by_condition": {
            "maximum_identical_token_run": 0,
            "repeated_novel_lexical_fourgrams": 0,
        },
        "language_model_eos_stops": 0,
        "inherited_r55_lexical_truncation_threshold": 1,
        "inherited_r55_lexical_truncations": 0,
        "alternate_tokens_selected": 0,
        "token_logits_modified": False,
        "weights_changed": False,
    }

    def generate(model, tokenizer, prompt, route, maximum, screen_device):
        return _generate(
            model,
            tokenizer,
            prompt,
            route,
            maximum,
            screen_device,
            telemetry=telemetry,
        )

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
        candidate_sha256=r55.CANDIDATE_SHA256,
        metadata_sha256=r55.METADATA_SHA256,
        parent_counts=r55.PARENT_COUNTS,
        result_format="abi-r59-irreversible-collapse-invariant-development-screen/1",
        campaign_name="R59",
        expected_training_seed=55_001,
        router_sha256=r55.ROUTER_SHA256,
        capability_to_route=fit_router.CAPABILITY_TO_INDEX,
        generation_fn=generate,
        extra_result_fields={"runtime_invariant": telemetry},
    )


if __name__ == "__main__":
    main()
