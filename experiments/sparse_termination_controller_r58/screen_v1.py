"""Screen the frozen R55 host with the frozen R58 stop controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.broad_payload_reconstruction_r51 import screen_v1a as common
from experiments.deep_sparse_adapters_r55 import screen_v1 as r55
from experiments.external_router_cake_r49.run_v1 import R49Error
from experiments.isolated_capability_cakes_r53 import fit_router
from experiments.sparse_termination_controller_r58.controller import (
    SparseTerminationController,
)


CONTROLLER_SHA256 = "3c9a7f5bc45b3a5eb5183ec29cd994c083b3d4fbfdfcb3898c17f82848b9ee22"
CONTROLLER_METADATA_SHA256 = "0de887077a3feee10e01f27e6d98e253e0ae87062bece6bce991ee43c92cc5b8"


def _preflight(host: Path, controller_dir: Path) -> dict[str, Any]:
    r55._preflight(host)
    controller_path = controller_dir / "termination_controller.safetensors"
    metadata_path = controller_dir / "metadata.json"
    if (
        common._sha256_file(controller_path) != CONTROLLER_SHA256
        or common._sha256_file(metadata_path) != CONTROLLER_METADATA_SHA256
    ):
        raise common.ScreenError("R58 controller identity changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    stored_manifest = unsigned.pop("manifest_sha256", None)
    computed_manifest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    controller = metadata.get("controller", {})
    training = metadata.get("training", {})
    inputs = metadata.get("inputs", {})
    frozen_host = metadata.get("frozen_host", {})
    if (
        stored_manifest != computed_manifest
        or metadata.get("status") != "TRAINED_NOT_YET_CERTIFIED"
        or controller.get("sha256") != CONTROLLER_SHA256
        or controller.get("routes") != 14
        or controller.get("width") != 768
        or controller.get("rank") != 16
        or controller.get("installed_parameters") != 172_494
        or controller.get("active_parameters_per_sequence") != 12_321
        or controller.get("maximum_active_routes_per_decision") != 1
        or controller.get("threshold_logit") != 0.0
        or controller.get("can_modify_token_logits") is not False
        or controller.get("can_select_or_rewrite_tokens") is not False
        or frozen_host.get("checkpoint_sha256") != r55.CANDIDATE_SHA256
        or frozen_host.get("metadata_sha256") != r55.METADATA_SHA256
        or frozen_host.get("unchanged_after_training") is not True
        or training.get("seed") != 58_001
        or training.get("successful_optimizer_steps") != 1_200
        or training.get("main_rows") != 16_134
        or training.get("anchor_rows") != 1_438
        or training.get("unique_main_rows_seen") != 16_134
        or training.get("unique_anchor_rows_seen") != 1_438
        or inputs.get("validation_or_final_outputs_used") != 0
        or inputs.get("teacher_calls") != 0
        or inputs.get("source_parameters_copied") != 0
    ):
        raise common.ScreenError("R58 controller training contract changed")
    return metadata


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    route: int,
    maximum: int,
    device: torch.device,
    *,
    controller: SparseTerminationController,
    telemetry: dict[str, Any],
):
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + maximum > model.config.max_tokens:
        raise R49Error("R58 validation prompt exceeds context")
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
        "next_hidden": result["hidden"][:, -1],
    }
    for _ in range(maximum):
        stop = controller.should_stop(state["next_hidden"], route)
        telemetry["decisions"] += 1
        if controller.last_calls != (route,):
            telemetry["selected_route_exact"] = False
            raise R49Error("R58 termination controller selected another route")
        if stop:
            telemetry["controller_stops"] += 1
            break
        selected = state["next_logits"][0].argmax(dim=-1).reshape(1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            telemetry["language_model_eos_stops"] += 1
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
            "next_hidden": result["hidden"][:, -1],
        }
    elapsed = time.perf_counter() - started
    output = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output, generated, elapsed, physical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
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
    host = args.host.resolve()
    controller_dir = args.controller.resolve()
    _preflight(host, controller_dir)
    device = torch.device("cuda")
    controller = SparseTerminationController().to(device)
    controller.load_state_dict(
        load_file(
            str(controller_dir / "termination_controller.safetensors"),
            device="cuda",
        ),
        strict=True,
    )
    controller.eval()
    telemetry: dict[str, Any] = {
        "artifact_sha256": CONTROLLER_SHA256,
        "metadata_sha256": CONTROLLER_METADATA_SHA256,
        "decisions": 0,
        "controller_stops": 0,
        "language_model_eos_stops": 0,
        "selected_route_exact": True,
        "maximum_active_routes_per_decision": 1,
        "token_logits_modified": False,
        "tokens_selected_or_rewritten": False,
    }

    def generate(model, tokenizer, prompt, route, maximum, screen_device):
        return _generate(
            model,
            tokenizer,
            prompt,
            route,
            maximum,
            screen_device,
            controller=controller,
            telemetry=telemetry,
        )

    common.run(
        host,
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
        result_format="abi-r58-sparse-termination-controller-development-screen/1",
        campaign_name="R58",
        expected_training_seed=55_001,
        router_sha256=r55.ROUTER_SHA256,
        capability_to_route=fit_router.CAPABILITY_TO_INDEX,
        generation_fn=generate,
        extra_result_fields={"termination_controller": telemetry},
    )
    if common._sha256_file(
        controller_dir / "termination_controller.safetensors"
    ) != CONTROLLER_SHA256:
        raise common.ScreenError("R58 controller changed during evaluation")


if __name__ == "__main__":
    main()
