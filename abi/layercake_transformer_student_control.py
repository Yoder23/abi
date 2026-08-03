"""Materialize a shared-initialization transformer-path diagnostic control."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from .artifacts import module_state_sha256
from .layercake_core_loader import load_layercake_core
from .layercake_full_core_acquisition import ARTIFACT_FORMAT, _manifest_sha
from .layercake_host import _is_within, _sha256_file


CONTROL_FORMAT = "abi-layercake-transformer-student-control-base/1"
CONTROL_ROLE = (
    "SHARED_INITIALIZATION_SAME_BUDGET_TRANSFORMER_STUDENT_CONTROL"
)


class TransformerStudentControlError(RuntimeError):
    """Raised when the diagnostic control cannot be made exact."""


def disable_task_cake_effects(model: torch.nn.Module) -> dict[str, Any]:
    """Make every post-transformer task cake the exact identity function."""

    if not getattr(model, "task_cakes", None):
        raise TransformerStudentControlError("task cakes are missing")
    with torch.no_grad():
        for cake in model.task_cakes:
            cake.up.weight.zero_()
    identity_count = sum(
        int(torch.count_nonzero(cake.up.weight).item()) == 0
        for cake in model.task_cakes
    )
    return {
        "installed_task_cakes": len(model.task_cakes),
        "identity_task_cakes": identity_count,
        "zeroed_up_projection_parameters": sum(
            cake.up.weight.numel() for cake in model.task_cakes
        ),
        "task_cake_effect_disabled_exact": identity_count
        == len(model.task_cakes),
        "task_cakes_state_sha256": module_state_sha256(model.task_cakes),
        "task_classifier_state_sha256": module_state_sha256(
            model.task_classifier
        ),
    }


@torch.inference_mode()
def _verify_plain_transformer_equivalence(
    model: torch.nn.Module,
    *,
    token_id: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [[token_id, token_id]], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    direct_hidden = model.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    direct_logits = F.linear(direct_hidden, model.output_weight)
    routed_logits = model(
        input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([2], device=device),
        task_routes=torch.tensor([0], device=device),
        use_cache=False,
    )["logits"]
    maximum_absolute_difference = float(
        (direct_logits - routed_logits).abs().max().item()
    )
    return {
        "probe_token_id": token_id,
        "maximum_absolute_logit_difference": maximum_absolute_difference,
        "exact": maximum_absolute_difference == 0.0,
    }


def materialize_transformer_student_control(
    *,
    parent_path: str | Path,
    layercake_root: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Copy the exact shared v41 transformer and disable only cake effects."""

    parent_path = Path(parent_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    abi_root = Path(__file__).resolve().parents[1]
    if not _is_within(parent_path, abi_root):
        raise TransformerStudentControlError(
            "diagnostic parent must be in the ABI evidence tree"
        )
    if _is_within(output_path, layercake_root):
        raise TransformerStudentControlError(
            "diagnostic output may not modify the sealed LayerCake tree"
        )
    if output_path.exists():
        raise TransformerStudentControlError(
            f"control artifact is immutable: {output_path}"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise TransformerStudentControlError("CUDA is unavailable")

    metadata_path = parent_path / "metadata.json"
    checkpoint_path = parent_path / "model.safetensors"
    parent_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parent_checkpoint_sha = _sha256_file(checkpoint_path)
    if parent_metadata.get("checkpoint", {}).get("sha256") != parent_checkpoint_sha:
        raise TransformerStudentControlError("parent checkpoint changed")
    canonical_abi_sha = _sha256_file(canonical_abi_path)
    if (
        parent_metadata.get("canonical_semantic_abi", {}).get("sha256")
        != canonical_abi_sha
    ):
        raise TransformerStudentControlError("canonical ABI binding changed")

    device = torch.device(device_name)
    model, tokenizer, _ = load_layercake_core(
        parent_path,
        layercake_root=layercake_root,
        device=device,
    )
    model.eval()
    shared_state_before = module_state_sha256(model.transformer)
    parent_state_before = module_state_sha256(model)
    cake_control = disable_task_cake_effects(model)
    shared_state_after = module_state_sha256(model.transformer)
    equivalence = _verify_plain_transformer_equivalence(
        model,
        token_id=int(tokenizer.eos_token_id),
    )
    if (
        shared_state_after != shared_state_before
        or cake_control["task_cake_effect_disabled_exact"] is not True
        or equivalence["exact"] is not True
    ):
        raise TransformerStudentControlError(
            "plain-transformer control equivalence did not verify"
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    shared_transformer_parameters = sum(
        parameter.numel() for parameter in model.transformer.parameters()
    )
    frozen_container_parameters = total_parameters - shared_transformer_parameters
    output_path.mkdir(parents=True, exist_ok=False)
    output_checkpoint = output_path / "model.safetensors"
    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
        str(output_checkpoint),
    )
    tokenizer.save_pretrained(output_path)
    output_tokenizer = output_path / "tokenizer.json"
    manifest: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "status": "MATERIALIZED_DIAGNOSTIC_NOT_TRAINED_NOT_PROMOTABLE",
        "architecture": copy.deepcopy(parent_metadata["architecture"]),
        "checkpoint": {
            "path": output_checkpoint.name,
            "sha256": _sha256_file(output_checkpoint),
            "bytes": output_checkpoint.stat().st_size,
        },
        "tokenizer": {
            "path": output_tokenizer.name,
            "sha256": _sha256_file(output_tokenizer),
        },
        "parent_layercake": {
            "path_at_materialization": str(parent_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "checkpoint_sha256": parent_checkpoint_sha,
            "logical_state_sha256_before": parent_state_before,
            "shared_transformer_state_sha256_before": shared_state_before,
            "shared_transformer_state_sha256_after": shared_state_after,
            "shared_transformer_tensors_preserved_exact": True,
            "unchanged_on_disk": _sha256_file(checkpoint_path)
            == parent_checkpoint_sha,
        },
        "acquired_core": {
            "logical_state_sha256_after": module_state_sha256(model),
            "total_parameter_count": total_parameters,
            "training_graph_parameter_count": total_parameters,
            "trainable_parameter_count": 0,
            "optimizer_parameter_element_count": 0,
            "frozen_parameter_count": total_parameters,
            "trainable_scope": "none_materialized_control",
            "active_parameter_count": shared_transformer_parameters,
            "graph_topology_changed": False,
            "parameter_shapes_changed": False,
            "task_cake_count": len(model.task_cakes),
            "maximum_active_task_cakes_per_sequence": 1,
            "physical_sparse_topology_preserved": True,
        },
        "canonical_semantic_abi": {
            "path_at_materialization": str(canonical_abi_path),
            "sha256": canonical_abi_sha,
            "changed": False,
        },
        "decoding": copy.deepcopy(parent_metadata.get("decoding")),
        "decoding_contract": copy.deepcopy(
            parent_metadata.get("decoding_contract")
        ),
        "foreign_source_boundary": {
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
            "source_parameters_copied": 0,
            "source_parameters_retained_exact": 0,
            "source_generated_text_retained_in_deployment": False,
            "teacher_tokenizer_required_at_inference": False,
        },
        "transformer_student_control": {
            "format": CONTROL_FORMAT,
            "role": CONTROL_ROLE,
            "purpose": "Quality and acquisition attribution only",
            "same_shared_initialization_as_layercake_parent": True,
            "shared_transformer_parameter_count": shared_transformer_parameters,
            "frozen_identity_container_parameter_count": (
                frozen_container_parameters
            ),
            "total_container_parameter_count": total_parameters,
            "task_classifier_retained_but_behaviorally_irrelevant": True,
            **cake_control,
            "plain_transformer_equivalence": equivalence,
            "inference_performance_baseline": False,
            "abi_transfer_credit": False,
            "promotion_eligible": False,
        },
        "claim_boundary": (
            "This diagnostic preserves the exact v41 shared transformer "
            "initialization and disables post-transformer cake effects. It is "
            "not an ABI transfer, LayerCake candidate, runtime baseline, or "
            "production artifact."
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    (output_path / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    result = materialize_transformer_student_control(
        parent_path=args.parent,
        layercake_root=args.layercake_root,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
