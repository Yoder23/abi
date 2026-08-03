import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from abi.layercake_component_rollback import (
    TOKENIZER_FILES,
    build_component_rollback_parent,
)
from abi.layercake_full_core_acquisition import FullCoreAcquisitionError


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path, *, route_five_value, shared_value=7.0):
    path.mkdir()
    checkpoint = path / "model.safetensors"
    save_file(
        {
            "shared.weight": torch.tensor([shared_value]),
            "task_cakes.2.weight": torch.tensor([2.0]),
            "task_cakes.5.weight": torch.tensor([route_five_value]),
        },
        str(checkpoint),
    )
    for index, filename in enumerate(TOKENIZER_FILES):
        (path / filename).write_text(
            f"tokenizer-{index}",
            encoding="utf-8",
        )
    metadata = {
        "format": "abi-layercake-full-english-core-acquisition/1",
        "status": "TEST",
        "architecture": {
            "vocab_size": 3,
            "width": 3,
            "layers": 3,
            "heads": 1,
            "max_tokens": 8,
            "task_cakes": 10,
            "task_cake_rank": 1,
            "architecture_version": "test",
        },
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "tokenizer": {
            "path": "tokenizer.json",
            "sha256": _sha256(path / "tokenizer.json"),
        },
        "canonical_semantic_abi": {"sha256": "a" * 64},
        "acquired_core": {
            "active_parameter_count": 2,
            "total_parameter_count": 3,
        },
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_component_rollback_replaces_only_selected_route(tmp_path):
    target = tmp_path / "target"
    donor = tmp_path / "donor"
    output = tmp_path / "output"
    _artifact(target, route_five_value=5.0)
    _artifact(donor, route_five_value=9.0)
    target_checkpoint_sha = _sha256(target / "model.safetensors")
    donor_checkpoint_sha = _sha256(donor / "model.safetensors")

    manifest = build_component_rollback_parent(
        target_path=target,
        donor_path=donor,
        output_path=output,
        selected_task_cake_routes=(5,),
    )
    target_state = load_file(str(target / "model.safetensors"))
    donor_state = load_file(str(donor / "model.safetensors"))
    output_state = load_file(str(output / "model.safetensors"))
    assert torch.equal(
        output_state["task_cakes.5.weight"],
        donor_state["task_cakes.5.weight"],
    )
    assert torch.equal(
        output_state["task_cakes.2.weight"],
        target_state["task_cakes.2.weight"],
    )
    assert torch.equal(
        output_state["shared.weight"],
        target_state["shared.weight"],
    )
    rollback = manifest["component_rollback"]
    assert rollback["selected_task_cake_routes"] == [5]
    assert rollback["selected_tensor_count"] == 1
    assert rollback["changed_tensor_count"] == 1
    assert rollback[
        "all_unselected_tensors_byte_identical_to_target"
    ] is True
    assert rollback[
        "all_selected_tensors_byte_identical_to_donor"
    ] is True
    assert manifest["acquired_core"]["trainable_parameter_count"] == 0
    assert _sha256(target / "model.safetensors") == target_checkpoint_sha
    assert _sha256(donor / "model.safetensors") == donor_checkpoint_sha


def test_component_rollback_rejects_architecture_mismatch(tmp_path):
    target = tmp_path / "target"
    donor = tmp_path / "donor"
    _artifact(target, route_five_value=5.0)
    _artifact(donor, route_five_value=9.0)
    metadata_path = donor / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["architecture"]["width"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(
        FullCoreAcquisitionError,
        match="architectures differ",
    ):
        build_component_rollback_parent(
            target_path=target,
            donor_path=donor,
            output_path=tmp_path / "output",
            selected_task_cake_routes=(5,),
        )


def test_component_rollback_rejects_unselected_route_or_noop(tmp_path):
    target = tmp_path / "target"
    donor = tmp_path / "donor"
    _artifact(target, route_five_value=5.0)
    _artifact(donor, route_five_value=5.0)
    with pytest.raises(
        FullCoreAcquisitionError,
        match="already equal",
    ):
        build_component_rollback_parent(
            target_path=target,
            donor_path=donor,
            output_path=tmp_path / "output",
            selected_task_cake_routes=(5,),
        )
