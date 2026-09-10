import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from experiments.preexisting_representation_r15b.generation_qualification import (
    parse_final_digit,
)
from experiments.preexisting_representation_r15b.isolated_worker import _decode
from experiments.preexisting_representation_r15b.isolation import build_capsule
from experiments.preexisting_representation_r15b.protocol import heldout_capabilities
from experiments.preexisting_representation_r15b.public_qualification import (
    DEPTHS,
    OPERATIONS,
    apply_program,
    build_rows,
    wilson_lower,
)
from experiments.preexisting_representation_r15b.representation import (
    RepresentationError,
    decode_labels,
    decode_transition,
    labels_to_operations,
)


def test_registered_operations_are_noncommutative_permutations() -> None:
    images = []
    for _name, multiplier, offset in OPERATIONS:
        image = tuple((multiplier * value + offset) % 8 for value in range(8))
        assert sorted(image) == list(range(8))
        images.append(image)
    left = apply_program(0, (0, 1))
    right = apply_program(0, (1, 0))
    assert left != right


def test_public_rows_are_deterministic_unique_and_correct() -> None:
    first = build_rows(rows_per_depth=32, seed=1515001)
    second = build_rows(rows_per_depth=32, seed=1515001)
    assert first == second
    assert len(first) == sum(min(32, 8 * 3**depth) for depth in DEPTHS)
    assert len({row["row_id"] for row in first}) == len(first)
    assert all(row["answer"] == apply_program(row["start"], tuple(row["program"])) for row in first)


def test_expression_prompt_is_a_distinct_public_formulation() -> None:
    instructions = build_rows(rows_per_depth=32, seed=1515001)
    expressions = build_rows(rows_per_depth=32, seed=1515001, prompt_style="expression")
    assert [row["answer"] for row in instructions] == [row["answer"] for row in expressions]
    assert [row["prompt_sha256"] for row in instructions] != [
        row["prompt_sha256"] for row in expressions
    ]
    reasoning = build_rows(
        rows_per_depth=32,
        seed=1515001,
        prompt_style="expression_reasoning",
    )
    assert all("FINAL:" in row["prompt"] for row in reasoning)
    indexed = build_rows(
        rows_per_depth=32,
        seed=1515001,
        prompt_style="indexed_steps",
    )
    assert all("x0 =" in row["prompt"] and "FINAL:" in row["prompt"] for row in indexed)


def test_wilson_lower_is_fail_closed_and_monotonic() -> None:
    assert wilson_lower(0, 100) < 1e-15
    assert wilson_lower(80, 100) < wilson_lower(90, 100) < wilson_lower(100, 100)


def test_final_digit_parser_is_strict_and_uses_last_marked_answer() -> None:
    assert parse_final_digit("reasoning\nFINAL: 6") == 6
    assert parse_final_digit("FINAL: 2\ncorrection FINAL: 5.") == 5
    assert parse_final_digit("answer is 6") is None
    assert parse_final_digit("FINAL: 9") is None


def test_representation_decoder_recovers_affine_transition() -> None:
    output_rows = torch.eye(8)
    labels = [1, 2, 0, 3, 7, 0]
    residuals = torch.eye(8)[labels].reshape(3, 2, 8)
    assert decode_labels(residuals, output_rows) == labels
    assert labels_to_operations(labels) == [(1, 1), (3, 0), (1, 7)]
    transition = decode_transition(residuals, output_rows)
    assert transition.shape == (3, 8, 8)
    assert torch.equal(transition.sum(dim=-1), torch.ones(3, 8))


def test_representation_decoder_rejects_non_affine_labels() -> None:
    output_rows = torch.eye(8)
    residuals = torch.eye(8)[[0, 2, 0, 3, 7, 0]].reshape(3, 2, 8)
    with pytest.raises((RepresentationError, RuntimeError)):
        decode_transition(residuals, output_rows)


def test_heldout_slot_permutations_are_committed_unique_and_deterministic() -> None:
    secret = bytes(range(32))
    commitment = hashlib.sha256(secret).hexdigest()
    first = heldout_capabilities(secret.hex(), expected_commitment=commitment, count=4)
    second = heldout_capabilities(secret.hex(), expected_commitment=commitment, count=4)
    assert [item.slot_order for item in first] == [item.slot_order for item in second]
    assert len({item.slot_order for item in first}) == 4
    assert len({item.capability.capability_id for item in first}) == 4


def test_pure_stdlib_capsule_decoder_reads_only_anonymous_bundle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "bundle.safetensors"
    output_rows = torch.eye(8)
    labels = [1, 2, 0, 3, 7, 0]
    residuals = torch.eye(8)[labels].reshape(3, 2, 8)
    save_file(
        {"residuals": residuals, "output_rows": output_rows},
        str(bundle),
        metadata={"format": "abi-r15b-anonymous-pre-answer-representation/1"},
    )
    capsule = tmp_path / "capsule"
    manifest = build_capsule(root=root, representation=bundle, capsule=capsule)
    assert manifest["prompts_included"] == 0
    assert manifest["answers_included"] == 0
    assert manifest["semantic_labels_included"] == 0
    assert _decode(capsule)["labels"] == labels
