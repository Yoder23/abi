from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from experiments.foreign_capability_r14.capability import (
    MODULUS,
    OPERATORS,
    generate_rows,
)
from experiments.foreign_neural_state_r15.frontend import (
    R15FrontendError,
    frontend_spec,
    labels_for_capability,
    load_frozen_frontend,
    predict_labels,
    transition_from_labels,
)
from experiments.foreign_neural_state_r15.isolated_worker import (
    IsolatedR15Error,
    _decode,
    _verify_manifest,
)
from experiments.foreign_neural_state_r15.isolation import build_capsule
from experiments.foreign_neural_state_r15.protocol import (
    capability_rows,
    heldout_capabilities,
)
from experiments.foreign_neural_state_r15.public_preflight import _public_capabilities
from experiments.foreign_neural_state_r15.recipient_worker import summarize
from experiments.foreign_neural_state_r15.verify import _probabilities
from experiments.native_isa_r11.core import transition_accuracy


def _exact_table_state(capability):
    width = OPERATORS * MODULUS
    readout = torch.zeros(OPERATORS, MODULUS, width)
    for operator in range(OPERATORS):
        for state in range(MODULUS):
            readout[operator, state, operator * MODULUS + state] = 1.0
    feature = torch.zeros(MODULUS, width)
    for operator in range(OPERATORS):
        for state in range(MODULUS):
            output = capability.apply(state, [operator])
            feature[output, operator * MODULUS + state] += 1.0
    state = {
        "format": "abi-r15a-frozen-weight-delta-affine-table-frontend/4",
        "input": "qwen_before_after_effective_output_weight_delta_only",
        "candidate_program_search": False,
        "behavioral_queries": 0,
        "oracle_calls_at_extraction": 0,
        "public_event_count": 256,
        "feature_elements": MODULUS * width,
        "epochs": 2000,
        "learning_rate": 0.05,
        "weight_decay": 0.001,
        "seed": 15015101,
        "output_decoding": "sum_24_atomic_scores_over_32_affine_codes_per_operator",
        "first_loss": 1.0,
        "final_loss": 0.0,
        "readout": readout,
        "bias": torch.zeros(OPERATORS, MODULUS, MODULUS),
    }
    return state, feature.flatten()


def test_affine_table_frontend_decodes_all_atomic_constraints():
    capability = _public_capabilities(1, seed=15150001, split="unit")[0]
    state, feature = _exact_table_state(capability)

    predicted = predict_labels(state, feature)

    assert torch.equal(predicted, labels_for_capability(capability))
    transition = transition_from_labels(predicted.tolist())
    rows = generate_rows(
        capability,
        split="unit_unseen",
        rows=512,
        depths=range(8, 16),
        seed=15150002,
    )
    assert transition_accuracy(transition, rows) == 1.0


def test_affine_table_frontend_fails_closed_on_search_contract_change():
    capability = _public_capabilities(1, seed=15150003, split="unit")[0]
    state, feature = _exact_table_state(capability)
    state["candidate_program_search"] = True

    with pytest.raises(R15FrontendError, match="contract changed"):
        predict_labels(state, feature)


def test_transition_rejects_non_permutation_affine_pair():
    with pytest.raises(R15FrontendError, match="non-permutation"):
        transition_from_labels([0, 2, 0, 1, 0, 1])


def test_frontend_spec_hash_is_stable_and_tensor_sensitive():
    capability = _public_capabilities(1, seed=15150004, split="unit")[0]
    state, _ = _exact_table_state(capability)
    original = frontend_spec(state)

    assert frontend_spec(state) == original
    state["readout"][0, 0, 0] += 0.25
    assert frontend_spec(state)["tensor_sha256"] != original["tensor_sha256"]


def test_frozen_frontend_round_trip_and_metadata_custody(tmp_path):
    capability = _public_capabilities(1, seed=15150006, split="unit")[0]
    state, feature = _exact_table_state(capability)
    spec = frontend_spec(state)
    path = tmp_path / "frontend.safetensors"
    save_file(
        {"readout": state["readout"], "bias": state["bias"]},
        str(path),
        metadata={
            "format": state["format"],
            "frontend_spec_sha256": spec["evidence_sha256"],
        },
    )

    loaded = load_frozen_frontend(path, spec)

    assert torch.equal(predict_labels(loaded, feature), labels_for_capability(capability))
    bad_spec = dict(spec)
    bad_spec["seed"] += 1
    with pytest.raises(R15FrontendError, match="hash changed"):
        load_frozen_frontend(path, bad_spec)


def test_stdlib_isolated_worker_decodes_only_the_anonymous_delta(tmp_path):
    capability = _public_capabilities(1, seed=15150007, split="unit")[0]
    state, feature = _exact_table_state(capability)
    spec = frontend_spec(state)
    frontend_path = tmp_path / "source_frontend.safetensors"
    delta_path = tmp_path / "source_delta.safetensors"
    save_file(
        {"readout": state["readout"], "bias": state["bias"]},
        str(frontend_path),
        metadata={
            "format": state["format"],
            "frontend_spec_sha256": spec["evidence_sha256"],
        },
    )
    save_file({"delta": feature}, str(delta_path))
    root = tmp_path / "capsule"
    build_capsule(
        worker=Path(__file__).parents[1]
        / "experiments/foreign_neural_state_r15/isolated_worker.py",
        frontend=frontend_path,
        frontend_spec=spec,
        delta=delta_path,
        capsule=root,
    )

    manifest = _verify_manifest(root)
    decoded = _decode(root)

    assert manifest["capability_reveals_included"] == 0
    assert decoded["labels"] == labels_for_capability(capability).tolist()
    assert decoded["behavioral_queries"] == 0
    (root / "delta.safetensors").write_bytes(
        (root / "delta.safetensors").read_bytes() + b"tamper"
    )
    with pytest.raises(IsolatedR15Error, match="layout invalid"):
        _decode(root)


def test_public_capabilities_are_deterministic_and_distinct():
    first = _public_capabilities(32, seed=15150005, split="unit")
    second = _public_capabilities(32, seed=15150005, split="unit")

    assert [item.operations for item in first] == [item.operations for item in second]
    assert len({item.operations for item in first}) == 32


def test_heldout_capability_rows_are_deterministic_distinct_and_disjoint():
    secret = bytes(range(32))
    commitment = hashlib.sha256(secret).hexdigest()
    capabilities = heldout_capabilities(
        secret.hex(), expected_commitment=commitment, count=8
    )
    config = {
        "data": {
            "atomic_seed": 15151001,
            "black_box_queries_per_capability": 16,
            "black_box_query_depths": [4, 5],
            "query_seed": 15151002,
            "evaluation_rows_per_capability": 64,
            "evaluation_depths": [13, 14],
            "evaluation_seed": 15151003,
            "counterfactual_pairs_per_capability": 8,
            "counterfactual_depth": 14,
            "counterfactual_seed": 15151004,
            "source_behavior_rows_per_capability": 32,
            "recipient_rows_per_capability": 32,
        }
    }

    generated = capability_rows(config, capabilities)

    assert len({item.operations for item in capabilities}) == 8
    assert all(len(item["atomic"]) == 24 for item in generated)
    for item in generated:
        keys = [
            row["program_key"]
            for split in ("atomic", "queries", "evaluation", "counterfactual")
            for row in item[split]
        ]
        assert len(keys) == len(set(keys))


def test_recipient_summary_treats_noncanonical_prediction_as_incorrect():
    evaluation = [[{"row_id": "row-1", "answer": 3}]]
    rows = []
    for condition in (
        "BASE",
        "AFTER",
        "REMOVED",
        "BACKEND_REMOVED",
        "CODEC_REMOVED",
    ):
        rows.append(
            {
                "capability_id": "capability-1",
                "condition": condition,
                "row_id": "row-1",
                "canonical_prediction": 3 if condition == "AFTER" else None,
                "prediction_token_id": 17 if condition == "AFTER" else 99,
            }
        )

    result = summarize(rows, evaluation)

    assert result["accuracy"]["capability-1/BASE"] == 0.0
    assert result["accuracy"]["capability-1/AFTER"] == 1.0
    assert result["removal_conditions_equal_base"] is True


def test_zero_intervention_is_a_finite_state_but_not_a_probability_simplex():
    zero = [0.0] * 8

    assert _probabilities(zero, normalized=False) == zero
    with pytest.raises(RuntimeError, match="probabilities invalid"):
        _probabilities(zero)
