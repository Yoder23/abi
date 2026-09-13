from __future__ import annotations

from pathlib import Path

from experiments.canonical_layercake_import_r25.protocol import (
    BUILD_INITIALIZATIONS,
    HOST_INITIALIZATIONS,
    NAMESPACES,
    prepared_rows,
    source_splits,
)
from experiments.factual_semantic_r16.package import load_package
from experiments.generative_transfer_r21 import run as base_run


def test_r25_frozen_source_has_two_exact_eight_record_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    package_paths = sorted(
        (root / "results/factual_semantic_r16/heldout_v2/extraction/packages").glob(
            "*.abipkg"
        )
    )
    packages = [load_package(path) for path in package_paths]
    assert len(packages) == 2
    assert {package["namespace"] for package in packages} == set(NAMESPACES)
    assert all(len(package["facts"]) == 8 for package in packages)


def test_r25_canonical_decoder_preserves_frozen_evaluation_answers() -> None:
    root = Path(__file__).resolve().parents[1]
    base_run._layercake(root)
    from layercake.models.canonical_factual import CanonicalFactualDecoder

    package_paths = (
        root / "results/factual_semantic_r16/heldout_v2/extraction/packages"
    ).glob("*.abipkg")
    models = {
        package["namespace"]: CanonicalFactualDecoder(
            package["namespace"], package["facts"]
        )
        for package in (load_package(path) for path in package_paths)
    }
    _, raw_evaluation = source_splits(
        root / "results/factual_semantic_r16/heldout_v2/source_observations.jsonl"
    )
    for row in prepared_rows(raw_evaluation):
        model = models[row["namespace"]]
        state = model.prefill_bytes(row["prompt"])
        while not state.complete:
            model.decode_step(state)
        output = model.tokenizer.decode_actions(
            state.generated_actions, state.source_lexemes
        ).decode("utf-8")
        assert output == row["response"]


def test_r25_initialization_sets_are_distinct() -> None:
    assert len(set(BUILD_INITIALIZATIONS)) == 3
    assert len(set(HOST_INITIALIZATIONS)) == 3
    assert set(BUILD_INITIALIZATIONS).isdisjoint(HOST_INITIALIZATIONS)
