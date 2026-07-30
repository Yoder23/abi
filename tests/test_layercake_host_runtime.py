from __future__ import annotations

import json

import numpy as np
import pytest

from abi.layercake_host import _symbolic_surface_output
from abi.layercake_host_runtime import (
    LayerCakeHostRuntimeError,
    _bootstrap_interval,
    _canonical_sha,
    _quality,
    _quantize_embedding_rows,
    _runtime_candidate_manifest_sha,
    _select_token,
    _summarize_native_semantics,
)
from abi.symbolic_runtime import symbolic_surface_output


def test_native_evidence_hash_is_canonical_and_order_independent():
    assert _canonical_sha({"b": 2, "a": 1}) == _canonical_sha(
        {"a": 1, "b": 2}
    )


def test_native_runtime_error_is_fail_closed():
    with pytest.raises(LayerCakeHostRuntimeError):
        raise LayerCakeHostRuntimeError("stale graph")


def test_native_candidate_identity_accepts_host_or_standalone_manifest():
    assert _runtime_candidate_manifest_sha(
        {"host": {"deployment_manifest_sha256": "a" * 64}}
    ) == "a" * 64
    assert _runtime_candidate_manifest_sha(
        {"host": {"manifest_sha256": "b" * 64}}
    ) == "b" * 64
    with pytest.raises(LayerCakeHostRuntimeError):
        _runtime_candidate_manifest_sha({"host": {}})


def test_lightweight_v5_rewriting_matches_training_runtime():
    contract = {
        "handlers": ["concise_delayed_project_review"],
        "grammar": {},
    }
    prompt = (
        "Rewrite as one concise sentence while preserving every fact: "
        "Project Z455UMA encountered a delay. Its review is now scheduled "
        "for Wednesday at 11:00."
    )
    assert symbolic_surface_output(contract, prompt=prompt, route=8) == (
        _symbolic_surface_output(contract, prompt=prompt, route=8)
    )


def test_native_quality_detects_repetition_without_token_heuristics():
    fluent = _quality(
        b"Clear evidence supports a careful decision with useful context."
    )
    collapsed = _quality(b"loop loop loop loop loop loop loop loop")
    assert fluent["valid_utf8"] == 1.0
    assert collapsed["repetition_rate"] > fluent["repetition_rate"]
    assert collapsed["word_diversity"] < fluent["word_diversity"]


def test_paired_prompt_bootstrap_is_deterministic():
    first = _bootstrap_interval([2.1, 2.2, 2.3])
    second = _bootstrap_interval([2.1, 2.2, 2.3])
    assert first == second
    assert first[0] >= 2.0


def test_native_decoding_blocks_only_repeated_ngram_continuations():
    logits = np.zeros((1, 8), dtype=np.float32)
    logits[0, 3] = 10.0
    logits[0, 4] = 9.0
    generated = [1, 2, 3, 1, 2]
    assert _select_token(logits, generated) == 3
    assert _select_token(
        logits, generated, no_repeat_ngram_size=3
    ) == 4


def test_indexed_greedy_selection_matches_copy_reference_and_restores_logits():
    generator = np.random.default_rng(9824)
    for sparse in (False, True):
        for _ in range(200):
            width = 257 if sparse else 300
            logits = generator.normal(size=(1, width)).astype(np.float32)
            before = logits.copy()
            output_ids = (
                np.sort(
                    generator.choice(
                        500, size=width, replace=False
                    ).astype(np.int64)
                )
                if sparse
                else None
            )
            local_map = (
                {
                    int(token): (index,)
                    for index, token in enumerate(output_ids)
                }
                if sparse
                else None
            )
            token_space = (
                output_ids.tolist()
                if output_ids is not None
                else list(range(width))
            )
            generated = [
                int(value)
                for value in generator.choice(
                    token_space, size=20, replace=True
                )
            ]
            blocked = {
                int(value)
                for value in generator.choice(
                    token_space, size=5, replace=False
                )
            }
            copied = logits[0].copy()
            repetition_indices = []
            for token in set(generated):
                local = (
                    token
                    if local_map is None
                    else local_map.get(token)
                )
                if local is None:
                    continue
                repetition_indices.extend(
                    [int(local)]
                    if isinstance(local, int)
                    else [int(value) for value in local]
                )
            selected = np.asarray(
                repetition_indices, dtype=np.int64
            )
            selected_values = copied[selected]
            copied[selected] = np.where(
                selected_values > 0,
                selected_values / 1.15,
                selected_values * 1.15,
            )
            blocked_indices = [
                token
                if local_map is None
                else local_map[token][0]
                for token in blocked
            ]
            copied[blocked_indices] = -np.inf
            expected_local = int(copied.argmax())
            expected = (
                expected_local
                if output_ids is None
                else int(output_ids[expected_local])
            )
            actual = _select_token(
                logits,
                generated,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                output_token_ids=output_ids,
                output_token_local_index=local_map,
                blocked_token_ids=blocked,
            )
            assert actual == expected
            assert np.array_equal(logits, before)


def test_embedding_quantization_uses_independent_token_row_scales():
    embedding = np.asarray(
        [[0.001, -0.002, 0.003], [100.0, -80.0, 40.0]],
        dtype=np.float32,
    )
    quantized, scales = _quantize_embedding_rows(embedding)
    restored = quantized.astype(np.float32) * scales[:, None]
    assert quantized.dtype == np.int8
    assert scales.shape == (2,)
    assert scales[0] < scales[1]
    assert np.all(np.abs(restored - embedding) <= scales[:, None])


def test_native_semantic_summary_requires_depth_quality_and_routing():
    observations = []
    capabilities = (
        "abstention",
        "cake_output_realization",
        "clarification",
        "coherence",
        "conversation",
        "domain_independent_reasoning",
        "email_drafting",
        "format_control",
        "grammar",
        "instruction_following",
        "prompt_grounding",
        "rewriting",
        "summarization",
        "tone_control",
    )
    for capability in capabilities:
        observations.extend(
            {
                "capability": capability,
                "source_passed": True,
                "layercake_passed": True,
                "route_correct": True,
            }
            for _ in range(100)
        )
    metrics, complete, passed = _summarize_native_semantics(
        observations
    )
    assert len(metrics) == 14
    assert complete is True
    assert passed is True
    observations[-1]["route_correct"] = False
    assert _summarize_native_semantics(observations)[2] is False


def test_lightweight_symbolic_runtime_matches_training_implementation():
    contract = {
        "handlers": [
            "exact_supplied_text",
            "conservative_grammar_inflection",
            "exact_json_item_count",
        ],
        "grammar": {
            "instruction_prefix": (
                "Correct the grammar and output only the corrected sentence: "
            ),
            "verb_inflections": {"walk": "walks"},
        },
    }
    probes = (
        (
            "Correct the grammar and output only the corrected sentence: "
            "She walk home.",
            0,
        ),
        (
            "Return only one JSON object, with no Markdown, using "
            "`item`='runtime-probe' and `count`=11.",
            4,
        ),
        ("No handler should accept this.", 7),
        ("Reply with exactly GROUND-941-ZÜRICH and nothing else.", 4),
    )
    for prompt, route in probes:
        assert symbolic_surface_output(
            contract, prompt=prompt, route=route
        ) == _symbolic_surface_output(
            contract, prompt=prompt, route=route
        )


def test_exact_supplied_text_handler_is_strict_and_teacher_free():
    contract = {
        "handlers": ["exact_supplied_text"],
        "source_teacher_text_retained": False,
    }
    accepted = "Reply with exactly GROUND-941-ZÜRICH and nothing else."
    assert symbolic_surface_output(
        contract, prompt=accepted, route=4
    ) == "GROUND-941-ZÜRICH"
    rejected = [
        ("reply with exactly X and nothing else.", 4),
        ("prefix Reply with exactly X and nothing else.", 4),
        ("Reply with exactly  and nothing else.", 4),
        ("Reply with exactly X and nothing else. suffix", 4),
        ("Reply with exactly X\nY and nothing else.", 4),
        ("Reply with exactly X\rY and nothing else.", 4),
        (
            "Reply with exactly "
            + ("X" * 257)
            + " and nothing else.",
            4,
        ),
        (accepted, 3),
        (accepted, 7),
    ]
    for prompt, route in rejected:
        assert symbolic_surface_output(
            contract, prompt=prompt, route=route
        ) is None
