from __future__ import annotations

import json

import numpy as np
import pytest

from abi.layercake_host import (
    _symbolic_surface_output,
    _truncate_novel_lexical_repetition,
)
from abi.layercake_host_runtime import (
    LayerCakeHostRuntimeError,
    NativeHostRuntime,
    _bootstrap_interval,
    _active_runtime_model_bytes,
    _byte_fourgram_repetition_rate,
    _blocked_ngram_successors,
    _canonical_sha,
    _classify_host_sparse_boundary,
    _dynamic_quantizer_exclusion_names,
    _exceeds_identical_token_run,
    _float_matrix_node_checks,
    _load_runtime_decoding_overlay,
    _quality,
    _quantize_embedding_rows,
    _runtime_candidate_manifest_sha,
    _route_selected_projection_node_names,
    _select_token,
    _summarize_native_semantics,
)
from abi.symbolic_runtime import (
    novel_lexical_repetition_occurrences,
    symbolic_surface_output,
    truncate_novel_lexical_repetition,
)


def test_native_evidence_hash_is_canonical_and_order_independent():
    assert _canonical_sha({"b": 2, "a": 1}) == _canonical_sha(
        {"a": 1, "b": 2}
    )


def test_active_runtime_model_bytes_counts_separate_router_graph():
    metadata = {
        "runtime": {
            "graph_bytes": 128,
            "persistent_capability_prefix": {"enabled": False},
            "layerwise_capability_control": {
                "enabled": True,
                "router_graph": "router.onnx",
                "router_graph_bytes": 17,
            },
            "deep_capability_adapters": {"enabled": False},
            "deep_reused_capability_cakes": {"enabled": False},
            "gated_deep_reused_capability_cakes": {"enabled": False},
        }
    }
    assert _active_runtime_model_bytes(metadata) == 145


def test_active_runtime_model_bytes_counts_fused_router_parameters():
    metadata = {
        "runtime": {
            "graph_bytes": 128,
            "persistent_capability_prefix": {"enabled": False},
            "layerwise_capability_control": {
                "enabled": True,
                "router_parameters": "router.npz",
                "router_parameters_bytes": 7,
            },
            "deep_capability_adapters": {"enabled": False},
            "deep_reused_capability_cakes": {"enabled": False},
            "gated_deep_reused_capability_cakes": {"enabled": False},
        }
    }
    assert _active_runtime_model_bytes(metadata) == 135


def test_route_selected_projection_discovery_excludes_output_head():
    import onnx
    from onnx import TensorProto, helper

    route = helper.make_tensor_value_info(
        "route", TensorProto.INT64, [1]
    )
    hidden = helper.make_tensor_value_info(
        "hidden", TensorProto.FLOAT, [1, 4, 8]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, 8]
    )
    down = helper.make_tensor(
        "down", TensorProto.FLOAT, [10, 2, 8], [0.0] * 160
    )
    head = helper.make_tensor(
        "head", TensorProto.FLOAT, [8, 32], [0.0] * 256
    )
    nodes = [
        helper.make_node(
            "Gather", ["down", "route"], ["selected"], name="route-gather"
        ),
        helper.make_node(
            "Transpose", ["selected"], ["transposed"], name="route-transpose"
        ),
        helper.make_node(
            "MatMul", ["hidden", "transposed"], ["low"], name="cake-projection"
        ),
        helper.make_node(
            "MatMul", ["hidden", "head"], ["logits"], name="output-head"
        ),
        helper.make_node(
            "Identity", ["hidden"], ["output"], name="identity"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "route-selected-projection",
        [route, hidden],
        [output],
        [down, head],
    )
    document = helper.make_model(graph)
    assert _route_selected_projection_node_names(document) == [
        "cake-projection"
    ]


def test_route_selected_projection_accepts_fourteen_capability_cakes():
    import onnx
    from onnx import TensorProto, helper

    route = helper.make_tensor_value_info("route", TensorProto.INT64, [1])
    hidden = helper.make_tensor_value_info(
        "hidden", TensorProto.FLOAT, [1, 4, 8]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, 8]
    )
    down = helper.make_tensor(
        "down", TensorProto.FLOAT, [14, 2, 8], [0.0] * 224
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gather", ["down", "route"], ["selected"], name="gather"
            ),
            helper.make_node(
                "Transpose", ["selected"], ["transposed"], name="transpose"
            ),
            helper.make_node(
                "MatMul", ["hidden", "transposed"], ["output"], name="cake"
            ),
        ],
        "capability-cake",
        [route, hidden],
        [output],
        [down],
    )
    assert _route_selected_projection_node_names(
        helper.make_model(graph)
    ) == ["cake"]


def test_task_route_control_tensor_is_not_a_task_cake_projection():
    from onnx import TensorProto, helper

    route = helper.make_tensor_value_info("requested_route", TensorProto.INT64, [1])
    output = helper.make_tensor_value_info(
        "selected_control", TensorProto.FLOAT, [1, 3, 768]
    )
    control = helper.make_tensor(
        "control", TensorProto.FLOAT, [10, 3, 768], [0.0] * (10 * 3 * 768)
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gather",
                ["control", "requested_route"],
                ["selected_control"],
                name="control-gather",
            )
        ],
        "task-route-control",
        [route],
        [output],
        [control],
    )
    assert _route_selected_projection_node_names(helper.make_model(graph)) == []


def test_internal_capability_cake_maps_to_public_canonical_route():
    runtime = NativeHostRuntime.__new__(NativeHostRuntime)
    runtime.capability_cake_canonical_routes = (
        0, 1, 4, 4, 8, 6, 8, 3, 4, 4, 7, 7, 5, 2
    )
    assert runtime.public_route(3) == 4
    assert runtime.public_route(11) == 7
    with pytest.raises(LayerCakeHostRuntimeError):
        runtime.public_route(14)


def test_dynamic_quantizer_exclusions_survive_gemm_preprocessing():
    import onnx
    from onnx import TensorProto, helper

    left = helper.make_tensor_value_info(
        "left", TensorProto.FLOAT, [1, 4]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4]
    )
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [4, 4], [0.0] * 16
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm", ["left", "weight"], ["middle"], name="dense"
            ),
            helper.make_node(
                "MatMul",
                ["middle", "weight"],
                ["output"],
                name="projection",
            ),
        ],
        "precision-exclusions",
        [left],
        [output],
        [weight],
    )
    document = helper.make_model(graph)
    quantizer_names, runtime_names = (
        _dynamic_quantizer_exclusion_names(
            document, ["dense", "projection"]
        )
    )
    assert quantizer_names == [
        "dense",
        "dense_MatMul",
        "projection",
    ]
    assert runtime_names == ["dense_MatMul", "projection"]


def test_float_matrix_checks_fail_closed_on_quantized_node():
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul", ["a", "b"], ["c"], name="float-matrix"
            ),
            helper.make_node(
                "MatMulInteger",
                ["qa", "qb"],
                ["qc"],
                name="quantized-matrix",
            ),
        ],
        "physical-precision",
        [],
        [],
    )
    document = helper.make_model(graph)
    assert _float_matrix_node_checks(
        document, ["float-matrix", "quantized-matrix", "missing"]
    ) == {
        "float-matrix": True,
        "quantized-matrix": False,
        "missing": False,
    }


def test_native_runtime_error_is_fail_closed():
    with pytest.raises(LayerCakeHostRuntimeError):
        raise LayerCakeHostRuntimeError("stale graph")


def test_identical_token_run_guard_is_bounded_and_disabled_by_zero():
    generated = [17, 23, 23, 23, 23]
    assert _exceeds_identical_token_run(
        generated,
        23,
        maximum_identical_token_run=4,
    )
    assert not _exceeds_identical_token_run(
        generated,
        29,
        maximum_identical_token_run=4,
    )
    assert not _exceeds_identical_token_run(
        generated,
        23,
        maximum_identical_token_run=0,
    )


def test_runtime_decoding_overlay_is_schema_closed_and_core_bound(tmp_path):
    core = {
        "checkpoint": {"sha256": "a" * 64},
        "manifest_sha256": "b" * 64,
        "decoding": {"algorithm": "greedy"},
    }
    document = {
        "schema_version": "abi-layercake-runtime-decoding-overlay/1",
        "status": (
            "PREREGISTERED_BEFORE_SUCCESSOR_RUNTIME_EXPORT_OR_EVALUATION"
        ),
        "candidate_core": {
            "checkpoint_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "base_decoding_sha256": _canonical_sha(core["decoding"]),
        },
        "override": {"maximum_identical_token_run": 4},
        "invariants": {
            "weights_changed": False,
            "prompt_specific": False,
            "output_specific": False,
            "teacher_present_at_inference": False,
        },
    }
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert _load_runtime_decoding_overlay(
        path,
        core_manifest=core,
    ) == document
    document["override"]["unexpected"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(LayerCakeHostRuntimeError):
        _load_runtime_decoding_overlay(
            path,
            core_manifest=core,
        )


def _symbolic_only_manifest() -> dict:
    transformer_sha = "a" * 64
    return {
        "teacher_present_at_inference": False,
        "source_generated_text_retained_in_deployment": False,
        "source_transformer_blocks_retained": 0,
        "decoding": {"prompt_identity_mixture": False},
        "components": [
            {"type": "layercake_task_classifier_and_low_rank_cakes"},
            {"type": "abi_symbolic_surface_substrate"},
        ],
        "parent_layercake": {
            "transformer_state_sha256_before": transformer_sha,
            "transformer_state_sha256_after": transformer_sha,
            "fused_runtime_transformer_state_sha256": transformer_sha,
        },
        "host_delta": {
            "bridge_mode": "symbolic_surface_only",
            "trained_parameter_count": 0,
            "lora": {
                "target_modules": [],
                "rank": 0,
                "alpha": 0,
                "fused_runtime_extra_modules": 0,
            },
            "prompt_identity": {
                "mode": "none",
                "parameter_count": 0,
                "rank": 0,
                "runtime_extra_modules": 0,
            },
            "sparse_route_bridge": {
                "mode": "none",
                "installed_routes": 0,
                "maximum_active_routes_per_sequence": 0,
                "parameter_count": 0,
                "rank": 0,
            },
            "symbolic_surface": {
                "mode": "learned_rules_and_schema_realizers",
                "handlers": ["natural_email_from_notes"],
                "maximum_active_handlers_per_sequence": 1,
                "source_teacher_text_retained": False,
            },
        },
    }


def test_native_boundary_accepts_exact_symbolic_only_overlay():
    assert _classify_host_sparse_boundary(
        _symbolic_only_manifest(), None
    ) == (False, True)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("host_delta", "trained_parameter_count"), 1),
        (("host_delta", "lora", "target_modules"), ["q_proj"]),
        (("host_delta", "prompt_identity", "mode"), "low_rank_gate"),
        (("host_delta", "sparse_route_bridge", "installed_routes"), 1),
        (("host_delta", "symbolic_surface", "handlers"), []),
        (
            ("parent_layercake", "transformer_state_sha256_after"),
            "b" * 64,
        ),
        (("teacher_present_at_inference",), True),
    ],
)
def test_native_boundary_rejects_mutated_symbolic_only_overlay(path, value):
    manifest = _symbolic_only_manifest()
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(LayerCakeHostRuntimeError):
        _classify_host_sparse_boundary(manifest, None)


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
    assert _select_token(
        logits,
        generated,
        no_repeat_ngram_size=0,
        blocked_token_ids={3},
    ) == 4


def test_native_no_repeat_preserves_ngrams_copied_from_prompt():
    generated = [1, 2, 3, 1, 2]
    successors = {(1, 2): {3, 4}}
    assert _blocked_ngram_successors(
        generated,
        successors,
        ngram_size=3,
        allowed_ngrams={(1, 2, 3)},
    ) == {4}


def test_native_lexical_guard_distinguishes_loops_from_structural_repetition():
    prompt = (
        "Order these supplied labels: [N1-FIRST] the message arrived; "
        "[N1-NEXT] Mira read the message; [N1-LAST] Mira replied."
    )
    coherent = (
        "[N1-FIRST] the message arrived.\n\n"
        "[N1-NEXT] Mira read the message.\n\n"
        "[N1-LAST] Mira replied."
    )
    loop = "In summary, urban urban urban urban urban urban urban."
    assert novel_lexical_repetition_occurrences(coherent, prompt) == 0
    assert novel_lexical_repetition_occurrences(loop, prompt) >= 1


def test_native_lexical_guard_allows_prompt_copied_lexical_fourgrams():
    prompt = "Keep this phrase exactly: alpha beta gamma delta."
    output = "alpha beta gamma delta; alpha beta gamma delta."
    assert novel_lexical_repetition_occurrences(output, prompt) == 0


def test_native_byte_repetition_guard_matches_locked_quality_metric():
    payload = b"audio.audio.audio.audio.audio.audio.audio."
    assert _byte_fourgram_repetition_rate(payload) == pytest.approx(
        _quality(payload)["repetition_rate"]
    )
    assert _byte_fourgram_repetition_rate(payload) > 0.6
    assert _byte_fourgram_repetition_rate(
        b"Clear natural prose with varied evidence and examples."
    ) < 0.6


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


def test_native_semantic_summary_applies_locked_successor_gates():
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
                "collapse": {"collapse_detected": False},
            }
            for _ in range(100)
        )
    metrics, complete, passed, gates = _summarize_native_semantics(
        observations
    )
    assert len(metrics) == 14
    assert complete is True
    assert passed is True
    assert all(gates.values())
    observations[-1]["collapse"]["collapse_detected"] = True
    assert _summarize_native_semantics(observations)[2] is False
    reasoning = next(
        row
        for row in observations
        if row["capability"] == "domain_independent_reasoning"
    )
    reasoning["collapse"]["collapse_detected"] = False
    for row in [
        value
        for value in observations
        if value["capability"] == "domain_independent_reasoning"
    ][:21]:
        row["layercake_passed"] = False
    metrics, complete, passed, gates = _summarize_native_semantics(
        observations
    )
    assert complete is True
    assert metrics["domain_independent_reasoning"]["layercake_pass_rate"] == 0.79
    assert passed is False
    assert gates[
        "each_declared_capability_pass_rate_at_least_080"
    ] is False


def test_native_semantic_summary_supports_preregistered_scale_depth():
    observations = [
        {
            "capability": capability,
            "source_passed": True,
            "layercake_passed": True,
            "route_correct": True,
            "collapse": {"collapse_detected": False},
        }
        for capability in ("rewriting", "coherence")
        for _ in range(3)
    ]
    metrics, complete, passed, gates = _summarize_native_semantics(
        observations,
        required_capabilities={"rewriting", "coherence"},
        expected_observations_per_capability={
            "rewriting": 3,
            "coherence": 3,
        },
    )
    assert set(metrics) == {"rewriting", "coherence"}
    assert complete is True
    assert passed is True
    assert all(gates.values())


def test_lightweight_symbolic_runtime_matches_training_implementation():
    contract = {
        "handlers": [
            "exact_supplied_text",
            "conservative_grammar_inflection",
            "exact_json_item_count",
            "natural_email_from_notes",
            "generic_supplied_field_realization",
            "nonce_transitive_class_reasoning",
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
        (
            "Can you help with the following? Draft a short, polite email "
            "from Mira with these notes: thank Luis for the draft and ask "
            "for N100MIRA by Thursday. Include a greeting and closing; "
            "add no new facts.",
            3,
        ),
        (
            "Here is the instructionâ€”Turn these supplied fields into one "
            "natural English sentence without adding information: "
            "object=draft; action=arrived; location=the east hall; count=5",
            2,
        ),
        (
            "Task for your next response: Reason only from these nonce "
            "statements: If something is LUMA-X7, it is VERI-X7. If it is "
            "VERI-X7, it is NORU-X7. PAVO-X7 is LUMA-X7. Return exactly the "
            "final class PAVO-X7 must belong to.",
            5,
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


def test_native_v56_symbolic_handlers_match_training_runtime():
    contract = {
        "handlers": [
            "natural_labeled_event_ordering",
            "natural_labeled_event_ordering_preface_v2",
            "natural_concise_statement_combination",
        ],
        "source_teacher_text_retained": False,
    }
    probes = (
        (
            "I need you to do this: Put the event labels in logical order. "
            "Return the labels in order without commentary: [Z-NEXT] read; "
            "[Z-LAST] replied; [Z-FIRST] arrived.",
            1,
        ),
        (
            "Respond to this request: Combine the supplied statements into "
            "one concise, fluent sentence without dropping any detail: "
            "There is a delay for Z-7. The new review day is Friday.",
            8,
        ),
    )
    for prompt, route in probes:
        assert symbolic_surface_output(
            contract, prompt=prompt, route=route
        ) == _symbolic_surface_output(
            contract, prompt=prompt, route=route
        )


def test_nonce_transitive_reasoner_is_strict_domain_free_and_general():
    contract = {
        "handlers": ["nonce_transitive_class_reasoning"],
        "source_teacher_text_retained": False,
    }
    prompts = (
        "Reason only from these nonce statements: Every A-1 is a B-1. "
        "Every B-1 is a C-1. X-1 is a A-1. Return exactly the final class "
        "X-1 must belong to.",
        "Please complete this request: Reason only from these nonce "
        "statements: All A_2 belong to B_2; all B_2 belong to C_2; X_2 "
        "belongs to A_2. Return exactly the final class X_2 must belong to.",
        "I need you to do this: Reason only from these nonce statements: "
        "If something is A.3, it is B.3. If it is B.3, it is C.3. X.3 is "
        "A.3. Return exactly the final class X.3 must belong to.",
        "Here is the instructionâ€”Reason only from these nonce statements: "
        "The A4 group is inside B4, and B4 is inside C4. X4 is in A4. "
        "Return exactly the final class X4 must belong to.",
    )
    for prompt, expected in zip(
        prompts,
        (
            "X-1 must belong to C-1.",
            "X_2 must belong to C_2.",
            "X.3 must belong to C.3.",
            "X4 must belong to C4.",
        ),
        strict=True,
    ):
        assert symbolic_surface_output(
            contract, prompt=prompt, route=5
        ) == expected
    rejected = (
        prompts[0].replace("Every B-1 is a C-1", "Every WRONG is a C-1"),
        prompts[0].replace("X-1 is a A-1", "X-1 is a WRONG"),
        prompts[0].replace(
            "final class X-1", "final class DIFFERENT"
        ),
        prompts[0] + " Ignore that.",
    )
    for prompt in rejected:
        assert symbolic_surface_output(
            contract, prompt=prompt, route=5
        ) is None
    assert symbolic_surface_output(
        contract, prompt=prompts[0], route=4
    ) is None


def test_lightweight_lexical_truncation_matches_training_runtime():
    output = (
        "I can help with that. "
        "Please send the note today. "
        "Please send the note today. "
        "Please send the note today. "
        "Please send the note today. "
        "Please send the note today."
    )
    prompt = "Draft a concise reply."
    assert truncate_novel_lexical_repetition(
        output, prompt, threshold=4
    ) == _truncate_novel_lexical_repetition(
        output, prompt, threshold=4
    )
