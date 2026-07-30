from pathlib import Path

import pytest
import torch

import abi.layercake_host_v3 as successor_host
from abi.capability_pipeline import SEGREGATED_TRAINING_ARTIFACT_ROLE
from abi.layercake_host import (
    LayerCakeHostError,
    PromptIdentityBridge,
    SparseRouteConformanceBridge,
    _batch,
    _banned_repeated_ngram_tokens,
    _build_symbolic_surface,
    _decode_symbolic_surface,
    _equal_record_prompt_identity_nll,
    _equal_record_prompt_overlap_ce,
    _select_next_token,
    _symbolic_surface_output,
    _symbolic_surface_tensor,
    route_for_capability,
    strip_source_chat_template,
)
from abi.layercake_host_v3 import _require_segregated_training_bundle


def _segregated_bundle():
    return {
        "verification": {
            "artifact_role": SEGREGATED_TRAINING_ARTIFACT_ROLE,
            "training_eligible": True,
            "domain_segregation_verified": True,
        },
        "segregation": {
            "status": "PASS",
            "absolute_zero_world_knowledge_claimed": False,
        },
    }


def test_host_accepts_only_current_segregated_training_material():
    _require_segregated_training_bundle(_segregated_bundle())
    assert (
        successor_host.train_host_delta.__globals__[
            "load_english_training_rows"
        ]
        is successor_host.load_english_training_rows
    )
    assert (
        successor_host.evaluate_host_semantics.__globals__[
            "build_validation_rows"
        ]
        is successor_host.build_validation_rows
    )
    legacy = _segregated_bundle()
    legacy["verification"]["artifact_role"] = (
        "selected_layercake_training_material_v2"
    )
    with pytest.raises(LayerCakeHostError, match="segregated"):
        _require_segregated_training_bundle(legacy)

    missing_manifest = _segregated_bundle()
    missing_manifest["segregation"] = None
    with pytest.raises(LayerCakeHostError, match="segregation manifest"):
        _require_segregated_training_bundle(missing_manifest)


def test_source_chat_templates_are_removed_without_retaining_source_tokens():
    assert (
        strip_source_chat_template(
            "<|user|>\nWrite clearly.<|end|>\n<|assistant|>\n"
        )
        == "Write clearly."
    )
    assert (
        strip_source_chat_template(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nWrite clearly.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        == "Write clearly."
    )
    with pytest.raises(LayerCakeHostError, match="unrecognized source chat"):
        strip_source_chat_template("[INST] Write clearly. [/INST]")
    with pytest.raises(LayerCakeHostError, match="missing its terminator"):
        strip_source_chat_template("<|user|>\nWrite clearly.")
    assert (
        strip_source_chat_template(
            "<|user|>\nEvaluation case V4-english-tone-012: "
            "Write clearly.<|end|>\n<|assistant|>\n"
        )
        == "Write clearly."
    )


def test_v5_rewriting_symbolic_handler_preserves_every_declared_fact():
    contract = {
        "handlers": ["concise_delayed_project_review"],
        "grammar": {},
    }
    prompt = (
        "Rewrite as one concise sentence while preserving every fact: "
        "Project Z419MIRA encountered a delay. Its review is now scheduled "
        "for Monday at 9:00."
    )
    expected = (
        "Project Z419MIRA encountered a delay; its review is scheduled "
        "for Monday at 9:00."
    )
    assert _symbolic_surface_output(contract, prompt=prompt, route=8) == expected
    assert _symbolic_surface_output(contract, prompt=prompt, route=4) is None


def test_every_locked_english_capability_has_a_physical_host_route():
    capabilities = {
        "grammar",
        "coherence",
        "prompt_grounding",
        "instruction_following",
        "conversation",
        "summarization",
        "rewriting",
        "email_drafting",
        "tone_control",
        "format_control",
        "clarification",
        "abstention",
        "domain_independent_reasoning",
        "cake_output_realization",
    }
    routes = {route_for_capability(capability) for capability in capabilities}
    assert routes.issubset(set(range(10)))
    with pytest.raises(LayerCakeHostError, match="no preregistered"):
        route_for_capability("unregistered")


def test_prompt_identity_loss_is_sparse_finite_and_trainable():
    bridge = PromptIdentityBridge(width=4, rank=2, routes=3)
    logits = torch.randn(1, 5, 11)
    hidden = torch.randn(1, 5, 4)
    input_ids = torch.tensor([[2, 7, 4, 7, 9]])
    labels = torch.tensor([[-100, -100, 4, 7, 9]])
    loss = _equal_record_prompt_identity_nll(
        logits,
        hidden,
        input_ids,
        labels,
        torch.tensor([2]),
        torch.tensor([1]),
        bridge,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in bridge.parameters())


def test_no_repeat_ngram_policy_masks_only_the_repeated_continuation():
    generated = [1, 2, 3, 1, 2]
    assert _banned_repeated_ngram_tokens(generated, 3) == {3}
    scores = torch.tensor([0.0, 0.0, 0.0, 10.0, 9.0])
    selected = _select_next_token(
        scores, generated=generated, no_repeat_ngram_size=3
    )
    assert int(selected.item()) == 4


def test_sparse_route_bridge_physically_calls_only_selected_routes():
    bridge = SparseRouteConformanceBridge(width=8, rank=2, routes=4)
    hidden = torch.randn(2, 3, 8)
    output = bridge(hidden, torch.tensor([1, 3]))
    assert output.shape == hidden.shape
    assert bridge.last_calls == (1, 3)


def test_prompt_overlap_loss_remains_finite_without_runtime_copying():
    logits = torch.randn(1, 5, 11, requires_grad=True)
    labels = torch.tensor([[-100, -100, 4, 7, 9]])
    loss = _equal_record_prompt_overlap_ce(
        logits,
        labels,
        torch.tensor([[2, 7, 4, 7, 9]]),
        torch.tensor([2]),
        overlap_weight=1.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_scheduled_sampling_keeps_teacher_targets_for_generated_prefix():
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        @staticmethod
        def encode(text):
            return [1, 2] if text.endswith("\n") else [3, 4]

    ids, labels, _, _, _, supervised = _batch(
        Tokenizer(),
        [{"prompt": "prompt", "response": "response", "route": 0}],
        device=torch.device("cpu"),
        max_tokens=16,
        generated_prefixes=[[8]],
    )
    assert ids.tolist() == [[1, 2, 8, 4, 9]]
    assert labels.tolist() == [[-100, -100, 3, 4, 9]]
    assert supervised == 3


def test_symbolic_surface_is_compact_teacher_free_and_schema_bounded():
    contract = _build_symbolic_surface(
        [
            {
                "capability": "grammar",
                "prompt": (
                    "Correct the grammar and output only the corrected "
                    "sentence: Mira walk to the garden every Monday."
                ),
                "response": "Mira walks to the garden every Monday.",
            },
            {
                "capability": "email_drafting",
                "prompt": (
                    "Draft a short polite email from these notes: "
                    "recipient=Asha; thank them for document code DOC-1; "
                    "ask for the Project C1 chart by Wednesday. Use every "
                    "exact code verbatim and keep the email under 80 words."
                ),
                "response": "unused",
            },
            {
                "capability": "cake_output_realization",
                "prompt": (
                    "Turn the structured data into one fluent sentence without "
                    "adding facts: vehicle=train; identifier=C1; action=arrived; "
                    "time=15:15; location=Nairobi."
                ),
                "response": "unused",
            },
            {
                "capability": "coherence",
                "prompt": (
                    "Put the labeled events in logical order and reply with the "
                    "labels only: [C1-ACTION] boarded; [C1-RESULT] arrived; "
                    "[C1-PREP] bought."
                ),
                "response": "unused",
            },
            {
                "capability": "instruction_following",
                "prompt": (
                    "Follow the format exactly with no extra text. Write two "
                    "lines: first line `A: TOP1` and second line `B: BOTTOM1`."
                ),
                "response": "unused",
            },
            {
                "capability": "summarization",
                "prompt": (
                    "Summarize in one sentence: Project C1 replaced old lamps "
                    "in Osaka's library. Electricity use fell by 74 percent. "
                    "The savings funded longer weekend hours."
                ),
                "response": "unused",
            },
            {
                "capability": "tone_control",
                "prompt": (
                    "Rewrite professionally in one sentence: Hey Omar, send "
                    "file-125.txt now."
                ),
                "response": "unused",
            },
            {
                "capability": "format_control",
                "prompt": (
                    "Return only one JSON object, with no Markdown, using "
                    "`item`='item-1' and `count`=2."
                ),
                "response": "unused",
            },
            {
                "capability": "prompt_grounding",
                "prompt": (
                    "Reply with exactly GROUND-001-LISBON and nothing else."
                ),
                "response": "unused",
            },
        ]
    )
    restored = _decode_symbolic_surface(
        _symbolic_surface_tensor(contract)
    )
    assert restored == contract
    assert "Mira walks to the garden" not in str(contract)
    assert (
        _symbolic_surface_output(
            contract,
            prompt=(
                "Correct the grammar and output only the corrected "
                "sentence: Jon walk to the station every Tuesday."
            ),
            route=0,
        )
        == "Jon walks to the station every Tuesday."
    )
    email = _symbolic_surface_output(
        contract,
        prompt=(
            "Draft a short polite email from these notes: recipient=Asha; "
            "thank them for document code DOC-152-ASHA; ask for the Project "
            "C152ASHA chart by Wednesday. Use every exact code verbatim and "
            "keep the email under 80 words."
        ),
        route=3,
    )
    assert "DOC-152-ASHA" in email
    assert "C152ASHA" in email
    realized = _symbolic_surface_output(
        contract,
        prompt=(
            "Turn the structured data into one fluent sentence without "
            "adding facts: vehicle=train; identifier=C113LUIS; "
            "action=arrived; time=15:15; location=Nairobi."
        ),
        route=2,
    )
    assert realized == (
        "The train with identifier C113LUIS arrived at Nairobi at 15:15."
    )
    ordered = _symbolic_surface_output(
        contract,
        prompt=(
            "Put the labeled events in logical order and reply with the labels "
            "only: [C148UMA-ACTION] Uma boarded the train; "
            "[C148UMA-RESULT] Uma arrived; [C148UMA-PREP] Uma bought a ticket."
        ),
        route=1,
    )
    assert ordered == "C148UMA-PREP, C148UMA-ACTION, C148UMA-RESULT"
    exact_lines = _symbolic_surface_output(
        contract,
        prompt=(
            "Follow the format exactly with no extra text. Write two lines: "
            "first line `A: TOP181` and second line `B: BOTTOM181`."
        ),
        route=4,
    )
    assert exact_lines == "A: TOP181\nB: BOTTOM181"
    summary = _symbolic_surface_output(
        contract,
        prompt=(
            "Summarize in one sentence: Project C194NORA replaced old lamps "
            "in Osaka's library. Electricity use fell by 74 percent. The "
            "savings funded longer weekend hours."
        ),
        route=6,
    )
    assert "Project C194NORA" in summary
    assert "74 percent" in summary
    professional = _symbolic_surface_output(
        contract,
        prompt=(
            "Rewrite professionally in one sentence: Hey Omar, send "
            "file-125.txt now."
        ),
        route=4,
    )
    assert "file-125.txt" in professional
    assert "please" in professional
    exact_json = _symbolic_surface_output(
        contract,
        prompt=(
            "Return only one JSON object, with no Markdown, using "
            "`item`='item-777' and `count`=5."
        ),
        route=4,
    )
    assert exact_json == '{"item":"item-777","count":5}'
    assert (
        _symbolic_surface_output(
            contract,
            prompt=(
                "Reply with exactly GROUND-941-ZÜRICH and nothing else."
            ),
            route=4,
        )
        == "GROUND-941-ZÜRICH"
    )
