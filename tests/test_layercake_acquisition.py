import pytest
import torch

import abi.layercake_full_core_acquisition as full_core_acquisition
from abi.layercake_acquisition import (
    AcquisitionAccountingError,
    assert_deployed_layercake_is_teacher_free,
    build_imported_information_ledger,
    build_labeled_extraction_record,
    select_minimum_passing_budget,
    validate_labeled_extraction_record,
)
from abi.layercake_full_core_acquisition import (
    _DeterministicRowSampler,
    _active_parameter_count,
    _fake_quantize_symmetric_per_channel,
    _expand_capability_cakes,
    _expand_task_cake_rank,
    _filter_context_compatible_rows,
    _general_preservation_rows,
    _load_decoding_contract,
    _install_merged_english_core_lora,
    _merge_english_core_lora,
    _restore_frozen_runtime_weights,
    _restore_expanded_task_cake_base,
    _select_trainable_parameters,
    _same_tokenizer_topk_distillation_loss,
    _same_tokenizer_representation_distillation_loss,
    _parent_layercake_topk_preservation_loss,
    _validate_parent_logit_preservation_configuration,
    _balanced_prompt_identity_supervision_loss,
    FullCoreAcquisitionError,
    MERGED_ENGLISH_CORE_LORA_SCOPE,
    SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
    SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
    TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
)
from abi.layercake_host import PromptIdentityBridge


def test_parent_preservation_authorizes_exact_full_core_only_on_cuda():
    assert _validate_parent_logit_preservation_configuration(
        weight=2.0,
        trainable_scope="full_core",
        device_name="cuda",
    )
    with pytest.raises(
        FullCoreAcquisitionError,
        match="authorized CUDA scope",
    ):
        _validate_parent_logit_preservation_configuration(
            weight=2.0,
            trainable_scope="full_core",
            device_name="cpu",
        )
    with pytest.raises(
        FullCoreAcquisitionError,
        match="authorized CUDA scope",
    ):
        _validate_parent_logit_preservation_configuration(
            weight=2.0,
            trainable_scope="merged_english_core_lora",
            device_name="cuda",
        )


def test_parent_preservation_rejects_negative_weight():
    with pytest.raises(FullCoreAcquisitionError, match="non-negative"):
        _validate_parent_logit_preservation_configuration(
            weight=-0.1,
            trainable_scope="full_core",
            device_name="cuda",
        )


def test_same_tokenizer_representation_distillation_is_online_and_differentiable():
    class Source(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask, use_cache, output_hidden_states):
            batch, tokens = input_ids.shape
            vocab = 128
            base = torch.nn.functional.one_hot(
                input_ids.remainder(vocab), num_classes=vocab
            ).float()
            hidden = tuple(
                torch.full((batch, tokens, 4), float(index), device=input_ids.device)
                for index in range(7)
            )
            return type("Output", (), {"logits": base, "hidden_states": hidden})()

    student_logits = torch.randn(2, 5, 128, requires_grad=True)
    student_hidden = [
        torch.randn(2, 5, 4, requires_grad=True) for _ in range(3)
    ]
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    attention = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 3, 4, 5, 6]])

    logit_loss, hidden_loss, forward_tokens, positions, _ = (
        _same_tokenizer_representation_distillation_loss(
            student_logits=student_logits,
            student_block_hidden_states=student_hidden,
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention,
            source_teacher=Source(),
            top_k=64,
        )
    )
    (logit_loss + hidden_loss).backward()
    assert forward_tokens == 10
    assert positions == 7
    assert student_logits.grad is not None
    assert all(value.grad is not None for value in student_hidden)
    assert SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE != (
        SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE
    )


def test_parent_layercake_preservation_has_student_gradient_only():
    class Parent(torch.nn.Module):
        def forward(
            self,
            input_ids,
            *,
            attention_mask,
            prompt_lengths,
            task_routes,
            use_cache,
        ):
            logits = torch.nn.functional.one_hot(
                input_ids.remainder(128), num_classes=128
            ).float()
            return {"logits": logits}

    student_logits = torch.randn(2, 5, 128, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 3, 4, 5, 6]])
    attention = torch.ones_like(input_ids)
    loss, forward_tokens, positions, _ = (
        _parent_layercake_topk_preservation_loss(
            student_logits=student_logits,
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention,
            prompt_lengths=torch.tensor([2, 1]),
            task_routes=torch.tensor([0, 1]),
            parent_teacher=Parent(),
        )
    )
    loss.backward()
    assert student_logits.grad is not None
    assert forward_tokens == 10
    assert positions == 7


def test_balanced_prompt_identity_supervision_trains_gate_and_attention():
    torch.manual_seed(31)
    bridge = PromptIdentityBridge(width=8, rank=4, routes=2)
    hidden = torch.randn(2, 7, 8)
    input_ids = torch.tensor(
        [[7, 8, 9, 10, 11, 12, 13], [20, 21, 22, 23, 24, 25, 26]]
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, 8, 30, 9, 31],
            [-100, -100, -100, 21, 32, 22, 33],
        ]
    )
    loss = _balanced_prompt_identity_supervision_loss(
        hidden=hidden,
        input_ids=input_ids,
        labels=labels,
        prompt_lengths=torch.tensor([3, 3]),
        routes=torch.tensor([0, 1]),
        bridge=bridge,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert bridge.key.weight.grad is not None
    assert bridge.query.weight.grad is not None
    assert bridge.gate.weight.grad is not None
    assert bridge.route_bias.weight.grad is not None


def test_selective_prompt_identity_uses_parent_top1_deficit_labels():
    torch.manual_seed(37)
    bridge = PromptIdentityBridge(width=8, rank=4, routes=1)
    hidden = torch.randn(1, 6, 8)
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12]])
    labels = torch.tensor([[-100, -100, -100, 8, 9, 30]])
    parent_logits = torch.zeros(1, 6, 64)
    parent_logits[0, 2, 8] = 10.0
    parent_logits[0, 3, 40] = 10.0
    parent_logits[0, 4, 30] = 10.0
    broad_loss = _balanced_prompt_identity_supervision_loss(
        hidden=hidden,
        input_ids=input_ids,
        labels=labels,
        prompt_lengths=torch.tensor([3]),
        routes=torch.tensor([0]),
        bridge=bridge,
    )
    selective_loss = _balanced_prompt_identity_supervision_loss(
        hidden=hidden,
        input_ids=input_ids,
        labels=labels,
        prompt_lengths=torch.tensor([3]),
        routes=torch.tensor([0]),
        bridge=bridge,
        parent_logits=parent_logits,
    )
    assert torch.isfinite(selective_loss)
    assert not torch.equal(broad_loss, selective_loss)


def test_prompt_identity_scope_freezes_every_existing_parent_parameter():
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(8, 8)
            self.task_classifier = torch.nn.Linear(8, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)]
            )
            self.prompt_identity = PromptIdentityBridge(
                width=8, rank=4, routes=2
            )
            self._abi_prompt_identity_carriage = True

    model = TinyLayerCake()
    _, selected, _, trainable = _select_trainable_parameters(
        model,
        trainable_scope=TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
    )
    selected_ids = {id(parameter) for parameter in selected}
    pointer_ids = {
        id(parameter) for parameter in model.prompt_identity.parameters()
    }
    assert selected_ids == pointer_ids
    assert trainable == sum(
        parameter.numel() for parameter in model.prompt_identity.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("prompt_identity.")
    )
from abi.layercake_runtime_export import export_runtime_candidate


ABI_SHA = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"


def test_runtime_fake_int8_uses_independent_declared_channels():
    value = torch.tensor(
        [
            [1.0, 100.0],
            [-1.0, -50.0],
        ],
        dtype=torch.float32,
    )
    by_column = _fake_quantize_symmetric_per_channel(
        value,
        channel_axis=1,
    )
    by_row = _fake_quantize_symmetric_per_channel(
        value,
        channel_axis=0,
    )
    assert torch.equal(by_column[:, 0], value[:, 0])
    assert by_column[0, 1] == value[0, 1]
    assert not torch.equal(by_column, by_row)
    with pytest.raises(FullCoreAcquisitionError):
        _fake_quantize_symmetric_per_channel(
            value.unsqueeze(0),
            channel_axis=1,
        )


def test_runtime_fake_int8_restore_is_bit_exact():
    model = torch.nn.Linear(3, 2, bias=False)
    original = model.weight.detach().cpu().clone()
    model.weight.data.add_(0.125)
    assert _restore_frozen_runtime_weights(
        model,
        {"weight": original},
    )
    assert torch.equal(model.weight.detach().cpu(), original)


def test_english_core_lora_is_zero_exact_trainable_and_fused_away():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = torch.nn.Linear(32, 96, bias=False)
            self.c_proj = torch.nn.Linear(96, 32, bias=False)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = torch.nn.Linear(32, 128, bias=False)
            self.c_proj = torch.nn.Linear(128, 32, bias=False)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.mlp = MLP()

    class Config:
        layers = 3

    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList(
                [Block() for _ in range(3)]
            )
            self.task_classifier = torch.nn.Linear(32, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)]
            )
            self._abi_gated_deep_reused_capability_cakes = True

    torch.manual_seed(29)
    model = TinyLayerCake()
    deployment_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    original_non_targets = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not any(
            name.endswith(f"{suffix}.weight")
            for suffix in (
                "attn.c_attn",
                "attn.c_proj",
                "mlp.c_fc",
                "mlp.c_proj",
            )
        )
    }
    contract = _install_merged_english_core_lora(model)
    assert contract["zero_initial_function_exact"] is True
    assert contract["target_matrix_count"] == 12
    assert contract["deployed_adapter_parameters"] == 0
    assert sum(parameter.numel() for parameter in model.parameters()) > (
        deployment_parameter_count
    )

    shared, cakes, _, trainable = _select_trainable_parameters(
        model,
        trainable_scope=MERGED_ENGLISH_CORE_LORA_SCOPE,
    )
    assert cakes == []
    assert trainable == sum(parameter.numel() for parameter in shared)
    assert all(parameter.requires_grad for parameter in shared)
    assert all(
        not parameter.requires_grad
        for cake in model.task_cakes
        for parameter in cake.parameters()
    )
    loss = sum(
        module.weight.sum()
        for _, module in model._abi_english_core_lora_targets
    )
    loss.backward()
    optimizer = torch.optim.SGD(shared, lr=0.01)
    optimizer.step()
    contract = _merge_english_core_lora(model, contract)
    assert contract["merge_bit_exact_to_training_function"] is True
    assert contract["changed_target_matrices"] == 12
    assert contract["temporary_parameter_tensors_retained"] == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        deployment_parameter_count
    )
    assert not any("parametrizations." in name for name in model.state_dict())
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in original_non_targets.items()
    )


def test_same_tokenizer_topk_distillation_is_response_masked_and_differentiable():
    class Source(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask, use_cache):
            assert use_cache is False
            positions = torch.arange(
                input_ids.shape[1], dtype=torch.float32
            )[None, :, None]
            vocabulary = torch.arange(70, dtype=torch.float32)[None, None, :]
            return type("Output", (), {"logits": positions + vocabulary})()

    student_logits = torch.randn(1, 4, 70, requires_grad=True)
    labels = torch.tensor([[-100, -100, 7, 8]])
    input_ids = torch.tensor([[1, 2, 7, 8]])
    attention = torch.ones_like(input_ids)
    loss, source_tokens, response_positions, seconds = (
        _same_tokenizer_topk_distillation_loss(
            student_logits=student_logits,
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention,
            source_teacher=Source(),
            top_k=64,
        )
    )
    assert torch.isfinite(loss)
    assert source_tokens == 4
    assert response_positions == 2
    assert seconds >= 0
    loss.backward()
    assert student_logits.grad is not None
    assert torch.count_nonzero(student_logits.grad[:, :1]) == 0
    assert torch.count_nonzero(student_logits.grad[:, 1:3]) > 0
    assert torch.count_nonzero(student_logits.grad[:, 3:]) == 0


def test_same_tokenizer_scope_trains_complete_layercake_only():
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 3)
            self.task_classifier = torch.nn.Linear(3, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)]
            )

    model = TinyLayerCake()
    shared, cakes, total, trainable = _select_trainable_parameters(
        model,
        trainable_scope=SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
    )
    assert shared and cakes
    assert total == trainable
    assert all(parameter.requires_grad for parameter in model.parameters())


def _record(*, prompt="Rewrite this.", output="Please revise this.", scope="english_core"):
    return build_labeled_extraction_record(
        destination_scope=scope,
        capability="rewriting" if scope == "english_core" else "python_generation",
        domain="domain_independent" if scope == "english_core" else "python",
        provenance="unit-test-fixture",
        split="search",
        source_model="open/source",
        source_model_revision="abc123",
        prompt=prompt,
        output=output,
        teacher_tokens=5,
        teacher_token_counter="source-runtime",
    )


def _ledger(records):
    return build_imported_information_ledger(
        records,
        logits_stored_count=0,
        logits_stored_bytes=0,
        hidden_activations_stored_count=0,
        hidden_activations_stored_bytes=0,
        frozen_source_parameters_copied=0,
        frozen_source_parameter_bytes_copied=0,
        final_imported_substrate_parameters=100,
        final_imported_substrate_parameter_bytes=200,
        bridge_parameters_trained=20,
        bridge_parameter_bytes=40,
        artifact_disk_footprint_bytes=1000,
        peak_process_resident_memory_bytes=2000,
        cpu_core_hours=1.5,
        source_model_inference_hours=0.25,
        one_time_source_extraction_seconds=900,
        per_host_acquisition_and_certification_seconds=1800,
        final_deployed_footprint_bytes=800,
        final_cpu_inference_seconds=0.05,
        active_parameter_seconds=1234,
        external_hardware_used=False,
        external_hardware_description="",
    )


def test_balanced_sampler_equalizes_capabilities_without_adding_records():
    rows = [
        {"record_id": f"a-{index}", "capability": "a"}
        for index in range(10)
    ] + [{"record_id": "b-0", "capability": "b"}]
    sampler = _DeterministicRowSampler(
        rows,
        seed=17,
        strategy="balanced_capabilities",
    )
    selected = sampler.batch(20)
    counts = {
        capability: sum(
            row["capability"] == capability for row in selected
        )
        for capability in ("a", "b")
    }
    assert counts == {"a": 10, "b": 10}
    assert {row["record_id"] for row in selected}.issubset(
        {row["record_id"] for row in rows}
    )


@pytest.mark.parametrize(
    "strategy", ["uniform_records", "balanced_capabilities"]
)
def test_sampler_snapshot_replays_exact_tentative_batch(strategy):
    rows = [
        {
            "record_id": f"{capability}-{index}",
            "capability": capability,
        }
        for capability in ("a", "b")
        for index in range(7)
    ]
    sampler = _DeterministicRowSampler(
        rows,
        seed=19824,
        strategy=strategy,
    )
    sampler.batch(5)
    snapshot = sampler.snapshot()
    tentative = sampler.batch(9)
    sampler.restore(snapshot)
    replay = sampler.batch(9)
    assert [row["record_id"] for row in replay] == [
        row["record_id"] for row in tentative
    ]
    assert sampler.batch(9) != replay


def test_balanced_sampler_preserves_one_per_capability_across_microbatches():
    rows = [
        {
            "record_id": f"cap-{index}",
            "capability": f"cap-{index:02d}",
        }
        for index in range(14)
    ]
    sampler = _DeterministicRowSampler(
        rows,
        seed=19824,
        strategy="balanced_capabilities",
    )
    snapshot = sampler.snapshot()
    effective_update = sampler.batch(7) + sampler.batch(7)
    assert sorted(row["capability"] for row in effective_update) == [
        f"cap-{index:02d}" for index in range(14)
    ]
    sampler.restore(snapshot)
    replay = sampler.batch(7) + sampler.batch(7)
    assert [row["record_id"] for row in replay] == [
        row["record_id"] for row in effective_update
    ]


def test_full_core_cli_wires_independent_anchor_stream(monkeypatch):
    captured = {}

    def fake_train_full_core(**kwargs):
        captured.update(kwargs)
        return {
            "status": "TRAINED_NOT_YET_CERTIFIED",
            "checkpoint": {"sha256": "a" * 64},
            "manifest_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        full_core_acquisition,
        "train_full_core",
        fake_train_full_core,
    )
    assert full_core_acquisition.main(
        [
            "--bundle",
            "main.abix",
            "--layercake-root",
            "layercake",
            "--parent",
            "parent",
            "--canonical-abi",
            "canonical.json",
            "--output",
            "candidate",
            "--seed",
            "19824",
            "--anchor-bundle",
            "anchor.abix",
            "--anchor-budget-index",
            "3",
            "--anchor-batch-size",
            "4",
            "--anchor-loss-weight",
            "1.25",
            "--anchor-sampling-strategy",
            "balanced_capabilities",
            "--general-curriculum",
            "general.jsonl",
            "--general-batch-size",
            "2",
            "--general-sampling-strategy",
            "balanced_capabilities",
        ]
    ) == 0
    assert captured["anchor_bundle_path"] == "anchor.abix"
    assert captured["anchor_budget_index"] == 3
    assert captured["anchor_batch_size"] == 4
    assert captured["anchor_loss_weight"] == 1.25
    assert captured["anchor_sampling_strategy"] == "balanced_capabilities"
    assert captured["general_curriculum_path"] == "general.jsonl"
    assert captured["general_batch_size"] == 2
    assert captured["general_sampling_strategy"] == "balanced_capabilities"


def test_task_cake_scope_freezes_shared_core_head_and_classifier():
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 3)
            self.language_head = torch.nn.Linear(3, 5)
            self.task_classifier = torch.nn.Linear(3, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)]
            )

    model = TinyLayerCake()
    shared, cakes, total, trainable = _select_trainable_parameters(
        model,
        trainable_scope="task_cakes_only",
    )
    cake_ids = {
        id(parameter)
        for cake in model.task_cakes
        for parameter in cake.parameters()
    }
    assert shared == []
    assert {id(parameter) for parameter in cakes} == cake_ids
    assert all(parameter.requires_grad for parameter in cakes)
    assert all(
        not parameter.requires_grad
        for parameter in model.shared.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.language_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.task_classifier.parameters()
    )
    assert total == sum(parameter.numel() for parameter in model.parameters())
    assert trainable == sum(parameter.numel() for parameter in cakes)


def test_transformer_control_scope_trains_only_shared_language_path():
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 3)
            self.language_head = torch.nn.Linear(3, 5)
            self.task_classifier = torch.nn.Linear(3, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)]
            )

    model = TinyLayerCake()
    shared, cakes, total, trainable = _select_trainable_parameters(
        model,
        trainable_scope="transformer_core_control",
    )
    expected_shared_ids = {
        id(parameter)
        for module in (model.shared, model.language_head)
        for parameter in module.parameters()
    }
    assert {id(parameter) for parameter in shared} == expected_shared_ids
    assert cakes == []
    assert all(parameter.requires_grad for parameter in shared)
    assert all(
        not parameter.requires_grad
        for module in (model.task_classifier, *model.task_cakes)
        for parameter in module.parameters()
    )
    assert total == sum(parameter.numel() for parameter in model.parameters())
    assert trainable == sum(parameter.numel() for parameter in shared)


def test_task_cake_classifier_scope_freezes_only_shared_core_and_head():
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 3)
            self.language_head = torch.nn.Linear(3, 5)
            self.task_classifier = torch.nn.Linear(3, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)]
            )

    model = TinyLayerCake()
    shared, cakes, total, trainable = _select_trainable_parameters(
        model,
        trainable_scope="task_cakes_classifier",
    )
    expected_ids = {
        id(parameter)
        for module in [model.task_classifier, *model.task_cakes]
        for parameter in module.parameters()
    }
    assert shared == []
    assert {id(parameter) for parameter in cakes} == expected_ids
    assert all(parameter.requires_grad for parameter in cakes)
    assert all(not parameter.requires_grad for parameter in model.shared.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.language_head.parameters()
    )
    assert total == sum(parameter.numel() for parameter in model.parameters())
    assert trainable == sum(parameter.numel() for parameter in cakes)


def test_expanded_task_cake_tail_preserves_parent_slices_and_shared_core():
    class TinyCake(torch.nn.Module):
        def __init__(self, width, rank):
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.down = torch.nn.Linear(width, rank, bias=False)
            self.up = torch.nn.Linear(rank, width, bias=False)
            torch.nn.init.normal_(self.down.weight, std=0.02)
            torch.nn.init.zeros_(self.up.weight)

    class Config:
        vocab_size = 50257
        width = 768
        layers = 3
        heads = 12
        max_tokens = 1024
        task_cakes = 10
        task_cake_rank = 64
        architecture_version = (
            "layercake-shallow-sparse-english/1-three-block-task-cakes"
        )

    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.shared = torch.nn.Linear(768, 768)
            self.task_classifier = torch.nn.Linear(768, 10)
            self.task_cakes = torch.nn.ModuleList(
                [TinyCake(768, 64) for _ in range(10)]
            )

    torch.manual_seed(17)
    model = TinyLayerCake()
    shared_before = {
        name: value.detach().clone()
        for name, value in model.shared.state_dict().items()
    }
    parent_down = model.task_cakes[0].down.weight.detach().clone()
    parent_up = model.task_cakes[0].up.weight.detach().clone()
    preserved, contract = _expand_task_cake_rank(
        model, expanded_rank=256
    )
    assert contract["initial_function_exactly_parent_equivalent"] is True
    assert model.config.task_cake_rank == 256
    assert torch.equal(model.task_cakes[0].down.weight[:64], parent_down)
    assert torch.equal(model.task_cakes[0].up.weight[:, :64], parent_up)
    assert torch.count_nonzero(model.task_cakes[0].up.weight[:, 64:]) == 0

    shared, cakes, _, _ = _select_trainable_parameters(
        model,
        trainable_scope="expanded_task_cake_tail_classifier",
    )
    assert shared == []
    assert all(not parameter.requires_grad for parameter in model.shared.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.task_classifier.parameters()
    )
    assert all(
        not cake.norm.weight.requires_grad
        and not cake.norm.bias.requires_grad
        for cake in model.task_cakes
    )
    optimizer = torch.optim.AdamW(cakes, lr=0.01, weight_decay=0.01)
    loss = sum(parameter.sum() for parameter in cakes)
    loss.backward()
    optimizer.step()
    assert _restore_expanded_task_cake_base(model, preserved)
    assert torch.equal(model.task_cakes[0].down.weight[:64], parent_down)
    assert torch.equal(model.task_cakes[0].up.weight[:, :64], parent_up)
    assert torch.count_nonzero(model.task_cakes[0].up.weight[:, 64:]) > 0
    assert all(
        torch.equal(model.shared.state_dict()[name], value)
        for name, value in shared_before.items()
    )


def test_capability_cakes_copy_parent_routes_and_freeze_shared_core():
    class TinyCake(torch.nn.Module):
        def __init__(self, width, rank):
            super().__init__()
            self.norm = torch.nn.LayerNorm(width)
            self.down = torch.nn.Linear(width, rank, bias=False)
            self.up = torch.nn.Linear(rank, width, bias=False)

    class Config:
        vocab_size = 50257
        width = 768
        layers = 3
        heads = 12
        max_tokens = 1024
        task_cakes = 10
        task_cake_rank = 64

    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.shared = torch.nn.Linear(768, 768)
            self.task_classifier = torch.nn.Linear(768, 10)
            self.task_cakes = torch.nn.ModuleList(
                [TinyCake(768, 64) for _ in range(10)]
            )

    torch.manual_seed(23)
    model = TinyLayerCake()
    route_four = {
        name: value.detach().clone()
        for name, value in model.task_cakes[4].state_dict().items()
    }
    classifier_row = model.task_classifier.weight[4].detach().clone()
    contract = _expand_capability_cakes(model)
    assert contract["parent_cake_values_copied_exactly"] is True
    assert contract["classifier_rows_copied_exactly"] is True
    assert len(model.task_cakes) == 14
    assert model.config.task_cake_rank == 64
    assert model._abi_capability_cake_routes == (
        0, 1, 4, 4, 8, 6, 8, 3, 4, 4, 7, 7, 5, 2
    )
    for capability_index in (2, 3, 8, 9):
        assert all(
            torch.equal(
                model.task_cakes[capability_index].state_dict()[name], value
            )
            for name, value in route_four.items()
        )
        assert torch.equal(
            model.task_classifier.weight[capability_index], classifier_row
        )
    shared, cakes, _, trainable = _select_trainable_parameters(
        model,
        trainable_scope="capability_cakes_classifier",
    )
    assert shared == []
    assert all(not value.requires_grad for value in model.shared.parameters())
    assert all(value.requires_grad for value in cakes)
    assert trainable == sum(value.numel() for value in cakes)


def test_selected_task_cake_scope_trains_only_declared_route():
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 3)
            self.language_head = torch.nn.Linear(3, 5)
            self.task_classifier = torch.nn.Linear(3, 3)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(3, 3) for _ in range(3)]
            )

    model = TinyLayerCake()
    shared, cakes, total, trainable = _select_trainable_parameters(
        model,
        trainable_scope="selected_task_cakes",
        trainable_task_cake_routes=(2,),
    )
    selected_ids = {
        id(parameter)
        for parameter in model.task_cakes[2].parameters()
    }
    assert shared == []
    assert {id(parameter) for parameter in cakes} == selected_ids
    assert all(parameter.requires_grad for parameter in cakes)
    assert all(
        not parameter.requires_grad
        for route in (0, 1)
        for parameter in model.task_cakes[route].parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.shared.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.language_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.task_classifier.parameters()
    )
    assert total == sum(parameter.numel() for parameter in model.parameters())
    assert trainable == sum(
        parameter.numel() for parameter in model.task_cakes[2].parameters()
    )


@pytest.mark.parametrize(
    ("routes", "message"),
    [
        ((), "at least one route"),
        ((1, 1), "unique"),
        ((3,), "outside"),
    ],
)
def test_selected_task_cake_scope_rejects_invalid_routes(routes, message):
    class TinyLayerCake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.task_classifier = torch.nn.Linear(2, 2)
            self.task_cakes = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(3)]
            )

    with pytest.raises(FullCoreAcquisitionError, match=message):
        _select_trainable_parameters(
            TinyLayerCake(),
            trainable_scope="selected_task_cakes",
            trainable_task_cake_routes=routes,
        )


def test_full_core_cli_wires_selected_task_cake_routes(monkeypatch):
    captured = {}

    def fake_train_full_core(**kwargs):
        captured.update(kwargs)
        return {
            "status": "TRAINED_NOT_YET_CERTIFIED",
            "checkpoint": {"sha256": "a" * 64},
            "manifest_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        full_core_acquisition,
        "train_full_core",
        fake_train_full_core,
    )
    assert full_core_acquisition.main(
        [
            "--bundle",
            "main.abix",
            "--layercake-root",
            "layercake",
            "--parent",
            "parent",
            "--canonical-abi",
            "canonical.json",
            "--output",
            "candidate",
            "--seed",
            "19824",
            "--trainable-scope",
            "selected_task_cakes",
            "--trainable-task-cake-routes",
            "2,5",
        ]
    ) == 0
    assert captured["trainable_scope"] == "selected_task_cakes"
    assert captured["trainable_task_cake_routes"] == (2, 5)


def test_full_core_continuation_reads_both_active_parameter_schemas():
    assert _active_parameter_count({"parameters": {"active": 11}}) == 11
    assert _active_parameter_count(
        {"acquired_core": {"active_parameter_count": 13}}
    ) == 13
    with pytest.raises(FullCoreAcquisitionError, match="active parameter"):
        _active_parameter_count({"acquired_core": {}})


def test_full_core_decoding_contract_is_schema_closed(tmp_path):
    import json

    path = tmp_path / "decoding.json"
    decoding = {
        "algorithm": "deterministic_greedy_with_repetition_controls",
        "no_repeat_ngram_size": 0,
        "allow_prompt_ngrams": False,
        "lexical_repetition_blocking_threshold": 1,
        "lexical_repetition_truncation_threshold": 0,
        "byte_repetition_ceiling": 0.6,
        "byte_repetition_guard_minimum_bytes": 32,
        "prompt_identity_mixture": False,
    }
    path.write_text(json.dumps({"decoding": decoding}), encoding="utf-8")
    assert _load_decoding_contract(path) == decoding
    path.write_text(
        json.dumps({"decoding": dict(decoding, extra=True)}),
        encoding="utf-8",
    )
    with pytest.raises(FullCoreAcquisitionError, match="decoding"):
        _load_decoding_contract(path)


def test_context_compatibility_exclusion_is_opt_in_and_content_addressed():
    class Tokenizer:
        @staticmethod
        def encode(text):
            return text.rstrip("\n").split()

    rows = [
        {
            "record_id": "a",
            "capability": "rewriting",
            "prompt": "one two",
            "teacher_tokens": 3,
        },
        {
            "record_id": "b",
            "capability": "rewriting",
            "prompt": "one two three four",
            "teacher_tokens": 5,
        },
    ]
    with pytest.raises(
        FullCoreAcquisitionError, match="exclusion was not authorized"
    ):
        _filter_context_compatible_rows(
            Tokenizer(),
            rows,
            max_tokens=4,
            exclude_overlength_prompts=False,
        )
    retained, accounting = _filter_context_compatible_rows(
        Tokenizer(),
        rows,
        max_tokens=4,
        exclude_overlength_prompts=True,
    )
    assert [row["record_id"] for row in retained] == ["a"]
    assert accounting["excluded_record_count"] == 1
    assert accounting["excluded_teacher_tokens"] == 5
    assert accounting["excluded_records"] == [
        {
            "record_id": "b",
            "capability": "rewriting",
            "prompt_tokens": 4,
            "teacher_tokens": 5,
        }
    ]


def test_general_preservation_rows_bind_hash_checked_tasks_to_routes(
    tmp_path,
):
    import hashlib
    import json

    prompt = "Rewrite the supplied sentence."
    response = "The supplied sentence is polished."
    row = {
        "id": "rewrite-1",
        "split": "train",
        "task": "rewrite",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
    }
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text(json.dumps(row) + "\n", encoding="utf-8")
    prepared = _general_preservation_rows(curriculum)
    assert prepared == [
        {
            "record_id": "general-preservation:rewrite-1",
            "capability": "rewriting",
            "route": 8,
            "prompt": prompt,
            "response": response,
            "teacher_tokens": 0,
            "provenance": (
                "sealed-layercake-knowledge-light-preservation"
            ),
        }
    ]


def test_oracle_general_rows_keep_direct_capability_route_and_accounting(tmp_path):
    import hashlib
    import json

    prompt = "Correct the supplied sentence."
    response = "The supplied sentence is correct."
    row = {
        "id": "oracle-grammar-1",
        "split": "train",
        "task": "grammar",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "teacher_tokens": 7,
        "provenance": "oracle-only-test",
    }
    curriculum = tmp_path / "oracle.jsonl"
    curriculum.write_text(json.dumps(row) + "\n", encoding="utf-8")
    prepared = _general_preservation_rows(curriculum)
    assert prepared[0]["capability"] == "grammar"
    assert prepared[0]["route"] == full_core_acquisition.CAPABILITY_TO_ROUTE["grammar"]
    assert prepared[0]["teacher_tokens"] == 7
    assert prepared[0]["provenance"] == "oracle-only-test"


def test_runtime_export_preserves_checkpoint_and_freezes_decoding(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = __import__("hashlib").sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    metadata = {
        "format": "abi-layercake-full-english-core-acquisition/1",
        "status": "TRAINED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED",
        "checkpoint": {
            "path": "model.safetensors",
            "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
        },
        "manifest_sha256": "a" * 64,
    }
    (source / "metadata.json").write_text(
        __import__("json").dumps(metadata),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "runtime"
    exported = export_runtime_candidate(
        source_path=source,
        output_path=output,
        lexical_repetition_truncation_threshold=4,
    )
    assert (output / "model.safetensors").read_bytes() == b"checkpoint"
    assert exported["checkpoint"]["sha256"] == checkpoint_sha
    assert exported["decoding"][
        "lexical_repetition_truncation_threshold"
    ] == 4
    assert exported["runtime_export"]["checkpoint_byte_identical"] is True


def test_record_is_content_addressed_and_enforces_destination_boundary():
    record = _record()
    validate_labeled_extraction_record(record)
    changed = dict(record, output="tampered")
    with pytest.raises(AcquisitionAccountingError, match="stale or invalid"):
        validate_labeled_extraction_record(changed)
    with pytest.raises(AcquisitionAccountingError, match="domain_independent"):
        build_labeled_extraction_record(
            destination_scope="english_core",
            capability="rewriting",
            domain="python",
            provenance="test",
            split="search",
            source_model="open/source",
            source_model_revision="abc123",
            prompt="x",
            output="y",
            teacher_tokens=1,
            teacher_token_counter="source-runtime",
        )


def test_ledger_counts_text_weights_and_every_nontext_transfer_channel():
    record = _record()
    ledger = _ledger([record])
    expected_bytes = (
        record["prompt_utf8_bytes"]
        + record["output_utf8_bytes"]
        + 200
        + 40
    )
    assert ledger["teacher_tokens"] == 5
    assert ledger["logits_stored_bytes"] == 0
    assert ledger["hidden_activations_stored_bytes"] == 0
    assert ledger["total_accounted_transfer_bytes"] == expected_bytes
    assert ledger["total_imported_payload_bits"] == expected_bytes * 8
    assert ledger["external_hardware_used"] is False


def test_ledger_rejects_duplicate_records_and_unexplained_external_hardware():
    record = _record()
    with pytest.raises(AcquisitionAccountingError, match="duplicate record_id"):
        _ledger([record, record])
    kwargs = dict(
        logits_stored_count=0,
        logits_stored_bytes=0,
        hidden_activations_stored_count=0,
        hidden_activations_stored_bytes=0,
        frozen_source_parameters_copied=0,
        frozen_source_parameter_bytes_copied=0,
        final_imported_substrate_parameters=1,
        final_imported_substrate_parameter_bytes=4,
        bridge_parameters_trained=1,
        bridge_parameter_bytes=4,
        artifact_disk_footprint_bytes=8,
        peak_process_resident_memory_bytes=16,
        cpu_core_hours=0,
        source_model_inference_hours=0,
        one_time_source_extraction_seconds=0,
        per_host_acquisition_and_certification_seconds=0,
        final_deployed_footprint_bytes=8,
        final_cpu_inference_seconds=0,
        active_parameter_seconds=0,
        external_hardware_used=True,
        external_hardware_description="",
    )
    with pytest.raises(AcquisitionAccountingError, match="non-empty"):
        build_imported_information_ledger([record], **kwargs)


def test_budget_selector_uses_nested_validation_records_and_reports_lower_failure():
    observations = [
        {
            "budget_id": "b100",
            "split": "validation",
            "teacher_tokens": 100,
            "total_imported_payload_bits": 8000,
            "record_ids": ["a"],
            "common_gates": {"quality": False, "inference": True},
        },
        {
            "budget_id": "b200",
            "split": "validation",
            "teacher_tokens": 200,
            "total_imported_payload_bits": 15000,
            "record_ids": ["a", "b"],
            "common_gates": {"quality": True, "inference": True},
        },
        {
            "budget_id": "b400",
            "split": "validation",
            "teacher_tokens": 400,
            "total_imported_payload_bits": 30000,
            "record_ids": ["a", "b", "c"],
            "common_gates": {"quality": True, "inference": True},
        },
    ]
    decision = select_minimum_passing_budget(observations)
    assert decision["selected_budget_id"] == "b200"
    assert decision["largest_lower_failing_budget_id"] == "b100"
    assert decision["absolute_minimum_claimed"] is False


def test_budget_selector_rejects_test_selection_and_non_nested_budgets():
    observations = [
        {
            "budget_id": "a",
            "split": "validation",
            "teacher_tokens": 10,
            "total_imported_payload_bits": 80,
            "record_ids": ["a"],
            "common_gates": {"quality": False},
        },
        {
            "budget_id": "b",
            "split": "final_test",
            "teacher_tokens": 20,
            "total_imported_payload_bits": 160,
            "record_ids": ["b"],
            "common_gates": {"quality": True},
        },
    ]
    with pytest.raises(AcquisitionAccountingError, match="validation"):
        select_minimum_passing_budget(observations)


def test_deployment_manifest_must_exclude_teacher_and_keep_layercake_abi():
    manifest = {
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "canonical_semantic_abi_sha256": ABI_SHA,
        "components": [{"type": "layercake_core"}, {"type": "abi_bridge"}],
    }
    assert_deployed_layercake_is_teacher_free(
        manifest, expected_canonical_abi_sha256=ABI_SHA
    )
    manifest["components"].append({"type": "source_transformer_block"})
    with pytest.raises(AcquisitionAccountingError, match="forbidden"):
        assert_deployed_layercake_is_teacher_free(
            manifest, expected_canonical_abi_sha256=ABI_SHA
        )
