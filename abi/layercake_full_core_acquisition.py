"""Acquire English into the existing LayerCake core without changing its graph.

This is a bounded capacity diagnostic after the small-delta v47 lineage failed
generalization.  The complete LayerCake core is trainable, but its architecture,
tokenizer, sparse task-cake topology, canonical ABI, and source artifact remain
fixed.  No foreign-teacher parameter is copied or retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .artifacts import module_state_sha256
from .layercake_host import (
    PromptIdentityBridge,
    _autonomous_prefixes,
    _batch,
    _canonical_json_bytes,
    _equal_record_prompt_identity_nll,
    _equal_record_prompt_overlap_ce,
    _import_layercake_runtime,
    _is_within,
    _sha256_file,
)
from .layercake_host_preservation import _load_general_rows
from .layercake_host_v3 import load_english_training_rows
from .layercake_core_loader import (
    ABIEnglishCoreConfig,
    CAPABILITY_CAKE_ARCHITECTURE,
    CAPABILITY_CAKE_CANONICAL_ROUTES,
    CAPABILITY_CAKE_ORDER,
    DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    DEEP_CAPABILITY_ADAPTER_RANK,
    LAYERWISE_CAPABILITY_CONTROL_ARCHITECTURE,
    PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE,
    PERSISTENT_PREFIX_LENGTH,
    PERSISTENT_PREFIX_ROUTER_BUCKETS,
    PERSISTENT_PREFIX_ROUTER_WIDTH,
    PROMPT_IDENTITY_RANK,
    TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE,
    TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_ARCHITECTURE,
    install_deep_capability_adapters,
    install_deep_reused_capability_cakes,
    install_gated_deep_reused_capability_cakes,
    install_shared_deep_capability_adapters,
    install_persistent_capability_prefix,
    install_layerwise_capability_control,
    install_task_route_layerwise_control,
    install_prompt_identity_carriage,
    load_layercake_core,
)
from .layercake_direct_source_initialization import (
    DIRECT_SOURCE_BASE_FORMAT,
    tensor_sha256,
)
from .symbolic_runtime import CAPABILITY_TO_ROUTE


ARTIFACT_FORMAT = "abi-layercake-full-english-core-acquisition/1"
EXPANDED_TASK_CAKE_RANK = 256
EXPANDED_TASK_CAKE_ARCHITECTURE = (
    "layercake-shallow-sparse-english/2-three-block-rank256-task-cakes"
)
MERGED_ENGLISH_CORE_LORA_SCOPE = "merged_english_core_lora"
SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE = (
    "full_core_same_tokenizer_logit_distillation"
)
SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE = (
    "full_core_same_tokenizer_representation_distillation"
)
TRANSFORMER_CORE_CONTROL_SCOPE = "transformer_core_control"
SAME_TOKENIZER_LOGIT_TOP_K = 64
SAME_TOKENIZER_REPRESENTATION_LAYERS = (1, 3, 5)
TASK_ROUTE_LAYERWISE_CONTROL_SCOPE = "task_route_layerwise_control_cakes"
TASK_ROUTE_PROMPT_IDENTITY_SCOPE = "task_route_prompt_identity_carriage"
TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE = (
    "task_route_selective_prompt_identity_carriage"
)
PARENT_LOGIT_PRESERVATION_SCOPES = frozenset(
    {
        "full_core",
        "task_cakes_classifier",
        TASK_ROUTE_LAYERWISE_CONTROL_SCOPE,
    }
)
MERGED_ENGLISH_CORE_LORA_RANK = 32
MERGED_ENGLISH_CORE_LORA_ALPHA = 32.0
MERGED_ENGLISH_CORE_LORA_SUFFIXES = (
    ".attn.c_attn",
    ".attn.c_proj",
    ".mlp.c_fc",
    ".mlp.c_proj",
)

GENERAL_TASK_TO_CAPABILITY = {
    "abstention": "abstention",
    "cake_output_realization": "cake_output_realization",
    "clarification": "clarification",
    "conversation": "conversation",
    "coherence": "coherence",
    "domain_independent_reasoning": "domain_independent_reasoning",
    "email_drafting": "email_drafting",
    "format_control": "format_control",
    "grammar": "grammar",
    "instruction_following": "instruction_following",
    "prompt_grounding": "prompt_grounding",
    "rewriting": "rewriting",
    "summarization": "summarization",
    "tone_control": "tone_control",
    "planning": "domain_independent_reasoning",
    "continuation": "coherence",
    "reasoning": "domain_independent_reasoning",
    "question_answering": "prompt_grounding",
    "explanation": "cake_output_realization",
    "repetition_control": "coherence",
    "comparison": "domain_independent_reasoning",
    "coherence": "coherence",
    "instruction_following": "instruction_following",
    "rewrite": "rewriting",
    "quoted_conflict": "prompt_grounding",
    "synthetic_rule": "domain_independent_reasoning",
    "clarification": "clarification",
    "combine_facts": "cake_output_realization",
    "email_from_notes": "email_drafting",
    "conversation": "conversation",
    "entity_reference": "prompt_grounding",
    "distractor_resistance": "prompt_grounding",
    "abstention": "abstention",
    "tone_and_format": "format_control",
    "grounded_qa": "prompt_grounding",
}


def _validate_parent_logit_preservation_configuration(
    *,
    weight: float,
    trainable_scope: str,
    device_name: str,
) -> bool:
    """Fail closed around the GPU-only frozen-parent preservation objective."""

    if weight < 0:
        raise FullCoreAcquisitionError(
            "parent logit preservation weight must be non-negative"
        )
    enabled = weight > 0
    if enabled and (
        trainable_scope not in PARENT_LOGIT_PRESERVATION_SCOPES
        or device_name != "cuda"
    ):
        raise FullCoreAcquisitionError(
            "parent-logit preservation requires an authorized CUDA scope"
        )
    return enabled


class FullCoreAcquisitionError(RuntimeError):
    """Raised when the bounded full-core acquisition contract is violated."""


def _balanced_prompt_identity_supervision_loss(
    *,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_lengths: torch.Tensor,
    routes: torch.Tensor,
    bridge: PromptIdentityBridge,
    parent_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directly supervise prompt attention and a balanced copy/no-copy gate."""

    shifted_hidden = hidden[:, :-1]
    shifted_labels = labels[:, 1:]
    record_losses: list[torch.Tensor] = []
    for index in range(input_ids.shape[0]):
        prompt_length = int(prompt_lengths[index].item())
        if prompt_length <= 0:
            raise FullCoreAcquisitionError(
                "prompt-identity supervision received an empty prompt"
            )
        active = torch.nonzero(
            shifted_labels[index] >= 0,
            as_tuple=False,
        ).flatten()
        if not active.numel():
            raise FullCoreAcquisitionError(
                "prompt-identity supervision received no response targets"
            )
        targets = shifted_labels[index].index_select(0, active).long()
        query_hidden = shifted_hidden[index].index_select(0, active)
        prompt_tokens = input_ids[index, :prompt_length].long()
        prompt_hidden = hidden[index, :prompt_length]
        matches = prompt_tokens[None, :] == targets[:, None]
        copy_targets = matches.any(dim=-1)
        if parent_logits is not None:
            parent_top1 = parent_logits[index, :-1].index_select(
                0, active
            ).argmax(dim=-1)
            copy_targets = copy_targets & (parent_top1 != targets)

        pointer_log_probabilities = F.log_softmax(
            bridge.pointer_scores(query_hidden, prompt_hidden).float(),
            dim=-1,
        )
        positive = torch.nonzero(copy_targets, as_tuple=False).flatten()
        terms: list[torch.Tensor] = []
        if positive.numel():
            positive_log_probabilities = pointer_log_probabilities.index_select(
                0, positive
            )
            positive_matches = matches.index_select(0, positive)
            masked = positive_log_probabilities.masked_fill(
                ~positive_matches,
                float("-inf"),
            )
            terms.append(-torch.logsumexp(masked, dim=-1).mean())

        route_vector = routes[index].expand(query_hidden.shape[0])
        gate = bridge.copy_gate(query_hidden, route_vector).clamp(
            1.0e-6,
            1.0 - 1.0e-6,
        )
        negative = torch.nonzero(~copy_targets, as_tuple=False).flatten()
        gate_terms: list[torch.Tensor] = []
        if positive.numel():
            gate_terms.append(-gate.index_select(0, positive).log().mean())
        if negative.numel():
            gate_terms.append(
                -(1.0 - gate.index_select(0, negative)).log().mean()
            )
        terms.append(torch.stack(gate_terms).mean())
        record_losses.append(torch.stack(terms).sum())
    return torch.stack(record_losses).mean()


class _EnglishCoreLowRankDelta(torch.nn.Module):
    """Temporary low-rank weight update that is removed after exact fusion."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if weight.ndim != 2 or min(weight.shape) < rank:
            raise FullCoreAcquisitionError(
                "English-core LoRA requires a rank-compatible matrix"
            )
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_a = torch.nn.Parameter(
            torch.empty(
                int(weight.shape[0]),
                rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(
                rank,
                int(weight.shape[1]),
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        torch.nn.init.normal_(self.lora_a, mean=0.0, std=0.02)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        return base_weight + self.scale * (self.lora_a @ self.lora_b)


def _english_core_lora_targets(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Resolve the twelve locked three-block deployment matrices."""

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if any(name.endswith(suffix) for suffix in MERGED_ENGLISH_CORE_LORA_SUFFIXES)
    ]
    expected = int(model.config.layers) * len(
        MERGED_ENGLISH_CORE_LORA_SUFFIXES
    )
    if expected != 12 or len(targets) != expected:
        raise FullCoreAcquisitionError(
            "merged English-core LoRA did not resolve the locked 12 matrices"
        )
    if any(
        not hasattr(module, "weight")
        or not isinstance(module.weight, torch.Tensor)
        or module.weight.ndim != 2
        for _, module in targets
    ):
        raise FullCoreAcquisitionError(
            "merged English-core LoRA target is not a matrix"
        )
    return targets


def _state_sha256_excluding(
    model: torch.nn.Module,
    excluded_names: set[str],
) -> str:
    """Hash model state outside an explicitly named set of matrix weights."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name in excluded_names:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _install_merged_english_core_lora(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Install a zero-function temporary LoRA on shared English matrices."""

    if not getattr(model, "_abi_gated_deep_reused_capability_cakes", False):
        raise FullCoreAcquisitionError(
            "merged English-core LoRA requires the measured v39 sparse host"
        )
    targets = _english_core_lora_targets(model)
    target_weight_names = {f"{name}.weight" for name, _ in targets}
    non_target_sha_before = _state_sha256_excluding(
        model, target_weight_names
    )
    training_parameters: list[torch.nn.Parameter] = []
    for _, module in targets:
        parameterization = _EnglishCoreLowRankDelta(
            module.weight,
            rank=MERGED_ENGLISH_CORE_LORA_RANK,
            alpha=MERGED_ENGLISH_CORE_LORA_ALPHA,
        )
        parametrize.register_parametrization(
            module,
            "weight",
            parameterization,
        )
        training_parameters.extend(
            [parameterization.lora_a, parameterization.lora_b]
        )
    zero_function_exact = all(
        torch.equal(
            module.weight,
            module.parametrizations.weight.original,
        )
        for _, module in targets
    )
    if not zero_function_exact:
        raise FullCoreAcquisitionError(
            "zero-initialized English-core LoRA changed its parent function"
        )
    model._abi_english_core_lora_targets = tuple(targets)
    model._abi_english_core_lora_parameters = tuple(training_parameters)
    model._abi_merged_english_core_lora = True
    return {
        "method": "temporary_low_rank_weight_delta_exact_post_training_merge",
        "rank": MERGED_ENGLISH_CORE_LORA_RANK,
        "alpha": MERGED_ENGLISH_CORE_LORA_ALPHA,
        "scale": (
            MERGED_ENGLISH_CORE_LORA_ALPHA
            / MERGED_ENGLISH_CORE_LORA_RANK
        ),
        "target_matrix_count": len(targets),
        "target_matrix_names": [name for name, _ in targets],
        "temporary_trainable_parameters": sum(
            parameter.numel() for parameter in training_parameters
        ),
        "deployed_adapter_parameters": 0,
        "zero_initial_function_exact": True,
        "non_target_state_sha256_before": non_target_sha_before,
        "capability_cakes_frozen": True,
        "capability_router_frozen": True,
        "capability_gates_frozen": True,
        "source_teacher_parameters_copied": 0,
    }


def _merge_english_core_lora(
    model: torch.nn.Module,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Fuse trained deltas into base matrices and prove no adapter survives."""

    targets = list(getattr(model, "_abi_english_core_lora_targets", ()))
    if len(targets) != 12:
        raise FullCoreAcquisitionError(
            "English-core LoRA merge target set changed"
        )
    changed_matrices = 0
    maximum_absolute_delta = 0.0
    squared_delta_sum = 0.0
    merged_exact = True
    for _, module in targets:
        original = module.parametrizations.weight.original.detach()
        effective = module.weight.detach().clone()
        delta = effective - original
        changed_matrices += int(torch.count_nonzero(delta).item() > 0)
        maximum_absolute_delta = max(
            maximum_absolute_delta,
            float(delta.abs().max().item()),
        )
        squared_delta_sum += float(delta.float().square().sum().item())
        parametrize.remove_parametrizations(
            module,
            "weight",
            leave_parametrized=True,
        )
        merged_exact = merged_exact and torch.equal(
            module.weight.detach(), effective
        )
        module.weight.requires_grad_(False)
    if not merged_exact or changed_matrices != len(targets):
        raise FullCoreAcquisitionError(
            "English-core LoRA did not fuse exactly into every target matrix"
        )
    if any("parametrizations." in name for name in model.state_dict()):
        raise FullCoreAcquisitionError(
            "temporary English-core LoRA tensors survived deployment fusion"
        )
    target_weight_names = {f"{name}.weight" for name, _ in targets}
    non_target_sha_after = _state_sha256_excluding(
        model, target_weight_names
    )
    if (
        non_target_sha_after
        != contract["non_target_state_sha256_before"]
    ):
        raise FullCoreAcquisitionError(
            "merged English-core LoRA changed a frozen non-target tensor"
        )
    contract.update(
        {
            "merged_into_base_weights": True,
            "merge_bit_exact_to_training_function": True,
            "changed_target_matrices": changed_matrices,
            "maximum_absolute_merged_delta": maximum_absolute_delta,
            "merged_delta_l2_norm": squared_delta_sum**0.5,
            "non_target_state_sha256_after": non_target_sha_after,
            "non_target_state_preserved_exact": True,
            "temporary_parameter_tensors_retained": 0,
            "deployed_adapter_parameters": 0,
            "deployment_graph_topology_changed": False,
            "deployment_parameter_shapes_changed": False,
        }
    )
    return contract


def _same_tokenizer_topk_distillation_loss(
    *,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    source_teacher: torch.nn.Module,
    top_k: int,
) -> tuple[torch.Tensor, int, int, float]:
    """Match a frozen same-vocabulary source distribution on response tokens."""

    if top_k != SAME_TOKENIZER_LOGIT_TOP_K:
        raise FullCoreAcquisitionError("source-logit top-k contract changed")
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        source_logits = source_teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    response_mask = labels[:, 1:] >= 0
    response_positions = int(response_mask.sum().item())
    if response_positions <= 0:
        raise FullCoreAcquisitionError(
            "same-tokenizer source distillation observed no response token"
        )
    student_response = student_logits[:, :-1][response_mask].float()
    source_response = source_logits[:, :-1][response_mask].float()
    top_values, top_indices = source_response.topk(top_k, dim=-1)
    source_probabilities = torch.softmax(top_values, dim=-1)
    student_log_probabilities = torch.log_softmax(
        student_response, dim=-1
    ).gather(-1, top_indices)
    loss = -(
        source_probabilities * student_log_probabilities
    ).sum(dim=-1).mean()
    return (
        loss,
        int(attention_mask.sum().item()),
        response_positions,
        inference_seconds,
    )


def _same_tokenizer_representation_distillation_loss(
    *,
    student_logits: torch.Tensor,
    student_block_hidden_states: Sequence[torch.Tensor],
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    source_teacher: torch.nn.Module,
    top_k: int,
    selected_source_layers: Sequence[int] = (
        SAME_TOKENIZER_REPRESENTATION_LAYERS
    ),
) -> tuple[torch.Tensor, torch.Tensor, int, int, float]:
    """Match source token distributions and mapped intermediate states online."""

    if top_k != SAME_TOKENIZER_LOGIT_TOP_K:
        raise FullCoreAcquisitionError("source-logit top-k contract changed")
    if len(student_block_hidden_states) != len(selected_source_layers):
        raise FullCoreAcquisitionError(
            "student/source representation depth mapping changed"
        )
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        source_result = source_teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    response_mask = labels[:, 1:] >= 0
    response_positions = int(response_mask.sum().item())
    if response_positions <= 0:
        raise FullCoreAcquisitionError(
            "same-tokenizer representation distillation observed no response token"
        )
    student_response = student_logits[:, :-1][response_mask].float()
    source_response = source_result.logits[:, :-1][response_mask].float()
    top_values, top_indices = source_response.topk(top_k, dim=-1)
    source_probabilities = torch.softmax(top_values, dim=-1)
    student_log_probabilities = torch.log_softmax(
        student_response, dim=-1
    ).gather(-1, top_indices)
    logit_loss = -(
        source_probabilities * student_log_probabilities
    ).sum(dim=-1).mean()

    hidden_losses = []
    source_hidden_states = source_result.hidden_states
    for student_hidden, source_layer in zip(
        student_block_hidden_states,
        selected_source_layers,
        strict=True,
    ):
        source_hidden = source_hidden_states[int(source_layer) + 1]
        student_values = F.normalize(
            student_hidden[:, :-1][response_mask].float(), dim=-1
        )
        source_values = F.normalize(
            source_hidden[:, :-1][response_mask].float(), dim=-1
        )
        hidden_losses.append(
            (1.0 - (student_values * source_values).sum(dim=-1)).mean()
        )
    representation_loss = torch.stack(hidden_losses).mean()
    return (
        logit_loss,
        representation_loss,
        int(attention_mask.sum().item()),
        response_positions,
        inference_seconds,
    )


def _parent_layercake_topk_preservation_loss(
    *,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    task_routes: torch.Tensor,
    parent_teacher: torch.nn.Module,
    top_k: int = SAME_TOKENIZER_LOGIT_TOP_K,
) -> tuple[torch.Tensor, int, int, float]:
    """Preserve a frozen parent LayerCake distribution while fitting its cakes."""

    if top_k != SAME_TOKENIZER_LOGIT_TOP_K:
        raise FullCoreAcquisitionError("parent-preservation top-k changed")
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        parent_logits = parent_teacher(
            input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            task_routes=task_routes,
            use_cache=False,
        )["logits"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    response_mask = labels[:, 1:] >= 0
    response_positions = int(response_mask.sum().item())
    if response_positions <= 0:
        raise FullCoreAcquisitionError(
            "parent preservation observed no response token"
        )
    student_response = student_logits[:, :-1][response_mask].float()
    parent_response = parent_logits[:, :-1][response_mask].float()
    parent_values, parent_indices = parent_response.topk(top_k, dim=-1)
    parent_probabilities = torch.softmax(parent_values, dim=-1)
    student_log_probabilities = torch.log_softmax(
        student_response, dim=-1
    ).gather(-1, parent_indices)
    loss = -(
        parent_probabilities * student_log_probabilities
    ).sum(dim=-1).mean()
    return (
        loss,
        int(attention_mask.sum().item()),
        response_positions,
        inference_seconds,
    )


class _DeterministicRowSampler:
    """Draw records uniformly or give every declared capability equal weight."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        strategy: str,
    ) -> None:
        if strategy not in {"uniform_records", "balanced_capabilities"}:
            raise FullCoreAcquisitionError("unknown sampling strategy")
        if not rows:
            raise FullCoreAcquisitionError("cannot sample an empty corpus")
        self.rows = rows
        self.rng = random.Random(seed)
        self.strategy = strategy
        self.order = list(range(len(rows)))
        self.cursor = len(self.order)
        self.by_capability: dict[str, list[int]] = {}
        self.capability_cursors: dict[str, int] = {}
        for index, row in enumerate(rows):
            self.by_capability.setdefault(
                str(row["capability"]), []
            ).append(index)
        self.capability_order = sorted(self.by_capability)
        self.capability_cursor = len(self.capability_order)
        for capability, indices in self.by_capability.items():
            self.capability_cursors[capability] = len(indices)

    def _next_uniform(self) -> int:
        if self.cursor >= len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def _next_capability(self) -> str:
        if self.capability_cursor >= len(self.capability_order):
            self.rng.shuffle(self.capability_order)
            self.capability_cursor = 0
        capability = self.capability_order[self.capability_cursor]
        self.capability_cursor += 1
        return capability

    def _next_balanced(self) -> int:
        capability = self._next_capability()
        indices = self.by_capability[capability]
        cursor = self.capability_cursors[capability]
        if cursor >= len(indices):
            self.rng.shuffle(indices)
            cursor = 0
        index = indices[cursor]
        self.capability_cursors[capability] = cursor + 1
        return index

    def batch(self, size: int) -> list[Mapping[str, Any]]:
        if size <= 0:
            raise FullCoreAcquisitionError("sample size must be positive")
        selector = (
            self._next_balanced
            if self.strategy == "balanced_capabilities"
            else self._next_uniform
        )
        return [self.rows[selector()] for _ in range(size)]

    def snapshot(self) -> dict[str, Any]:
        """Capture deterministic sampler state before a tentative AMP step."""

        return {
            "rng_state": self.rng.getstate(),
            "order": list(self.order),
            "cursor": self.cursor,
            "by_capability": {
                key: list(value)
                for key, value in self.by_capability.items()
            },
            "capability_cursors": dict(self.capability_cursors),
            "capability_order": list(self.capability_order),
            "capability_cursor": self.capability_cursor,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore a tentative batch that produced no optimizer update."""

        self.rng.setstate(state["rng_state"])
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.by_capability = {
            str(key): list(value)
            for key, value in state["by_capability"].items()
        }
        self.capability_cursors = {
            str(key): int(value)
            for key, value in state["capability_cursors"].items()
        }
        self.capability_order = list(state["capability_order"])
        self.capability_cursor = int(state["capability_cursor"])


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _transformer_control_task_state(model: torch.nn.Module) -> dict[str, Any]:
    """Hash and verify the frozen identity-cake control container."""

    up_weights = [cake.up.weight for cake in model.task_cakes]
    zero_elements = sum(
        int(torch.count_nonzero(weight.detach()).item()) == 0
        for weight in up_weights
    )
    return {
        "task_cakes_state_sha256": module_state_sha256(model.task_cakes),
        "task_classifier_state_sha256": module_state_sha256(
            model.task_classifier
        ),
        "identity_cake_count": zero_elements,
        "all_task_cake_up_weights_zero_exact": (
            zero_elements == len(up_weights)
        ),
    }


def _select_trainable_parameters(
    model: torch.nn.Module,
    *,
    trainable_scope: str,
    trainable_task_cake_routes: Sequence[int] = (),
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    int,
    int,
]:
    """Select full-core, all-cake, or explicitly isolated route parameters."""

    if trainable_scope not in {
        "full_core",
        "task_cakes_only",
        "task_cakes_classifier",
        "selected_task_cakes",
        "expanded_task_cake_tail_classifier",
        "capability_cakes_classifier",
        "persistent_capability_prefix_cakes",
        "layerwise_capability_control_cakes",
        "deep_capability_adapter_cakes",
        "shared_deep_capability_adapter_cakes",
        "deep_reused_capability_cakes",
        "gated_deep_reused_capability_cakes",
        MERGED_ENGLISH_CORE_LORA_SCOPE,
        SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
        SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
        TASK_ROUTE_LAYERWISE_CONTROL_SCOPE,
        TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
        TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
        TRANSFORMER_CORE_CONTROL_SCOPE,
    }:
        raise FullCoreAcquisitionError("unknown trainable scope")
    all_parameters = list(model.parameters())
    task_cakes = list(model.task_cakes)
    task_cake_parameters = [
        parameter
        for cake in task_cakes
        for parameter in cake.parameters()
    ]
    if not task_cake_parameters:
        raise FullCoreAcquisitionError("LayerCake task cakes are incomplete")
    selected_routes = tuple(int(route) for route in trainable_task_cake_routes)
    if len(set(selected_routes)) != len(selected_routes):
        raise FullCoreAcquisitionError(
            "selected task-cake routes must be unique"
        )
    if trainable_scope == "selected_task_cakes":
        if not selected_routes:
            raise FullCoreAcquisitionError(
                "selected task-cake scope requires at least one route"
            )
        if any(
            route < 0 or route >= len(task_cakes)
            for route in selected_routes
        ):
            raise FullCoreAcquisitionError(
                "selected task-cake route is outside the installed topology"
            )
    elif selected_routes:
        raise FullCoreAcquisitionError(
            "task-cake routes are valid only for selected_task_cakes scope"
        )
    for parameter in all_parameters:
        parameter.requires_grad_(
            trainable_scope
            in {
                "full_core",
                SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
                SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
            }
        )
    if trainable_scope == TRANSFORMER_CORE_CONTROL_SCOPE:
        classifier_parameters = list(model.task_classifier.parameters())
        frozen_control_parameters = classifier_parameters + task_cake_parameters
        frozen_control_ids = {
            id(parameter) for parameter in frozen_control_parameters
        }
        shared_parameters = [
            parameter
            for parameter in all_parameters
            if id(parameter) not in frozen_control_ids
        ]
        for parameter in shared_parameters:
            parameter.requires_grad_(True)
        cake_parameters = []
    elif trainable_scope == MERGED_ENGLISH_CORE_LORA_SCOPE:
        if not getattr(model, "_abi_merged_english_core_lora", False):
            raise FullCoreAcquisitionError(
                "merged English-core LoRA scope is not installed"
            )
        shared_parameters = list(model._abi_english_core_lora_parameters)
        for parameter in shared_parameters:
            parameter.requires_grad_(True)
        cake_parameters = []
    elif trainable_scope in {
        "capability_cakes_classifier",
        "persistent_capability_prefix_cakes",
        "layerwise_capability_control_cakes",
        "deep_capability_adapter_cakes",
        "shared_deep_capability_adapter_cakes",
        "deep_reused_capability_cakes",
        "gated_deep_reused_capability_cakes",
    }:
        if (
            int(model.config.task_cake_rank) != 64
            or len(task_cakes) != len(CAPABILITY_CAKE_ORDER)
            or tuple(
                getattr(model, "_abi_capability_cake_routes", ())
            )
            != CAPABILITY_CAKE_CANONICAL_ROUTES
        ):
            raise FullCoreAcquisitionError(
                "capability-cake scope requires the locked 14-by-rank64 topology"
            )
        if trainable_scope in {
            "persistent_capability_prefix_cakes",
            "layerwise_capability_control_cakes",
            "deep_capability_adapter_cakes",
            "shared_deep_capability_adapter_cakes",
            "deep_reused_capability_cakes",
            "gated_deep_reused_capability_cakes",
        }:
            expected_marker = {
                "persistent_capability_prefix_cakes": (
                    "_abi_persistent_capability_prefix"
                ),
                "layerwise_capability_control_cakes": (
                    "_abi_layerwise_capability_control"
                ),
                "deep_capability_adapter_cakes": (
                    "_abi_deep_capability_adapters"
                ),
                "shared_deep_capability_adapter_cakes": (
                    "_abi_shared_deep_capability_adapters"
                ),
                "deep_reused_capability_cakes": (
                    "_abi_deep_reused_capability_cakes"
                ),
                "gated_deep_reused_capability_cakes": (
                    "_abi_gated_deep_reused_capability_cakes"
                ),
            }[trainable_scope]
            if not getattr(model, expected_marker, False):
                raise FullCoreAcquisitionError(
                    "persistent conditioning scope requires its locked topology"
                )
            conditioning_parameters: list[torch.nn.Parameter]
            if trainable_scope == "persistent_capability_prefix_cakes":
                conditioning_parameters = [
                    model.capability_prefix_keys,
                    model.capability_prefix_values,
                ]
            elif trainable_scope == "layerwise_capability_control_cakes":
                conditioning_parameters = [model.capability_control_vectors]
            elif trainable_scope == "deep_capability_adapter_cakes":
                conditioning_parameters = [
                    parameter
                    for layer in model.capability_layer_adapters
                    for adapter in layer
                    for parameter in adapter.parameters()
                ]
            elif trainable_scope == "shared_deep_capability_adapter_cakes":
                conditioning_parameters = [
                    parameter
                    for adapter in model.capability_shared_adapters
                    for parameter in adapter.parameters()
                ]
            elif trainable_scope == "deep_reused_capability_cakes":
                conditioning_parameters = []
            else:
                conditioning_parameters = [
                    model.capability_deep_cake_gates
                ]
            cake_parameters = (
                task_cake_parameters
                + list(model.capability_router_embedding.parameters())
                + list(model.capability_router.parameters())
                + conditioning_parameters
            )
        else:
            cake_parameters = list(model.task_classifier.parameters()) + (
                task_cake_parameters
            )
        for parameter in cake_parameters:
            parameter.requires_grad_(True)
        shared_parameters = []
    elif trainable_scope == "expanded_task_cake_tail_classifier":
        if int(model.config.task_cake_rank) != EXPANDED_TASK_CAKE_RANK:
            raise FullCoreAcquisitionError(
                "expanded task-cake scope requires rank 256"
            )
        classifier_parameters = list(model.task_classifier.parameters())
        cake_parameters = classifier_parameters + [
            parameter
            for cake in task_cakes
            for parameter in (cake.down.weight, cake.up.weight)
        ]
        for parameter in cake_parameters:
            parameter.requires_grad_(True)
        shared_parameters = []
    elif trainable_scope == TASK_ROUTE_LAYERWISE_CONTROL_SCOPE:
        cake_parameters = task_cake_parameters + [
            model.task_route_control_vectors
        ]
        for parameter in cake_parameters:
            parameter.requires_grad_(True)
        shared_parameters = []
    elif trainable_scope in {
        TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
        TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
    }:
        if not getattr(model, "_abi_prompt_identity_carriage", False):
            raise FullCoreAcquisitionError(
                "prompt-identity carriage scope is not installed"
            )
        cake_parameters = list(model.prompt_identity.parameters())
        for parameter in cake_parameters:
            parameter.requires_grad_(True)
        shared_parameters = []
    elif trainable_scope == "task_cakes_classifier":
        cake_parameters = (
            list(model.task_classifier.parameters()) + task_cake_parameters
        )
        for parameter in cake_parameters:
            parameter.requires_grad_(True)
        shared_parameters = []
    elif trainable_scope in {
        "task_cakes_only",
        "selected_task_cakes",
    }:
        cake_parameters = (
            task_cake_parameters
            if trainable_scope == "task_cakes_only"
            else [
                parameter
                for route in selected_routes
                for parameter in task_cakes[route].parameters()
            ]
        )
        for parameter in cake_parameters:
            parameter.requires_grad_(True)
        shared_parameters: list[torch.nn.Parameter] = []
    else:
        classifier_parameters = list(
            model.task_classifier.parameters()
        )
        cake_parameters = (
            classifier_parameters + task_cake_parameters
        )
        cake_ids = {id(parameter) for parameter in cake_parameters}
        shared_parameters = [
            parameter
            for parameter in all_parameters
            if id(parameter) not in cake_ids
        ]
        if not shared_parameters or not cake_parameters:
            raise FullCoreAcquisitionError(
                "LayerCake parameter groups are incomplete"
            )
    total_parameter_count = sum(
        parameter.numel() for parameter in all_parameters
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in all_parameters
        if parameter.requires_grad
    )
    return (
        shared_parameters,
        cake_parameters,
        total_parameter_count,
        trainable_parameter_count,
    )


def _active_parameter_count(parent_metadata: Mapping[str, Any]) -> int:
    """Read the active graph size from original or continued-core metadata."""

    original = parent_metadata.get("parameters")
    if isinstance(original, Mapping):
        value = original.get("active")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    continued = parent_metadata.get("acquired_core")
    if isinstance(continued, Mapping):
        value = continued.get("active_parameter_count")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise FullCoreAcquisitionError(
        "parent metadata does not declare a positive active parameter count"
    )


def _expand_task_cake_rank(
    model: torch.nn.Module,
    *,
    expanded_rank: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    """Add a trainable cake tail while preserving every parent cake value."""

    parent_rank = int(model.config.task_cake_rank)
    if (
        parent_rank != 64
        or expanded_rank != EXPANDED_TASK_CAKE_RANK
        or int(model.config.layers) != 3
        or len(model.task_cakes) != 10
    ):
        raise FullCoreAcquisitionError(
            "expanded cake acquisition requires the three-block rank-64 parent"
        )
    preserved: dict[int, dict[str, torch.Tensor]] = {}
    expanded = []
    for route, parent_cake in enumerate(model.task_cakes):
        cake = type(parent_cake)(
            int(model.config.width), expanded_rank
        ).to(
            device=parent_cake.down.weight.device,
            dtype=parent_cake.down.weight.dtype,
        )
        with torch.no_grad():
            cake.norm.weight.copy_(parent_cake.norm.weight)
            cake.norm.bias.copy_(parent_cake.norm.bias)
            cake.down.weight[:parent_rank].copy_(
                parent_cake.down.weight
            )
            cake.up.weight[:, :parent_rank].copy_(
                parent_cake.up.weight
            )
            cake.up.weight[:, parent_rank:].zero_()
        preserved[route] = {
            "norm_weight": parent_cake.norm.weight.detach().clone(),
            "norm_bias": parent_cake.norm.bias.detach().clone(),
            "down": parent_cake.down.weight.detach().clone(),
            "up": parent_cake.up.weight.detach().clone(),
        }

        def mask_down(
            gradient: torch.Tensor,
            *,
            locked: int = parent_rank,
        ) -> torch.Tensor:
            masked = gradient.clone()
            masked[:locked].zero_()
            return masked

        def mask_up(
            gradient: torch.Tensor,
            *,
            locked: int = parent_rank,
        ) -> torch.Tensor:
            masked = gradient.clone()
            masked[:, :locked].zero_()
            return masked

        cake.down.weight.register_hook(mask_down)
        cake.up.weight.register_hook(mask_up)
        expanded.append(cake)
    model.task_cakes = torch.nn.ModuleList(expanded)
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(model.config.vocab_size),
        width=int(model.config.width),
        layers=int(model.config.layers),
        heads=int(model.config.heads),
        max_tokens=int(model.config.max_tokens),
        task_cakes=int(model.config.task_cakes),
        task_cake_rank=expanded_rank,
        architecture_version=EXPANDED_TASK_CAKE_ARCHITECTURE,
    )
    added_per_cake = 2 * int(model.config.width) * (
        expanded_rank - parent_rank
    )
    return preserved, {
        "parent_rank": parent_rank,
        "expanded_rank": expanded_rank,
        "added_parameters_per_installed_cake": added_per_cake,
        "added_total_parameters": added_per_cake * len(expanded),
        "added_active_parameters": added_per_cake,
        "base_norm_and_rank64_slices_locked": True,
        "new_down_initialization": "deterministic_normal",
        "new_up_initialization": "zero",
        "initial_function_exactly_parent_equivalent": True,
    }


def _expand_capability_cakes(model: torch.nn.Module) -> dict[str, Any]:
    """Split collided canonical routes into one rank-64 cake per capability."""

    if (
        int(model.config.layers) != 3
        or int(model.config.task_cake_rank) != 64
        or len(model.task_cakes) != 10
        or int(model.task_classifier.out_features) != 10
    ):
        raise FullCoreAcquisitionError(
            "capability isolation requires the three-block 10-route parent"
        )
    parent_cakes = list(model.task_cakes)
    parent_classifier = model.task_classifier
    device = parent_classifier.weight.device
    dtype = parent_classifier.weight.dtype
    cakes = []
    for canonical_route in CAPABILITY_CAKE_CANONICAL_ROUTES:
        parent_cake = parent_cakes[int(canonical_route)]
        cake = type(parent_cake)(
            int(model.config.width), 64
        ).to(device=device, dtype=dtype)
        cake.load_state_dict(
            {
                name: value.detach().clone()
                for name, value in parent_cake.state_dict().items()
            },
            strict=True,
        )
        cakes.append(cake)
    classifier = torch.nn.Linear(
        int(model.config.width), len(CAPABILITY_CAKE_ORDER)
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        for capability_index, canonical_route in enumerate(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        ):
            classifier.weight[capability_index].copy_(
                parent_classifier.weight[int(canonical_route)]
            )
            classifier.bias[capability_index].copy_(
                parent_classifier.bias[int(canonical_route)]
            )
    model.task_cakes = torch.nn.ModuleList(cakes)
    model.task_classifier = classifier
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(model.config.vocab_size),
        width=int(model.config.width),
        layers=int(model.config.layers),
        heads=int(model.config.heads),
        max_tokens=int(model.config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        ),
        architecture_version=CAPABILITY_CAKE_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = (
        CAPABILITY_CAKE_CANONICAL_ROUTES
    )
    exact_copies = all(
        all(
            torch.equal(value, parent_cakes[canonical_route].state_dict()[name])
            for name, value in cake.state_dict().items()
        )
        for cake, canonical_route in zip(
            model.task_cakes, CAPABILITY_CAKE_CANONICAL_ROUTES
        )
    )
    exact_classifier_rows = all(
        torch.equal(
            model.task_classifier.weight[index],
            parent_classifier.weight[canonical_route],
        )
        and torch.equal(
            model.task_classifier.bias[index],
            parent_classifier.bias[canonical_route],
        )
        for index, canonical_route in enumerate(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        )
    )
    if not exact_copies or not exact_classifier_rows:
        raise FullCoreAcquisitionError(
            "capability-cake initialization changed its parent route function"
        )
    per_cake = sum(
        parameter.numel() for parameter in model.task_cakes[0].parameters()
    )
    return {
        "capability_order": list(CAPABILITY_CAKE_ORDER),
        "capability_to_canonical_route": list(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        ),
        "parent_canonical_routes": 10,
        "installed_capability_cakes": len(CAPABILITY_CAKE_ORDER),
        "capability_cake_rank": 64,
        "maximum_active_capability_cakes_per_sequence": 1,
        "parent_cake_values_copied_exactly": exact_copies,
        "classifier_rows_copied_exactly": exact_classifier_rows,
        "initial_selected_cake_function_parent_equivalent": True,
        "added_installed_cake_parameters": per_cake * 4,
        "added_active_cake_parameters": 0,
        "canonical_abi_changed": False,
    }


def _restore_expanded_task_cake_base(
    model: torch.nn.Module,
    preserved: Mapping[int, Mapping[str, torch.Tensor]],
) -> bool:
    """Undo optimizer decay on locked slices and prove exact preservation."""

    parent_rank = 64
    with torch.no_grad():
        for route, values in preserved.items():
            cake = model.task_cakes[int(route)]
            cake.norm.weight.copy_(values["norm_weight"])
            cake.norm.bias.copy_(values["norm_bias"])
            cake.down.weight[:parent_rank].copy_(values["down"])
            cake.up.weight[:, :parent_rank].copy_(values["up"])
    return all(
        torch.equal(
            model.task_cakes[int(route)].norm.weight,
            values["norm_weight"],
        )
        and torch.equal(
            model.task_cakes[int(route)].norm.bias,
            values["norm_bias"],
        )
        and torch.equal(
            model.task_cakes[int(route)].down.weight[:parent_rank],
            values["down"],
        )
        and torch.equal(
            model.task_cakes[int(route)].up.weight[:, :parent_rank],
            values["up"],
        )
        for route, values in preserved.items()
    )


def _frozen_shared_state_sha256(model: torch.nn.Module) -> str:
    """Hash every state tensor outside cakes and the trainable classifier."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if (
            name.startswith("task_cakes.")
            or (
                name.startswith("task_classifier.")
                and not getattr(
                    model, "_abi_persistent_capability_prefix", False
                )
                and not getattr(
                    model, "_abi_layerwise_capability_control", False
                )
                and not getattr(
                    model, "_abi_deep_capability_adapters", False
                )
                and not getattr(
                    model, "_abi_deep_reused_capability_cakes", False
                )
                and not getattr(
                    model,
                    "_abi_gated_deep_reused_capability_cakes",
                    False,
                )
            )
            or name.startswith("capability_router_embedding.")
            or name.startswith("capability_router.")
            or name.startswith("capability_layer_adapters.")
            or name.startswith("capability_shared_adapters.")
            or name == "capability_deep_cake_gates"
            or name.startswith("prompt_identity.")
            or name in {
                "capability_prefix_keys",
                "capability_prefix_values",
                "capability_control_vectors",
                "task_route_control_vectors",
            }
        ):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_sha256_excluding_prefixes(
    model: torch.nn.Module,
    prefixes: Sequence[str],
) -> str:
    """Hash an exact parent state while excluding newly installed modules."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        if tensor.dtype is torch.bfloat16:
            digest.update(tensor.view(torch.int16).numpy().tobytes())
        else:
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fake_quantize_symmetric_per_channel(
    value: torch.Tensor,
    *,
    channel_axis: int,
) -> torch.Tensor:
    """Model deployment QInt8 weights while retaining GPU gradients downstream."""

    if value.ndim != 2 or channel_axis not in {0, 1}:
        raise FullCoreAcquisitionError(
            "runtime fake-int8 supports only two-dimensional channel weights"
        )
    reduction_axis = 1 - channel_axis
    scale = torch.clamp(
        value.detach().abs().amax(
            dim=reduction_axis,
            keepdim=True,
        )
        / 127.0,
        min=torch.finfo(torch.float32).eps,
    )
    return torch.clamp(
        torch.round(value.detach() / scale),
        min=-127,
        max=127,
    ) * scale


def _apply_frozen_runtime_fake_int8(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Temporarily fake-quantize frozen deployment matrices in-place."""

    originals: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}
    expected_suffixes = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        if name == "transformer.wte.weight":
            channel_axis = 0
        elif (
            name.startswith("transformer.h.")
            and name.endswith(expected_suffixes)
        ):
            channel_axis = 1
        else:
            continue
        originals[name] = parameter.detach().cpu().clone()
        quantized = _fake_quantize_symmetric_per_channel(
            parameter,
            channel_axis=channel_axis,
        )
        errors[name] = float(
            (parameter.detach() - quantized).abs().max().item()
        )
        parameter.data.copy_(quantized)
    if len(originals) != 13 or "transformer.wte.weight" not in originals:
        raise FullCoreAcquisitionError(
            "runtime fake-int8 did not isolate the expected embedding and "
            "twelve transformer deployment matrices"
        )
    return originals, {
        "method": "symmetric_qint8_per_deployment_channel",
        "fake_quantized_tensor_count": len(originals),
        "fake_quantized_tensors": sorted(originals),
        "maximum_absolute_error_by_tensor": dict(sorted(errors.items())),
        "embedding_channel_axis": 0,
        "transformer_matrix_channel_axis": 1,
        "activation_quantization_modeled": False,
        "frozen_weights_restored_before_checkpoint": True,
    }


def _restore_frozen_runtime_weights(
    model: torch.nn.Module,
    originals: Mapping[str, torch.Tensor],
) -> bool:
    """Restore and prove exact frozen weights before hashing or saving."""

    parameters = dict(model.named_parameters())
    if set(originals) - set(parameters):
        raise FullCoreAcquisitionError(
            "runtime fake-int8 restore references missing parameters"
        )
    for name, original in originals.items():
        parameters[name].data.copy_(
            original.to(
                device=parameters[name].device,
                dtype=parameters[name].dtype,
            )
        )
    return all(
        torch.equal(parameters[name].detach().cpu(), original)
        for name, original in originals.items()
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_decoding_contract(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoding = payload.get("decoding")
    required = {
        "algorithm",
        "no_repeat_ngram_size",
        "allow_prompt_ngrams",
        "lexical_repetition_blocking_threshold",
        "lexical_repetition_truncation_threshold",
        "byte_repetition_ceiling",
        "byte_repetition_guard_minimum_bytes",
        "prompt_identity_mixture",
    }
    if (
        not isinstance(decoding, dict)
        or set(decoding) != required
        or decoding["algorithm"]
        not in {"greedy", "deterministic_greedy_with_repetition_controls"}
        or isinstance(decoding["no_repeat_ngram_size"], bool)
        or not isinstance(decoding["no_repeat_ngram_size"], int)
        or decoding["no_repeat_ngram_size"] < 0
        or not isinstance(decoding["allow_prompt_ngrams"], bool)
        or isinstance(
            decoding["lexical_repetition_blocking_threshold"], bool
        )
        or not isinstance(
            decoding["lexical_repetition_blocking_threshold"], int
        )
        or decoding["lexical_repetition_blocking_threshold"] < 0
        or isinstance(
            decoding["lexical_repetition_truncation_threshold"], bool
        )
        or not isinstance(
            decoding["lexical_repetition_truncation_threshold"], int
        )
        or decoding["lexical_repetition_truncation_threshold"] < 0
        or isinstance(decoding["byte_repetition_ceiling"], bool)
        or not isinstance(decoding["byte_repetition_ceiling"], (int, float))
        or not 0.0 <= float(decoding["byte_repetition_ceiling"]) <= 1.0
        or isinstance(
            decoding["byte_repetition_guard_minimum_bytes"], bool
        )
        or not isinstance(
            decoding["byte_repetition_guard_minimum_bytes"], int
        )
        or decoding["byte_repetition_guard_minimum_bytes"] < 0
        or decoding["prompt_identity_mixture"] is not False
        or (
            decoding["allow_prompt_ngrams"]
            and decoding["no_repeat_ngram_size"] <= 0
        )
    ):
        raise FullCoreAcquisitionError("decoding contract is invalid")
    return dict(decoding)


def _general_preservation_rows(path: Path) -> list[dict[str, Any]]:
    """Bind the locked knowledge-light curriculum to canonical cake routes."""

    rows = _load_general_rows(path, split="train")
    prepared = []
    for row in rows:
        task = str(row["task"])
        capability = GENERAL_TASK_TO_CAPABILITY.get(task)
        if capability is None:
            raise FullCoreAcquisitionError(
                f"general preservation task is unmapped: {task}"
            )
        prepared.append(
            {
                "record_id": f"general-preservation:{row['id']}",
                "capability": capability,
                "route": CAPABILITY_TO_ROUTE[capability],
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "teacher_tokens": int(row.get("teacher_tokens", 0)),
                "provenance": str(
                    row.get(
                        "provenance",
                        "sealed-layercake-knowledge-light-preservation",
                    )
                ),
            }
        )
    return prepared


def _filter_context_compatible_rows(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    exclude_overlength_prompts: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preflight the immutable source rows against the exact host tokenizer."""

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        prompt_tokens = len(
            tokenizer.encode(str(row["prompt"]) + "\n")
        )
        if prompt_tokens >= max_tokens:
            excluded.append(
                {
                    "record_id": str(row["record_id"]),
                    "capability": str(row["capability"]),
                    "prompt_tokens": prompt_tokens,
                    "teacher_tokens": int(row["teacher_tokens"]),
                }
            )
        else:
            retained.append(dict(row))
    excluded.sort(key=lambda row: row["record_id"])
    if excluded and not exclude_overlength_prompts:
        raise FullCoreAcquisitionError(
            f"{len(excluded)} source prompts exceed the locked context "
            "budget; exclusion was not authorized"
        )
    original_capabilities = {
        str(row["capability"]) for row in rows
    }
    retained_capabilities = {
        str(row["capability"]) for row in retained
    }
    if retained_capabilities != original_capabilities:
        raise FullCoreAcquisitionError(
            "context compatibility exclusion removed complete capability "
            "coverage"
        )
    excluded_ids = "\n".join(
        str(row["record_id"]) for row in excluded
    ).encode("utf-8")
    return retained, {
        "policy": (
            "exclude_prompt_token_count_greater_than_or_equal_to_max_tokens"
            if exclude_overlength_prompts
            else "fail_closed"
        ),
        "bound_max_tokens": max_tokens,
        "original_record_count": len(rows),
        "retained_record_count": len(retained),
        "excluded_record_count": len(excluded),
        "excluded_teacher_tokens": sum(
            int(row["teacher_tokens"]) for row in excluded
        ),
        "excluded_record_ids_sha256": hashlib.sha256(
            excluded_ids
        ).hexdigest(),
        "excluded_records": excluded,
    }


def _apply_target_control(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a deterministic response derangement for a causal control run."""

    if mode not in {"identity", "deterministic_derangement"}:
        raise FullCoreAcquisitionError("unknown target-control mode")
    materialized = [dict(row) for row in rows]
    if mode == "identity":
        return materialized, {
            "mode": mode,
            "seed": None,
            "derangement_offset": 0,
            "all_targets_changed": False,
            "mapping_sha256": None,
        }
    if not seed.strip() or len(materialized) < 2:
        raise FullCoreAcquisitionError(
            "target derangement requires a seed and at least two rows"
        )
    ordered = sorted(
        materialized,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['record_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected_offset = None
    for offset in range(1, len(ordered)):
        if all(
            str(row["response"])
            != str(ordered[(index + offset) % len(ordered)]["response"])
            for index, row in enumerate(ordered)
        ):
            selected_offset = offset
            break
    if selected_offset is None:
        raise FullCoreAcquisitionError(
            "no exact response derangement exists for these rows"
        )
    mapping = []
    for index, row in enumerate(ordered):
        donor = ordered[(index + selected_offset) % len(ordered)]
        original_response = str(row["response"])
        row["response"] = str(donor["response"])
        mapping.append(
            {
                "record_id": str(row["record_id"]),
                "donor_record_id": str(donor["record_id"]),
                "original_response_sha256": hashlib.sha256(
                    original_response.encode("utf-8")
                ).hexdigest(),
                "controlled_response_sha256": hashlib.sha256(
                    str(row["response"]).encode("utf-8")
                ).hexdigest(),
            }
        )
    mapping_sha256 = hashlib.sha256(
        _canonical_json_bytes(mapping)
    ).hexdigest()
    return sorted(ordered, key=lambda row: str(row["record_id"])), {
        "mode": mode,
        "seed": seed,
        "derangement_offset": selected_offset,
        "all_targets_changed": True,
        "mapping_sha256": mapping_sha256,
        "mapping_count": len(mapping),
    }


def train_full_core(
    *,
    bundle_path: str | Path,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    budget_index: int,
    seed: int,
    steps: int,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    trainable_scope: str = "full_core",
    trainable_task_cake_routes: Sequence[int] = (),
    expanded_task_cake_rank: int | None = None,
    shared_learning_rate: float = 2.0e-5,
    cake_learning_rate: float = 1.0e-4,
    classifier_loss_weight: float = 0.25,
    prompt_overlap_loss_weight: float = 1.0,
    max_tokens: int = 256,
    recovery_start_step: int = 400,
    recovery_interval: int = 8,
    recovery_horizons: Sequence[int] = (8, 16, 32),
    sampling_strategy: str = "uniform_records",
    general_curriculum_path: str | Path | None = None,
    general_batch_size: int = 0,
    general_loss_weight: float = 1.0,
    general_sampling_strategy: str = "uniform_records",
    anchor_bundle_path: str | Path | None = None,
    anchor_budget_index: int = -1,
    anchor_batch_size: int = 0,
    anchor_loss_weight: float = 1.0,
    anchor_sampling_strategy: str = "balanced_capabilities",
    decoding_contract_path: str | Path | None = None,
    exclude_overlength_prompts: bool = False,
    frozen_runtime_fake_int8: bool = False,
    same_tokenizer_source_path: str | Path | None = None,
    source_distillation_weight: float = 0.5,
    source_distillation_top_k: int = SAME_TOKENIZER_LOGIT_TOP_K,
    source_representation_distillation_weight: float = 1.0,
    parent_logit_preservation_weight: float = 0.0,
    prompt_identity_loss_weight: float = 0.0,
    target_control_mode: str = "identity",
    target_derangement_seed: str = "",
    device_name: str = "cuda",
) -> dict[str, Any]:
    if min(
        steps,
        batch_size,
        gradient_accumulation_steps,
        max_tokens,
    ) <= 0:
        raise FullCoreAcquisitionError(
            "steps, batch size, gradient accumulation, and token limit "
            "must be positive"
        )
    if target_control_mode not in {
        "identity",
        "deterministic_derangement",
    }:
        raise FullCoreAcquisitionError("unknown target-control mode")
    if (
        target_control_mode == "deterministic_derangement"
    ) != bool(target_derangement_seed.strip()):
        raise FullCoreAcquisitionError(
            "target derangement mode and seed must be supplied together"
        )
    if min(shared_learning_rate, cake_learning_rate) <= 0:
        raise FullCoreAcquisitionError("learning rates must be positive")
    if classifier_loss_weight < 0 or prompt_overlap_loss_weight < 0:
        raise FullCoreAcquisitionError("loss weights must be non-negative")
    if prompt_identity_loss_weight < 0:
        raise FullCoreAcquisitionError(
            "prompt-identity loss weight must be non-negative"
        )
    if recovery_start_step < 0 or recovery_interval < 0:
        raise FullCoreAcquisitionError("recovery schedule is invalid")
    if recovery_interval and (
        not recovery_horizons
        or any(int(horizon) <= 0 for horizon in recovery_horizons)
    ):
        raise FullCoreAcquisitionError("recovery horizons must be positive")
    if general_batch_size < 0 or general_loss_weight <= 0:
        raise FullCoreAcquisitionError(
            "general preservation size or loss weight is invalid"
        )
    if (general_curriculum_path is None) != (general_batch_size == 0):
        raise FullCoreAcquisitionError(
            "general curriculum and positive batch size must be supplied together"
        )
    if anchor_batch_size < 0 or anchor_loss_weight <= 0:
        raise FullCoreAcquisitionError(
            "anchor batch size or loss weight is invalid"
        )
    if (anchor_bundle_path is None) != (anchor_batch_size == 0):
        raise FullCoreAcquisitionError(
            "anchor bundle and positive batch size must be supplied together"
        )
    if anchor_sampling_strategy not in {
        "uniform_records",
        "balanced_capabilities",
    }:
        raise FullCoreAcquisitionError("unknown anchor sampling strategy")
    if sampling_strategy not in {
        "uniform_records",
        "balanced_capabilities",
    }:
        raise FullCoreAcquisitionError("unknown sampling strategy")
    if general_sampling_strategy not in {
        "uniform_records",
        "balanced_capabilities",
    }:
        raise FullCoreAcquisitionError(
            "unknown general sampling strategy"
        )
    if trainable_scope not in {
        "full_core",
        "task_cakes_only",
        "task_cakes_classifier",
        "selected_task_cakes",
        "expanded_task_cake_tail_classifier",
        "capability_cakes_classifier",
        "persistent_capability_prefix_cakes",
        "layerwise_capability_control_cakes",
        "deep_capability_adapter_cakes",
        "shared_deep_capability_adapter_cakes",
        "deep_reused_capability_cakes",
        "gated_deep_reused_capability_cakes",
        MERGED_ENGLISH_CORE_LORA_SCOPE,
        SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
        SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
        TASK_ROUTE_LAYERWISE_CONTROL_SCOPE,
        TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
        TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
        TRANSFORMER_CORE_CONTROL_SCOPE,
    }:
        raise FullCoreAcquisitionError("unknown trainable scope")
    if (
        trainable_scope == TRANSFORMER_CORE_CONTROL_SCOPE
        and classifier_loss_weight != 0.0
    ):
        raise FullCoreAcquisitionError(
            "transformer-core control requires zero classifier loss"
        )
    if trainable_scope == "expanded_task_cake_tail_classifier":
        if expanded_task_cake_rank != EXPANDED_TASK_CAKE_RANK:
            raise FullCoreAcquisitionError(
                "expanded task-cake scope requires --expanded-task-cake-rank 256"
            )
    elif expanded_task_cake_rank is not None:
        raise FullCoreAcquisitionError(
            "expanded task-cake rank requires the isolated expansion scope"
        )
    selected_task_cake_routes = tuple(
        int(route) for route in trainable_task_cake_routes
    )
    if (
        trainable_scope == "selected_task_cakes"
        and not selected_task_cake_routes
    ):
        raise FullCoreAcquisitionError(
            "selected task-cake scope requires at least one route"
        )
    if (
        trainable_scope != "selected_task_cakes"
        and selected_task_cake_routes
    ):
        raise FullCoreAcquisitionError(
            "task-cake routes require selected_task_cakes scope"
        )
    if frozen_runtime_fake_int8 and (
        trainable_scope != "selected_task_cakes"
        or device_name != "cuda"
    ):
        raise FullCoreAcquisitionError(
            "frozen runtime fake-int8 requires CUDA selected-task-cake training"
        )
    if (
        trainable_scope == MERGED_ENGLISH_CORE_LORA_SCOPE
        and device_name != "cuda"
    ):
        raise FullCoreAcquisitionError(
            "ABI English-core acquisition is GPU-trained; CPU is certification-only"
        )
    if trainable_scope == "task_cakes_classifier" and device_name != "cuda":
        raise FullCoreAcquisitionError(
            "LayerCake sparse conformance acquisition requires CUDA"
        )
    if (
        trainable_scope == TASK_ROUTE_LAYERWISE_CONTROL_SCOPE
        and device_name != "cuda"
    ):
        raise FullCoreAcquisitionError(
            "task-route layerwise-control acquisition requires CUDA"
        )
    prompt_identity_enabled = trainable_scope in {
        TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
        TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
    }
    selective_prompt_identity_enabled = (
        trainable_scope == TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE
    )
    if prompt_identity_enabled and (
        device_name != "cuda" or prompt_identity_loss_weight <= 0
    ):
        raise FullCoreAcquisitionError(
            "prompt-identity carriage requires CUDA and a positive direct loss"
        )
    if not prompt_identity_enabled and prompt_identity_loss_weight:
        raise FullCoreAcquisitionError(
            "prompt-identity loss requires its isolated carriage scope"
        )
    parent_logit_preservation_enabled = (
        _validate_parent_logit_preservation_configuration(
            weight=parent_logit_preservation_weight,
            trainable_scope=trainable_scope,
            device_name=device_name,
        )
    )
    source_representation_distillation_enabled = (
        trainable_scope
        == SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE
    )
    source_distillation_enabled = trainable_scope in {
        SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
        SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
    }
    if source_distillation_enabled:
        if same_tokenizer_source_path is None or device_name != "cuda":
            raise FullCoreAcquisitionError(
                "same-tokenizer source distillation requires a local source and CUDA"
            )
        if (
            source_distillation_weight <= 0
            or source_distillation_top_k != SAME_TOKENIZER_LOGIT_TOP_K
            or (
                source_representation_distillation_enabled
                and source_representation_distillation_weight <= 0
            )
        ):
            raise FullCoreAcquisitionError(
                "same-tokenizer source distillation contract changed"
            )
    elif same_tokenizer_source_path is not None:
        raise FullCoreAcquisitionError(
            "a same-tokenizer source requires its isolated trainable scope"
        )

    bundle_path = Path(bundle_path).resolve()
    general_curriculum_path = (
        Path(general_curriculum_path).resolve()
        if general_curriculum_path is not None
        else None
    )
    anchor_bundle_path = (
        Path(anchor_bundle_path).resolve()
        if anchor_bundle_path is not None
        else None
    )
    decoding_contract_path = (
        Path(decoding_contract_path).resolve()
        if decoding_contract_path is not None
        else None
    )
    decoding_contract = _load_decoding_contract(decoding_contract_path)
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    same_tokenizer_source_path = (
        Path(same_tokenizer_source_path).resolve()
        if same_tokenizer_source_path is not None
        else None
    )
    parent_in_sealed_tree = _is_within(parent_path, layercake_root)
    abi_root = Path(__file__).resolve().parents[1]
    if not parent_in_sealed_tree and not _is_within(parent_path, abi_root):
        raise FullCoreAcquisitionError(
            "parent must belong to LayerCake or the ABI evidence tree"
        )
    if _is_within(output_path, layercake_root):
        raise FullCoreAcquisitionError("ABI acquisition may not modify the sealed LayerCake tree")
    if output_path.exists():
        raise FullCoreAcquisitionError(f"core artifact is immutable: {output_path}")
    if not canonical_abi_path.is_file():
        raise FullCoreAcquisitionError("canonical semantic ABI is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise FullCoreAcquisitionError("CUDA was requested but is unavailable")

    archive_sha_before = _sha256_file(bundle_path)
    rows, budget, bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    rows, target_control = _apply_target_control(
        rows,
        mode=target_control_mode,
        seed=target_derangement_seed,
    )
    anchor_archive_sha_before = (
        _sha256_file(anchor_bundle_path)
        if anchor_bundle_path is not None
        else None
    )
    if anchor_bundle_path is not None:
        anchor_rows, anchor_budget, anchor_bundle = (
            load_english_training_rows(
                anchor_bundle_path,
                budget_index=anchor_budget_index,
            )
        )
    else:
        anchor_rows, anchor_budget, anchor_bundle = [], None, None
    general_rows = (
        _general_preservation_rows(general_curriculum_path)
        if general_curriculum_path is not None
        else []
    )
    verification = bundle["verification"]
    if (
        verification["domain_segregation_verified"] is not True
        or verification["training_eligible"] is not True
    ):
        raise FullCoreAcquisitionError("training bundle did not pass segregation")
    if anchor_bundle is not None:
        anchor_verification = anchor_bundle["verification"]
        if (
            anchor_verification["domain_segregation_verified"] is not True
            or anchor_verification["training_eligible"] is not True
        ):
            raise FullCoreAcquisitionError(
                "anchor bundle did not pass segregation"
            )

    parent_metadata_path = parent_path / "metadata.json"
    parent_checkpoint_path = parent_path / "model.safetensors"
    parent_metadata = json.loads(
        parent_metadata_path.read_text(encoding="utf-8")
    )
    parent_checkpoint_sha = _sha256_file(parent_checkpoint_path)
    if parent_metadata["checkpoint"]["sha256"] != parent_checkpoint_sha:
        raise FullCoreAcquisitionError("sealed parent checkpoint hash changed")
    parent_active_parameter_count = _active_parameter_count(parent_metadata)
    allowed_abi_parent_formats = {
        "abi-layercake-capability-naive-training-base/1",
        "abi-layercake-six-block-capacity-base/1",
        "abi-layercake-three-block-depth-compression-base/1",
        DIRECT_SOURCE_BASE_FORMAT,
        ARTIFACT_FORMAT,
    }
    if not parent_in_sealed_tree and (
        parent_metadata.get("format") not in allowed_abi_parent_formats
        or parent_metadata.get("canonical_semantic_abi", {}).get("sha256")
        != _sha256_file(canonical_abi_path)
    ):
        raise FullCoreAcquisitionError(
            "ABI-owned parent is not an allowed hash-bound LayerCake base"
        )

    device = torch.device(device_name)
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    _import_layercake_runtime(layercake_root)
    model, tokenizer, _ = load_layercake_core(
        parent_path,
        layercake_root=layercake_root,
        device=device,
    )
    transformer_student_control: dict[str, Any] | None = None
    transformer_control_task_state_before: dict[str, Any] | None = None
    if trainable_scope == TRANSFORMER_CORE_CONTROL_SCOPE:
        transformer_student_control = copy.deepcopy(
            parent_metadata.get("transformer_student_control")
        )
        transformer_control_task_state_before = (
            _transformer_control_task_state(model)
        )
        if (
            not isinstance(transformer_student_control, dict)
            or transformer_student_control.get("role")
            != "SHARED_INITIALIZATION_SAME_BUDGET_TRANSFORMER_STUDENT_CONTROL"
            or transformer_student_control.get(
                "task_cake_effect_disabled_exact"
            )
            is not True
            or transformer_control_task_state_before[
                "all_task_cake_up_weights_zero_exact"
            ]
            is not True
        ):
            raise FullCoreAcquisitionError(
                "transformer-core control parent is not an exact identity-cake control"
            )
    direct_source_contract = copy.deepcopy(
        parent_metadata.get("direct_source_initialization")
    )
    direct_source_checkpoint_path: Path | None = None
    if parent_metadata.get("format") == DIRECT_SOURCE_BASE_FORMAT:
        if not isinstance(direct_source_contract, dict):
            raise FullCoreAcquisitionError(
                "direct-source parent omitted its import ledger"
            )
        copied_parameters = int(
            direct_source_contract.get(
                "source_parameters_copied_at_initialization", 0
            )
        )
        copied_blocks = int(
            direct_source_contract.get(
                "source_transformer_blocks_retained_exact_at_initialization",
                0,
            )
        )
        copied_tensors = direct_source_contract.get("copied_target_tensors")
        if (
            copied_parameters <= 0
            or copied_blocks != int(model.config.layers)
            or not isinstance(copied_tensors, list)
            or not copied_tensors
            or parent_metadata.get("foreign_source_boundary", {}).get(
                "source_parameters_copied"
            )
            != copied_parameters
        ):
            raise FullCoreAcquisitionError(
                "direct-source parent accounting changed"
            )
        direct_source_checkpoint_path = Path(
            direct_source_contract["source_path_at_initialization"]
        ).resolve() / "model.safetensors"
        if (
            not direct_source_checkpoint_path.is_file()
            or _sha256_file(direct_source_checkpoint_path)
            != direct_source_contract["source_checkpoint_sha256"]
        ):
            raise FullCoreAcquisitionError(
                "direct-source checkpoint identity changed"
            )
    elif direct_source_contract is not None:
        raise FullCoreAcquisitionError(
            "non-direct parent unexpectedly declares source initialization"
        )
    source_teacher: torch.nn.Module | None = None
    source_distillation_contract: dict[str, Any] | None = None
    if source_distillation_enabled:
        assert same_tokenizer_source_path is not None
        source_checkpoint_path = (
            same_tokenizer_source_path / "model.safetensors"
        )
        source_tokenizer_path = (
            same_tokenizer_source_path / "tokenizer.json"
        )
        source_config_path = same_tokenizer_source_path / "config.json"
        for required in (
            source_checkpoint_path,
            source_tokenizer_path,
            source_config_path,
        ):
            if not required.is_file():
                raise FullCoreAcquisitionError(
                    f"same-tokenizer source file is absent: {required.name}"
                )
        source_tokenizer = AutoTokenizer.from_pretrained(
            same_tokenizer_source_path,
            local_files_only=True,
        )
        if (
            source_tokenizer.get_vocab() != tokenizer.get_vocab()
            or source_tokenizer.eos_token_id != tokenizer.eos_token_id
        ):
            raise FullCoreAcquisitionError(
                "source and LayerCake token-id vocabularies are not exact"
            )
        source_teacher = AutoModelForCausalLM.from_pretrained(
            same_tokenizer_source_path,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to(device)
        source_teacher.eval()
        source_teacher.requires_grad_(False)
        if (
            int(source_teacher.config.vocab_size)
            != int(model.config.vocab_size)
            or int(
                getattr(
                    source_teacher.config,
                    "n_positions",
                    getattr(source_teacher.config, "max_position_embeddings", 0),
                )
            )
            < max_tokens
        ):
            raise FullCoreAcquisitionError(
                "same-tokenizer source vocabulary or context is incompatible"
            )
        source_distillation_contract = {
            "method": (
                "online_same_tokenizer_topk_and_mapped_hidden_state_distillation"
                if source_representation_distillation_enabled
                else "online_same_tokenizer_topk_next_token_distillation"
            ),
            "path_at_training": str(same_tokenizer_source_path),
            "checkpoint_sha256": _sha256_file(source_checkpoint_path),
            "checkpoint_bytes": source_checkpoint_path.stat().st_size,
            "config_sha256": _sha256_file(source_config_path),
            "tokenizer_sha256": _sha256_file(source_tokenizer_path),
            "token_id_vocabulary_exact": True,
            "source_parameter_count": sum(
                parameter.numel()
                for parameter in source_teacher.parameters()
            ),
            "source_precision_during_acquisition": "float16",
            "top_k": source_distillation_top_k,
            "loss_weight": source_distillation_weight,
            "representation_loss_weight": (
                source_representation_distillation_weight
                if source_representation_distillation_enabled
                else 0.0
            ),
            "selected_source_hidden_layer_indices": (
                list(SAME_TOKENIZER_REPRESENTATION_LAYERS)
                if source_representation_distillation_enabled
                else []
            ),
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "source_transformer_blocks_retained": 0,
            "teacher_present_at_inference": False,
            "source_teacher_forward_tokens": 0,
            "source_teacher_response_positions_distilled": 0,
            "source_model_inference_seconds": 0.0,
        }
        del source_tokenizer
    parent_preservation_teacher: torch.nn.Module | None = None
    parent_preservation_contract: dict[str, Any] | None = None
    if parent_logit_preservation_enabled:
        parent_preservation_teacher = copy.deepcopy(model)
        parent_preservation_teacher.eval()
        parent_preservation_teacher.requires_grad_(False)
        parent_preservation_contract = {
            "method": "online_frozen_parent_top64_response_distribution_preservation",
            "parent_path_at_training": str(parent_path),
            "parent_checkpoint_sha256": parent_checkpoint_sha,
            "weight": parent_logit_preservation_weight,
            "top_k": SAME_TOKENIZER_LOGIT_TOP_K,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "parent_parameter_aliases_in_student": 0,
            "parent_teacher_present_at_inference": False,
            "parent_teacher_forward_tokens": 0,
            "parent_teacher_response_positions": 0,
            "parent_teacher_inference_seconds": 0.0,
        }
    architecture_manifest = copy.deepcopy(parent_metadata["architecture"])
    pre_training_deployment_state_sha = module_state_sha256(model)
    expanded_base: dict[int, dict[str, torch.Tensor]] = {}
    task_cake_expansion: dict[str, Any] | None = None
    capability_cake_expansion: dict[str, Any] | None = None
    task_route_layerwise_control: dict[str, Any] | None = None
    prompt_identity_parent_state_sha256_before: str | None = None
    english_core_lora: dict[str, Any] | None = None
    if trainable_scope == "expanded_task_cake_tail_classifier":
        expanded_base, task_cake_expansion = _expand_task_cake_rank(
            model,
            expanded_rank=int(expanded_task_cake_rank),
        )
        architecture_manifest.update(
            {
                "task_cake_rank": EXPANDED_TASK_CAKE_RANK,
                "architecture_version": EXPANDED_TASK_CAKE_ARCHITECTURE,
            }
        )
        parent_active_parameter_count += int(
            task_cake_expansion["added_active_parameters"]
        )
    elif trainable_scope == TASK_ROUTE_LAYERWISE_CONTROL_SCOPE:
        install_task_route_layerwise_control(model, initialize=True)
        task_route_layerwise_control = {
            "installed_controls": int(model.config.task_cakes),
            "control_layers": int(model.config.layers),
            "control_width": int(model.config.width),
            "installed_control_parameters": (
                int(model.config.task_cakes)
                * int(model.config.layers)
                * int(model.config.width)
            ),
            "active_control_parameters": (
                int(model.config.layers) * int(model.config.width)
            ),
            "maximum_active_control_paths_per_sequence": 1,
            "initial_control_is_zero_exact": True,
            "existing_task_cakes_preserved": True,
            "existing_task_classifier_frozen": True,
            "automatic_route_uses_unconditioned_parent_classifier": True,
            "extra_kv_positions": 0,
            "conditions_every_real_token_kv_write": True
        }
        architecture_manifest = model.config.canonical_dict()
        parent_active_parameter_count += int(
            task_route_layerwise_control["active_control_parameters"]
        )
    elif trainable_scope in {
        TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
        TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
    }:
        if not getattr(model, "_abi_task_route_layerwise_control", False):
            raise FullCoreAcquisitionError(
                "prompt-identity carriage requires the v51 route-control parent"
            )
        install_prompt_identity_carriage(
            model,
            initialize=True,
            selective=selective_prompt_identity_enabled,
        )
        prompt_identity_parent_state_sha256_before = (
            pre_training_deployment_state_sha
        )
        prompt_identity_parameter_count = sum(
            parameter.numel()
            for parameter in model.prompt_identity.parameters()
        )
        task_route_layerwise_control = copy.deepcopy(
            parent_metadata.get("acquired_core", {}).get(
                "task_route_layerwise_control"
            )
        )
        architecture_manifest = model.config.canonical_dict()
        parent_active_parameter_count += prompt_identity_parameter_count
    elif trainable_scope in {
        "capability_cakes_classifier",
        "persistent_capability_prefix_cakes",
        "layerwise_capability_control_cakes",
        "deep_capability_adapter_cakes",
        "shared_deep_capability_adapter_cakes",
        "deep_reused_capability_cakes",
        "gated_deep_reused_capability_cakes",
    }:
        capability_cake_expansion = _expand_capability_cakes(model)
        if trainable_scope == "persistent_capability_prefix_cakes":
            install_persistent_capability_prefix(model, initialize=True)
            capability_cake_expansion.update(
                {
                    "persistent_prefix_length": PERSISTENT_PREFIX_LENGTH,
                    "installed_prefixes": len(CAPABILITY_CAKE_ORDER),
                    "maximum_active_prefixes_per_sequence": 1,
                    "installed_prefix_parameters": (
                        2
                        * len(CAPABILITY_CAKE_ORDER)
                        * int(model.config.layers)
                        * int(model.config.heads)
                        * PERSISTENT_PREFIX_LENGTH
                        * (int(model.config.width) // int(model.config.heads))
                    ),
                    "active_prefix_parameters": (
                        2
                        * int(model.config.layers)
                        * int(model.config.heads)
                        * PERSISTENT_PREFIX_LENGTH
                        * (int(model.config.width) // int(model.config.heads))
                    ),
                    "router_buckets": PERSISTENT_PREFIX_ROUTER_BUCKETS,
                    "router_width": PERSISTENT_PREFIX_ROUTER_WIDTH,
                    "prefix_enters_persistent_transformer_state": True,
                    "public_cache_excludes_prefix": True,
                }
            )
        elif trainable_scope == "layerwise_capability_control_cakes":
            install_layerwise_capability_control(model, initialize=True)
            capability_cake_expansion.update(
                {
                    "installed_control_parameters": (
                        len(CAPABILITY_CAKE_ORDER)
                        * int(model.config.layers)
                        * int(model.config.width)
                    ),
                    "active_control_parameters": (
                        int(model.config.layers) * int(model.config.width)
                    ),
                    "maximum_active_control_paths_per_sequence": 1,
                    "router_buckets": PERSISTENT_PREFIX_ROUTER_BUCKETS,
                    "router_width": PERSISTENT_PREFIX_ROUTER_WIDTH,
                    "control_conditions_every_real_token_kv_write": True,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                }
            )
        elif trainable_scope == "deep_capability_adapter_cakes":
            install_deep_capability_adapters(model, initialize=True)
            adapter_parameters = (
                len(CAPABILITY_CAKE_ORDER)
                * int(model.config.layers)
                * (
                    2 * int(model.config.width)
                    + 2
                    * int(model.config.width)
                    * DEEP_CAPABILITY_ADAPTER_RANK
                )
            )
            active_adapter_parameters = (
                int(model.config.layers)
                * (
                    2 * int(model.config.width)
                    + 2
                    * int(model.config.width)
                    * DEEP_CAPABILITY_ADAPTER_RANK
                )
            )
            capability_cake_expansion.update(
                {
                    "installed_adapter_parameters": adapter_parameters,
                    "active_adapter_parameters": active_adapter_parameters,
                    "installed_deep_adapters": (
                        len(CAPABILITY_CAKE_ORDER)
                        * int(model.config.layers)
                    ),
                    "maximum_active_deep_adapters_per_sequence": (
                        int(model.config.layers)
                    ),
                    "adapter_rank": DEEP_CAPABILITY_ADAPTER_RANK,
                    "router_buckets": PERSISTENT_PREFIX_ROUTER_BUCKETS,
                    "router_width": PERSISTENT_PREFIX_ROUTER_WIDTH,
                    "adapter_conditions_every_real_token_kv_write": True,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                }
            )
        elif trainable_scope == "shared_deep_capability_adapter_cakes":
            install_shared_deep_capability_adapters(
                model, initialize=True
            )
            adapter_parameters = (
                len(CAPABILITY_CAKE_ORDER)
                * (
                    2 * int(model.config.width)
                    + 2
                    * int(model.config.width)
                    * DEEP_CAPABILITY_ADAPTER_RANK
                )
            )
            active_adapter_parameters = (
                2 * int(model.config.width)
                + 2
                * int(model.config.width)
                * DEEP_CAPABILITY_ADAPTER_RANK
            )
            capability_cake_expansion.update(
                {
                    "installed_adapter_parameters": adapter_parameters,
                    "active_adapter_parameters": active_adapter_parameters,
                    "installed_deep_adapters": len(CAPABILITY_CAKE_ORDER),
                    "active_adapter_invocations_per_sequence": int(
                        model.config.layers
                    ),
                    "shared_adapter_weights_across_layers": True,
                    "adapter_rank": DEEP_CAPABILITY_ADAPTER_RANK,
                    "router_buckets": PERSISTENT_PREFIX_ROUTER_BUCKETS,
                    "router_width": PERSISTENT_PREFIX_ROUTER_WIDTH,
                    "adapter_conditions_every_real_token_kv_write": True,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                }
            )
        elif trainable_scope == "deep_reused_capability_cakes":
            install_deep_reused_capability_cakes(
                model, initialize=True
            )
            capability_cake_expansion.update(
                {
                    "initial_selected_cake_function_parent_equivalent": False,
                    "intentional_function_change": (
                        "selected nonzero capability cake relocated from "
                        "post-transformer to all three pre-block positions"
                    ),
                    "installed_adapter_parameters": 0,
                    "active_adapter_parameters": 0,
                    "selected_unique_cakes_per_sequence": 1,
                    "selected_cake_invocations_per_sequence": int(
                        model.config.layers
                    ),
                    "shared_cake_weights_across_layers": True,
                    "final_post_transformer_cake_invocations": 0,
                    "router_buckets": PERSISTENT_PREFIX_ROUTER_BUCKETS,
                    "router_width": PERSISTENT_PREFIX_ROUTER_WIDTH,
                    "cake_conditions_every_real_token_kv_write": True,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                }
            )
        elif trainable_scope == "gated_deep_reused_capability_cakes":
            install_gated_deep_reused_capability_cakes(
                model, initialize=True
            )
            capability_cake_expansion.update(
                {
                    "installed_adapter_parameters": 0,
                    "active_adapter_parameters": int(model.config.layers),
                    "installed_scalar_gate_parameters": (
                        len(CAPABILITY_CAKE_ORDER)
                        * int(model.config.layers)
                    ),
                    "active_scalar_gate_parameters": int(
                        model.config.layers
                    ),
                    "scalar_gate_shape": [
                        len(CAPABILITY_CAKE_ORDER),
                        int(model.config.layers),
                    ],
                    "scalar_gate_initialization": 0.0,
                    "selected_unique_cakes_per_sequence": 1,
                    "pre_block_selected_cake_invocations": int(
                        model.config.layers
                    ),
                    "final_selected_cake_invocations": 1,
                    "shared_cake_weights_across_all_invocations": True,
                    "router_buckets": PERSISTENT_PREFIX_ROUTER_BUCKETS,
                    "router_width": PERSISTENT_PREFIX_ROUTER_WIDTH,
                    "gated_cake_conditions_every_real_token_kv_write": True,
                    "extra_kv_positions": 0,
                    "public_cache_contains_only_real_tokens": True,
                }
            )
        architecture_manifest = model.config.canonical_dict()
        if trainable_scope in {
            "persistent_capability_prefix_cakes",
            "layerwise_capability_control_cakes",
            "deep_capability_adapter_cakes",
            "shared_deep_capability_adapter_cakes",
            "deep_reused_capability_cakes",
            "gated_deep_reused_capability_cakes",
        }:
            parent_active_parameter_count += (
                -10 * (int(model.config.width) + 1)
                + PERSISTENT_PREFIX_ROUTER_BUCKETS
                * PERSISTENT_PREFIX_ROUTER_WIDTH
                + PERSISTENT_PREFIX_ROUTER_WIDTH
                * len(CAPABILITY_CAKE_ORDER)
                + len(CAPABILITY_CAKE_ORDER)
                + (
                    2
                    * int(model.config.layers)
                    * int(model.config.heads)
                    * PERSISTENT_PREFIX_LENGTH
                    * (int(model.config.width) // int(model.config.heads))
                    if trainable_scope
                    == "persistent_capability_prefix_cakes"
                    else (
                        int(model.config.layers) * int(model.config.width)
                        if trainable_scope
                        == "layerwise_capability_control_cakes"
                        else int(
                            capability_cake_expansion[
                                "active_adapter_parameters"
                            ]
                        )
                    )
                )
            )
        else:
            parent_active_parameter_count += (
                len(CAPABILITY_CAKE_ORDER) - 10
            ) * (int(model.config.width) + 1)
        capability_index = {
            capability: index
            for index, capability in enumerate(CAPABILITY_CAKE_ORDER)
        }
        for collection in (rows, anchor_rows, general_rows):
            for row in collection:
                capability = str(row["capability"])
                if capability not in capability_index:
                    raise FullCoreAcquisitionError(
                        f"capability-cake record is unmapped: {capability}"
                    )
                row["route"] = capability_index[capability]
    if trainable_scope == MERGED_ENGLISH_CORE_LORA_SCOPE:
        english_core_lora = _install_merged_english_core_lora(model)
        capability_index = {
            capability: index
            for index, capability in enumerate(CAPABILITY_CAKE_ORDER)
        }
        for collection in (rows, anchor_rows, general_rows):
            for row in collection:
                capability = str(row["capability"])
                if capability not in capability_index:
                    raise FullCoreAcquisitionError(
                        f"merged-core record is unmapped: {capability}"
                    )
                row["route"] = capability_index[capability]
    rows, context_compatibility = _filter_context_compatible_rows(
        tokenizer,
        rows,
        max_tokens=max_tokens,
        exclude_overlength_prompts=exclude_overlength_prompts,
    )
    anchor_context_compatibility = None
    if anchor_rows:
        (
            anchor_rows,
            anchor_context_compatibility,
        ) = _filter_context_compatible_rows(
            tokenizer,
            anchor_rows,
            max_tokens=max_tokens,
            exclude_overlength_prompts=exclude_overlength_prompts,
        )
    context_compatibility["parent_tokenizer_sha256"] = _sha256_file(
        parent_path / "tokenizer.json"
    )
    if anchor_context_compatibility is not None:
        anchor_context_compatibility[
            "parent_tokenizer_sha256"
        ] = context_compatibility["parent_tokenizer_sha256"]
    model.train()
    student_block_capture: list[torch.Tensor | None] = [
        None for _ in model.transformer.h
    ]
    student_block_hook_handles = []
    if source_representation_distillation_enabled:
        def _capture_block(index: int):
            def hook(_module, _inputs, output):
                student_block_capture[index] = (
                    output[0] if isinstance(output, tuple) else output
                )
            return hook

        student_block_hook_handles = [
            block.register_forward_hook(_capture_block(index))
            for index, block in enumerate(model.transformer.h)
        ]
    frozen_shared_state_sha256_before = _frozen_shared_state_sha256(model)
    initial_state_sha = (
        pre_training_deployment_state_sha
        if trainable_scope == MERGED_ENGLISH_CORE_LORA_SCOPE
        else module_state_sha256(model)
    )

    (
        shared_parameters,
        cake_parameters,
        total_parameter_count,
        trainable_parameter_count,
    ) = _select_trainable_parameters(
        model,
        trainable_scope=trainable_scope,
        trainable_task_cake_routes=selected_task_cake_routes,
    )
    optimizer_parameter_count = trainable_parameter_count
    effective_trainable_parameter_count = trainable_parameter_count
    if task_cake_expansion is not None:
        effective_trainable_parameter_count = (
            sum(
                parameter.numel()
                for parameter in model.task_classifier.parameters()
            )
            + int(task_cake_expansion["added_total_parameters"])
        )
    frozen_runtime_originals: dict[str, torch.Tensor] = {}
    runtime_aware_training: dict[str, Any] = {
        "enabled": False,
        "method": None,
        "frozen_weights_restored_before_checkpoint": True,
    }
    if frozen_runtime_fake_int8:
        (
            frozen_runtime_originals,
            runtime_aware_training,
        ) = _apply_frozen_runtime_fake_int8(model)
        runtime_aware_training["enabled"] = True
    optimizer_groups: list[dict[str, Any]] = []
    if shared_parameters:
        optimizer_groups.append(
            {
                "params": shared_parameters,
                "lr": shared_learning_rate,
            }
        )
    if cake_parameters:
        optimizer_groups.append(
            {
                "params": cake_parameters,
                "lr": cake_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=0.01,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else (lambda: nullcontext())
    )

    sampler = _DeterministicRowSampler(
        rows,
        seed=seed,
        strategy=sampling_strategy,
    )
    general_sampler = (
        _DeterministicRowSampler(
            general_rows,
            seed=seed + 1_000_003,
            strategy=general_sampling_strategy,
        )
        if general_rows
        else None
    )
    anchor_sampler = (
        _DeterministicRowSampler(
            anchor_rows,
            seed=seed + 2_000_033,
            strategy=anchor_sampling_strategy,
        )
        if anchor_rows
        else None
    )
    unique_seen: set[str] = set()
    unique_general_seen: set[str] = set()
    unique_anchor_seen: set[str] = set()
    sampled_records_by_capability: Counter[str] = Counter()
    sampled_general_records_by_capability: Counter[str] = Counter()
    sampled_anchor_records_by_capability: Counter[str] = Counter()
    supervised_tokens_seen = 0
    general_supervised_tokens_seen = 0
    anchor_supervised_tokens_seen = 0
    raw_utf8_bytes_seen = 0
    general_raw_utf8_bytes_seen = 0
    anchor_raw_utf8_bytes_seen = 0
    autonomous_prefix_tokens_seen = 0
    source_teacher_forward_tokens = 0
    source_teacher_response_positions = 0
    source_model_inference_seconds = 0.0
    parent_teacher_forward_tokens = 0
    parent_teacher_response_positions = 0
    parent_teacher_inference_seconds = 0.0
    recovery_batches = 0
    horizon_counts = {
        str(int(horizon)): 0 for horizon in recovery_horizons
    }
    successful_steps = 0
    skipped_amp_steps = 0
    attempted_batches = 0
    attempted_microbatches = 0
    curves: list[dict[str, Any]] = []
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    cpu_before = process.cpu_times()
    started = time.perf_counter()

    while successful_steps < steps:
        attempted_batches += 1
        if attempted_batches > steps + 1000:
            raise FullCoreAcquisitionError("too many non-finite optimizer attempts")
        sampler_snapshot = sampler.snapshot()
        general_sampler_snapshot = (
            general_sampler.snapshot()
            if general_sampler is not None
            else None
        )
        anchor_sampler_snapshot = (
            anchor_sampler.snapshot()
            if anchor_sampler is not None
            else None
        )
        generated_prefix_horizon = None
        if (
            recovery_interval > 0
            and successful_steps >= recovery_start_step
            and (successful_steps - recovery_start_step) % recovery_interval == 0
        ):
            horizon = int(
                recovery_horizons[
                    recovery_batches % len(recovery_horizons)
                ]
            )
            generated_prefix_horizon = horizon
        optimizer.zero_grad(set_to_none=True)
        scale_before = scaler.get_scale()
        selected_for_update: list[Mapping[str, Any]] = []
        selected_general_for_update: list[Mapping[str, Any]] = []
        selected_anchor_for_update: list[Mapping[str, Any]] = []
        observed_for_update = 0
        general_observed_for_update = 0
        anchor_observed_for_update = 0
        generated_prefix_tokens_for_update = 0
        loss_sums = {
            "total": 0.0,
            "language": 0.0,
            "classifier": 0.0,
            "general_language": 0.0,
            "general_classifier": 0.0,
            "anchor_language": 0.0,
            "anchor_classifier": 0.0,
            "source_distillation": 0.0,
            "source_representation_distillation": 0.0,
            "parent_logit_preservation": 0.0,
            "prompt_identity": 0.0,
            "anchor_prompt_identity": 0.0,
            "general_prompt_identity": 0.0,
        }
        source_distillation_observations = 0
        general_loss_observations = 0
        anchor_loss_observations = 0
        for _ in range(gradient_accumulation_steps):
            attempted_microbatches += 1
            selected = sampler.batch(batch_size)
            selected_general = (
                general_sampler.batch(general_batch_size)
                if general_sampler is not None
                else []
            )
            selected_anchor = (
                anchor_sampler.batch(anchor_batch_size)
                if anchor_sampler is not None
                else []
            )
            generated_prefixes = (
                _autonomous_prefixes(
                    model,
                    tokenizer,
                    selected,
                    horizon=generated_prefix_horizon,
                    device=device,
                )
                if generated_prefix_horizon is not None
                else None
            )
            (
                ids,
                labels,
                attention,
                prompt_lengths,
                routes,
                observed,
            ) = _batch(
                tokenizer,
                selected,
                device=device,
                max_tokens=max_tokens,
                generated_prefixes=generated_prefixes,
            )
            general_batch = (
                _batch(
                    tokenizer,
                    selected_general,
                    device=device,
                    max_tokens=max_tokens,
                )
                if selected_general
                else None
            )
            anchor_batch = (
                _batch(
                    tokenizer,
                    selected_anchor,
                    device=device,
                    max_tokens=max_tokens,
                )
                if selected_anchor
                else None
            )
            with autocast():
                if source_representation_distillation_enabled:
                    student_block_capture[:] = [
                        None for _ in student_block_capture
                    ]
                result = model(
                    ids,
                    attention_mask=attention,
                    prompt_lengths=prompt_lengths,
                    task_routes=routes,
                    use_cache=False,
                )
                language_loss = _equal_record_prompt_overlap_ce(
                    result["logits"],
                    labels,
                    ids,
                    prompt_lengths,
                    overlap_weight=prompt_overlap_loss_weight,
                )
                classifier_loss = F.cross_entropy(
                    result["task_logits"], routes
                )
                prompt_identity_loss = None
                if prompt_identity_enabled:
                    mixture_loss = (
                        None
                        if selective_prompt_identity_enabled
                        else _equal_record_prompt_identity_nll(
                            result["logits"],
                            result["hidden"],
                            ids,
                            labels,
                            prompt_lengths,
                            routes,
                            model.prompt_identity,
                        )
                    )
                    direct_supervision_loss = (
                        _balanced_prompt_identity_supervision_loss(
                            hidden=result["hidden"],
                            input_ids=ids,
                            labels=labels,
                            prompt_lengths=prompt_lengths,
                            routes=routes,
                            bridge=model.prompt_identity,
                            parent_logits=(
                                result["logits"]
                                if selective_prompt_identity_enabled
                                else None
                            ),
                        )
                    )
                    prompt_identity_loss = (
                        direct_supervision_loss
                        if mixture_loss is None
                        else mixture_loss + direct_supervision_loss
                    )
                source_distillation_loss = None
                source_representation_distillation_loss = None
                if source_teacher is not None:
                    if source_representation_distillation_enabled:
                        if any(
                            value is None for value in student_block_capture
                        ):
                            raise FullCoreAcquisitionError(
                                "student block capture is incomplete"
                            )
                        (
                            source_distillation_loss,
                            source_representation_distillation_loss,
                            source_forward_tokens,
                            source_response_positions,
                            source_inference_seconds,
                        ) = _same_tokenizer_representation_distillation_loss(
                            student_logits=result["logits"],
                            student_block_hidden_states=[
                                value
                                for value in student_block_capture
                                if value is not None
                            ],
                            labels=labels,
                            input_ids=ids,
                            attention_mask=attention,
                            source_teacher=source_teacher,
                            top_k=source_distillation_top_k,
                        )
                    else:
                        (
                            source_distillation_loss,
                            source_forward_tokens,
                            source_response_positions,
                            source_inference_seconds,
                        ) = _same_tokenizer_topk_distillation_loss(
                            student_logits=result["logits"],
                            labels=labels,
                            input_ids=ids,
                            attention_mask=attention,
                            source_teacher=source_teacher,
                            top_k=source_distillation_top_k,
                        )
                    source_teacher_forward_tokens += source_forward_tokens
                    source_teacher_response_positions += (
                        source_response_positions
                    )
                    source_model_inference_seconds += source_inference_seconds
                    source_distillation_observations += 1
                parent_logit_preservation_loss = None
                if parent_preservation_teacher is not None:
                    (
                        parent_logit_preservation_loss,
                        parent_forward_tokens,
                        parent_response_positions,
                        parent_inference_seconds,
                    ) = _parent_layercake_topk_preservation_loss(
                        student_logits=result["logits"],
                        labels=labels,
                        input_ids=ids,
                        attention_mask=attention,
                        prompt_lengths=prompt_lengths,
                        task_routes=routes,
                        parent_teacher=parent_preservation_teacher,
                    )
                    parent_teacher_forward_tokens += parent_forward_tokens
                    parent_teacher_response_positions += (
                        parent_response_positions
                    )
                    parent_teacher_inference_seconds += (
                        parent_inference_seconds
                    )
                loss = (
                    language_loss
                    + classifier_loss_weight * classifier_loss
                    + (
                        source_distillation_weight
                        * source_distillation_loss
                        if source_distillation_loss is not None
                        else 0.0
                    )
                    + (
                        source_representation_distillation_weight
                        * source_representation_distillation_loss
                        if source_representation_distillation_loss is not None
                        else 0.0
                    )
                    + (
                        parent_logit_preservation_weight
                        * parent_logit_preservation_loss
                        if parent_logit_preservation_loss is not None
                        else 0.0
                    )
                    + (
                        prompt_identity_loss_weight
                        * prompt_identity_loss
                        if prompt_identity_loss is not None
                        else 0.0
                    )
                )
                anchor_language_loss = None
                anchor_classifier_loss = None
                anchor_prompt_identity_loss = None
                if anchor_batch is not None:
                    (
                        anchor_ids,
                        anchor_labels,
                        anchor_attention,
                        anchor_prompt_lengths,
                        anchor_routes,
                        anchor_observed,
                    ) = anchor_batch
                    anchor_result = model(
                        anchor_ids,
                        attention_mask=anchor_attention,
                        prompt_lengths=anchor_prompt_lengths,
                        task_routes=anchor_routes,
                        use_cache=False,
                    )
                    anchor_language_loss = (
                        _equal_record_prompt_overlap_ce(
                            anchor_result["logits"],
                            anchor_labels,
                            anchor_ids,
                            anchor_prompt_lengths,
                            overlap_weight=prompt_overlap_loss_weight,
                        )
                    )
                    anchor_classifier_loss = F.cross_entropy(
                        anchor_result["task_logits"],
                        anchor_routes,
                    )
                    if prompt_identity_enabled:
                        anchor_direct_loss = (
                            _balanced_prompt_identity_supervision_loss(
                                hidden=anchor_result["hidden"],
                                input_ids=anchor_ids,
                                labels=anchor_labels,
                                prompt_lengths=anchor_prompt_lengths,
                                routes=anchor_routes,
                                bridge=model.prompt_identity,
                                parent_logits=(
                                    anchor_result["logits"]
                                    if selective_prompt_identity_enabled
                                    else None
                                ),
                            )
                        )
                        anchor_prompt_identity_loss = (
                            anchor_direct_loss
                            if selective_prompt_identity_enabled
                            else anchor_direct_loss
                            + _equal_record_prompt_identity_nll(
                                anchor_result["logits"],
                                anchor_result["hidden"],
                                anchor_ids,
                                anchor_labels,
                                anchor_prompt_lengths,
                                anchor_routes,
                                model.prompt_identity,
                            )
                        )
                    loss = loss + anchor_loss_weight * (
                        anchor_language_loss
                        + classifier_loss_weight
                        * anchor_classifier_loss
                        + (
                            prompt_identity_loss_weight
                            * anchor_prompt_identity_loss
                            if anchor_prompt_identity_loss is not None
                            else 0.0
                        )
                    )
                general_language_loss = None
                general_classifier_loss = None
                general_prompt_identity_loss = None
                if general_batch is not None:
                    (
                        general_ids,
                        general_labels,
                        general_attention,
                        general_prompt_lengths,
                        general_routes,
                        general_observed,
                    ) = general_batch
                    general_result = model(
                        general_ids,
                        attention_mask=general_attention,
                        prompt_lengths=general_prompt_lengths,
                        task_routes=general_routes,
                        use_cache=False,
                    )
                    general_language_loss = (
                        _equal_record_prompt_overlap_ce(
                            general_result["logits"],
                            general_labels,
                            general_ids,
                            general_prompt_lengths,
                            overlap_weight=prompt_overlap_loss_weight,
                        )
                    )
                    general_classifier_loss = F.cross_entropy(
                        general_result["task_logits"], general_routes
                    )
                    if prompt_identity_enabled:
                        general_direct_loss = (
                            _balanced_prompt_identity_supervision_loss(
                                hidden=general_result["hidden"],
                                input_ids=general_ids,
                                labels=general_labels,
                                prompt_lengths=general_prompt_lengths,
                                routes=general_routes,
                                bridge=model.prompt_identity,
                                parent_logits=(
                                    general_result["logits"]
                                    if selective_prompt_identity_enabled
                                    else None
                                ),
                            )
                        )
                        general_prompt_identity_loss = (
                            general_direct_loss
                            if selective_prompt_identity_enabled
                            else general_direct_loss
                            + _equal_record_prompt_identity_nll(
                                general_result["logits"],
                                general_result["hidden"],
                                general_ids,
                                general_labels,
                                general_prompt_lengths,
                                general_routes,
                                model.prompt_identity,
                            )
                        )
                    loss = loss + general_loss_weight * (
                        general_language_loss
                        + classifier_loss_weight
                        * general_classifier_loss
                        + (
                            prompt_identity_loss_weight
                            * general_prompt_identity_loss
                            if general_prompt_identity_loss is not None
                            else 0.0
                        )
                    )
            scaler.scale(
                loss / gradient_accumulation_steps
            ).backward()
            selected_for_update.extend(selected)
            selected_general_for_update.extend(selected_general)
            selected_anchor_for_update.extend(selected_anchor)
            observed_for_update += observed
            if general_batch is not None:
                general_observed_for_update += general_observed
                general_loss_observations += 1
            if anchor_batch is not None:
                anchor_observed_for_update += anchor_observed
                anchor_loss_observations += 1
            generated_prefix_tokens_for_update += sum(
                len(prefix) for prefix in generated_prefixes or []
            )
            loss_sums["total"] += float(loss.detach())
            loss_sums["language"] += float(language_loss.detach())
            loss_sums["classifier"] += float(classifier_loss.detach())
            if source_distillation_loss is not None:
                loss_sums["source_distillation"] += float(
                    source_distillation_loss.detach()
                )
            if source_representation_distillation_loss is not None:
                loss_sums["source_representation_distillation"] += float(
                    source_representation_distillation_loss.detach()
                )
            if parent_logit_preservation_loss is not None:
                loss_sums["parent_logit_preservation"] += float(
                    parent_logit_preservation_loss.detach()
                )
            if prompt_identity_loss is not None:
                loss_sums["prompt_identity"] += float(
                    prompt_identity_loss.detach()
                )
            if general_language_loss is not None:
                loss_sums["general_language"] += float(
                    general_language_loss.detach()
                )
            if general_classifier_loss is not None:
                loss_sums["general_classifier"] += float(
                    general_classifier_loss.detach()
                )
            if anchor_language_loss is not None:
                loss_sums["anchor_language"] += float(
                    anchor_language_loss.detach()
                )
            if anchor_classifier_loss is not None:
                loss_sums["anchor_classifier"] += float(
                    anchor_classifier_loss.detach()
                )
            if anchor_prompt_identity_loss is not None:
                loss_sums["anchor_prompt_identity"] += float(
                    anchor_prompt_identity_loss.detach()
                )
            if general_prompt_identity_loss is not None:
                loss_sums["general_prompt_identity"] += float(
                    general_prompt_identity_loss.detach()
                )
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if expanded_base and not _restore_expanded_task_cake_base(
            model, expanded_base
        ):
            raise FullCoreAcquisitionError(
                "expanded task-cake optimization changed a locked base slice"
            )
        if scaler.get_scale() < scale_before:
            sampler.restore(sampler_snapshot)
            if (
                general_sampler is not None
                and general_sampler_snapshot is not None
            ):
                general_sampler.restore(general_sampler_snapshot)
            if (
                anchor_sampler is not None
                and anchor_sampler_snapshot is not None
            ):
                anchor_sampler.restore(anchor_sampler_snapshot)
            skipped_amp_steps += 1
            continue
        successful_steps += 1
        sampled_records_by_capability.update(
            str(row["capability"]) for row in selected_for_update
        )
        sampled_general_records_by_capability.update(
            str(row["capability"])
            for row in selected_general_for_update
        )
        sampled_anchor_records_by_capability.update(
            str(row["capability"]) for row in selected_anchor_for_update
        )
        if generated_prefix_horizon is not None:
            recovery_batches += 1
            horizon_counts[str(generated_prefix_horizon)] += 1
            autonomous_prefix_tokens_seen += (
                generated_prefix_tokens_for_update
            )
        unique_seen.update(
            str(row["record_id"]) for row in selected_for_update
        )
        unique_general_seen.update(
            str(row["record_id"]) for row in selected_general_for_update
        )
        unique_anchor_seen.update(
            str(row["record_id"]) for row in selected_anchor_for_update
        )
        supervised_tokens_seen += observed_for_update
        general_supervised_tokens_seen += general_observed_for_update
        anchor_supervised_tokens_seen += anchor_observed_for_update
        raw_utf8_bytes_seen += sum(
            len((str(row["prompt"]) + str(row["response"])).encode("utf-8"))
            for row in selected_for_update
        )
        general_raw_utf8_bytes_seen += sum(
            len((str(row["prompt"]) + str(row["response"])).encode("utf-8"))
            for row in selected_general_for_update
        )
        anchor_raw_utf8_bytes_seen += sum(
            len((str(row["prompt"]) + str(row["response"])).encode("utf-8"))
            for row in selected_anchor_for_update
        )
        if (
            successful_steps == 1
            or successful_steps % 100 == 0
            or successful_steps == steps
        ):
            curve = {
                "step": successful_steps,
                "total_loss": (
                    loss_sums["total"] / gradient_accumulation_steps
                ),
                "language_loss": (
                    loss_sums["language"] / gradient_accumulation_steps
                ),
                "classifier_loss": (
                    loss_sums["classifier"]
                    / gradient_accumulation_steps
                ),
                "source_distillation_loss": (
                    loss_sums["source_distillation"]
                    / source_distillation_observations
                    if source_distillation_observations
                    else None
                ),
                "source_representation_distillation_loss": (
                    loss_sums["source_representation_distillation"]
                    / source_distillation_observations
                    if source_representation_distillation_enabled
                    and source_distillation_observations
                    else None
                ),
                "parent_logit_preservation_loss": (
                    loss_sums["parent_logit_preservation"]
                    / gradient_accumulation_steps
                    if parent_logit_preservation_enabled
                    else None
                ),
                "prompt_identity_loss": (
                    loss_sums["prompt_identity"]
                    / gradient_accumulation_steps
                    if prompt_identity_enabled
                    else None
                ),
                "general_language_loss": (
                    loss_sums["general_language"]
                    / general_loss_observations
                    if general_loss_observations
                    else None
                ),
                "general_classifier_loss": (
                    loss_sums["general_classifier"]
                    / general_loss_observations
                    if general_loss_observations
                    else None
                ),
                "anchor_language_loss": (
                    loss_sums["anchor_language"]
                    / anchor_loss_observations
                    if anchor_loss_observations
                    else None
                ),
                "anchor_classifier_loss": (
                    loss_sums["anchor_classifier"]
                    / anchor_loss_observations
                    if anchor_loss_observations
                    else None
                ),
                "anchor_prompt_identity_loss": (
                    loss_sums["anchor_prompt_identity"]
                    / anchor_loss_observations
                    if anchor_loss_observations and prompt_identity_enabled
                    else None
                ),
                "general_prompt_identity_loss": (
                    loss_sums["general_prompt_identity"]
                    / general_loss_observations
                    if general_loss_observations and prompt_identity_enabled
                    else None
                ),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)

    for handle in student_block_hook_handles:
        handle.remove()
    student_block_hook_handles.clear()
    student_block_capture.clear()

    if source_teacher is not None:
        assert source_distillation_contract is not None
        model_parameter_ids = {
            id(parameter) for parameter in model.parameters()
        }
        if any(
            id(parameter) in model_parameter_ids
            for parameter in source_teacher.parameters()
        ):
            raise FullCoreAcquisitionError(
                "source teacher parameter alias entered the LayerCake host"
            )
        source_distillation_contract.update(
            {
                "source_teacher_forward_tokens": (
                    source_teacher_forward_tokens
                ),
                "source_teacher_response_positions_distilled": (
                    source_teacher_response_positions
                ),
                "source_model_inference_seconds": (
                    source_model_inference_seconds
                ),
                "source_model_inference_hours": (
                    source_model_inference_seconds / 3600.0
                ),
                "source_teacher_removed_before_checkpoint": True,
            }
        )
        del source_teacher
        source_teacher = None
        torch.cuda.empty_cache()

    if parent_preservation_teacher is not None:
        assert parent_preservation_contract is not None
        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        alias_count = sum(
            id(parameter) in model_parameter_ids
            for parameter in parent_preservation_teacher.parameters()
        )
        if alias_count:
            raise FullCoreAcquisitionError(
                "frozen parent parameter alias entered the LayerCake host"
            )
        parent_preservation_contract.update(
            {
                "parent_teacher_forward_tokens": parent_teacher_forward_tokens,
                "parent_teacher_response_positions": (
                    parent_teacher_response_positions
                ),
                "parent_teacher_inference_seconds": (
                    parent_teacher_inference_seconds
                ),
                "parent_teacher_inference_hours": (
                    parent_teacher_inference_seconds / 3600.0
                ),
                "parent_teacher_removed_before_checkpoint": True,
            }
        )
        del parent_preservation_teacher
        parent_preservation_teacher = None
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    frozen_runtime_weights_restored_exact = True
    if frozen_runtime_originals:
        frozen_runtime_weights_restored_exact = (
            _restore_frozen_runtime_weights(
                model,
                frozen_runtime_originals,
            )
        )
        if not frozen_runtime_weights_restored_exact:
            raise FullCoreAcquisitionError(
                "frozen runtime fake-int8 weights were not restored exactly"
            )
    runtime_aware_training[
        "frozen_weights_restored_exact"
    ] = frozen_runtime_weights_restored_exact
    expanded_base_preserved_exact = True
    if expanded_base:
        expanded_base_preserved_exact = _restore_expanded_task_cake_base(
            model, expanded_base
        )
        if not expanded_base_preserved_exact:
            raise FullCoreAcquisitionError(
                "expanded task-cake base slices were not preserved exactly"
            )
    if english_core_lora is not None:
        english_core_lora = _merge_english_core_lora(
            model, english_core_lora
        )
        frozen_shared_state_sha256_before = english_core_lora[
            "non_target_state_sha256_before"
        ]
        frozen_shared_state_sha256_after = english_core_lora[
            "non_target_state_sha256_after"
        ]
    else:
        frozen_shared_state_sha256_after = _frozen_shared_state_sha256(model)
    if (
        trainable_scope in {
            "task_cakes_classifier",
            TASK_ROUTE_LAYERWISE_CONTROL_SCOPE,
            TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
            TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
            "task_cakes_only",
            "selected_task_cakes",
            "expanded_task_cake_tail_classifier",
            "capability_cakes_classifier",
            "persistent_capability_prefix_cakes",
            "layerwise_capability_control_cakes",
            "deep_capability_adapter_cakes",
            "shared_deep_capability_adapter_cakes",
            "deep_reused_capability_cakes",
            "gated_deep_reused_capability_cakes",
        }
        and frozen_shared_state_sha256_after
        != frozen_shared_state_sha256_before
    ):
        raise FullCoreAcquisitionError(
            "expanded task-cake acquisition changed the frozen shared core"
        )
    transformer_control_task_state_after: dict[str, Any] | None = None
    if trainable_scope == TRANSFORMER_CORE_CONTROL_SCOPE:
        assert transformer_control_task_state_before is not None
        transformer_control_task_state_after = (
            _transformer_control_task_state(model)
        )
        if (
            transformer_control_task_state_after
            != transformer_control_task_state_before
            or transformer_control_task_state_after[
                "all_task_cake_up_weights_zero_exact"
            ]
            is not True
        ):
            raise FullCoreAcquisitionError(
                "transformer-core control changed its frozen identity-cake container"
            )
    prompt_identity_parent_state_sha256_after = None
    if prompt_identity_enabled:
        prompt_identity_parent_state_sha256_after = (
            _state_sha256_excluding_prefixes(
                model,
                ("prompt_identity.",),
            )
        )
        if (
            prompt_identity_parent_state_sha256_after
            != prompt_identity_parent_state_sha256_before
        ):
            raise FullCoreAcquisitionError(
                "prompt-identity acquisition changed an existing parent tensor"
            )
    model.eval()
    deployment_total_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    final_state_sha = module_state_sha256(model)
    if final_state_sha == initial_state_sha:
        raise FullCoreAcquisitionError("full-core acquisition changed no parameters")
    if _sha256_file(bundle_path) != archive_sha_before:
        raise FullCoreAcquisitionError("imported ABI artifact changed during training")
    if (
        anchor_bundle_path is not None
        and _sha256_file(anchor_bundle_path) != anchor_archive_sha_before
    ):
        raise FullCoreAcquisitionError(
            "broad behavior anchor artifact changed during training"
        )
    if _sha256_file(parent_checkpoint_path) != parent_checkpoint_sha:
        raise FullCoreAcquisitionError("sealed parent changed during training")
    direct_source_conformance: dict[str, Any] | None = None
    if direct_source_contract is not None:
        final_named_state = model.state_dict()
        retained_exact_parameters = 0
        retained_exact_parameter_tensors: list[str] = []
        for item in direct_source_contract["copied_target_tensors"]:
            target_name = str(item["target_tensor"])
            if target_name not in final_named_state:
                raise FullCoreAcquisitionError(
                    f"direct-source target tensor disappeared: {target_name}"
                )
            retained = (
                tensor_sha256(final_named_state[target_name])
                == item["sha256_after_copy"]
            )
            if retained and item["kind"] == "parameter":
                retained_exact_parameters += int(item["numel"])
                retained_exact_parameter_tensors.append(target_name)
        final_block_hashes = [
            module_state_sha256(block) for block in model.transformer.h
        ]
        initial_block_hashes = direct_source_contract[
            "target_block_state_sha256_after_copy"
        ]
        retained_exact_blocks = sum(
            before == after
            for before, after in zip(
                initial_block_hashes, final_block_hashes, strict=True
            )
        )
        if retained_exact_blocks:
            raise FullCoreAcquisitionError(
                "GPU conformance retained an exact imported source block"
            )
        direct_source_conformance = {
            **direct_source_contract,
            "gpu_conformance_applied": True,
            "source_parameters_retained_exact_after_conformance": (
                retained_exact_parameters
            ),
            "source_parameter_tensors_retained_exact_after_conformance": (
                retained_exact_parameter_tensors
            ),
            "source_transformer_blocks_retained_exact_after_conformance": 0,
            "target_block_state_sha256_after_conformance": final_block_hashes,
            "source_teacher_present_during_conformance": (
                source_distillation_enabled
            ),
            "source_teacher_removed_before_checkpoint": True,
        }
        assert direct_source_checkpoint_path is not None
        if (
            _sha256_file(direct_source_checkpoint_path)
            != direct_source_contract["source_checkpoint_sha256"]
        ):
            raise FullCoreAcquisitionError(
                "direct-source checkpoint changed during GPU conformance"
            )

    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint_path),
    )
    tokenizer.save_pretrained(output_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    tokenizer_path = output_path / "tokenizer.json"
    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )
    manifest: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "status": "TRAINED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED",
        "architecture": architecture_manifest,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
        },
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": _sha256_file(tokenizer_path),
        },
        "parent_layercake": {
            "path_at_training": str(parent_path),
            "checkpoint_sha256": parent_checkpoint_sha,
            "metadata_sha256": _sha256_file(parent_metadata_path),
            "logical_state_sha256_before": initial_state_sha,
            "unchanged_on_disk": True,
        },
        "acquired_core": {
            "logical_state_sha256_after": final_state_sha,
            "total_parameter_count": deployment_total_parameter_count,
            "training_graph_parameter_count": total_parameter_count,
            "trainable_parameter_count": effective_trainable_parameter_count,
            "optimizer_parameter_element_count": optimizer_parameter_count,
            "frozen_parameter_count": (
                total_parameter_count - effective_trainable_parameter_count
            ),
            "trainable_scope": trainable_scope,
            "trainable_task_cake_routes": list(
                selected_task_cake_routes
            ),
            "active_parameter_count": parent_active_parameter_count,
            "graph_topology_changed": (
                capability_cake_expansion is not None
                or task_route_layerwise_control is not None
                or prompt_identity_enabled
            ),
            "parameter_shapes_changed": (
                task_cake_expansion is not None
                or capability_cake_expansion is not None
                or task_route_layerwise_control is not None
                or prompt_identity_enabled
            ),
            "task_cake_rank_expanded": task_cake_expansion is not None,
            "task_cake_expansion": task_cake_expansion,
            "capability_cake_expansion": capability_cake_expansion,
            "task_route_layerwise_control": task_route_layerwise_control,
            "prompt_identity_carriage": (
                {
                    "rank": PROMPT_IDENTITY_RANK,
                    "parameter_count": sum(
                        parameter.numel()
                        for parameter in model.prompt_identity.parameters()
                    ),
                    "prompt_tokens_only": True,
                    "balanced_copy_gate_supervision": True,
                    "selective_parent_top1_deficit_labels": (
                        selective_prompt_identity_enabled
                    ),
                    "positive_prompt_position_attention_supervision": True,
                    "all_parent_parameters_frozen": True,
                    "parent_state_sha256_before": (
                        prompt_identity_parent_state_sha256_before
                    ),
                    "parent_state_sha256_after": (
                        prompt_identity_parent_state_sha256_after
                    ),
                    "parent_state_preserved_exact": (
                        prompt_identity_parent_state_sha256_before
                        == prompt_identity_parent_state_sha256_after
                    ),
                    "persistent_prompt_keys": True,
                }
                if prompt_identity_enabled
                else None
            ),
            "english_core_lora": english_core_lora,
            "expanded_base_preserved_exact": (
                expanded_base_preserved_exact
            ),
            "frozen_shared_state_sha256_before": (
                frozen_shared_state_sha256_before
            ),
            "frozen_shared_state_sha256_after": (
                frozen_shared_state_sha256_after
            ),
            "frozen_shared_state_preserved_exact": (
                frozen_shared_state_sha256_before
                == frozen_shared_state_sha256_after
            ),
            "task_cake_count": int(
                architecture_manifest["task_cakes"]
            ),
            "maximum_active_task_cakes_per_sequence": 1,
            "physical_sparse_topology_preserved": True,
        },
        "canonical_semantic_abi": {
            "path_at_training": str(canonical_abi_path),
            "sha256": _sha256_file(canonical_abi_path),
            "changed": False,
        },
        "decoding": (
            {
                **copy.deepcopy(decoding_contract),
                "prompt_identity_mixture": prompt_identity_enabled,
            }
            if decoding_contract is not None
            else {
                "algorithm": "greedy",
                "no_repeat_ngram_size": 0,
                "allow_prompt_ngrams": False,
                "lexical_repetition_blocking_threshold": 0,
                "lexical_repetition_truncation_threshold": 0,
                "byte_repetition_ceiling": 0.0,
                "byte_repetition_guard_minimum_bytes": 0,
                "prompt_identity_mixture": prompt_identity_enabled,
            }
        ),
        "decoding_contract": {
            "path_at_training": (
                str(decoding_contract_path)
                if decoding_contract_path is not None
                else None
            ),
            "sha256": (
                _sha256_file(decoding_contract_path)
                if decoding_contract_path is not None
                else None
            ),
            "bound_before_training": decoding_contract_path is not None,
            "weights_changed_by_decoding": False,
        },
        "imported_artifact": {
            "path_at_training": str(bundle_path),
            "archive_sha256_before": archive_sha_before,
            "archive_sha256_after": _sha256_file(bundle_path),
            "manifest_sha256": verification["manifest_sha256"],
            "artifact_role": verification["artifact_role"],
            "domain_segregation_verified": True,
            "budget_id": budget["budget_id"],
            "budget_index": budget_index,
            "selected_english_records": len(rows),
            "unique_selected_records_seen": len(unique_seen),
            "all_selected_records_seen": len(unique_seen) == len(rows),
            "selected_teacher_tokens": sum(
                int(row["teacher_tokens"]) for row in rows
            ),
            "selected_teacher_output_bytes": sum(
                len(str(row["response"]).encode("utf-8")) for row in rows
            ),
            "teacher_logits_stored": 0,
            "teacher_hidden_activation_bytes_stored": 0,
            "target_control": target_control,
        },
        "broad_behavior_anchor": {
            "enabled": anchor_bundle_path is not None,
            "path_at_training": (
                str(anchor_bundle_path)
                if anchor_bundle_path is not None
                else None
            ),
            "archive_sha256_before": anchor_archive_sha_before,
            "archive_sha256_after": (
                _sha256_file(anchor_bundle_path)
                if anchor_bundle_path is not None
                else None
            ),
            "manifest_sha256": (
                anchor_bundle["verification"]["manifest_sha256"]
                if anchor_bundle is not None
                else None
            ),
            "artifact_role": (
                anchor_bundle["verification"]["artifact_role"]
                if anchor_bundle is not None
                else None
            ),
            "domain_segregation_verified": (
                True if anchor_bundle is not None else None
            ),
            "budget_id": (
                anchor_budget["budget_id"]
                if anchor_budget is not None
                else None
            ),
            "budget_index": (
                anchor_budget_index
                if anchor_bundle_path is not None
                else None
            ),
            "selected_english_records": len(anchor_rows),
            "unique_selected_records_seen": len(unique_anchor_seen),
            "all_selected_records_seen": (
                len(unique_anchor_seen) == len(anchor_rows)
                if anchor_rows
                else True
            ),
            "selected_teacher_tokens": sum(
                int(row["teacher_tokens"]) for row in anchor_rows
            ),
            "selected_teacher_output_bytes": sum(
                len(str(row["response"]).encode("utf-8"))
                for row in anchor_rows
            ),
            "context_compatibility": anchor_context_compatibility,
            "teacher_logits_stored": 0,
            "teacher_hidden_activation_bytes_stored": 0,
            "source_parameters_copied": 0,
            "source_transformer_blocks_retained": 0,
            "teacher_present_at_inference": False,
        },
        "context_compatibility": context_compatibility,
        "foreign_source_boundary": {
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
            "source_parameters_copied": (
                int(
                    direct_source_contract[
                        "source_parameters_copied_at_initialization"
                    ]
                )
                if direct_source_contract is not None
                else 0
            ),
            "source_parameters_retained_exact": (
                int(
                    direct_source_conformance[
                        "source_parameters_retained_exact_after_conformance"
                    ]
                )
                if direct_source_conformance is not None
                else 0
            ),
            "source_generated_text_retained_in_deployment": False,
            "teacher_tokenizer_required_at_inference": False,
        },
        "direct_source_initialization": direct_source_conformance,
        "same_tokenizer_source_distillation": (
            source_distillation_contract
        ),
        "parent_logit_preservation": parent_preservation_contract,
        "transformer_student_control": (
            {
                **transformer_student_control,
                "training_scope": TRANSFORMER_CORE_CONTROL_SCOPE,
                "language_model_shared_tensors_trainable": True,
                "task_classifier_trainable": False,
                "task_cakes_trainable": False,
                "classifier_loss_weight": classifier_loss_weight,
                "task_state_before": transformer_control_task_state_before,
                "task_state_after": transformer_control_task_state_after,
                "frozen_task_container_preserved_exact": (
                    transformer_control_task_state_before
                    == transformer_control_task_state_after
                ),
                "task_cake_effect_disabled_exact": True,
                "inference_performance_baseline": False,
            }
            if transformer_student_control is not None
            else None
        ),
        "general_english_preservation": {
            "enabled": general_curriculum_path is not None,
            "path_at_training": (
                str(general_curriculum_path)
                if general_curriculum_path is not None
                else None
            ),
            "sha256": (
                _sha256_file(general_curriculum_path)
                if general_curriculum_path is not None
                else None
            ),
            "source_provenances": sorted(
                {
                    str(row.get("provenance", "unspecified"))
                    for row in general_rows
                }
            ),
            "teacher_tokens_imported": sum(
                int(row.get("teacher_tokens", 0))
                for row in general_rows
            ),
            "teacher_outputs_imported": sum(
                int(row.get("teacher_tokens", 0)) > 0
                for row in general_rows
            ),
            "domain_artifacts_imported": 0,
            "training_rows": len(general_rows),
            "unique_rows_seen": len(unique_general_seen),
            "all_rows_seen": (
                len(unique_general_seen) == len(general_rows)
                if general_rows
                else True
            ),
            "claim_boundary": (
                "This stream is accounted from its record-level provenance. "
                "It does not establish ABI transfer, production eligibility, "
                "specialist-domain isolation, or absolute zero world knowledge."
            ),
        },
        "training": {
            "seed": seed,
            "device": str(device),
            "steps": steps,
            "successful_optimizer_steps": successful_steps,
            "skipped_amp_optimizer_steps": skipped_amp_steps,
            "attempted_batches": attempted_batches,
            "attempted_microbatches": attempted_microbatches,
            "batch_size": (
                batch_size * gradient_accumulation_steps
            ),
            "microbatch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_teacher_batch_size": (
                batch_size * gradient_accumulation_steps
            ),
            "anchor_batch_size": (
                anchor_batch_size * gradient_accumulation_steps
            ),
            "anchor_microbatch_size": anchor_batch_size,
            "effective_anchor_batch_size": (
                anchor_batch_size * gradient_accumulation_steps
            ),
            "anchor_loss_weight": anchor_loss_weight,
            "anchor_sampling_strategy": anchor_sampling_strategy,
            "general_preservation_batch_size": (
                general_batch_size * gradient_accumulation_steps
            ),
            "general_preservation_microbatch_size": general_batch_size,
            "effective_general_preservation_batch_size": (
                general_batch_size * gradient_accumulation_steps
            ),
            "total_examples_per_step": (
                (
                    batch_size
                    + anchor_batch_size
                    + general_batch_size
                )
                * gradient_accumulation_steps
            ),
            "general_preservation_loss_weight": general_loss_weight,
            "general_sampling_strategy": general_sampling_strategy,
            "sampling_strategy": sampling_strategy,
            "trainable_scope": trainable_scope,
            "trainable_task_cake_routes": list(
                selected_task_cake_routes
            ),
            "target_control": target_control,
            "sampled_records_by_capability": dict(
                sorted(sampled_records_by_capability.items())
            ),
            "sampled_general_records_by_capability": dict(
                sorted(sampled_general_records_by_capability.items())
            ),
            "sampled_anchor_records_by_capability": dict(
                sorted(sampled_anchor_records_by_capability.items())
            ),
            "shared_learning_rate": shared_learning_rate,
            "cake_learning_rate": cake_learning_rate,
            "classifier_loss_weight": classifier_loss_weight,
            "prompt_overlap_loss_weight": prompt_overlap_loss_weight,
            "source_distillation_weight": (
                source_distillation_weight
                if source_distillation_enabled
                else 0.0
            ),
            "source_distillation_top_k": (
                source_distillation_top_k
                if source_distillation_enabled
                else 0
            ),
            "source_representation_distillation_weight": (
                source_representation_distillation_weight
                if source_representation_distillation_enabled
                else 0.0
            ),
            "source_representation_layer_mapping": (
                list(SAME_TOKENIZER_REPRESENTATION_LAYERS)
                if source_representation_distillation_enabled
                else []
            ),
            "parent_logit_preservation_weight": (
                parent_logit_preservation_weight
            ),
            "prompt_identity_loss_weight": prompt_identity_loss_weight,
            "parent_teacher_forward_tokens": parent_teacher_forward_tokens,
            "parent_teacher_response_positions": (
                parent_teacher_response_positions
            ),
            "parent_teacher_inference_seconds": (
                parent_teacher_inference_seconds
            ),
            "source_teacher_forward_tokens": (
                source_teacher_forward_tokens
            ),
            "source_teacher_response_positions_distilled": (
                source_teacher_response_positions
            ),
            "source_model_inference_seconds": (
                source_model_inference_seconds
            ),
            "weight_decay": 0.01,
            "max_tokens": max_tokens,
            "supervised_layercake_tokens_seen": supervised_tokens_seen,
            "anchor_supervised_layercake_tokens_seen": (
                anchor_supervised_tokens_seen
            ),
            "general_preservation_tokens_seen": (
                general_supervised_tokens_seen
            ),
            "raw_utf8_bytes_seen": raw_utf8_bytes_seen,
            "anchor_raw_utf8_bytes_seen": anchor_raw_utf8_bytes_seen,
            "general_preservation_utf8_bytes_seen": (
                general_raw_utf8_bytes_seen
            ),
            "self_generated_prefix_recovery": {
                "start_step": recovery_start_step,
                "interval": recovery_interval,
                "horizons": [int(value) for value in recovery_horizons],
                "batches": recovery_batches,
                "horizon_batches": horizon_counts,
                "autonomous_prefix_tokens_seen": autonomous_prefix_tokens_seen,
            },
            "wall_seconds": elapsed,
            "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0,
            "cpu_seconds": cpu_seconds,
            "cpu_core_hours": cpu_seconds / 3600,
            "active_parameter_seconds": (
                effective_trainable_parameter_count * elapsed
            ),
            "optimizer_parameter_element_count": optimizer_parameter_count,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(process.memory_info().rss),
            "peak_device_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "runtime_aware_training": runtime_aware_training,
            "curves": curves,
        },
        "claim_boundary": (
            "This capacity diagnostic preserves the existing LayerCake graph "
            + (
                "and trains the complete core. "
                if trainable_scope == "full_core"
                else (
                    "and GPU-trains the complete core against segregated "
                    "English response targets plus online top-64 next-token "
                    "probabilities from a frozen same-tokenizer source, then "
                    "removes that source before checkpointing. "
                    + (
                        "Its parent contains a fully accounted direct source "
                        "initialization; conformance retains zero exact source "
                        "blocks and records every still-exact tensor. "
                        if direct_source_conformance is not None
                        else "No source parameter, block, logit file, or "
                        "activation is copied or retained. "
                    )
                    if trainable_scope
                    in {
                        SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
                        SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
                    }
                    else (
                    "and GPU-trains temporary rank-32 updates on only the "
                    "twelve shared English transformer matrices, then fuses "
                    "those updates into the existing matrix shapes while "
                    "retaining no adapter and leaving capability cakes, "
                    "routing, gates, embeddings, and the language head "
                    "bit-exact. "
                    if trainable_scope == MERGED_ENGLISH_CORE_LORA_SCOPE
                    else (
                        "and freezes every shared-core tensor and each parent "
                        "rank-64 cake slice while training only the expanded "
                        "physically selected cake tails and route classifier. "
                        if trainable_scope
                        == "expanded_task_cake_tail_classifier"
                        else (
                            "and freezes every shared-core tensor while splitting "
                            "collided canonical routes into 14 internally selected "
                            "rank-64 capability cakes. "
                            if trainable_scope in {
                                "capability_cakes_classifier",
                                "persistent_capability_prefix_cakes",
                                "layerwise_capability_control_cakes",
                                "deep_capability_adapter_cakes",
                                "shared_deep_capability_adapter_cakes",
                                "deep_reused_capability_cakes",
                                "gated_deep_reused_capability_cakes",
                            }
                            else (
                                "and freezes every existing v51 parameter while "
                                "GPU-training only one integrated rank-32 prompt-position "
                                "carriage bridge with direct balanced gate and attention "
                                "supervision. "
                                if trainable_scope in {
                                    TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
                                    TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
                                }
                                else (
                                "and GPU-trains only the existing sparse task-cake residuals and "
                                "their route classifier. "
                                if trainable_scope == "task_cakes_classifier"
                                else (
                                "and freezes the shared core, embeddings, language head, "
                                "and route classifier while training "
                                + (
                                    "only explicitly selected sparse task-cake routes "
                                    f"{list(selected_task_cake_routes)}. "
                                    if trainable_scope == "selected_task_cakes"
                                    else (
                                        "only existing sparse task-cake residual "
                                        "parameters. "
                                    )
                                )
                                )
                                )
                            )
                        )
                    ))
                )
            )
            + (
                "It is not a fluent-core claim or runtime certification."
            )
        ),
    }
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    _write_json(output_path / "metadata.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1
    )
    parser.add_argument(
        "--trainable-scope",
        choices=(
            "full_core",
            "task_cakes_only",
            "task_cakes_classifier",
            "selected_task_cakes",
            "expanded_task_cake_tail_classifier",
            "capability_cakes_classifier",
            "persistent_capability_prefix_cakes",
            "layerwise_capability_control_cakes",
            "deep_capability_adapter_cakes",
            "shared_deep_capability_adapter_cakes",
            "deep_reused_capability_cakes",
            "gated_deep_reused_capability_cakes",
            MERGED_ENGLISH_CORE_LORA_SCOPE,
            SAME_TOKENIZER_LOGIT_DISTILLATION_SCOPE,
            SAME_TOKENIZER_REPRESENTATION_DISTILLATION_SCOPE,
            TASK_ROUTE_LAYERWISE_CONTROL_SCOPE,
            TASK_ROUTE_PROMPT_IDENTITY_SCOPE,
            TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
            TRANSFORMER_CORE_CONTROL_SCOPE,
        ),
        default="full_core",
    )
    parser.add_argument(
        "--trainable-task-cake-routes",
        default="",
        help=(
            "comma-separated installed route indices; required only for "
            "selected_task_cakes"
        ),
    )
    parser.add_argument(
        "--expanded-task-cake-rank",
        type=int,
        choices=(EXPANDED_TASK_CAKE_RANK,),
    )
    parser.add_argument("--shared-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--cake-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--classifier-loss-weight", type=float, default=0.25)
    parser.add_argument("--prompt-overlap-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--recovery-start-step", type=int, default=400)
    parser.add_argument("--recovery-interval", type=int, default=8)
    parser.add_argument("--recovery-horizons", default="8,16,32")
    parser.add_argument(
        "--sampling-strategy",
        choices=("uniform_records", "balanced_capabilities"),
        default="uniform_records",
    )
    parser.add_argument("--general-curriculum")
    parser.add_argument(
        "--general-batch-size", type=int, default=0
    )
    parser.add_argument(
        "--general-loss-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--general-sampling-strategy",
        choices=("uniform_records", "balanced_capabilities"),
        default="uniform_records",
    )
    parser.add_argument("--anchor-bundle")
    parser.add_argument("--anchor-budget-index", type=int, default=-1)
    parser.add_argument("--anchor-batch-size", type=int, default=0)
    parser.add_argument("--anchor-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--anchor-sampling-strategy",
        choices=("uniform_records", "balanced_capabilities"),
        default="balanced_capabilities",
    )
    parser.add_argument("--decoding-contract")
    parser.add_argument(
        "--exclude-overlength-prompts",
        action="store_true",
        help=(
            "Exclude, enumerate, and hash source prompts whose tokenized "
            "length leaves no supervised token inside --max-tokens."
        ),
    )
    parser.add_argument(
        "--frozen-runtime-fake-int8",
        action="store_true",
        help=(
            "On CUDA, train only selected task cakes against temporary "
            "deployment-channel fake-QInt8 frozen weights, then restore "
            "every frozen weight exactly before checkpointing."
        ),
    )
    parser.add_argument("--same-tokenizer-source")
    parser.add_argument(
        "--source-distillation-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--source-distillation-top-k",
        type=int,
        default=SAME_TOKENIZER_LOGIT_TOP_K,
    )
    parser.add_argument(
        "--source-representation-distillation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--parent-logit-preservation-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--prompt-identity-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--target-control",
        choices=("identity", "deterministic_derangement"),
        default="identity",
        help=(
            "identity for the real run, or a deterministic response "
            "derangement for a causal negative control"
        ),
    )
    parser.add_argument("--target-derangement-seed", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    manifest = train_full_core(
        bundle_path=args.bundle,
        layercake_root=args.layercake_root,
        parent_path=args.parent,
        canonical_abi_path=args.canonical_abi,
        output_path=args.output,
        budget_index=args.budget_index,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        trainable_scope=args.trainable_scope,
        trainable_task_cake_routes=tuple(
            int(value)
            for value in args.trainable_task_cake_routes.split(",")
            if value.strip()
        ),
        expanded_task_cake_rank=args.expanded_task_cake_rank,
        shared_learning_rate=args.shared_learning_rate,
        cake_learning_rate=args.cake_learning_rate,
        classifier_loss_weight=args.classifier_loss_weight,
        prompt_overlap_loss_weight=args.prompt_overlap_loss_weight,
        max_tokens=args.max_tokens,
        recovery_start_step=args.recovery_start_step,
        recovery_interval=args.recovery_interval,
        recovery_horizons=tuple(
            int(value)
            for value in args.recovery_horizons.split(",")
            if value.strip()
        ),
        sampling_strategy=args.sampling_strategy,
        general_curriculum_path=args.general_curriculum,
        general_batch_size=args.general_batch_size,
        general_loss_weight=args.general_loss_weight,
        general_sampling_strategy=args.general_sampling_strategy,
        anchor_bundle_path=args.anchor_bundle,
        anchor_budget_index=args.anchor_budget_index,
        anchor_batch_size=args.anchor_batch_size,
        anchor_loss_weight=args.anchor_loss_weight,
        anchor_sampling_strategy=args.anchor_sampling_strategy,
        decoding_contract_path=args.decoding_contract,
        exclude_overlength_prompts=args.exclude_overlength_prompts,
        frozen_runtime_fake_int8=args.frozen_runtime_fake_int8,
        same_tokenizer_source_path=args.same_tokenizer_source,
        source_distillation_weight=args.source_distillation_weight,
        source_distillation_top_k=args.source_distillation_top_k,
        source_representation_distillation_weight=(
            args.source_representation_distillation_weight
        ),
        parent_logit_preservation_weight=(
            args.parent_logit_preservation_weight
        ),
        prompt_identity_loss_weight=args.prompt_identity_loss_weight,
        target_control_mode=args.target_control,
        target_derangement_seed=args.target_derangement_seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
