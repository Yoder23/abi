"""Versioned loader for three- and six-block LayerCake English cores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MethodType
from typing import Any

from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DynamicCache

from .layercake_host import (
    PromptIdentityBridge,
    _canonical_json_bytes,
    _import_layercake_runtime,
    _sha256_file,
)


def _load_symbolic_surface_substrate(
    path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Load one exact non-neural English-form substrate, if declared."""

    declared = metadata.get("symbolic_surface_substrate")
    if declared is None:
        return None
    if (
        not isinstance(declared, dict)
        or declared.get("format")
        not in {
            "abi-layercake-symbolic-substrate-graft/1",
            "abi-layercake-symbolic-substrate-extension/1",
        }
        or declared.get("teacher_present_at_inference") is not False
        or declared.get("source_teacher_text_retained") is not False
        or declared.get("source_neural_parameters_copied") != 0
        or declared.get("source_task_cakes_copied") != 0
        or declared.get("source_classifier_parameters_copied") != 0
        or declared.get("maximum_active_handlers_per_sequence") != 1
    ):
        raise RuntimeError("symbolic substrate contract is invalid")
    payload_path = (path / str(declared.get("path", ""))).resolve()
    if payload_path.parent != path or not payload_path.is_file():
        raise RuntimeError("symbolic substrate path escaped its core artifact")
    payload = payload_path.read_bytes()
    if (
        len(payload) != int(declared.get("payload_bytes", -1))
        or hashlib.sha256(payload).hexdigest()
        != declared.get("payload_sha256")
    ):
        raise RuntimeError("symbolic substrate bytes are stale or tampered")
    contract = json.loads(payload.decode("utf-8"))
    if (
        _canonical_json_bytes(contract) != payload
        or list(contract.get("handlers", ()))
        != list(declared.get("handlers", ()))
    ):
        raise RuntimeError("symbolic substrate canonical contract changed")
    return contract


CAPABILITY_CAKE_ORDER = (
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
)
CAPABILITY_CAKE_CANONICAL_ROUTES = (
    0, 1, 4, 4, 8, 6, 8, 3, 4, 4, 7, 7, 5, 2
)
CAPABILITY_CAKE_ARCHITECTURE = (
    "layercake-shallow-sparse-english/3-three-block-"
    "rank64-capability-cakes"
)
PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE = (
    "layercake-shallow-sparse-english/4-three-block-"
    "persistent-capability-prefix-p8-rank64-cakes"
)
LAYERWISE_CAPABILITY_CONTROL_ARCHITECTURE = (
    "layercake-shallow-sparse-english/5-three-block-"
    "layerwise-capability-control-rank64-cakes"
)
TASK_ROUTE_LAYERWISE_CONTROL_ARCHITECTURE = (
    "layercake-shallow-sparse-english/10-three-block-"
    "task-route-layerwise-control-rank64-cakes"
)
TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE = (
    "layercake-shallow-sparse-english/11-three-block-"
    "task-route-layerwise-control-rank64-cakes-prompt-pointer32"
)
TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_ARCHITECTURE = (
    "layercake-shallow-sparse-english/12-three-block-"
    "task-route-layerwise-control-rank64-cakes-selective-prompt-pointer32"
)
DEEP_CAPABILITY_ADAPTER_ARCHITECTURE = (
    "layercake-shallow-sparse-english/6-three-block-"
    "deep-rank32-capability-adapters-rank64-cakes"
)
SHARED_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE = (
    "layercake-shallow-sparse-english/7-three-block-"
    "shared-deep-rank32-capability-adapters-rank64-cakes"
)
DEEP_REUSED_CAPABILITY_CAKE_ARCHITECTURE = (
    "layercake-shallow-sparse-english/8-three-block-"
    "deep-reused-rank64-capability-cakes"
)
GATED_DEEP_REUSED_CAPABILITY_CAKE_ARCHITECTURE = (
    "layercake-shallow-sparse-english/9-three-block-"
    "gated-deep-reused-rank64-capability-cakes"
)
DEEP_CAPABILITY_ADAPTER_RANK = 32
PERSISTENT_PREFIX_LENGTH = 8
PERSISTENT_PREFIX_ROUTER_BUCKETS = 4096
PERSISTENT_PREFIX_ROUTER_WIDTH = 32
PROMPT_IDENTITY_RANK = 32


@dataclass(frozen=True)
class ABIEnglishCoreConfig:
    vocab_size: int = 50257
    width: int = 768
    layers: int = 3
    heads: int = 12
    max_tokens: int = 1024
    task_cakes: int = 10
    task_cake_rank: int = 64
    capability_cake_order: tuple[str, ...] | list[str] = ()
    capability_cake_canonical_routes: tuple[int, ...] | list[int] = ()
    capability_prefix_length: int = 0
    capability_router_buckets: int = 0
    capability_router_width: int = 0
    capability_control_width: int = 0
    task_route_layerwise_control: bool = False
    prompt_identity_rank: int = 0
    prompt_identity_selective: bool = False
    capability_adapter_rank: int = 0
    capability_adapter_shared_across_layers: bool = False
    deep_reused_capability_cakes: bool = False
    deep_cake_gate_layers: int = 0
    architecture_version: str = (
        "layercake-shallow-sparse-english/1-three-block-task-cakes"
    )

    def __post_init__(self) -> None:
        if self.layers not in {3, 6}:
            raise ValueError("ABI English core layers must be three or six")
        if self.width != 768 or self.heads != 12:
            raise ValueError("ABI English core width or head count changed")
        if self.width % self.heads:
            raise ValueError("width must divide evenly across attention heads")
        capability_topology = self.task_cakes == 14
        if self.task_cakes not in {10, 14} or self.task_cake_rank not in {64, 256}:
            raise ValueError("instruction-cake topology changed")
        if capability_topology and (
            self.layers != 3
            or self.task_cake_rank != 64
            or tuple(self.capability_cake_order) != CAPABILITY_CAKE_ORDER
            or tuple(self.capability_cake_canonical_routes)
            != CAPABILITY_CAKE_CANONICAL_ROUTES
        ):
            raise ValueError("capability-cake topology changed")
        if not capability_topology and (
            self.capability_cake_order
            or self.capability_cake_canonical_routes
        ):
            raise ValueError("canonical task cakes declare capability routing")
        prefix_topology = self.capability_prefix_length > 0
        task_route_control_topology = self.task_route_layerwise_control
        prompt_identity_topology = self.prompt_identity_rank > 0
        if self.prompt_identity_selective and not prompt_identity_topology:
            raise ValueError("selective prompt identity bridge is absent")
        control_topology = (
            self.capability_control_width > 0
            and not task_route_control_topology
        )
        adapter_topology = self.capability_adapter_rank > 0
        reused_cake_topology = self.deep_reused_capability_cakes
        gated_reused_cake_topology = self.deep_cake_gate_layers > 0
        if self.capability_adapter_shared_across_layers and not (
            adapter_topology
        ):
            raise ValueError("shared capability adapter is absent")
        if sum((prefix_topology, control_topology, adapter_topology)) > 1:
            raise ValueError("capability conditioning topologies conflict")
        if prefix_topology and (
            not capability_topology
            or self.capability_prefix_length != PERSISTENT_PREFIX_LENGTH
            or self.capability_router_buckets
            != PERSISTENT_PREFIX_ROUTER_BUCKETS
            or self.capability_router_width
            != PERSISTENT_PREFIX_ROUTER_WIDTH
        ):
            raise ValueError("persistent capability-prefix topology changed")
        if not prefix_topology and (
            (self.capability_router_buckets or self.capability_router_width)
            and not (
                control_topology
                or adapter_topology
                or reused_cake_topology
                or gated_reused_cake_topology
            )
        ):
            raise ValueError("capability router exists without a prefix")
        if control_topology and (
            not capability_topology
            or self.capability_control_width != self.width
            or self.capability_router_buckets
            != PERSISTENT_PREFIX_ROUTER_BUCKETS
            or self.capability_router_width
            != PERSISTENT_PREFIX_ROUTER_WIDTH
        ):
            raise ValueError("layerwise capability-control topology changed")
        if task_route_control_topology and (
            capability_topology
            or self.task_cakes != 10
            or self.task_cake_rank != 64
            or self.capability_control_width != self.width
            or self.capability_router_buckets
            or self.capability_router_width
            or self.capability_cake_order
            or self.capability_cake_canonical_routes
        ):
            raise ValueError("task-route layerwise-control topology changed")
        if prompt_identity_topology and (
            not task_route_control_topology
            or self.prompt_identity_rank != PROMPT_IDENTITY_RANK
            or capability_topology
            or self.task_cakes != 10
            or self.task_cake_rank != 64
        ):
            raise ValueError("prompt-identity carriage topology changed")
        if adapter_topology and (
            not capability_topology
            or self.capability_adapter_rank
            != DEEP_CAPABILITY_ADAPTER_RANK
            or self.capability_router_buckets
            != PERSISTENT_PREFIX_ROUTER_BUCKETS
            or self.capability_router_width
            != PERSISTENT_PREFIX_ROUTER_WIDTH
        ):
            raise ValueError("deep capability-adapter topology changed")
        if reused_cake_topology and (
            not capability_topology
            or prefix_topology
            or control_topology
            or adapter_topology
            or self.capability_router_buckets
            != PERSISTENT_PREFIX_ROUTER_BUCKETS
            or self.capability_router_width
            != PERSISTENT_PREFIX_ROUTER_WIDTH
        ):
            raise ValueError("deep reused capability-cake topology changed")
        if gated_reused_cake_topology and (
            not capability_topology
            or self.deep_cake_gate_layers != self.layers
            or prefix_topology
            or control_topology
            or adapter_topology
            or reused_cake_topology
            or self.capability_router_buckets
            != PERSISTENT_PREFIX_ROUTER_BUCKETS
            or self.capability_router_width
            != PERSISTENT_PREFIX_ROUTER_WIDTH
        ):
            raise ValueError(
                "gated deep reused capability-cake topology changed"
            )
        if self.task_cake_rank == 256 and self.layers != 3:
            raise ValueError("expanded instruction cakes require three blocks")
        expected_architecture = (
            TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_ARCHITECTURE
            if self.prompt_identity_selective
            else TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE
            if prompt_identity_topology
            else TASK_ROUTE_LAYERWISE_CONTROL_ARCHITECTURE
            if task_route_control_topology
            else GATED_DEEP_REUSED_CAPABILITY_CAKE_ARCHITECTURE
            if gated_reused_cake_topology
            else DEEP_REUSED_CAPABILITY_CAKE_ARCHITECTURE
            if reused_cake_topology
            else (
                SHARED_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE
                if self.capability_adapter_shared_across_layers
                else DEEP_CAPABILITY_ADAPTER_ARCHITECTURE
            )
            if adapter_topology
            else (
                LAYERWISE_CAPABILITY_CONTROL_ARCHITECTURE
                if control_topology
                else (
                    PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE
                    if prefix_topology
                    else (
                        CAPABILITY_CAKE_ARCHITECTURE
                        if capability_topology
                        else (
                            "layercake-shallow-sparse-english/1-three-block-task-cakes"
                            if self.layers == 3 and self.task_cake_rank == 64
                            else (
                                "layercake-shallow-sparse-english/2-six-block-task-cakes"
                                if self.layers == 6 and self.task_cake_rank == 64
                                else (
                                    "layercake-shallow-sparse-english/2-three-block-"
                                    "rank256-task-cakes"
                                )
                            )
                        )
                    )
                )
            )
        )
        if self.architecture_version != expected_architecture:
            raise ValueError("instruction-cake architecture version changed")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)


def _legacy_cache(past_key_values: Any) -> tuple[tuple[torch.Tensor, ...], ...]:
    if past_key_values is None:
        return ()
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    return tuple(past_key_values)


def _persistent_prefix_forward(
    self,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    task_routes: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
) -> dict[str, Any]:
    """Run one selected KV prefix while exposing only real tokens in cache."""

    if input_ids.ndim != 2:
        raise ValueError("input ids must be [batch, tokens]")
    batch, tokens = input_ids.shape
    real_past = _legacy_cache(past_key_values)
    real_past_length = (
        int(real_past[0][0].shape[2]) if real_past else 0
    )
    router_hidden = self.capability_router_embedding(
        torch.remainder(input_ids, self.config.capability_router_buckets)
    )
    positions = torch.arange(tokens, device=input_ids.device)[None]
    if prompt_lengths is not None:
        router_mask = positions < prompt_lengths[:, None]
    elif attention_mask is not None:
        router_mask = attention_mask.to(dtype=torch.bool)
    else:
        router_mask = torch.ones(
            batch, tokens, dtype=torch.bool, device=input_ids.device
        )
    router_weights = router_mask.to(router_hidden.dtype)
    router_summary = (
        router_hidden * router_weights[:, :, None]
    ).sum(dim=1) / router_weights.sum(dim=1, keepdim=True).clamp_min(1)
    task_logits = self.capability_router(router_summary)
    if task_routes is None:
        if real_past:
            raise ValueError("cached decode requires the prefill task route")
        task_routes = task_logits.argmax(dim=-1)
    else:
        task_routes = task_routes.to(input_ids.device).long().flatten()
    selected_keys = self.capability_prefix_keys.index_select(0, task_routes)
    selected_values = self.capability_prefix_values.index_select(
        0, task_routes
    )
    prefixed_past = []
    for layer in range(int(self.config.layers)):
        if real_past:
            real_key, real_value = real_past[layer]
        else:
            real_key = selected_keys[:, layer, :, :0, :]
            real_value = selected_values[:, layer, :, :0, :]
        prefixed_past.append(
            (
                torch.cat((selected_keys[:, layer], real_key), dim=2),
                torch.cat((selected_values[:, layer], real_value), dim=2),
            )
        )
    position_ids = torch.arange(
        real_past_length,
        real_past_length + tokens,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    transformer_attention = None
    if attention_mask is not None:
        prefix_and_past = torch.ones(
            batch,
            int(self.config.capability_prefix_length) + real_past_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        transformer_attention = torch.cat(
            (prefix_and_past, attention_mask), dim=1
        )
    result = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=transformer_attention,
        past_key_values=DynamicCache.from_legacy_cache(
            tuple(prefixed_past)
        ),
        use_cache=use_cache,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    adapted = self._dispatch(hidden, task_routes)
    logits = F.linear(adapted, self.output_weight)
    public_cache = None
    if use_cache:
        prefixed_result = _legacy_cache(result.past_key_values)
        prefix_length = int(self.config.capability_prefix_length)
        public_cache = tuple(
            (key[:, :, prefix_length:], value[:, :, prefix_length:])
            for key, value in prefixed_result
        )
    self.last_task_logits = task_logits
    self.last_task_routes = task_routes.detach()
    return {
        "logits": logits,
        "past_key_values": public_cache,
        "task_logits": task_logits,
        "task_routes": task_routes,
        "hidden": adapted,
    }


def install_persistent_capability_prefix(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Attach the locked p8 prefix tensors and hashed neural router."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.capability_router_embedding = torch.nn.Embedding(
        PERSISTENT_PREFIX_ROUTER_BUCKETS,
        PERSISTENT_PREFIX_ROUTER_WIDTH,
    ).to(device=device, dtype=dtype)
    model.capability_router = torch.nn.Linear(
        PERSISTENT_PREFIX_ROUTER_WIDTH,
        len(CAPABILITY_CAKE_ORDER),
    ).to(device=device, dtype=dtype)
    shape = (
        len(CAPABILITY_CAKE_ORDER),
        int(config.layers),
        int(config.heads),
        PERSISTENT_PREFIX_LENGTH,
        int(config.width) // int(config.heads),
    )
    model.capability_prefix_keys = torch.nn.Parameter(
        torch.empty(shape, device=device, dtype=dtype)
    )
    model.capability_prefix_values = torch.nn.Parameter(
        torch.empty(shape, device=device, dtype=dtype)
    )
    if initialize:
        torch.nn.init.normal_(
            model.capability_router_embedding.weight, std=0.02
        )
        torch.nn.init.zeros_(model.capability_router.weight)
        torch.nn.init.zeros_(model.capability_router.bias)
        torch.nn.init.normal_(model.capability_prefix_keys, std=0.02)
        torch.nn.init.zeros_(model.capability_prefix_values)
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        ),
        capability_prefix_length=PERSISTENT_PREFIX_LENGTH,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        architecture_version=PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = CAPABILITY_CAKE_CANONICAL_ROUTES
    model._abi_persistent_capability_prefix = True
    model.forward = MethodType(_persistent_prefix_forward, model)


def _add_layerwise_control(module, args, kwargs):
    """Inject the already selected control before one frozen block."""

    control = getattr(module, "_abi_selected_capability_control", None)
    if control is None:
        raise RuntimeError("layerwise capability control was not selected")
    return (args[0] + control[:, None], *args[1:]), kwargs


def _layerwise_control_forward(
    self,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    task_routes: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
) -> dict[str, Any]:
    """Condition every real-token KV write without adding cache positions."""

    if input_ids.ndim != 2:
        raise ValueError("input ids must be [batch, tokens]")
    batch, tokens = input_ids.shape
    router_hidden = self.capability_router_embedding(
        torch.remainder(input_ids, self.config.capability_router_buckets)
    )
    positions = torch.arange(tokens, device=input_ids.device)[None]
    if prompt_lengths is not None:
        router_mask = positions < prompt_lengths[:, None]
    elif attention_mask is not None:
        router_mask = attention_mask.to(dtype=torch.bool)
    else:
        router_mask = torch.ones(
            batch, tokens, dtype=torch.bool, device=input_ids.device
        )
    weights = router_mask.to(router_hidden.dtype)
    summary = (router_hidden * weights[:, :, None]).sum(dim=1) / (
        weights.sum(dim=1, keepdim=True).clamp_min(1)
    )
    task_logits = self.capability_router(summary)
    if task_routes is None:
        if past_key_values is not None:
            raise ValueError("cached decode requires the prefill task route")
        task_routes = task_logits.argmax(dim=-1)
    else:
        task_routes = task_routes.to(input_ids.device).long().flatten()
    selected = self.capability_control_vectors.index_select(0, task_routes)
    for layer, block in enumerate(self.transformer.h):
        block._abi_selected_capability_control = selected[:, layer]
    result = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    adapted = self._dispatch(hidden, task_routes)
    logits = F.linear(adapted, self.output_weight)
    self.last_task_logits = task_logits
    self.last_task_routes = task_routes.detach()
    return {
        "logits": logits,
        "past_key_values": result.past_key_values,
        "task_logits": task_logits,
        "task_routes": task_routes,
        "hidden": adapted,
    }


def install_layerwise_capability_control(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Attach three physically selected controls and the hashed router."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.capability_router_embedding = torch.nn.Embedding(
        PERSISTENT_PREFIX_ROUTER_BUCKETS,
        PERSISTENT_PREFIX_ROUTER_WIDTH,
    ).to(device=device, dtype=dtype)
    model.capability_router = torch.nn.Linear(
        PERSISTENT_PREFIX_ROUTER_WIDTH,
        len(CAPABILITY_CAKE_ORDER),
    ).to(device=device, dtype=dtype)
    model.capability_control_vectors = torch.nn.Parameter(
        torch.empty(
            len(CAPABILITY_CAKE_ORDER),
            int(config.layers),
            int(config.width),
            device=device,
            dtype=dtype,
        )
    )
    if initialize:
        torch.nn.init.normal_(
            model.capability_router_embedding.weight, std=0.02
        )
        torch.nn.init.zeros_(model.capability_router.weight)
        torch.nn.init.zeros_(model.capability_router.bias)
        torch.nn.init.zeros_(model.capability_control_vectors)
    for block in model.transformer.h:
        block.register_forward_pre_hook(
            _add_layerwise_control, with_kwargs=True
        )
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        ),
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        capability_control_width=int(config.width),
        architecture_version=LAYERWISE_CAPABILITY_CONTROL_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = CAPABILITY_CAKE_CANONICAL_ROUTES
    model._abi_layerwise_capability_control = True
    model.forward = MethodType(_layerwise_control_forward, model)


def _task_route_layerwise_control_forward(
    self,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    task_routes: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
) -> dict[str, Any]:
    """Use the proven 10-route classifier, then condition persistent KV state."""

    if input_ids.ndim != 2:
        raise ValueError("input ids must be [batch, tokens]")
    task_logits = None
    if task_routes is None:
        if past_key_values is not None:
            raise ValueError("cached decode requires the prefill task route")
        zero_control = torch.zeros(
            input_ids.shape[0],
            int(self.config.width),
            dtype=self.task_route_control_vectors.dtype,
            device=input_ids.device,
        )
        for block in self.transformer.h:
            block._abi_selected_capability_control = zero_control
        routing_result = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        routing_summary = self._prompt_summary(
            routing_result.last_hidden_state,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )
        task_logits = self.task_classifier(routing_summary)
        task_routes = task_logits.argmax(dim=-1)
    else:
        task_routes = task_routes.to(input_ids.device).long().flatten()
    selected = self.task_route_control_vectors.index_select(0, task_routes)
    for layer, block in enumerate(self.transformer.h):
        block._abi_selected_capability_control = selected[:, layer]
    result = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    if task_logits is None and past_key_values is None:
        summary = self._prompt_summary(
            hidden,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )
        task_logits = self.task_classifier(summary)
    adapted = self._dispatch(hidden, task_routes)
    logits = F.linear(adapted, self.output_weight)
    self.last_task_logits = task_logits
    self.last_task_routes = task_routes.detach()
    return {
        "logits": logits,
        "past_key_values": result.past_key_values,
        "task_logits": task_logits,
        "task_routes": task_routes,
        "hidden": adapted,
    }


def install_task_route_layerwise_control(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Attach one selected three-vector control to each existing task route."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.task_route_control_vectors = torch.nn.Parameter(
        torch.empty(
            int(config.task_cakes),
            int(config.layers),
            int(config.width),
            device=device,
            dtype=dtype,
        )
    )
    if initialize:
        torch.nn.init.zeros_(model.task_route_control_vectors)
    for block in model.transformer.h:
        block.register_forward_pre_hook(
            _add_layerwise_control, with_kwargs=True
        )
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=int(config.task_cakes),
        task_cake_rank=int(config.task_cake_rank),
        capability_control_width=int(config.width),
        task_route_layerwise_control=True,
        architecture_version=TASK_ROUTE_LAYERWISE_CONTROL_ARCHITECTURE,
    )
    model._abi_layerwise_capability_control = True
    model._abi_task_route_layerwise_control = True
    model._abi_capability_cake_routes = tuple(
        range(int(config.task_cakes))
    )
    model.forward = MethodType(_task_route_layerwise_control_forward, model)


def install_prompt_identity_carriage(
    model: torch.nn.Module,
    *,
    initialize: bool,
    selective: bool = False,
) -> None:
    """Attach a sparse prompt-position pointer to the proven route-control core."""

    if not getattr(model, "_abi_task_route_layerwise_control", False):
        raise ValueError(
            "prompt-identity carriage requires task-route layerwise control"
        )
    config = model.config
    bridge = PromptIdentityBridge(
        width=int(config.width),
        rank=PROMPT_IDENTITY_RANK,
        routes=int(config.task_cakes),
    ).to(
        device=model.transformer.wte.weight.device,
        dtype=model.transformer.wte.weight.dtype,
    )
    if not initialize:
        # The checkpoint load immediately replaces these deterministic shapes.
        bridge.requires_grad_(False)
    model.prompt_identity = bridge
    # Keep one registered state-dict path.  The runtime alias must not register
    # the same module a second time under another checkpoint key.
    object.__setattr__(model, "_abi_prompt_identity_bridge", bridge)
    model._abi_prompt_identity_carriage = True
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=int(config.task_cakes),
        task_cake_rank=int(config.task_cake_rank),
        capability_control_width=int(config.capability_control_width),
        task_route_layerwise_control=True,
        prompt_identity_rank=PROMPT_IDENTITY_RANK,
        prompt_identity_selective=bool(selective),
        architecture_version=(
            TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_ARCHITECTURE
            if selective
            else TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE
        ),
    )


def _make_deep_adapter_hook(adapters):
    """Create a traceable route-selected nonlinear pre-block adapter."""

    def hook(module, args, kwargs):
        hidden = args[0]
        routes = getattr(module, "_abi_selected_capability_routes", None)
        if routes is None:
            raise RuntimeError("deep capability adapter route was not selected")
        norm_weight = torch.stack(
            [adapter.norm.weight for adapter in adapters]
        ).index_select(0, routes)
        norm_bias = torch.stack(
            [adapter.norm.bias for adapter in adapters]
        ).index_select(0, routes)
        down_weight = torch.stack(
            [adapter.down.weight for adapter in adapters]
        ).index_select(0, routes)
        up_weight = torch.stack(
            [adapter.up.weight for adapter in adapters]
        ).index_select(0, routes)
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
        normalized = (hidden - mean) * torch.rsqrt(variance + 1.0e-5)
        normalized = (
            normalized * norm_weight[:, None] + norm_bias[:, None]
        )
        low = torch.bmm(normalized, down_weight.transpose(1, 2))
        update = torch.bmm(F.silu(low), up_weight.transpose(1, 2))
        return (hidden + update, *args[1:]), kwargs

    return hook


def _deep_adapter_forward(
    self,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    task_routes: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
) -> dict[str, Any]:
    """Apply one nonlinear capability adapter before every frozen block."""

    if input_ids.ndim != 2:
        raise ValueError("input ids must be [batch, tokens]")
    batch, tokens = input_ids.shape
    router_hidden = self.capability_router_embedding(
        torch.remainder(input_ids, self.config.capability_router_buckets)
    )
    positions = torch.arange(tokens, device=input_ids.device)[None]
    if prompt_lengths is not None:
        router_mask = positions < prompt_lengths[:, None]
    elif attention_mask is not None:
        router_mask = attention_mask.to(dtype=torch.bool)
    else:
        router_mask = torch.ones(
            batch, tokens, dtype=torch.bool, device=input_ids.device
        )
    weights = router_mask.to(router_hidden.dtype)
    summary = (router_hidden * weights[:, :, None]).sum(dim=1) / (
        weights.sum(dim=1, keepdim=True).clamp_min(1)
    )
    task_logits = self.capability_router(summary)
    if task_routes is None:
        if past_key_values is not None:
            raise ValueError("cached decode requires the prefill task route")
        task_routes = task_logits.argmax(dim=-1)
    else:
        task_routes = task_routes.to(input_ids.device).long().flatten()
    for block in self.transformer.h:
        block._abi_selected_capability_routes = task_routes
    result = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    adapted = self._dispatch(hidden, task_routes)
    logits = F.linear(adapted, self.output_weight)
    self.last_task_logits = task_logits
    self.last_task_routes = task_routes.detach()
    return {
        "logits": logits,
        "past_key_values": result.past_key_values,
        "task_logits": task_logits,
        "task_routes": task_routes,
        "hidden": adapted,
    }


def install_deep_capability_adapters(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Attach three rank-32 nonlinear adapters per capability."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.capability_router_embedding = torch.nn.Embedding(
        PERSISTENT_PREFIX_ROUTER_BUCKETS,
        PERSISTENT_PREFIX_ROUTER_WIDTH,
    ).to(device=device, dtype=dtype)
    model.capability_router = torch.nn.Linear(
        PERSISTENT_PREFIX_ROUTER_WIDTH,
        len(CAPABILITY_CAKE_ORDER),
    ).to(device=device, dtype=dtype)
    cake_type = type(model.task_cakes[0])
    model.capability_layer_adapters = torch.nn.ModuleList(
        torch.nn.ModuleList(
            cake_type(int(config.width), DEEP_CAPABILITY_ADAPTER_RANK).to(
                device=device, dtype=dtype
            )
            for _ in CAPABILITY_CAKE_ORDER
        )
        for _ in range(int(config.layers))
    )
    if initialize:
        torch.nn.init.normal_(
            model.capability_router_embedding.weight, std=0.02
        )
        torch.nn.init.zeros_(model.capability_router.weight)
        torch.nn.init.zeros_(model.capability_router.bias)
        for layer in model.capability_layer_adapters:
            for adapter in layer:
                torch.nn.init.zeros_(adapter.up.weight)
    for block, adapters in zip(
        model.transformer.h, model.capability_layer_adapters
    ):
        block.register_forward_pre_hook(
            _make_deep_adapter_hook(adapters), with_kwargs=True
        )
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        capability_adapter_rank=DEEP_CAPABILITY_ADAPTER_RANK,
        architecture_version=DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = CAPABILITY_CAKE_CANONICAL_ROUTES
    model._abi_deep_capability_adapters = True
    model.forward = MethodType(_deep_adapter_forward, model)


def install_shared_deep_capability_adapters(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Attach one selected rank-32 adapter reused before all three blocks."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.capability_router_embedding = torch.nn.Embedding(
        PERSISTENT_PREFIX_ROUTER_BUCKETS,
        PERSISTENT_PREFIX_ROUTER_WIDTH,
    ).to(device=device, dtype=dtype)
    model.capability_router = torch.nn.Linear(
        PERSISTENT_PREFIX_ROUTER_WIDTH,
        len(CAPABILITY_CAKE_ORDER),
    ).to(device=device, dtype=dtype)
    cake_type = type(model.task_cakes[0])
    model.capability_shared_adapters = torch.nn.ModuleList(
        cake_type(int(config.width), DEEP_CAPABILITY_ADAPTER_RANK).to(
            device=device, dtype=dtype
        )
        for _ in CAPABILITY_CAKE_ORDER
    )
    if initialize:
        torch.nn.init.normal_(
            model.capability_router_embedding.weight, std=0.02
        )
        torch.nn.init.zeros_(model.capability_router.weight)
        torch.nn.init.zeros_(model.capability_router.bias)
        for adapter in model.capability_shared_adapters:
            torch.nn.init.zeros_(adapter.up.weight)
    for block in model.transformer.h:
        block.register_forward_pre_hook(
            _make_deep_adapter_hook(model.capability_shared_adapters),
            with_kwargs=True,
        )
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        capability_adapter_rank=DEEP_CAPABILITY_ADAPTER_RANK,
        capability_adapter_shared_across_layers=True,
        architecture_version=SHARED_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = CAPABILITY_CAKE_CANONICAL_ROUTES
    model._abi_deep_capability_adapters = True
    model._abi_shared_deep_capability_adapters = True
    model.forward = MethodType(_deep_adapter_forward, model)


def _deep_reused_cake_forward(
    self,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    task_routes: torch.Tensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
) -> dict[str, Any]:
    """Reuse one selected rank-64 capability cake before every block."""

    if input_ids.ndim != 2:
        raise ValueError("input ids must be [batch, tokens]")
    batch, tokens = input_ids.shape
    router_hidden = self.capability_router_embedding(
        torch.remainder(input_ids, self.config.capability_router_buckets)
    )
    positions = torch.arange(tokens, device=input_ids.device)[None]
    if prompt_lengths is not None:
        router_mask = positions < prompt_lengths[:, None]
    elif attention_mask is not None:
        router_mask = attention_mask.to(dtype=torch.bool)
    else:
        router_mask = torch.ones(
            batch, tokens, dtype=torch.bool, device=input_ids.device
        )
    weights = router_mask.to(router_hidden.dtype)
    summary = (router_hidden * weights[:, :, None]).sum(dim=1) / (
        weights.sum(dim=1, keepdim=True).clamp_min(1)
    )
    task_logits = self.capability_router(summary)
    if task_routes is None:
        if past_key_values is not None:
            raise ValueError("cached decode requires the prefill task route")
        task_routes = task_logits.argmax(dim=-1)
    else:
        task_routes = task_routes.to(input_ids.device).long().flatten()
    for block in self.transformer.h:
        block._abi_selected_capability_routes = task_routes
    result = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=True,
    )
    hidden = result.last_hidden_state
    logits = F.linear(hidden, self.output_weight)
    self.last_task_logits = task_logits
    self.last_task_routes = task_routes.detach()
    return {
        "logits": logits,
        "past_key_values": result.past_key_values,
        "task_logits": task_logits,
        "task_routes": task_routes,
        "hidden": hidden,
    }


def install_deep_reused_capability_cakes(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Relocate the selected installed cake to all three block inputs."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.capability_router_embedding = torch.nn.Embedding(
        PERSISTENT_PREFIX_ROUTER_BUCKETS,
        PERSISTENT_PREFIX_ROUTER_WIDTH,
    ).to(device=device, dtype=dtype)
    model.capability_router = torch.nn.Linear(
        PERSISTENT_PREFIX_ROUTER_WIDTH,
        len(CAPABILITY_CAKE_ORDER),
    ).to(device=device, dtype=dtype)
    if initialize:
        torch.nn.init.normal_(
            model.capability_router_embedding.weight, std=0.02
        )
        torch.nn.init.zeros_(model.capability_router.weight)
        torch.nn.init.zeros_(model.capability_router.bias)
    for block in model.transformer.h:
        block.register_forward_pre_hook(
            _make_deep_adapter_hook(model.task_cakes),
            with_kwargs=True,
        )
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        deep_reused_capability_cakes=True,
        architecture_version=DEEP_REUSED_CAPABILITY_CAKE_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = CAPABILITY_CAKE_CANONICAL_ROUTES
    model._abi_deep_reused_capability_cakes = True
    model.forward = MethodType(_deep_reused_cake_forward, model)


def _make_gated_reused_cake_hook(cakes, gates, layer_index: int):
    """Select one shared cake and scale its update for one block."""

    def hook(module, args, kwargs):
        hidden = args[0]
        routes = getattr(module, "_abi_selected_capability_routes", None)
        if routes is None:
            raise RuntimeError("gated reused cake route was not selected")
        norm_weight = torch.stack(
            [cake.norm.weight for cake in cakes]
        ).index_select(0, routes)
        norm_bias = torch.stack(
            [cake.norm.bias for cake in cakes]
        ).index_select(0, routes)
        down_weight = torch.stack(
            [cake.down.weight for cake in cakes]
        ).index_select(0, routes)
        up_weight = torch.stack(
            [cake.up.weight for cake in cakes]
        ).index_select(0, routes)
        selected_gate = gates.index_select(0, routes)[:, layer_index]
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
        normalized = (hidden - mean) * torch.rsqrt(variance + 1.0e-5)
        normalized = normalized * norm_weight[:, None] + norm_bias[:, None]
        low = torch.bmm(normalized, down_weight.transpose(1, 2))
        update = torch.bmm(F.silu(low), up_weight.transpose(1, 2))
        return (
            hidden + selected_gate[:, None, None] * update,
            *args[1:],
        ), kwargs

    return hook


def install_gated_deep_reused_capability_cakes(
    model: torch.nn.Module,
    *,
    initialize: bool,
) -> None:
    """Reuse the selected final cake before blocks through 42 gates."""

    config = model.config
    device = model.transformer.wte.weight.device
    dtype = model.transformer.wte.weight.dtype
    model.capability_router_embedding = torch.nn.Embedding(
        PERSISTENT_PREFIX_ROUTER_BUCKETS,
        PERSISTENT_PREFIX_ROUTER_WIDTH,
    ).to(device=device, dtype=dtype)
    model.capability_router = torch.nn.Linear(
        PERSISTENT_PREFIX_ROUTER_WIDTH,
        len(CAPABILITY_CAKE_ORDER),
    ).to(device=device, dtype=dtype)
    model.capability_deep_cake_gates = torch.nn.Parameter(
        torch.zeros(
            len(CAPABILITY_CAKE_ORDER),
            int(config.layers),
            device=device,
            dtype=dtype,
        )
    )
    if initialize:
        torch.nn.init.normal_(
            model.capability_router_embedding.weight, std=0.02
        )
        torch.nn.init.zeros_(model.capability_router.weight)
        torch.nn.init.zeros_(model.capability_router.bias)
    for layer_index, block in enumerate(model.transformer.h):
        block.register_forward_pre_hook(
            _make_gated_reused_cake_hook(
                model.task_cakes,
                model.capability_deep_cake_gates,
                layer_index,
            ),
            with_kwargs=True,
        )
    model.config = ABIEnglishCoreConfig(
        vocab_size=int(config.vocab_size),
        width=int(config.width),
        layers=int(config.layers),
        heads=int(config.heads),
        max_tokens=int(config.max_tokens),
        task_cakes=len(CAPABILITY_CAKE_ORDER),
        task_cake_rank=64,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        deep_cake_gate_layers=int(config.layers),
        architecture_version=GATED_DEEP_REUSED_CAPABILITY_CAKE_ARCHITECTURE,
    )
    model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
    model._abi_capability_cake_routes = CAPABILITY_CAKE_CANONICAL_ROUTES
    model._abi_gated_deep_reused_capability_cakes = True
    model.forward = MethodType(_deep_adapter_forward, model)


def load_layercake_core(
    path: str | Path,
    *,
    layercake_root: str | Path,
    device: str | Any = "cpu",
):
    """Load a hash-bound ABI core through the sealed LayerCake model class."""

    path = Path(path).resolve()
    metadata_path = path / "metadata.json"
    checkpoint_path = path / "model.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint", {}).get("sha256") != _sha256_file(
        checkpoint_path
    ):
        raise RuntimeError("LayerCake core checkpoint identity changed")
    _import_layercake_runtime(Path(layercake_root).resolve())
    from layercake.models.shallow_sparse_english import ShallowSparseEnglishCore

    config = ABIEnglishCoreConfig(**metadata["architecture"])
    model = ShallowSparseEnglishCore(config)
    if config.capability_prefix_length:
        install_persistent_capability_prefix(model, initialize=False)
    elif config.task_route_layerwise_control:
        install_task_route_layerwise_control(model, initialize=False)
    elif config.capability_control_width:
        install_layerwise_capability_control(model, initialize=False)
    elif config.capability_adapter_rank:
        if config.capability_adapter_shared_across_layers:
            install_shared_deep_capability_adapters(
                model, initialize=False
            )
        else:
            install_deep_capability_adapters(model, initialize=False)
    elif config.deep_reused_capability_cakes:
        install_deep_reused_capability_cakes(model, initialize=False)
    elif config.deep_cake_gate_layers:
        install_gated_deep_reused_capability_cakes(
            model, initialize=False
        )
    if config.prompt_identity_rank:
        install_prompt_identity_carriage(
            model,
            initialize=False,
            selective=bool(config.prompt_identity_selective),
        )
    model.load_state_dict(
        load_file(str(checkpoint_path), device=str(device)),
        strict=True,
    )
    if config.task_cakes == 14:
        model._abi_capability_cake_order = CAPABILITY_CAKE_ORDER
        model._abi_capability_cake_routes = (
            CAPABILITY_CAKE_CANONICAL_ROUTES
        )
    decoding = metadata.get(
        "decoding",
        {
            "algorithm": "greedy",
            "no_repeat_ngram_size": 0,
            "allow_prompt_ngrams": False,
            "lexical_repetition_blocking_threshold": 0,
            "lexical_repetition_truncation_threshold": 0,
            "byte_repetition_ceiling": 0.0,
            "byte_repetition_guard_minimum_bytes": 0,
            "prompt_identity_mixture": False,
        },
    )
    if (
        not isinstance(decoding, dict)
        or decoding.get("algorithm")
        not in {"greedy", "deterministic_greedy_with_repetition_controls"}
        or not isinstance(decoding.get("no_repeat_ngram_size"), int)
        or int(decoding["no_repeat_ngram_size"]) < 0
        or not isinstance(decoding.get("allow_prompt_ngrams"), bool)
        or not isinstance(
            decoding.get("lexical_repetition_blocking_threshold", 0), int
        )
        or int(decoding.get("lexical_repetition_blocking_threshold", 0)) < 0
        or not isinstance(
            decoding.get("lexical_repetition_truncation_threshold"), int
        )
        or int(decoding["lexical_repetition_truncation_threshold"]) < 0
        or not isinstance(
            decoding.get("byte_repetition_ceiling", 0.0), (int, float)
        )
        or not 0.0
        <= float(decoding.get("byte_repetition_ceiling", 0.0))
        <= 1.0
        or not isinstance(
            decoding.get("byte_repetition_guard_minimum_bytes", 0), int
        )
        or int(decoding.get("byte_repetition_guard_minimum_bytes", 0)) < 0
        or not isinstance(decoding.get("prompt_identity_mixture"), bool)
        or bool(decoding.get("prompt_identity_mixture"))
        != bool(config.prompt_identity_rank)
        or (
            decoding["allow_prompt_ngrams"]
            and int(decoding["no_repeat_ngram_size"]) <= 0
        )
    ):
        raise RuntimeError("LayerCake core decoding contract is invalid")
    model._abi_decoding = dict(decoding)
    model._abi_symbolic_surface = _load_symbolic_surface_substrate(
        path, metadata
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1_000_000_000
    return model.to(device).eval(), tokenizer, metadata
