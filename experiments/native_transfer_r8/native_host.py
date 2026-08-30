"""Frozen Hugging Face recipients and the generic R8 neural bridge."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

OPERATORS = ("vok", "narel", "tem")
MODULUS = 8
MAXIMUM_PROMPT_TOKENS = 128


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class NativeHostError(RuntimeError):
    """Raised when a host, bridge, or neural execution is not admissible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module: nn.Module) -> str:
    return tensor_state_sha256(module.state_dict())


@dataclass(frozen=True)
class HostSpec:
    key: str
    model_id: str
    revision: str
    architecture_family: str
    encoder_decoder: bool = False


SPECS = {
    "source": HostSpec("source", "distilgpt2", "2290a62682d06624634c1f46a6ad5be0f47f38aa", "GPT2"),
    "qwen2": HostSpec(
        "qwen2",
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "Qwen2",
    ),
    "pythia": HostSpec(
        "pythia",
        "EleutherAI/pythia-160m",
        "50f5173d932e8e61f858120bcb800b97af589f46",
        "GPTNeoX",
    ),
    "t5": HostSpec(
        "t5",
        "t5-base",
        "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
        "T5EncoderDecoder",
        True,
    ),
}


def snapshot_inventory(path: Path) -> dict[str, Any]:
    files = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_file():
            files.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    if not files:
        raise NativeHostError(f"empty model snapshot: {path}")
    return {
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


class FrozenNeuralHost:
    """A frozen model whose full-vocabulary logits own every scored answer."""

    def __init__(self, spec: HostSpec, *, device: str = "cuda") -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.spec = spec
        self.device = torch.device(device)
        snapshot = Path(
            snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True)
        ).resolve()
        if snapshot.name != spec.revision:
            raise NativeHostError(f"snapshot revision changed for {spec.key}")
        self.snapshot = snapshot
        self.inventory = snapshot_inventory(snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.truncation_side = "left"
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        loader = AutoModelForSeq2SeqLM if spec.encoder_decoder else AutoModelForCausalLM
        self.model = loader.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding = self.model.get_input_embeddings()
        self.hidden_width = int(self.embedding.weight.shape[1])
        self.target_token_ids, self.target_text = self._target_tokens()
        self.model_state_sha256 = module_sha256(self.model)

    def _target_tokens(self) -> tuple[list[int], list[str]]:
        choices = (" {}", "{}") if not self.spec.encoder_decoder else ("{}", " {}")
        ids: list[int] = []
        texts: list[str] = []
        for value in range(MODULUS):
            selected: tuple[int, str] | None = None
            for template in choices:
                text = template.format(value)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) == 1:
                    selected = (int(tokens[0]), text)
                    break
            if selected is None:
                raise NativeHostError(
                    f"{self.spec.key} has no single-token canonical digit {value}"
                )
            ids.append(selected[0])
            texts.append(selected[1])
        if len(set(ids)) != MODULUS:
            raise NativeHostError(f"{self.spec.key} digit token IDs are not unique")
        return ids, texts

    def encode(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAXIMUM_PROMPT_TOKENS,
            add_special_tokens=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def logits(
        self,
        prompts: Sequence[str],
        *,
        prefix: torch.Tensor | None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        encoded = self.encode(prompts)
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        token_embeddings = self.embedding(ids)
        prefix_length = 0
        if prefix is not None:
            if prefix.ndim == 2:
                prefix = prefix.unsqueeze(0).expand(ids.shape[0], -1, -1)
            if prefix.shape[0] != ids.shape[0] or prefix.shape[2] != self.hidden_width:
                raise NativeHostError("neural prefix shape changed")
            prefix = prefix.to(device=self.device, dtype=token_embeddings.dtype)
            prefix_length = int(prefix.shape[1])
            token_embeddings = torch.cat((prefix, token_embeddings), dim=1)
            mask = torch.cat(
                (
                    torch.ones((ids.shape[0], prefix_length), dtype=mask.dtype, device=self.device),
                    mask,
                ),
                dim=1,
            )
        if self.spec.encoder_decoder:
            start = int(self.model.config.decoder_start_token_id)
            decoder_ids = torch.full((ids.shape[0], 1), start, dtype=torch.long, device=self.device)
            output = self.model(
                inputs_embeds=token_embeddings,
                attention_mask=mask,
                decoder_input_ids=decoder_ids,
                use_cache=False,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            return output.logits[:, 0, :].float(), output
        output = self.model(
            inputs_embeds=token_embeddings,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last = prefix_length + mask[:, prefix_length:].sum(dim=1) - 1
        batch = torch.arange(ids.shape[0], device=self.device)
        return output.logits[batch, last].float(), output

    def target_ids(self, answers: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [self.target_token_ids[int(value)] for value in answers],
            dtype=torch.long,
            device=self.device,
        )

    def canonical_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor(self.target_token_ids, dtype=torch.long, device=logits.device)
        return torch.softmax(logits.index_select(-1, indices), dim=-1)

    def predictions(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def verify_frozen(self) -> None:
        adapter = getattr(self, "_generic_adapter_set", None)
        if adapter is not None:
            adapter.verify_base_frozen()
            if any(parameter.requires_grad for parameter in adapter.parameters()):
                raise NativeHostError(
                    f"generic adapter remains trainable after freeze: {self.spec.key}"
                )
            return
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise NativeHostError(f"recipient parameters became trainable: {self.spec.key}")
        if module_sha256(self.model) != self.model_state_sha256:
            raise NativeHostError(f"recipient weights changed: {self.spec.key}")


class CanonicalLatentBridge(nn.Module):
    """Host-private generic map from a 3x8x8 package to a soft neural prefix.

    The bridge receives no prompt, program, row identifier, or answer. It can
    encode atomic transition state but cannot execute a program.
    """

    def __init__(self, host: FrozenNeuralHost) -> None:
        super().__init__()
        width = host.hidden_width
        slots = len(OPERATORS) * MODULUS
        self.host_key = host.spec.key
        self.hidden_width = width
        self.base_slots = nn.Parameter(torch.zeros(slots, width))
        self.value_codes = nn.Parameter(torch.zeros(MODULUS, width))
        self.gain = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            value_ids = torch.tensor(host.target_token_ids, device=host.device)
            values = host.embedding(value_ids).detach().float()
            self.value_codes.copy_(values.to(self.value_codes.device))
            for slot in range(slots):
                operator = OPERATORS[slot // MODULUS]
                source = slot % MODULUS
                tokens = host.tokenizer.encode(f" {operator} {source}", add_special_tokens=False)
                if tokens:
                    token_ids = torch.tensor(tokens, device=host.device)
                    base = host.embedding(token_ids).detach().float().mean(dim=0)
                    self.base_slots[slot].copy_(base.to(self.base_slots.device))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        expected = (len(OPERATORS), MODULUS, MODULUS)
        if tuple(latent.shape[1:]) != expected:
            raise NativeHostError(f"canonical latent shape changed: {tuple(latent.shape)}")
        if not torch.isfinite(latent).all():
            raise NativeHostError("canonical latent is non-finite")
        probabilities = latent.float().clamp_min(0)
        normalizer = probabilities.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            normalizer > 0,
            probabilities / normalizer.clamp_min(torch.finfo(torch.float32).tiny),
            torch.full_like(probabilities, 1.0 / MODULUS),
        )
        flat = probabilities.reshape(probabilities.shape[0], -1, MODULUS)
        encoded = torch.einsum("bsv,vh->bsh", flat, self.value_codes.float())
        return (self.base_slots.float().unsqueeze(0) + self.gain * encoded).to(latent.device)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def verify_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise NativeHostError(f"bridge is trainable after freeze: {self.host_key}")


class GenericLoRALinear(nn.Module):
    """Host-private LoRA inserted into recipient computation before reveal."""

    def __init__(self, base: nn.Linear, *, rank: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise NativeHostError("invalid generic recipient LoRA target")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.a = nn.Parameter(torch.empty(base.in_features, self.rank))
        self.b = nn.Parameter(torch.zeros(self.rank, base.out_features))
        nn.init.normal_(self.a, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        update = torch.matmul(torch.matmul(inputs.float(), self.a.float()), self.b.float())
        return base + update.to(base.dtype) / self.rank


class GenericRecipientAdapterSet:
    """Capability-blind LoRA installed inside a frozen recipient.

    These weights are host-private and meta-trained across public capabilities.
    They never enter a capability package.  The LM head is deliberately excluded
    so the recipient's unchanged output projection remains causally downstream.
    """

    def __init__(self, host: FrozenNeuralHost, *, rank: int) -> None:
        if hasattr(host, "_generic_adapter_set"):
            raise NativeHostError(f"generic adapter already installed: {host.spec.key}")
        self.host = host
        self.expected_base_sha256 = host.model_state_sha256
        self.modules: dict[str, GenericLoRALinear] = {}
        excluded = {"lm_head", "embed_out"}
        targets = [
            (name, module)
            for name, module in host.model.named_modules()
            if isinstance(module, nn.Linear)
            and name
            and name.split(".")[-1] not in excluded
        ]
        if not targets:
            raise NativeHostError(f"no generic LoRA targets found: {host.spec.key}")
        for name, module in targets:
            replacement = GenericLoRALinear(module, rank=rank).to(module.weight.device)
            self._replace(name, replacement)
            self.modules[name] = replacement
        host._generic_adapter_set = self
        self.verify_base_frozen()

    def _replace(self, name: str, replacement: nn.Module) -> None:
        parent = self.host.model
        pieces = name.split(".")
        for piece in pieces[:-1]:
            parent = getattr(parent, piece)
        setattr(parent, pieces[-1], replacement)

    def parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for module in self.modules.values()
            for parameter in (module.a, module.b)
        ]

    def state(self) -> dict[str, torch.Tensor]:
        return {
            f"{name}/{suffix}": parameter.detach().cpu().contiguous()
            for name, module in sorted(self.modules.items())
            for suffix, parameter in (("lora_a", module.a), ("lora_b", module.b))
        }

    def load_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {
            f"{name}/{suffix}"
            for name in self.modules
            for suffix in ("lora_a", "lora_b")
        }
        if set(state) != expected:
            raise NativeHostError("generic recipient adapter tensor inventory changed")
        with torch.no_grad():
            for name, module in self.modules.items():
                module.a.copy_(state[f"{name}/lora_a"].to(module.a))
                module.b.copy_(state[f"{name}/lora_b"].to(module.b))

    def base_state_sha256(self) -> str:
        normalized: dict[str, torch.Tensor] = {}
        for name, value in self.host.model.state_dict().items():
            matched = False
            for module_name in self.modules:
                if name in {f"{module_name}.a", f"{module_name}.b"}:
                    matched = True
                    break
                prefix = f"{module_name}.base."
                if name.startswith(prefix):
                    normalized[f"{module_name}." + name.removeprefix(prefix)] = value
                    matched = True
                    break
            if not matched:
                normalized[name] = value
        return tensor_state_sha256(normalized)

    def verify_base_frozen(self) -> None:
        if self.base_state_sha256() != self.expected_base_sha256:
            raise NativeHostError(
                f"recipient base state changed after adapter install: {self.host.spec.key}"
            )
        for module in self.modules.values():
            if any(parameter.requires_grad for parameter in module.base.parameters()):
                raise NativeHostError("recipient base parameter became trainable")

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def inventory(self) -> dict[str, Any]:
        rows = [
            {
                "module": name,
                "rank": module.rank,
                "input_width": module.a.shape[0],
                "output_width": module.b.shape[1],
            }
            for name, module in sorted(self.modules.items())
        ]
        return {
            "module_count": len(rows),
            "trainable_parameters": sum(value.numel() for value in self.parameters()),
            "modules": rows,
            "inventory_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        }


class CapabilityConditionedBridge(CanonicalLatentBridge):
    """Static package prefix plus a capability-blind in-model LoRA interface.

    The LoRA is trained only on public meta-capabilities.  At evaluation the
    bridge still receives only the immutable package, never prompts or labels;
    prompt/package interaction happens in the recipient's own adapted layers.
    """

    def __init__(self, host: FrozenNeuralHost, *, rank: int) -> None:
        super().__init__(host)
        object.__setattr__(self, "_adapters", GenericRecipientAdapterSet(host, rank=rank))

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        yield from super().parameters(recurse=recurse)
        yield from self._adapters.parameters()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
        prefix = super().state_dict(*args, **kwargs)
        return {
            **{f"prefix/{name}": value for name, value in prefix.items()},
            **{f"adapter/{name}": value for name, value in self._adapters.state().items()},
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        prefix = {
            name.removeprefix("prefix/"): value
            for name, value in state_dict.items()
            if name.startswith("prefix/")
        }
        adapters = {
            name.removeprefix("adapter/"): value
            for name, value in state_dict.items()
            if name.startswith("adapter/")
        }
        expected_count = len(prefix) + len(adapters)
        if strict and expected_count != len(state_dict):
            raise NativeHostError("unknown capability-conditioned bridge tensor")
        result = super().load_state_dict(prefix, strict=strict, assign=assign)
        self._adapters.load_state(adapters)
        return result

    def freeze(self) -> None:
        super().freeze()
        self._adapters.freeze()

    def verify_frozen(self) -> None:
        super().verify_frozen()
        self._adapters.verify_base_frozen()

    def adapter_inventory(self) -> dict[str, Any]:
        return self._adapters.inventory()


class StructuredCapabilityBridge(nn.Module):
    """Expose each canonical transition as four ordered neural table tokens.

    The layout is host-private but capability-independent.  It converts no
    prompt and computes no answer: each package row becomes an operator token,
    source-state token, relation token, and expected target-state embedding.
    The recipient's meta-trained internal adapters must learn how to use that
    table while processing the ordinary prompt.
    """

    def __init__(self, host: FrozenNeuralHost, *, rank: int) -> None:
        super().__init__()
        self.host_key = host.spec.key
        self.hidden_width = host.hidden_width
        self.operator_codes = nn.Parameter(torch.empty(len(OPERATORS), self.hidden_width))
        self.source_codes = nn.Parameter(torch.empty(MODULUS, self.hidden_width))
        self.relation_code = nn.Parameter(torch.empty(self.hidden_width))
        self.value_codes = nn.Parameter(torch.empty(MODULUS, self.hidden_width))
        self.role_offsets = nn.Parameter(torch.zeros(4, self.hidden_width))
        with torch.no_grad():
            for index, operator in enumerate(OPERATORS):
                ids = host.tokenizer.encode(f" {operator}", add_special_tokens=False)
                self.operator_codes[index].copy_(
                    host.embedding(torch.tensor(ids, device=host.device))
                    .detach()
                    .float()
                    .mean(dim=0)
                    .to(self.operator_codes)
                )
            values = host.embedding(
                torch.tensor(host.target_token_ids, device=host.device)
            ).detach().float()
            self.source_codes.copy_(values.to(self.source_codes))
            self.value_codes.copy_(values.to(self.value_codes))
            relation_ids = host.tokenizer.encode(" ->", add_special_tokens=False)
            self.relation_code.copy_(
                host.embedding(torch.tensor(relation_ids, device=host.device))
                .detach()
                .float()
                .mean(dim=0)
                .to(self.relation_code)
            )
        object.__setattr__(self, "_adapters", GenericRecipientAdapterSet(host, rank=rank))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        expected = (len(OPERATORS), MODULUS, MODULUS)
        if tuple(latent.shape[1:]) != expected or not torch.isfinite(latent).all():
            raise NativeHostError("structured canonical latent changed")
        probabilities = latent.float().clamp_min(0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        batch = latent.shape[0]
        operators = (
            self.operator_codes[:, None, :]
            .expand(len(OPERATORS), MODULUS, self.hidden_width)
            .reshape(1, -1, self.hidden_width)
            .expand(batch, -1, -1)
        )
        sources = (
            self.source_codes[None, :, :]
            .expand(len(OPERATORS), MODULUS, self.hidden_width)
            .reshape(1, -1, self.hidden_width)
            .expand(batch, -1, -1)
        )
        relations = self.relation_code.reshape(1, 1, -1).expand(
            batch, len(OPERATORS) * MODULUS, -1
        )
        values = torch.einsum(
            "bosv,vh->bosh", probabilities, self.value_codes.float()
        ).reshape(batch, -1, self.hidden_width)
        rows = torch.stack((operators, sources, relations, values), dim=2)
        rows = rows + self.role_offsets.reshape(1, 1, 4, -1)
        return rows.reshape(batch, -1, self.hidden_width).to(latent.device)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        yield from super().parameters(recurse=recurse)
        yield from self._adapters.parameters()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
        prefix = super().state_dict(*args, **kwargs)
        return {
            **{f"prefix/{name}": value for name, value in prefix.items()},
            **{f"adapter/{name}": value for name, value in self._adapters.state().items()},
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        prefix = {
            name.removeprefix("prefix/"): value
            for name, value in state_dict.items()
            if name.startswith("prefix/")
        }
        adapters = {
            name.removeprefix("adapter/"): value
            for name, value in state_dict.items()
            if name.startswith("adapter/")
        }
        if strict and len(prefix) + len(adapters) != len(state_dict):
            raise NativeHostError("unknown structured bridge tensor")
        result = super().load_state_dict(prefix, strict=strict, assign=assign)
        self._adapters.load_state(adapters)
        return result

    def freeze(self) -> None:
        self.eval()
        for parameter in super().parameters():
            parameter.requires_grad_(False)
        self._adapters.freeze()

    def verify_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise NativeHostError(f"bridge is trainable after freeze: {self.host_key}")
        self._adapters.verify_base_frozen()

    def adapter_inventory(self) -> dict[str, Any]:
        return self._adapters.inventory()


def build_bridge(
    host: FrozenNeuralHost, config: Mapping[str, Any]
) -> CanonicalLatentBridge:
    method = str(config["training"].get("bridge_method", "static_prefix"))
    if method == "static_prefix":
        return CanonicalLatentBridge(host)
    if method == "meta_trained_lora_prefix":
        return CapabilityConditionedBridge(
            host, rank=int(config["training"]["bridge_lora_rank"])
        )
    if method == "structured_table_lora":
        return StructuredCapabilityBridge(
            host, rank=int(config["training"]["bridge_lora_rank"])
        )
    raise NativeHostError(f"unknown bridge method: {method}")
