"""Shared, deterministic machinery for capability-compiler Phase 2.

This module contains no experiment-specific thresholds.  Those live in the
hash-bound Phase 2 protocol.  It deliberately implements LoRA internally so
the exact target graph, accounting, and merge behavior remain auditable.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PHASE1_IR_SHA256 = "a246a52bcf27609b46cdb0530f1daaefe749b7c4a1000f9578f20e505a596f20"
CAPABILITIES = (
    "grammar",
    "coherence",
    "prompt_grounding",
    "instruction_following",
    "conversation",
    "supplied_text_summarization",
    "rewriting",
    "email_drafting_from_notes",
    "tone_control",
    "format_control",
    "clarification",
    "abstention",
    "fact_free_reasoning",
    "fluent_realization",
)
LORA_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


class Phase2Error(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_phase1_records(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != PHASE1_IR_SHA256:
        raise Phase2Error("Phase 1 IR identity changed")
    with zipfile.ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines()]
    if len(rows) != 7_000:
        raise Phase2Error("Phase 1 record depth changed")
    counts = {capability: 0 for capability in CAPABILITIES}
    for row in rows:
        capability = str(row.get("capability"))
        if capability not in counts or row.get("destination") != "english_core":
            raise Phase2Error("Phase 1 English inventory changed")
        if not row.get("functional_pass"):
            raise Phase2Error("ineligible Phase 1 record reached Phase 2")
        counts[capability] += 1
    if set(counts.values()) != {500}:
        raise Phase2Error("Phase 1 per-capability depth changed")
    return sorted(rows, key=lambda row: (str(row["capability"]), str(row["selection_key"])))


def load_catalog(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("probes"), list):
        raise Phase2Error("invalid frozen catalog")
    return value


@dataclass(frozen=True)
class TokenExample:
    record_id: str
    capability: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    response_positions: tuple[int, ...]
    authoritative_response_tokens: int


@dataclass(frozen=True)
class PackedBatch:
    pack_id: str
    capability: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    response_positions: tuple[int, ...]
    record_ids: tuple[str, ...]


def tokenize_records(records: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[TokenExample]:
    eos = int(tokenizer.eos_token_id)
    examples: list[TokenExample] = []
    for row in records:
        prompt_ids = tuple(int(value) for value in tokenizer(
            str(row["rendered_generation_prompt"]), add_special_tokens=False
        ).input_ids)
        response = tuple(int(value) for value in row["authoritative_generated_token_ids"])
        if not response:
            raise Phase2Error("empty authoritative response")
        sequence = prompt_ids + response + (eos,)
        labels = (-100,) * len(prompt_ids) + response + (eos,)
        positions = tuple(range(len(prompt_ids) - 1, len(sequence) - 1))
        examples.append(TokenExample(
            record_id=str(row["ir_record_id"]),
            capability=str(row["capability"]),
            input_ids=sequence,
            labels=labels,
            response_positions=positions,
            authoritative_response_tokens=len(response),
        ))
    return examples


def pack_examples(
    examples: Sequence[TokenExample], *, max_tokens: int, seed: int
) -> list[PackedBatch]:
    if max_tokens < 32:
        raise Phase2Error("packing context is too small")
    grouped: dict[str, list[TokenExample]] = {capability: [] for capability in CAPABILITIES}
    for example in examples:
        grouped[example.capability].append(example)
    packs: list[PackedBatch] = []
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: row.record_id)
        random.Random(stable_seed(seed, capability, "packing")).shuffle(rows)
        current_ids: list[int] = []
        current_labels: list[int] = []
        current_records: list[str] = []
        for row in rows:
            if len(row.input_ids) > max_tokens:
                raise Phase2Error(f"record exceeds packing context: {row.record_id}")
            if current_ids and len(current_ids) + len(row.input_ids) > max_tokens:
                response_positions = tuple(index - 1 for index, label in enumerate(current_labels) if label != -100)
                payload = canonical_json_bytes({
                    "capability": capability,
                    "input_ids": current_ids,
                    "labels": current_labels,
                    "record_ids": current_records,
                })
                packs.append(PackedBatch(sha256_bytes(payload), capability, tuple(current_ids), tuple(current_labels), response_positions, tuple(current_records)))
                current_ids, current_labels, current_records = [], [], []
            current_ids.extend(row.input_ids)
            current_labels.extend(row.labels)
            current_records.append(row.record_id)
        if current_ids:
            response_positions = tuple(index - 1 for index, label in enumerate(current_labels) if label != -100)
            payload = canonical_json_bytes({
                "capability": capability,
                "input_ids": current_ids,
                "labels": current_labels,
                "record_ids": current_records,
            })
            packs.append(PackedBatch(sha256_bytes(payload), capability, tuple(current_ids), tuple(current_labels), response_positions, tuple(current_records)))
    return packs


def pack_manifest(packs: Sequence[PackedBatch]) -> dict[str, Any]:
    records = [
        {
            "pack_id": pack.pack_id,
            "capability": pack.capability,
            "tokens": len(pack.input_ids),
            "response_tokens": sum(label != -100 for label in pack.labels),
            "record_ids": list(pack.record_ids),
            "input_sha256": sha256_bytes(np.asarray(pack.input_ids, dtype=np.int32).tobytes()),
            "labels_sha256": sha256_bytes(np.asarray(pack.labels, dtype=np.int32).tobytes()),
        }
        for pack in packs
    ]
    return {
        "format": "abi-phase2-pack-manifest/1",
        "packs": records,
        "pack_count": len(records),
        "record_count": sum(len(row["record_ids"]) for row in records),
        "input_tokens": sum(row["tokens"] for row in records),
        "response_tokens": sum(row["response_tokens"] for row in records),
        "content_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0 or base.weight.ndim != 2:
            raise Phase2Error("invalid LoRA target")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scale

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device=self.lora_a.device).manual_seed(seed)
        with torch.no_grad():
            bound = math.sqrt(5)
            self.lora_a.uniform_(-1.0 / bound, 1.0 / bound, generator=generator)
            self.lora_b.zero_()


def _resolve_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    return parent, parts[-1]


def install_lora(model: nn.Module, *, rank: int, alpha: float, dropout: float) -> list[str]:
    targets = [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and name.endswith(LORA_SUFFIXES)]
    if len(targets) != 128:
        raise Phase2Error(f"Phi-3 LoRA target graph changed: {len(targets)}")
    for name in targets:
        parent, child = _resolve_parent(model, name)
        base = getattr(parent, child)
        setattr(parent, child, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    return targets


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def reset_lora(model: nn.Module, *, seed: int, capability: str) -> None:
    for index, (_, module) in enumerate(lora_modules(model)):
        module.reset_parameters(stable_seed(seed, capability, index))


def capture_lora(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model):
        # clone() is required when the live adapter is already on CPU; without
        # it a captured state aliases the parameters and silently mutates on a
        # later reset or training step.
        state[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous().clone()
        state[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous().clone()
    return state


def load_lora(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, module in lora_modules(model):
            module.lora_a.copy_(state[f"{name}.lora_a"].to(module.lora_a))
            module.lora_b.copy_(state[f"{name}.lora_b"].to(module.lora_b))


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CompactTransformerLM(nn.Module):
    """A tied-embedding causal transformer matched to the host byte envelope."""

    def __init__(
        self,
        *,
        vocab_size: int = 32_064,
        hidden_size: int = 256,
        intermediate_size: int = 768,
        layers: int = 4,
        heads: int = 8,
        max_positions: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise Phase2Error("hidden size must divide attention heads")
        self.spec = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "layers": layers,
            "heads": heads,
            "max_positions": max_positions,
            "dropout": dropout,
            "tied_embeddings": True,
        }
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.position_embedding.num_embeddings:
            raise Phase2Error("student context exceeded")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        length = input_ids.shape[1]
        mask = torch.full((length, length), float("-inf"), device=input_ids.device, dtype=hidden.dtype)
        mask = torch.triu(mask, diagonal=1)
        hidden = self.blocks(hidden, mask=mask, is_causal=True)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight, self.output_bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def response_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)


def sparse_topk_kl(
    student_logits: torch.Tensor,
    positions: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    selected = student_logits[0, positions].float() / temperature
    student_log_probs = F.log_softmax(selected, dim=-1).gather(-1, teacher_indices.long())
    teacher_probs = F.softmax(teacher_values.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_values.float() / temperature, dim=-1)
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean() * (temperature**2)


def greedy_generate(
    model: nn.Module,
    prompt_ids: Sequence[int],
    *,
    eos_token_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    generated = list(int(value) for value in prompt_ids)
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor([generated[-768:]], dtype=torch.long, device=device)
            token = int(model(inputs)[0, -1].argmax().item())
            if token == eos_token_id:
                break
            generated.append(token)
    return generated[len(prompt_ids):]


def evaluate_functional(output: str, evaluator: Mapping[str, Any]) -> bool:
    # Use the exact evaluator that certified the frozen Phase 1 catalog.  A
    # second implementation risks changing defaults (notably case handling)
    # and would make teacher/student comparisons non-paired in substance.
    from .hf_extraction import evaluate_output

    return bool(evaluate_output(output, dict(evaluator))[0])


def repetition_collapse(output: str) -> bool:
    words = output.casefold().split()
    if len(words) < 8:
        return False
    for width in (1, 2, 3, 4):
        grams = [tuple(words[index:index + width]) for index in range(len(words) - width + 1)]
        if grams and max(grams.count(value) for value in set(grams)) >= 5:
            return True
    return False
