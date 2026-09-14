"""Portable stateful neural span bridge used by R85."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import _sha256_file


PACKAGE_FORMAT = "abi-r85-stateful-neural-span-package/1"
BRIDGE_ARCHITECTURE = "stateful-bidirectional-span-selector/1"
WIDTH = 128
LAYERS = 2
HEADS = 4
FEEDFORWARD_WIDTH = 512
MAX_TOKENS = 256
MAX_SPAN_TOKENS = 16
ROUTES = 14


class SpanPackageError(RuntimeError):
    pass


class StatefulSpanBridge(torch.nn.Module):
    """Reason over frozen host states and select one contiguous prompt span."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = torch.nn.Linear(768, WIDTH)
        self.position_embedding = torch.nn.Embedding(MAX_TOKENS, WIDTH)
        self.route_embedding = torch.nn.Embedding(ROUTES, WIDTH)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=WIDTH,
            nhead=HEADS,
            dim_feedforward=FEEDFORWARD_WIDTH,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.normalization = torch.nn.LayerNorm(WIDTH)
        self.start_head = torch.nn.Linear(WIDTH, 1)
        self.length_head = torch.nn.Linear(WIDTH, MAX_SPAN_TOKENS)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        routes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden.ndim != 3
            or attention_mask.shape != hidden.shape[:2]
            or hidden.shape[1] > MAX_TOKENS
            or routes.shape != (hidden.shape[0],)
        ):
            raise SpanPackageError("stateful span input contract changed")
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        state = (
            self.input_projection(hidden)
            + self.position_embedding(positions)[None, :, :]
            + self.route_embedding(routes.long())[:, None, :]
        )
        state = self.encoder(
            state, src_key_padding_mask=~attention_mask.bool()
        )
        state = self.normalization(state)
        start_logits = self.start_head(state).squeeze(-1)
        start_logits = start_logits.masked_fill(~attention_mask.bool(), -torch.inf)
        lengths = attention_mask.long().sum(dim=1) - 1
        pooled = state[torch.arange(state.shape[0], device=state.device), lengths]
        return start_logits, self.length_head(pooled)


def bridge_parameter_count() -> int:
    return sum(parameter.numel() for parameter in StatefulSpanBridge().parameters())


def validate_metadata(metadata: dict[str, Any], root: Path) -> None:
    unsigned = dict(metadata)
    claimed = unsigned.pop("manifest_sha256", None)
    bridge = metadata.get("bridge", {})
    boundary = metadata.get("source_boundary", {})
    if (
        _manifest_sha(unsigned) != claimed
        or metadata.get("format") != PACKAGE_FORMAT
        or metadata.get("status") != "TRAINED_NOT_YET_PROSPECTIVELY_CERTIFIED"
        or bridge.get("architecture") != BRIDGE_ARCHITECTURE
        or bridge.get("width") != WIDTH
        or bridge.get("layers") != LAYERS
        or bridge.get("heads") != HEADS
        or bridge.get("feedforward_width") != FEEDFORWARD_WIDTH
        or bridge.get("maximum_tokens") != MAX_TOKENS
        or bridge.get("maximum_span_tokens") != MAX_SPAN_TOKENS
        or bridge.get("parameter_count") != bridge_parameter_count()
        or boundary.get("teacher_present_at_inference") is not False
        or boundary.get("source_parameters_copied") != 0
        or boundary.get("source_transformer_blocks_retained") != 0
        or boundary.get("source_logits_stored") != 0
        or boundary.get("source_hidden_activations_stored") != 0
        or metadata.get("checkpoint", {}).get("sha256")
        != _sha256_file(root / "bridge.safetensors")
    ):
        raise SpanPackageError("R85 package metadata is invalid or stale")


def load_span_package(
    path: str | Path, *, device: str | torch.device
) -> tuple[StatefulSpanBridge, dict[str, Any]]:
    root = Path(path).resolve()
    metadata_path = root / "metadata.json"
    checkpoint_path = root / "bridge.safetensors"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise SpanPackageError("R85 package files are absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, root)
    bridge = StatefulSpanBridge().to(device)
    bridge.load_state_dict(load_file(str(checkpoint_path), device=str(device)), strict=True)
    bridge.eval()
    return bridge, metadata


@torch.inference_mode()
def realize_span(
    *,
    bridge: StatefulSpanBridge,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    routes: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: Any,
) -> tuple[str, list[int], int, int]:
    start_logits, length_logits = bridge(hidden, attention_mask, routes)
    start = int(start_logits[0].argmax().item())
    length = int(length_logits[0].argmax().item()) + 1
    available = int(attention_mask[0].sum().item())
    selected = input_ids[0, start : min(start + length, available)].tolist()
    output = tokenizer.decode(
        selected, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return output, tokenizer.encode(output), start, length
