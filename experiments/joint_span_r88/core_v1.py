"""Portable lexical-invariant joint-span bridge used by R88."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import _sha256_file


PACKAGE_FORMAT = "abi-r88-lexical-invariant-joint-span-package/1"
BRIDGE_ARCHITECTURE = "lexical-invariant-bidirectional-joint-span-selector/1"
WIDTH = 128
LAYERS = 2
HEADS = 4
FEEDFORWARD_WIDTH = 512
MAX_TOKENS = 256
MAX_SPAN_TOKENS = 16
ROUTES = 14


class JointSpanPackageError(RuntimeError):
    pass


class JointSpanBridge(torch.nn.Module):
    """Reason over frozen host states and score start/end pairs."""

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
        self.end_head = torch.nn.Linear(WIDTH, 1)

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
            raise JointSpanPackageError("R88 bridge input contract changed")
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        state = (
            self.input_projection(hidden)
            + self.position_embedding(positions)[None, :, :]
            + self.route_embedding(routes.long())[:, None, :]
        )
        state = self.encoder(state, src_key_padding_mask=~attention_mask.bool())
        state = self.normalization(state)
        invalid = ~attention_mask.bool()
        start = self.start_head(state).squeeze(-1).masked_fill(invalid, -torch.inf)
        end = self.end_head(state).squeeze(-1).masked_fill(invalid, -torch.inf)
        return start, end


def joint_span_scores(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        start_logits.shape != end_logits.shape
        or start_logits.shape != attention_mask.shape
        or start_logits.ndim != 2
    ):
        raise JointSpanPackageError("R88 joint score contract changed")
    tokens = start_logits.shape[1]
    starts = torch.arange(tokens, device=start_logits.device)[:, None]
    ends = torch.arange(tokens, device=start_logits.device)[None, :]
    legal_geometry = (ends >= starts) & (ends - starts < MAX_SPAN_TOKENS)
    legal = (
        attention_mask.bool()[:, :, None]
        & attention_mask.bool()[:, None, :]
        & legal_geometry[None, :, :]
    )
    scores = start_logits[:, :, None] + end_logits[:, None, :]
    return scores.masked_fill(~legal, -torch.inf)


def bridge_parameter_count() -> int:
    return sum(parameter.numel() for parameter in JointSpanBridge().parameters())


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
        raise JointSpanPackageError("R88 package metadata is invalid or stale")


def load_joint_span_package(
    path: str | Path, *, device: str | torch.device
) -> tuple[JointSpanBridge, dict[str, Any]]:
    root = Path(path).resolve()
    metadata_path = root / "metadata.json"
    checkpoint_path = root / "bridge.safetensors"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise JointSpanPackageError("R88 package files are absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, root)
    bridge = JointSpanBridge().to(device)
    bridge.load_state_dict(load_file(str(checkpoint_path), device=str(device)), strict=True)
    bridge.eval()
    return bridge, metadata


@torch.inference_mode()
def realize_joint_span(
    *,
    bridge: JointSpanBridge,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    routes: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: Any,
) -> tuple[str, list[int], int, int]:
    start_logits, end_logits = bridge(hidden, attention_mask, routes)
    scores = joint_span_scores(start_logits, end_logits, attention_mask)
    tokens = scores.shape[1]
    flat = int(scores[0].argmax().item())
    start, end = divmod(flat, tokens)
    selected = input_ids[0, start : end + 1].tolist()
    output = tokenizer.decode(
        selected, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return output, tokenizer.encode(output, add_special_tokens=False), start, end
