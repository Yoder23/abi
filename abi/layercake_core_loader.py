"""Versioned loader for three- and six-block LayerCake English cores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from transformers import AutoTokenizer

from .layercake_host import _import_layercake_runtime, _sha256_file


@dataclass(frozen=True)
class ABIEnglishCoreConfig:
    vocab_size: int = 50257
    width: int = 768
    layers: int = 3
    heads: int = 12
    max_tokens: int = 1024
    task_cakes: int = 10
    task_cake_rank: int = 64
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
        if self.task_cakes != 10 or self.task_cake_rank != 64:
            raise ValueError("instruction-cake topology changed")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    model.load_state_dict(
        load_file(str(checkpoint_path), device=str(device)),
        strict=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1_000_000_000
    return model.to(device).eval(), tokenizer, metadata
