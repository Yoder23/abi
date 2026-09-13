"""Additive R83 prompt-pointer extension over the hash-frozen ABI runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

import abi.layercake_full_core_acquisition as acquisition
from abi.layercake_core_loader import (
    ABIEnglishCoreConfig,
    PROMPT_IDENTITY_RANK,
    _import_layercake_runtime,
    _load_symbolic_surface_substrate,
)
from abi.layercake_host import PromptIdentityBridge, _sha256_file


GENERIC_SCOPE = "selective_prompt_identity_carriage"
GENERIC_ARCHITECTURE = (
    "abi-r83-additive/six-block-r47-selective-prompt-pointer32"
)
BASE_ARCHITECTURE = "layercake-shallow-sparse-english/2-six-block-task-cakes"
PARENT_SHA256 = "65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b"
PARENT_METADATA_SHA256 = "9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6"


class R83ExtensionError(RuntimeError):
    pass


def install_generic_prompt_identity(
    model: torch.nn.Module, *, initialize: bool, selective: bool = True
) -> None:
    """Register only the pointer; leave the sealed parent config and graph intact."""

    if not selective:
        raise R83ExtensionError("R83 is locked to selective pointer supervision")
    config = model.config
    if (
        int(config.layers) != 6
        or int(config.width) != 768
        or int(config.task_cakes) != 10
        or str(config.architecture_version) != BASE_ARCHITECTURE
        or getattr(model, "_abi_prompt_identity_carriage", False)
    ):
        raise R83ExtensionError("R83 parent topology changed")
    bridge = PromptIdentityBridge(
        width=int(config.width),
        rank=PROMPT_IDENTITY_RANK,
        routes=int(config.task_cakes),
    ).to(
        device=model.transformer.wte.weight.device,
        dtype=model.transformer.wte.weight.dtype,
    )
    if not initialize:
        bridge.requires_grad_(False)
    model.prompt_identity = bridge
    object.__setattr__(model, "_abi_prompt_identity_bridge", bridge)
    model._abi_prompt_identity_carriage = True


def _validate_extension_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(metadata)
    claimed_manifest = unsigned.pop("manifest_sha256", None)
    extension = metadata.get("r83_additive_extension")
    architecture = metadata.get("architecture", {})
    pointer = metadata.get("acquired_core", {}).get("prompt_identity_carriage") or {}
    if (
        acquisition._manifest_sha(unsigned) != claimed_manifest
        or not isinstance(extension, dict)
        or extension.get("format") != "abi-r83-generic-prompt-identity-extension/1"
        or extension.get("architecture") != GENERIC_ARCHITECTURE
        or extension.get("base_architecture") != BASE_ARCHITECTURE
        or extension.get("sealed_runtime_files_modified") is not False
        or extension.get("extension_sha256")
        != _sha256_file(Path(__file__).resolve())
        or extension.get("protocol_sha256")
        != _sha256_file(Path(__file__).with_name("PROTOCOL.md"))
        or extension.get("screen_sha256")
        != _sha256_file(Path(__file__).with_name("screen_host_v1.py"))
        or architecture.get("architecture_version") != BASE_ARCHITECTURE
        or architecture.get("layers") != 6
        or architecture.get("task_cakes") != 10
        or metadata.get("training", {}).get("trainable_scope") != GENERIC_SCOPE
        or metadata.get("acquired_core", {}).get("trainable_scope") != GENERIC_SCOPE
        or pointer.get("rank") != PROMPT_IDENTITY_RANK
        or pointer.get("parameter_count") != 49_931
        or pointer.get("selective_parent_top1_deficit_labels") is not True
        or pointer.get("parent_state_preserved_exact") is not True
        or metadata.get("decoding", {}).get("prompt_identity_mixture") is not True
    ):
        raise R83ExtensionError("R83 metadata is invalid, stale, or incomplete")
    return extension


def load_r83_core(
    path: str | Path,
    *,
    layercake_root: str | Path,
    device: str | torch.device = "cpu",
):
    """Fail-closed loader for the additive pointer checkpoint."""

    path = Path(path).resolve()
    metadata_path = path / "metadata.json"
    checkpoint_path = path / "model.safetensors"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise R83ExtensionError("R83 checkpoint or metadata is absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_extension_metadata(metadata)
    if metadata.get("checkpoint", {}).get("sha256") != _sha256_file(checkpoint_path):
        raise R83ExtensionError("R83 checkpoint identity changed")
    _import_layercake_runtime(Path(layercake_root).resolve())
    from layercake.models.shallow_sparse_english import ShallowSparseEnglishCore

    config = ABIEnglishCoreConfig(**metadata["architecture"])
    model = ShallowSparseEnglishCore(config)
    install_generic_prompt_identity(model, initialize=False, selective=True)
    model.load_state_dict(
        load_file(str(checkpoint_path), device=str(device)), strict=True
    )
    decoding = metadata.get("decoding")
    if (
        not isinstance(decoding, dict)
        or decoding.get("algorithm") != "greedy"
        or decoding.get("no_repeat_ngram_size") != 0
        or decoding.get("allow_prompt_ngrams") is not False
        or decoding.get("prompt_identity_mixture") is not True
    ):
        raise R83ExtensionError("R83 decoding contract changed")
    model._abi_decoding = dict(decoding)
    model._abi_symbolic_surface = _load_symbolic_surface_substrate(path, metadata)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1_000_000_000
    return model.to(device).eval(), tokenizer, metadata


def _finalize_metadata(output: Path) -> None:
    metadata_path = output / "metadata.json"
    checkpoint_path = output / "model.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    claimed = unsigned.pop("manifest_sha256", None)
    if (
        acquisition._manifest_sha(unsigned) != claimed
        or metadata.get("training", {}).get("trainable_scope")
        != acquisition.TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE
        or metadata.get("architecture", {}).get("architecture_version")
        != BASE_ARCHITECTURE
        or metadata.get("checkpoint", {}).get("sha256")
        != _sha256_file(checkpoint_path)
    ):
        raise R83ExtensionError("sealed acquisition output changed before finalization")
    metadata["training"]["trainable_scope"] = GENERIC_SCOPE
    metadata["acquired_core"]["trainable_scope"] = GENERIC_SCOPE
    metadata["r83_additive_extension"] = {
        "format": "abi-r83-generic-prompt-identity-extension/1",
        "architecture": GENERIC_ARCHITECTURE,
        "base_architecture": BASE_ARCHITECTURE,
        "rank": PROMPT_IDENTITY_RANK,
        "selective": True,
        "sealed_runtime_files_modified": False,
        "extension_sha256": _sha256_file(Path(__file__).resolve()),
        "protocol_sha256": _sha256_file(Path(__file__).with_name("PROTOCOL.md")),
        "screen_sha256": _sha256_file(Path(__file__).with_name("screen_host_v1.py")),
    }
    metadata["claim_boundary"] = (
        "R83 additively trains only one selective rank-32 prompt-position pointer "
        "over the byte-identical R47 six-block parent. The sealed ABI runtime and "
        "acquisition implementations remain unmodified. This is a bounded research "
        "candidate, not an English, domain, production, or moonshot claim."
    )
    metadata.pop("manifest_sha256", None)
    metadata["manifest_sha256"] = acquisition._manifest_sha(metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _validate_extension_metadata(metadata)


def train_main(argv: Sequence[str] | None = None) -> int:
    """Run the sealed trainer through one scoped additive installation hook."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--trainable-scope" in arguments:
        raise R83ExtensionError("R83 owns and locks the underlying trainable scope")
    try:
        output = Path(arguments[arguments.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise R83ExtensionError("R83 requires one explicit --output") from error
    original_load = acquisition.load_layercake_core
    original_install = acquisition.install_prompt_identity_carriage

    def patched_load(*args: Any, **kwargs: Any):
        model, tokenizer, metadata = original_load(*args, **kwargs)
        loaded_path = Path(args[0] if args else kwargs["path"]).resolve()
        if (
            _sha256_file(loaded_path / "model.safetensors") != PARENT_SHA256
            or _sha256_file(loaded_path / "metadata.json") != PARENT_METADATA_SHA256
            or str(model.config.architecture_version) != BASE_ARCHITECTURE
            or getattr(model, "_abi_task_route_layerwise_control", False)
        ):
            raise R83ExtensionError("R83 loaded a parent other than frozen R47")
        # Satisfy only the sealed trainer's branch guard. The additive installer
        # consumes and clears this non-state marker before any model execution.
        model._abi_task_route_layerwise_control = True
        return model, tokenizer, metadata

    def patched_install(
        model: torch.nn.Module, *, initialize: bool, selective: bool = False
    ) -> None:
        if getattr(model, "_abi_task_route_layerwise_control", None) is not True:
            raise R83ExtensionError("sealed prompt branch guard was not reached")
        model._abi_task_route_layerwise_control = False
        install_generic_prompt_identity(
            model, initialize=initialize, selective=selective
        )

    acquisition.load_layercake_core = patched_load
    acquisition.install_prompt_identity_carriage = patched_install
    try:
        result = acquisition.main(
            arguments
            + [
                "--trainable-scope",
                acquisition.TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_SCOPE,
            ]
        )
    finally:
        acquisition.load_layercake_core = original_load
        acquisition.install_prompt_identity_carriage = original_install
    if result != 0:
        return result
    _finalize_metadata(output)
    print(json.dumps({"r83_finalized": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
