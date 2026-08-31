"""Small install/remove/restore API for R10 canonical capability packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .runtime import CanonicalTransitionVM, CopyPasteRuntimeError, load_package, sha256_file


class CanonicalCapabilitySlot:
    """One host-independent slot whose behavior is owned by the pasted package."""

    def __init__(self) -> None:
        self._vm = CanonicalTransitionVM()
        self._latent: torch.Tensor | None = None
        self._package_sha256: str | None = None
        self._package_path: Path | None = None

    @property
    def installed(self) -> bool:
        return self._latent is not None

    @property
    def package_sha256(self) -> str | None:
        return self._package_sha256

    def paste(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        package, latent = load_package(resolved)
        digest = sha256_file(resolved)
        self._latent = latent.clone()
        self._package_sha256 = digest
        self._package_path = resolved
        return {
            "operation": "PASTE",
            "package_sha256": digest,
            "latent_sha256": package["latent_sha256"],
            "interpreter_abi": package["interpreter_abi"],
            "learned_parameters": self._vm.learned_parameters,
        }

    def remove(self) -> dict[str, Any]:
        prior = self._package_sha256
        self._latent = None
        self._package_sha256 = None
        self._package_path = None
        return {"operation": "REMOVE", "prior_package_sha256": prior, "installed": False}

    def execute(self, prompts: Sequence[str]) -> torch.Tensor:
        if self._latent is None:
            raise CopyPasteRuntimeError("no capability package is installed")
        if self._package_path is None or sha256_file(self._package_path) != self._package_sha256:
            raise CopyPasteRuntimeError("installed package bytes changed after paste")
        return self._vm.execute(self._latent, prompts)
