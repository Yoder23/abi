"""Zero-parameter canonical capability VM and frozen-host logit codec for R10."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

PACKAGE_FORMAT = "abi-copy-paste-r10-transition-package/1"
PACKAGE_KEYS = {
    "format",
    "family",
    "interpreter_abi",
    "latent_dtype",
    "latent_hex",
    "latent_sha256",
    "latent_shape",
    "provenance",
}
OPERATORS = ("vok", "narel", "tem")
OPERATOR_INDEX = {value: index for index, value in enumerate(OPERATORS)}
PROGRAM_RE = re.compile(
    r"^Opaque program: start ([0-7]) ; apply "
    r"((?:vok|narel|tem)(?: (?:vok|narel|tem))*) ; result =$"
)


class CopyPasteRuntimeError(RuntimeError):
    """Raised when a package, prompt, or host codec violates the R10 ABI."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latent_bytes(latent: torch.Tensor) -> bytes:
    value = latent.detach().cpu().float().contiguous()
    if tuple(value.shape) != (3, 8, 8) or not torch.isfinite(value).all():
        raise CopyPasteRuntimeError("canonical latent shape or numerics changed")
    return struct.pack("<192f", *[float(item) for item in value.flatten().tolist()])


def build_package(latent: torch.Tensor, provenance: Mapping[str, str]) -> dict[str, Any]:
    payload = latent_bytes(latent)
    if set(provenance) != {"source_receipt_sha256", "extraction_receipt_sha256"}:
        raise CopyPasteRuntimeError("package provenance schema changed")
    return {
        "format": PACKAGE_FORMAT,
        "family": "opaque_modular_micro_language/1",
        "interpreter_abi": "canonical-transition-vm/1",
        "latent_dtype": "float32-little-endian",
        "latent_hex": payload.hex(),
        "latent_sha256": sha256_bytes(payload),
        "latent_shape": [3, 8, 8],
        "provenance": dict(provenance),
    }


def write_package_once(
    directory: Path,
    latent: torch.Tensor,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    package = build_package(latent, provenance)
    content = canonical_json_bytes(package)
    package_sha256 = sha256_bytes(content)
    path = directory / f"sha256-{package_sha256}.abipkg"
    if path.exists():
        raise CopyPasteRuntimeError(f"immutable package exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.name,
        "sha256": package_sha256,
        "bytes": len(content),
        "latent_sha256": package["latent_sha256"],
    }


def load_package(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CopyPasteRuntimeError(f"package unavailable: {path}") from exc
    if not isinstance(value, dict) or set(value) != PACKAGE_KEYS:
        raise CopyPasteRuntimeError("package field inventory changed")
    if payload != canonical_json_bytes(value) or sha256_bytes(payload) != path.stem.removeprefix(
        "sha256-"
    ):
        raise CopyPasteRuntimeError("package content address changed")
    if (
        value.get("format") != PACKAGE_FORMAT
        or value.get("family") != "opaque_modular_micro_language/1"
        or value.get("interpreter_abi") != "canonical-transition-vm/1"
        or value.get("latent_dtype") != "float32-little-endian"
        or value.get("latent_shape") != [3, 8, 8]
    ):
        raise CopyPasteRuntimeError("package ABI changed")
    forbidden = {
        "prompt",
        "answer",
        "row_id",
        "solver",
        "model_id",
        "tokenizer_id",
        "hidden_width",
        "host_matrix",
    }
    if forbidden & set(value):
        raise CopyPasteRuntimeError("package contains forbidden execution or host data")
    try:
        raw = bytes.fromhex(str(value["latent_hex"]))
    except ValueError as exc:
        raise CopyPasteRuntimeError("package latent is not hexadecimal") from exc
    if len(raw) != 192 * 4 or sha256_bytes(raw) != value.get("latent_sha256"):
        raise CopyPasteRuntimeError("package latent identity changed")
    latent = torch.tensor(struct.unpack("<192f", raw), dtype=torch.float32).reshape(3, 8, 8)
    if not torch.isfinite(latent).all():
        raise CopyPasteRuntimeError("package latent is non-finite")
    return value, latent


class CanonicalTransitionVM:
    """Execute one registered capability-family IR with zero learned parameters."""

    learned_parameters = 0
    abi = "canonical-transition-vm/1"

    @staticmethod
    def parse(prompt: str) -> tuple[int, tuple[int, ...]]:
        marker = prompt.rfind("Opaque program:")
        if marker < 0:
            raise CopyPasteRuntimeError("registered program marker missing")
        surface = prompt[marker:]
        match = PROGRAM_RE.fullmatch(surface)
        if match is None:
            raise CopyPasteRuntimeError("registered program grammar changed")
        words = tuple(match.group(2).strip().split())
        if not words or any(word not in OPERATOR_INDEX for word in words):
            raise CopyPasteRuntimeError("registered operator sequence changed")
        return int(match.group(1)), tuple(OPERATOR_INDEX[word] for word in words)

    @staticmethod
    def normalize(latent: torch.Tensor) -> torch.Tensor:
        if tuple(latent.shape) != (3, 8, 8) or not torch.isfinite(latent).all():
            raise CopyPasteRuntimeError("VM latent shape or numerics changed")
        value = latent.float().clamp_min(0)
        denominator = value.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(value, 1.0 / 8.0)
        return torch.where(denominator > 0, value / denominator.clamp_min(1e-12), uniform)

    def execute(self, latent: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        transition = self.normalize(latent)
        outputs = []
        for prompt in prompts:
            start, program = self.parse(str(prompt))
            state = torch.nn.functional.one_hot(torch.tensor(start), num_classes=8).float()
            for operation in program:
                state = state @ transition[operation]
            outputs.append(state / state.sum().clamp_min(torch.finfo(torch.float32).tiny))
        return torch.stack(outputs)


def apply_host_codec(
    logits: torch.Tensor,
    distributions: torch.Tensor,
    target_token_ids: Sequence[int],
    *,
    margin: float,
) -> torch.Tensor:
    if logits.ndim != 2 or distributions.shape != (logits.shape[0], 8):
        raise CopyPasteRuntimeError("host codec tensor geometry changed")
    if len(target_token_ids) != 8 or len(set(int(value) for value in target_token_ids)) != 8:
        raise CopyPasteRuntimeError("host canonical token map changed")
    if not torch.isfinite(logits).all() or not torch.isfinite(distributions).all():
        raise CopyPasteRuntimeError("host codec received non-finite values")
    probabilities = distributions.to(logits.device).float().clamp_min(1e-12)
    log_probabilities = torch.log(probabilities)
    log_probabilities -= log_probabilities.max(dim=-1, keepdim=True).values
    peak = logits.max(dim=-1, keepdim=True).values + float(margin)
    canonical_logits = peak + log_probabilities
    result = logits.clone()
    indices = torch.tensor(target_token_ids, dtype=torch.long, device=logits.device)
    result.scatter_(
        1, indices.unsqueeze(0).expand(logits.shape[0], -1), canonical_logits.to(logits)
    )
    return result


def canonical_prediction(token_id: int, target_token_ids: Sequence[int]) -> int | None:
    mapping = {int(token): index for index, token in enumerate(target_token_ids)}
    return mapping.get(int(token_id))


def probability_tv(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise CopyPasteRuntimeError("distribution width changed")
    value = 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right))
    if not math.isfinite(value):
        raise CopyPasteRuntimeError("distribution distance is non-finite")
    return value


def discover_canonical_token_map(
    tokenizer: Any, *, encoder_decoder: bool
) -> tuple[list[int], list[str]]:
    """Find one native token per digit whose isolated decode is byte-exact."""
    templates = ("{}", " {}") if encoder_decoder else (" {}", "{}")
    token_ids: list[int] = []
    texts: list[str] = []
    for value in range(8):
        canonical = str(value)
        candidates: list[int] = []
        for template in templates:
            encoded = tokenizer.encode(template.format(value), add_special_tokens=False)
            for token_id in encoded:
                decoded = tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if decoded == canonical:
                    candidates.append(int(token_id))
        if not candidates:
            raise CopyPasteRuntimeError(
                f"host tokenizer cannot realize canonical digit {value} in one token"
            )
        token_ids.append(min(candidates))
        texts.append(canonical)
    if len(set(token_ids)) != 8:
        raise CopyPasteRuntimeError("host canonical digit tokens are not unique")
    return token_ids, texts
