"""Frozen generic frontend mapping Qwen weight deltas to canonical transitions."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from experiments.foreign_capability_r14.capability import MODULUS, OPERATORS, AffineCapability
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes


class R15FrontendError(RuntimeError):
    """Raised when the learned weight-delta frontend is invalid."""


def load_frozen_frontend(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Load a content-bound frontend artifact and reconstruct its strict state."""
    if not path.is_file() or spec.get("format") not in {
        "abi-r15a-frozen-weight-delta-affine-table-frontend/4"
    }:
        raise R15FrontendError("required frozen R15A frontend is unavailable")
    try:
        tensors = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except (OSError, RuntimeError) as exc:
        raise R15FrontendError("frozen R15A frontend is unreadable") from exc
    if set(tensors) != {"readout", "bias"} or metadata != {
        "format": spec["format"],
        "frontend_spec_sha256": spec["evidence_sha256"],
    }:
        raise R15FrontendError("frozen R15A frontend artifact schema changed")
    width = int(spec.get("feature_elements", 0)) // MODULUS
    if (
        int(spec.get("feature_elements", 0)) % MODULUS
        or tuple(tensors["readout"].shape) != (OPERATORS, MODULUS, width)
        or tuple(tensors["bias"].shape) != (OPERATORS, MODULUS, MODULUS)
        or not all(torch.isfinite(value).all() for value in tensors.values())
    ):
        raise R15FrontendError("frozen R15A frontend tensor contract changed")
    state = {
        key: value
        for key, value in spec.items()
        if key not in {"tensor_sha256", "evidence_sha256"}
    }
    state.update(tensors)
    if frontend_spec(state) != spec:
        raise R15FrontendError("frozen R15A frontend hash changed")
    return state


def labels_for_capability(capability: AffineCapability) -> torch.Tensor:
    """Encode each operation by outputs at states zero and one."""
    labels = []
    for operator in range(OPERATORS):
        labels.extend((capability.apply(0, [operator]), capability.apply(1, [operator])))
    return torch.tensor(labels, dtype=torch.long)


def transition_from_labels(labels: Sequence[int]) -> torch.Tensor:
    if len(labels) != 2 * OPERATORS:
        raise R15FrontendError("frontend label inventory changed")
    transition = torch.zeros(OPERATORS, MODULUS, MODULUS, dtype=torch.float32)
    for operator in range(OPERATORS):
        at_zero = int(labels[2 * operator])
        at_one = int(labels[2 * operator + 1])
        multiplier = (at_one - at_zero) % MODULUS
        if multiplier not in {1, 3, 5, 7}:
            raise R15FrontendError("frontend emitted a non-permutation affine operation")
        for state in range(MODULUS):
            transition[operator, state, (multiplier * state + at_zero) % MODULUS] = 1.0
    return transition


def _normalize_rows(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=1, keepdim=True)
    return centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _kernel(left: torch.Tensor, right: torch.Tensor, *, gamma: float) -> torch.Tensor:
    cosine = torch.matmul(_normalize_rows(left), _normalize_rows(right).t()).clamp(-1.0, 1.0)
    return torch.exp(-float(gamma) * (1.0 - cosine))


def train_frontend(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
    ridge: float,
) -> dict[str, Any]:
    """Fit a generic kernel decoder on public learning-event pairs only."""
    if features.ndim != 2 or labels.shape != (features.shape[0], 2 * OPERATORS):
        raise R15FrontendError("public frontend training shapes changed")
    if features.shape[0] < 8 or ridge <= 0 or gamma <= 0:
        raise R15FrontendError("insufficient or invalid public frontend training")
    x = features.detach().float().cpu().contiguous()
    y = labels.detach().long().cpu().contiguous()
    kernel = _kernel(x, x, gamma=gamma)
    identity = torch.eye(len(x), dtype=kernel.dtype)
    coefficients = []
    for head in range(y.shape[1]):
        target = torch.nn.functional.one_hot(y[:, head], num_classes=MODULUS).float()
        coefficients.append(torch.linalg.solve(kernel + float(ridge) * identity, target))
    state = {
        "format": "abi-r15a-frozen-weight-delta-kernel-frontend/1",
        "input": "qwen_before_after_effective_output_weight_delta_only",
        "candidate_program_search": False,
        "behavioral_queries": 0,
        "oracle_calls_at_extraction": 0,
        "public_event_count": int(x.shape[0]),
        "feature_elements": int(x.shape[1]),
        "gamma": float(gamma),
        "ridge": float(ridge),
        "train_features": x,
        "coefficients": torch.stack(coefficients),
    }
    return state


def _structured_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] % MODULUS != 0:
        raise R15FrontendError("weight-delta shape cannot be structured by output row")
    rows = features.float().reshape(features.shape[0], MODULUS, -1)
    rows = rows - rows.mean(dim=1, keepdim=True)
    return rows / rows.flatten(1).norm(dim=1).reshape(-1, 1, 1).clamp_min(1e-12)


def train_bilinear_frontend(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 2000,
    learning_rate: float = 0.05,
    weight_decay: float = 0.001,
    seed: int = 15015101,
    affine_codebook: bool = False,
) -> dict[str, Any]:
    """Learn six generic neural directions that read the eight delta rows."""
    if features.ndim != 2 or labels.shape != (features.shape[0], 2 * OPERATORS):
        raise R15FrontendError("public bilinear frontend shapes changed")
    x = _structured_features(features.detach().cpu())
    y = labels.detach().long().cpu().contiguous()
    torch.manual_seed(int(seed))
    readout = torch.nn.Parameter(torch.randn(2 * OPERATORS, x.shape[2]) * 0.01)
    bias = torch.nn.Parameter(torch.zeros(2 * OPERATORS, MODULUS))
    optimizer = torch.optim.AdamW(
        [readout, bias],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    first_loss = None
    final_loss = None
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        scores = torch.einsum("ncd,hd->nhc", x, readout) + bias
        loss = torch.nn.functional.cross_entropy(scores.reshape(-1, MODULUS), y.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if first_loss is None:
            first_loss = final_loss
    return {
        "format": (
            "abi-r15a-frozen-weight-delta-affine-codebook-frontend/2"
            if affine_codebook
            else "abi-r15a-frozen-weight-delta-bilinear-frontend/1"
        ),
        "input": "qwen_before_after_effective_output_weight_delta_only",
        "candidate_program_search": False,
        "behavioral_queries": 0,
        "oracle_calls_at_extraction": 0,
        "public_event_count": int(x.shape[0]),
        "feature_elements": int(features.shape[1]),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "output_decoding": (
            "argmax_over_32_valid_affine_codes_per_operator"
            if affine_codebook
            else "independent_digit_argmax"
        ),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "readout": readout.detach().contiguous(),
        "bias": bias.detach().contiguous(),
    }


def _affine_codes() -> tuple[tuple[int, int], ...]:
    return tuple(
        (at_zero, (at_zero + multiplier) % MODULUS)
        for at_zero in range(MODULUS)
        for multiplier in (1, 3, 5, 7)
    )


def _affine_code_targets(labels: torch.Tensor) -> torch.Tensor:
    codes = {code: index for index, code in enumerate(_affine_codes())}
    targets = torch.empty(labels.shape[0], OPERATORS, dtype=torch.long)
    for row in range(labels.shape[0]):
        for operator in range(OPERATORS):
            pair = (
                int(labels[row, 2 * operator]),
                int(labels[row, 2 * operator + 1]),
            )
            if pair not in codes:
                raise R15FrontendError("public label is not an affine permutation")
            targets[row, operator] = codes[pair]
    return targets


def _full_table_targets(labels: torch.Tensor) -> torch.Tensor:
    targets = torch.empty(labels.shape[0], OPERATORS, MODULUS, dtype=torch.long)
    for row in range(labels.shape[0]):
        for operator in range(OPERATORS):
            at_zero = int(labels[row, 2 * operator])
            at_one = int(labels[row, 2 * operator + 1])
            multiplier = (at_one - at_zero) % MODULUS
            if multiplier not in {1, 3, 5, 7}:
                raise R15FrontendError("public label is not an affine permutation")
            for state in range(MODULUS):
                targets[row, operator, state] = (
                    multiplier * state + at_zero
                ) % MODULUS
    return targets


def _affine_pair_scores(
    rows: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    cross: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    codes = _affine_codes()
    zero = torch.tensor([code[0] for code in codes], dtype=torch.long)
    one = torch.tensor([code[1] for code in codes], dtype=torch.long)
    zero_rows = rows[:, zero, :]
    one_rows = rows[:, one, :]
    return (
        torch.einsum("nkd,od->nok", zero_rows, left)
        + torch.einsum("nkd,od->nok", one_rows, right)
        + torch.einsum("nkd,od->nok", zero_rows * one_rows, cross)
        + bias.unsqueeze(0)
    )


def train_affine_pair_frontend(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 3000,
    learning_rate: float = 0.05,
    weight_decay: float = 0.001,
    seed: int = 15015101,
) -> dict[str, Any]:
    """Fit a compact pairwise decoder directly against valid affine codes.

    Each operator uses three learned directions over a pair of output-weight-delta
    rows: one direction for f(0), one for f(1), and one for their elementwise
    interaction.  This fixes the objective/decoder mismatch in frontend v2
    without adding source queries or a global search over capability programs.
    """
    if features.ndim != 2 or labels.shape != (features.shape[0], 2 * OPERATORS):
        raise R15FrontendError("public affine-pair frontend shapes changed")
    rows = _structured_features(features.detach().cpu())
    targets = _affine_code_targets(labels.detach().long().cpu().contiguous())
    torch.manual_seed(int(seed))
    width = rows.shape[2]
    left = torch.nn.Parameter(torch.randn(OPERATORS, width) * 0.01)
    right = torch.nn.Parameter(torch.randn(OPERATORS, width) * 0.01)
    cross = torch.nn.Parameter(torch.randn(OPERATORS, width) * 0.01)
    bias = torch.nn.Parameter(torch.zeros(OPERATORS, len(_affine_codes())))
    optimizer = torch.optim.AdamW(
        [left, right, cross, bias],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    first_loss = None
    final_loss = None
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        scores = _affine_pair_scores(rows, left, right, cross, bias)
        loss = torch.nn.functional.cross_entropy(
            scores.reshape(-1, len(_affine_codes())), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if first_loss is None:
            first_loss = final_loss
    return {
        "format": "abi-r15a-frozen-weight-delta-affine-pair-frontend/3",
        "input": "qwen_before_after_effective_output_weight_delta_only",
        "candidate_program_search": False,
        "behavioral_queries": 0,
        "oracle_calls_at_extraction": 0,
        "public_event_count": int(rows.shape[0]),
        "feature_elements": int(features.shape[1]),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "output_decoding": "argmax_over_32_valid_affine_codes_per_operator",
        "first_loss": first_loss,
        "final_loss": final_loss,
        "left": left.detach().contiguous(),
        "right": right.detach().contiguous(),
        "cross": cross.detach().contiguous(),
        "bias": bias.detach().contiguous(),
    }


def train_affine_table_frontend(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 2000,
    learning_rate: float = 0.05,
    weight_decay: float = 0.001,
    seed: int = 15015101,
) -> dict[str, Any]:
    """Decode all 24 atomic transitions, then enforce local affine consistency."""
    if features.ndim != 2 or labels.shape != (features.shape[0], 2 * OPERATORS):
        raise R15FrontendError("public affine-table frontend shapes changed")
    rows = _structured_features(features.detach().cpu())
    targets = _full_table_targets(labels.detach().long().cpu().contiguous())
    torch.manual_seed(int(seed))
    width = rows.shape[2]
    readout = torch.nn.Parameter(torch.randn(OPERATORS, MODULUS, width) * 0.01)
    bias = torch.nn.Parameter(torch.zeros(OPERATORS, MODULUS, MODULUS))
    optimizer = torch.optim.AdamW(
        [readout, bias],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    first_loss = None
    final_loss = None
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        scores = torch.einsum("ncd,osd->nosc", rows, readout) + bias
        loss = torch.nn.functional.cross_entropy(
            scores.reshape(-1, MODULUS), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if first_loss is None:
            first_loss = final_loss
    return {
        "format": "abi-r15a-frozen-weight-delta-affine-table-frontend/4",
        "input": "qwen_before_after_effective_output_weight_delta_only",
        "candidate_program_search": False,
        "behavioral_queries": 0,
        "oracle_calls_at_extraction": 0,
        "public_event_count": int(rows.shape[0]),
        "feature_elements": int(features.shape[1]),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "output_decoding": "sum_24_atomic_scores_over_32_affine_codes_per_operator",
        "first_loss": first_loss,
        "final_loss": final_loss,
        "readout": readout.detach().contiguous(),
        "bias": bias.detach().contiguous(),
    }


def predict_labels(state: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    format_name = state.get("format")
    if state.get("candidate_program_search") is not False:
        raise R15FrontendError("frontend state contract changed")
    if format_name == "abi-r15a-frozen-weight-delta-kernel-frontend/1":
        query = features.detach().float().cpu().reshape(1, -1)
        train = state["train_features"]
        scores = torch.einsum(
            "n,hnc->hc",
            _kernel(query, train, gamma=float(state["gamma"]))[0],
            state["coefficients"],
        )
    elif format_name in {
        "abi-r15a-frozen-weight-delta-bilinear-frontend/1",
        "abi-r15a-frozen-weight-delta-affine-codebook-frontend/2",
    }:
        query = _structured_features(features.detach().float().cpu().reshape(1, -1))
        scores = torch.einsum("ncd,hd->nhc", query, state["readout"])[0] + state["bias"]
    elif format_name == "abi-r15a-frozen-weight-delta-affine-pair-frontend/3":
        query = _structured_features(features.detach().float().cpu().reshape(1, -1))
        scores = _affine_pair_scores(
            query,
            state["left"],
            state["right"],
            state["cross"],
            state["bias"],
        )[0]
        codes = _affine_codes()
        labels = []
        for operator in range(OPERATORS):
            at_zero, at_one = codes[int(scores[operator].argmax())]
            labels.extend((at_zero, at_one))
        return torch.tensor(labels, dtype=torch.long)
    elif format_name == "abi-r15a-frozen-weight-delta-affine-table-frontend/4":
        query = _structured_features(features.detach().float().cpu().reshape(1, -1))
        scores = (
            torch.einsum("ncd,osd->nosc", query, state["readout"])[0]
            + state["bias"]
        )
        codes = _affine_codes()
        labels = []
        states = torch.arange(MODULUS)
        for operator in range(OPERATORS):
            candidate_scores = []
            for at_zero, at_one in codes:
                multiplier = (at_one - at_zero) % MODULUS
                outputs = (multiplier * states + at_zero) % MODULUS
                candidate_scores.append(
                    scores[operator, states, outputs].sum().item()
                )
            at_zero, at_one = codes[int(torch.tensor(candidate_scores).argmax())]
            labels.extend((at_zero, at_one))
        return torch.tensor(labels, dtype=torch.long)
    else:
        raise R15FrontendError("unknown frontend state format")
    if format_name != "abi-r15a-frozen-weight-delta-affine-codebook-frontend/2":
        return scores.argmax(dim=-1)
    labels = []
    for operator in range(OPERATORS):
        left = scores[2 * operator]
        right = scores[2 * operator + 1]
        candidates = []
        for at_zero in range(MODULUS):
            for multiplier in (1, 3, 5, 7):
                at_one = (at_zero + multiplier) % MODULUS
                candidates.append((float(left[at_zero] + right[at_one]), at_zero, at_one))
        _, at_zero, at_one = max(candidates)
        labels.extend((at_zero, at_one))
    return torch.tensor(labels, dtype=torch.long)


def frontend_spec(state: dict[str, Any]) -> dict[str, Any]:
    common = (
        "format",
        "input",
        "candidate_program_search",
        "behavioral_queries",
        "oracle_calls_at_extraction",
        "public_event_count",
        "feature_elements",
    )
    extras = (
        ("gamma", "ridge")
        if state["format"] == "abi-r15a-frozen-weight-delta-kernel-frontend/1"
        else (
            "epochs",
            "learning_rate",
            "weight_decay",
            "seed",
            "output_decoding",
            "first_loss",
            "final_loss",
        )
    )
    value = {key: state[key] for key in common + extras}
    tensor_hash = hashlib.sha256()
    if state["format"] == "abi-r15a-frozen-weight-delta-kernel-frontend/1":
        tensor_names = ("train_features", "coefficients")
    elif state["format"] == "abi-r15a-frozen-weight-delta-affine-pair-frontend/3":
        tensor_names = ("left", "right", "cross", "bias")
    else:
        tensor_names = ("readout", "bias")
    for name in tensor_names:
        tensor = state[name].detach().cpu().contiguous()
        tensor_hash.update(name.encode() + b"\0")
        tensor_hash.update(tensor.numpy().tobytes())
    value["tensor_sha256"] = tensor_hash.hexdigest()
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value
