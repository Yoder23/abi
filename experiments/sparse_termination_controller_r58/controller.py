"""Sparse route-selected termination controller for the frozen R55 host."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


ROUTES = 14
WIDTH = 768
RANK = 16


class SparseTerminationController(torch.nn.Module):
    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            down = torch.randn(ROUTES, RANK, WIDTH, generator=generator) * 0.02
            up = torch.randn(ROUTES, RANK, generator=generator) * 0.02
        else:
            down = torch.empty(ROUTES, RANK, WIDTH)
            up = torch.empty(ROUTES, RANK)
            torch.nn.init.normal_(down, std=0.02)
            torch.nn.init.normal_(up, std=0.02)
        self.down_weight = torch.nn.Parameter(down)
        self.down_bias = torch.nn.Parameter(torch.zeros(ROUTES, RANK))
        self.up_weight = torch.nn.Parameter(up)
        self.up_bias = torch.nn.Parameter(torch.zeros(ROUTES))
        self.last_calls: tuple[int, ...] = ()

    def forward(
        self,
        hidden: torch.Tensor,
        routes: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != WIDTH:
            raise ValueError("termination hidden state has invalid shape")
        if routes.ndim != 1 or routes.shape[0] != hidden.shape[0]:
            raise ValueError("termination routes have invalid shape")
        if bool((routes < 0).any()) or bool((routes >= ROUTES).any()):
            raise ValueError("termination route is outside the ABI catalog")
        selected_down = self.down_weight.index_select(0, routes)
        selected_down_bias = self.down_bias.index_select(0, routes)
        low = torch.bmm(hidden, selected_down.transpose(1, 2))
        low = F.gelu(low + selected_down_bias[:, None, :])
        selected_up = self.up_weight.index_select(0, routes)
        selected_up_bias = self.up_bias.index_select(0, routes)
        self.last_calls = tuple(int(route) for route in routes.detach().cpu())
        return (low * selected_up[:, None, :]).sum(dim=-1) + selected_up_bias[:, None]

    def should_stop(self, hidden: torch.Tensor, route: int) -> bool:
        if hidden.ndim == 1:
            hidden = hidden[None, None, :]
        elif hidden.ndim == 2:
            hidden = hidden[:, None, :]
        route_tensor = torch.tensor([route], dtype=torch.long, device=hidden.device)
        return bool(self(hidden.float(), route_tensor)[0, -1].item() >= 0.0)


def balanced_terminal_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    terminal_token_id: int,
) -> torch.Tensor:
    """Give one terminal target and all content targets equal record mass."""

    shifted_labels = labels[:, 1:]
    if logits.shape != shifted_labels.shape:
        raise ValueError("termination logits and shifted labels differ")
    record_losses = []
    for index in range(logits.shape[0]):
        active = shifted_labels[index] >= 0
        terminal = active & shifted_labels[index].eq(int(terminal_token_id))
        if int(terminal.sum().item()) != 1:
            raise ValueError("termination training requires one observable EOS")
        content = active & ~terminal
        if not bool(content.any()):
            raise ValueError("termination training requires response content")
        positive = F.binary_cross_entropy_with_logits(
            logits[index][terminal],
            torch.ones_like(logits[index][terminal]),
        )
        negative = F.binary_cross_entropy_with_logits(
            logits[index][content],
            torch.zeros_like(logits[index][content]),
        )
        record_losses.append(0.5 * positive + 0.5 * negative)
    return torch.stack(record_losses).mean()


def complete_terminal_rows(
    tokenizer,
    rows: Sequence[dict],
    *,
    max_tokens: int,
) -> list[dict]:
    return [
        dict(row)
        for row in rows
        if len(tokenizer.encode(str(row["prompt"]) + "\n"))
        + len(tokenizer.encode(str(row["response"])))
        < max_tokens
    ]
