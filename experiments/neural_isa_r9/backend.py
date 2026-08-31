"""Neural recipient backend for the R9 capability-specific diagnostic."""

from __future__ import annotations

import torch
from torch import nn


class PackageConditionedGRUBackend(nn.Module):
    """Read recipient states under a package and return canonical logit residuals.

    This class deliberately has no tokenizer, program parser, transition solver,
    row lookup, or capability metadata. Gate A permits its parameters to be
    capability-specific; Gate B will not.
    """

    def __init__(self, host_width: int, *, hidden_width: int = 128) -> None:
        super().__init__()
        self.host_width = int(host_width)
        self.hidden_width = int(hidden_width)
        self.package_encoder = nn.Sequential(
            nn.Linear(3 * 8 * 8, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(host_width),
            nn.Linear(host_width, hidden_width),
            nn.GELU(),
        )
        self.sequence = nn.GRU(hidden_width, hidden_width, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 8),
        )

    def forward(
        self,
        recipient_states: torch.Tensor,
        lengths: torch.Tensor,
        package: torch.Tensor,
    ) -> torch.Tensor:
        if recipient_states.ndim != 3 or recipient_states.shape[-1] != self.host_width:
            raise ValueError("recipient state shape changed")
        if lengths.shape != (recipient_states.shape[0],):
            raise ValueError("recipient length shape changed")
        if package.ndim == 3:
            package = package.unsqueeze(0)
        if tuple(package.shape[1:]) != (3, 8, 8):
            raise ValueError("canonical package shape changed")
        if package.shape[0] == 1 and recipient_states.shape[0] != 1:
            package = package.expand(recipient_states.shape[0], -1, -1, -1)
        if package.shape[0] != recipient_states.shape[0]:
            raise ValueError("package batch changed")
        if not torch.isfinite(recipient_states).all() or not torch.isfinite(package).all():
            raise ValueError("non-finite neural input")

        package_state = self.package_encoder(package.float().reshape(package.shape[0], -1))
        projected = self.token_projection(recipient_states.float())
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final = self.sequence(packed, package_state.unsqueeze(0))
        return self.head(torch.cat((final.squeeze(0), package_state), dim=-1))

