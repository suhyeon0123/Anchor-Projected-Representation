"""AnchorProjector — frozen per-model projection from hidden space into ACS.

Given anchor activations A_m in R^{N x d_m} for model m, this module implements
the two projections defined in the paper (Section 3.1):

    pi_pt(h)  = A_m_hat * norm(h - mu_m)     # point projection (absolute state)
    pi_dir(v) = A_m_hat * norm(v)            # directional projection (difference)

where A_m_hat is the row-normalized version of (A_m - mu_m), with
mu_m = (1/N) sum_i A_m[i].

Both projections produce R^N coordinates in the Anchor Coordinate Space (ACS).
The projector is frozen (no trainable parameters).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnchorProjector(nn.Module):
    """Frozen projection R^{d_m} -> R^N via centered cosine similarity to N anchors.

    Parameters
    ----------
    anchor_acts : torch.Tensor of shape [N, d_m], float32
        Anchor activations forwarded through model m at layer L_m.
    """

    def __init__(self, anchor_acts: torch.Tensor):
        super().__init__()
        mu = anchor_acts.mean(dim=0, keepdim=True)                      # [1, d_m]
        anc_c = anchor_acts - mu                                        # [N, d_m]
        anc_norm = anc_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
        anc_unit = anc_c / anc_norm                                     # [N, d_m]
        self.register_buffer("mu", mu)
        self.register_buffer("anc_unit", anc_unit)
        self.register_buffer("anc_c", anc_c)
        self.N = anchor_acts.shape[0]
        self.d_m = anchor_acts.shape[1]

    @torch.no_grad()
    def project(self, x: torch.Tensor, treat_as_point: bool = True) -> torch.Tensor:
        """Project an activation x in R^{d_m} into ACS (R^N).

        Parameters
        ----------
        x : torch.Tensor of shape [..., d_m]
        treat_as_point : bool
            If True, subtract mu_m before normalization (point projection,
            Equation 2 in the paper). If False, do not subtract mu_m (directional
            projection, used for difference vectors v_a^{(m)}).
        """
        if treat_as_point:
            x = x - self.mu
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / x_norm
        return x_unit @ self.anc_unit.T

    def forward(self, x: torch.Tensor, treat_as_point: bool = True) -> torch.Tensor:
        return self.project(x, treat_as_point=treat_as_point)

    @torch.no_grad()
    def back_project(self, r: torch.Tensor, treat_as_point: bool = False) -> torch.Tensor:
        """Back-project an ACS vector r in R^N into the native hidden space R^{d_m}.

        Implements Equation 5 (paper Section 3.4): v_recon = A_m_hat^T * c_a.
        This is the linear adjoint of the unit-anchor projector, used to
        reconstruct a canonical direction in an unseen model's hidden space
        from anchor activations alone.

        Parameters
        ----------
        r : torch.Tensor of shape [..., N]
        treat_as_point : bool
            If True, add mu_m back after back-projection (only meaningful for
            absolute states; directions should leave this False).
        """
        v = r @ self.anc_unit                                           # [..., d_m]
        if treat_as_point:
            v = v + self.mu
        return v
