"""Direction extraction in ACS.

Implements the three direction primitives from the paper:
  - native_direction:    v_a^{(m)} = mean(pos) - mean(neg) in model m's hidden space
                         (Equation 3).
  - canonical_direction: c_a = (1 / |S|) * sum_{m in S} pi_m^{dir}(v_a^{(m)})
                         in the shared ACS (Equation 4).
  - reconstruct:         v_a^{(m_u, recon)} = A_{m_u}^T * c_a in the unseen
                         model's native hidden space (Equation 5).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch

from acs.projector import AnchorProjector


def _to_tensor(x, device: str = "cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    import numpy as np
    return torch.from_numpy(np.asarray(x)).to(device)


def native_direction(pos_acts, neg_acts) -> torch.Tensor:
    """Mean-difference direction in a model's native hidden space.

    Implements Equation 3:
        v_a^{(m)} = (1/|P|) sum pos - (1/|N|) sum neg,    then unit-normalized.

    Parameters
    ----------
    pos_acts, neg_acts : array-like or torch.Tensor of shape [n, d_m]
        Activations for axis-positive and axis-negative prompts at layer L_m.

    Returns
    -------
    torch.Tensor of shape [d_m], unit-normalized.
    """
    pos = _to_tensor(pos_acts).float()
    neg = _to_tensor(neg_acts).float()
    v = pos.mean(dim=0) - neg.mean(dim=0)
    return v / (v.norm() + 1e-8)


def canonical_direction(
    native_dirs: Mapping[str, torch.Tensor],
    projectors: Mapping[str, AnchorProjector],
) -> torch.Tensor:
    """Average projected source directions into a canonical ACS direction.

    Implements Equation 4:
        c_a = (1 / |S|) * sum_{m in S} pi_m^{dir}(v_a^{(m)}).

    Parameters
    ----------
    native_dirs : dict[str, Tensor[d_m]]
        Native axis directions per source model (output of `native_direction`).
        Each direction must come from the same axis.
    projectors : dict[str, AnchorProjector]
        Anchor projectors per source model. Keys must match `native_dirs`.

    Returns
    -------
    torch.Tensor of shape [N], unit-normalized canonical direction in ACS.
    """
    assert set(native_dirs) == set(projectors), (
        "native_dirs and projectors must share the same model keys"
    )
    N = next(iter(projectors.values())).N
    proj_dirs = []
    for m, v in native_dirs.items():
        u = projectors[m](v.to(projectors[m].anc_unit.device), treat_as_point=False)
        u = u / (u.norm() + 1e-8)
        proj_dirs.append(u)
    stack = torch.stack(proj_dirs, dim=0)                              # [|S|, N]
    c = stack.mean(dim=0)                                              # [N]
    c = c / (c.norm() + 1e-8)
    return c


def reconstruct(
    canonical: torch.Tensor,
    projector: AnchorProjector,
    unit_normalize: bool = True,
) -> torch.Tensor:
    """Reconstruct a canonical ACS direction in an unseen model's native space.

    Implements Equation 5:
        v_a^{(m_u, recon)} = A_{m_u}^T * c_a.

    Parameters
    ----------
    canonical : torch.Tensor of shape [N]
        Canonical direction in ACS (output of `canonical_direction`).
    projector : AnchorProjector
        Anchor projector for the unseen target model m_u.
    unit_normalize : bool
        If True (default), unit-normalize the resulting native-space vector.

    Returns
    -------
    torch.Tensor of shape [d_{m_u}], unit-normalized (if requested).
    """
    v = projector.back_project(canonical.to(projector.anc_unit.device),
                               treat_as_point=False)
    if unit_normalize:
        v = v / (v.norm() + 1e-8)
    return v
