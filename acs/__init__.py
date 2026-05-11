"""ACS — Anchor-Projected Coordinate Space.

A shared coordinate system for behavioral-axis directions across LLM families.

Public API:
    AnchorProjector       — frozen per-model anchor projection (R^{d_m} -> R^N).
    native_direction      — mean(pos) - mean(neg) in a model's native space.
    canonical_direction   — average of source-model anchor-projected directions.
    reconstruct           — back-projection of a canonical direction into an
                            unseen model's native hidden space.
"""
from acs.projector import AnchorProjector
from acs.canonical import native_direction, canonical_direction, reconstruct

__all__ = [
    "AnchorProjector",
    "native_direction",
    "canonical_direction",
    "reconstruct",
]
