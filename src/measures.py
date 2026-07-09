"""Compatibility wrapper for causal-analysis helpers.

The implementation now lives in :mod:`src.causal_analysis`.
"""

from __future__ import annotations

from .causal_analysis import (
    compute_ccm,
    compute_dtw,
    compute_te,
    granger_strength,
    run_granger_causality_report,
    sweep_ccm_convergence,
    sweep_ccm_convergence_steps,
    sweep_ccm_grid,
    sweep_ccm_grid_all,
    sweep_transfer_entropy,
    sweep_transfer_entropy_grid,
)

__all__ = [
    "compute_ccm",
    "compute_dtw",
    "compute_te",
    "granger_strength",
    "run_granger_causality_report",
    "sweep_ccm_convergence",
    "sweep_ccm_convergence_steps",
    "sweep_ccm_grid",
    "sweep_ccm_grid_all",
    "sweep_transfer_entropy",
    "sweep_transfer_entropy_grid",
]
