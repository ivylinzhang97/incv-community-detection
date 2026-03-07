"""
INCV – Inductive Node-split Cross-Validation for Community Detection
=====================================================================

A Python implementation of the methods described in
*"Inductive Node-split Cross-Validation in Networks"*.

Modules
-------
core
    Core INCV functions: community simulation, spectral clustering,
    SBM probability estimation, f-fold and random-split NSCV.
competitors
    Competitor methods: NCV (Node CV) and ECV (Edge CV).
applications
    Data loaders for the International Trade and 108th U.S. Senate datasets.
simulations
    Simulation runners for fold comparison, community comparison, and
    method comparison experiments.
plotting
    Visualization utilities for CV loss curves, network plots, and
    simulation result figures.
"""

__version__ = "0.1.0"

# Core public API
from .core import (
    community_sim,
    community_sim_sbm,
    sbm_spectral_clustering,
    sbm_prob,
    nscv_f_fold,
    nscv_random_split,
    edge_index_map,
    neglog,
)

# Competitors
from .competitors import ncv_select, ecv_block

# Applications
from .applications import load_trade_data, load_senate_data

# Plotting
from .plotting import (
    plot_cv_loss,
    plot_network,
    plot_fold_comparison,
    plot_community_comparison,
)

__all__ = [
    # core
    "community_sim",
    "community_sim_sbm",
    "sbm_spectral_clustering",
    "sbm_prob",
    "nscv_f_fold",
    "nscv_random_split",
    "edge_index_map",
    "neglog",
    # competitors
    "ncv_select",
    "ecv_block",
    # applications
    "load_trade_data",
    "load_senate_data",
    # plotting
    "plot_cv_loss",
    "plot_network",
    "plot_fold_comparison",
    "plot_community_comparison",
]
