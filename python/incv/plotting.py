"""
Plotting utilities for INCV community detection results.

Provides functions to visualize cross-validation loss curves,
network graphs with community coloring, and simulation comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---------------------------------------------------------------------------
# CV loss plots (matches senate.R / international_trade.R loss figures)
# ---------------------------------------------------------------------------

def plot_cv_loss(k_vec, cv_loss, cv_mse=None, k_best_loss=None, k_best_mse=None,
                 title_prefix="INCV", figsize=(8, 10), save_path=None):
    """
    Plot NLL and MSE cross-validation curves.

    Parameters
    ----------
    k_vec : array-like
        Candidate K values.
    cv_loss : array-like
        Mean negative log-likelihood for each K.
    cv_mse : array-like or None
        Mean MSE for each K.  If None only the NLL panel is drawn.
    k_best_loss, k_best_mse : int or None
        Highlighted best K (green marker).
    title_prefix : str
    figsize : tuple
    save_path : str or None
        If given, save figure to this path.
    """
    n_panels = 2 if cv_mse is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # NLL panel
    ax = axes[0]
    ax.plot(k_vec, cv_loss, "k-o", markersize=8, linewidth=1.5)
    if k_best_loss is not None:
        idx = list(k_vec).index(k_best_loss)
        ax.plot(k_best_loss, cv_loss[idx], "o", color="lime",
                markersize=18, alpha=0.5, zorder=5)
    ax.set_ylabel("Neg-Log-Likelihood", fontsize=12)
    ax.set_title(f"{title_prefix}: NLL", fontsize=14)
    ax.set_xticks(k_vec)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # MSE panel
    if cv_mse is not None:
        ax = axes[1]
        ax.plot(k_vec, cv_mse, "k-o", markersize=8, linewidth=1.5)
        if k_best_mse is not None:
            idx = list(k_vec).index(k_best_mse)
            ax.plot(k_best_mse, cv_mse[idx], "o", color="lime",
                    markersize=18, alpha=0.5, zorder=5)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title(f"{title_prefix}: MSE", fontsize=14)
        ax.set_xticks(k_vec)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Candidate K", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# Network visualization
# ---------------------------------------------------------------------------

def plot_network(A, labels=None, node_names=None,
                 colors=None, layout="spring", figsize=(10, 10),
                 node_size=300, font_size=8, title=None, save_path=None):
    """
    Plot a network graph with optional community coloring.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Adjacency matrix.
    labels : np.ndarray or None
        1-based community labels.
    node_names : list[str] or None
    colors : list[str] or None
        One color per community.
    layout : {"spring", "kamada_kawai", "circular"}
    figsize, node_size, font_size : display parameters.
    title : str or None
    save_path : str or None

    Returns
    -------
    fig, ax
    """
    if not _HAS_NX:
        raise ImportError("networkx is required for plot_network. "
                          "Install it with: pip install networkx")

    G = nx.from_numpy_array(A)
    n = A.shape[0]

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    default_colors = ["#87CEEB", "#9ACD32", "#FA8072", "#FFD700",
                      "#DDA0DD", "#FF6347", "#40E0D0", "#EE82EE"]

    if labels is not None:
        k = int(labels.max())
        if colors is None:
            colors = default_colors[:k]
        node_colors = [colors[(labels[i] - 1) % len(colors)] for i in range(n)]
    else:
        node_colors = ["#FFD700"] * n

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        font_size=font_size,
        labels={i: node_names[i] for i in range(n)} if node_names else None,
        edge_color="gray",
        alpha=0.9,
        width=0.5,
    )
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Simulation result plots (fold comparison, community comparison)
# ---------------------------------------------------------------------------

def plot_fold_comparison(results_df, p_set, q_set, fold_set,
                         criterion="nll", figsize=(16, 10), save_path=None):
    """
    Plot success rate vs network size for different fold numbers.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns: ``p``, ``q``, ``fold``, ``n``, ``criterion``, ``success_rate``.
    p_set, q_set : list[float]
        Parameter combinations for panels.
    fold_set : list[int]
    criterion : {"nll", "mse"}
    figsize : tuple
    save_path : str or None
    """
    import pandas as pd

    data = results_df[results_df["criterion"] == criterion]
    n_panels = len(p_set)
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    color_map = ["blue", "red", "orange", "green"]
    marker_map = ["s", "o", "^", "D"]

    for idx, (p0, q0) in enumerate(zip(p_set, q_set)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        panel = data[(data["p"] == p0) & (data["q"] == q0)]

        for j, f in enumerate(fold_set):
            df = panel[panel["fold"] == f].sort_values("n")
            ax.plot(df["n"], df["success_rate"], "-o",
                    color=color_map[j % len(color_map)],
                    marker=marker_map[j % len(marker_map)],
                    label=f"{f}-fold", linewidth=2, markersize=6)

        ax.set_title(f"{criterion.upper()}: p={p0}, q={q0}", fontsize=11)
        ax.set_xlabel("Network Size n")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused axes
    for idx in range(len(p_set), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


def plot_community_comparison(results_df, k_set, n1_func,
                               criterion="nll", figsize=(16, 10), save_path=None):
    """
    Plot success rate vs sparsity q for different community sizes.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns: ``q``, ``k_true``, ``n1``, ``criterion``, ``success_rate``.
    k_set : list[int]
    n1_func : callable
        Maps k -> list of n1 values.
    criterion : {"nll", "mse"}
    figsize, save_path : see above.
    """
    data = results_df[results_df["criterion"] == criterion]
    n_panels = len(k_set)
    ncols = min(n_panels, 4)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    color_map = ["blue", "red", "green"]
    marker_map = ["s", "o", "^"]

    for idx, k in enumerate(k_set):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        panel = data[data["k_true"] == k]
        n1_set = n1_func(k)

        for j, n_min in enumerate(n1_set):
            df = panel[panel["n1"] == n_min].sort_values("q")
            ax.plot(df["q"], df["success_rate"], "-o",
                    color=color_map[j % len(color_map)],
                    marker=marker_map[j % len(marker_map)],
                    label=f"n_min={n_min}", linewidth=2, markersize=6)

        ax.set_title(f"{criterion.upper()}: K*={k}", fontsize=11)
        ax.set_xlabel("Network Sparsity q")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(len(k_set), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes
