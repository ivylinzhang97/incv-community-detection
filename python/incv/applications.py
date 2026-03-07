"""
Real-data application helpers for the INCV package.

Provides data loading and adjacency-matrix construction for the two
application datasets used in the paper:

1. **International Trade** (Section 5.1): builds a binary trade network
   from the Trade.csv data.
2. **108th U.S. Senate** (Section 5.2): builds a co-sponsorship network
   from the senate_108.csv data.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _resolve_path(filename, data_dir=None):
    if data_dir is not None:
        return os.path.join(data_dir, filename)
    return os.path.join(_DATA_DIR, filename)


# ---------------------------------------------------------------------------
# International Trade
# ---------------------------------------------------------------------------

def load_trade_data(year=20, quantile=0.75, data_dir=None):
    """
    Load the international trade dataset and construct an adjacency matrix.

    An edge between countries *i* and *j* is set to 1 when the total
    (symmetrised) log-trade exceeds the ``quantile``-th percentile of
    *either* country's trade distribution.

    Parameters
    ----------
    year : int
        Time period index in the data (default 20, corresponding to year 2000
        in the original dataset).
    quantile : float
        Percentile threshold for binarisation (default 0.75).
    data_dir : str or None
        Directory containing ``Trade.csv`` and ``TradeAttributes.csv``.

    Returns
    -------
    A : np.ndarray of shape (N, N)
        Symmetric binary adjacency matrix.
    W : np.ndarray of shape (N, N)
        Symmetrised weight matrix.
    attributes : pd.DataFrame
        Node attributes (country name, continent).
    """
    trade_path = _resolve_path("Trade.csv", data_dir)
    attr_path = _resolve_path("TradeAttributes.csv", data_dir)

    trade = pd.read_csv(trade_path)
    trade = trade[trade["t"] == year].copy()
    trade = trade.sort_values("exporter")
    trade = trade[["i", "j", "log_trade"]].rename(
        columns={"i": "from", "j": "to", "log_trade": "weight"}
    )

    N = max(trade["from"].max(), trade["to"].max())
    W = np.zeros((N, N))
    for _, row in trade.iterrows():
        W[int(row["from"]) - 1, int(row["to"]) - 1] = row["weight"]

    # Symmetrise
    for i in range(N):
        for j in range(i + 1, N):
            W[i, j] = W[i, j] + W[j, i]
            W[j, i] = W[i, j]

    # Binarise
    A = np.zeros((N, N), dtype=int)
    for i in range(N - 1):
        for j in range(i + 1, N):
            qi = np.quantile(W[i, :], quantile)
            qj = np.quantile(W[:, j], quantile)
            if W[i, j] > qi or W[i, j] > qj:
                A[i, j] = 1
            A[j, i] = A[i, j]

    attributes = pd.read_csv(attr_path) if os.path.exists(attr_path) else None
    return A, W, attributes


# ---------------------------------------------------------------------------
# 108th U.S. Senate
# ---------------------------------------------------------------------------

def load_senate_data(threshold=4, data_dir=None):
    """
    Load the 108th U.S. Senate co-sponsorship data and construct an adjacency matrix.

    An edge between senators *i* and *j* is set to 1 when their
    symmetrised co-sponsorship count exceeds ``threshold``.

    Parameters
    ----------
    threshold : int
        Minimum symmetrised co-sponsorship count for an edge (default 4).
    data_dir : str or None
        Directory containing ``senate_108.csv`` and
        ``senate_108_attribute.csv``.

    Returns
    -------
    A : np.ndarray of shape (N, N)
        Symmetric binary adjacency matrix.
    W : np.ndarray of shape (N, N)
        Symmetrised co-sponsorship count matrix.
    attributes : pd.DataFrame
        Node attributes (senator names).
    """
    senate_path = _resolve_path("senate_108.csv", data_dir)
    attr_path = _resolve_path("senate_108_attribute.csv", data_dir)

    senate = pd.read_csv(senate_path)
    N = max(senate["i"].max(), senate["j"].max())
    W = np.zeros((N, N))
    for _, row in senate.iterrows():
        W[int(row["i"]) - 1, int(row["j"]) - 1] = row["count"]

    # Symmetrise
    for i in range(N):
        for j in range(i + 1, N):
            W[i, j] = W[i, j] + W[j, i]
            W[j, i] = W[i, j]

    # Binarise
    A = np.zeros((N, N), dtype=int)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if W[i, j] > threshold:
                A[i, j] = 1
            A[j, i] = A[i, j]

    attributes = pd.read_csv(attr_path) if os.path.exists(attr_path) else None
    return A, W, attributes
