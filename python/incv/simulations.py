"""
Simulation runners for INCV community detection.

Translates the three simulation scripts:
- sim_folds.R: compare different fold numbers.
- sim_community.R: compare different community structures.
- sim_compare.R: compare INCV against competitor CV methods.
"""

import numpy as np
import pandas as pd
from .core import community_sim, nscv_f_fold
from .competitors import ncv_select, ecv_block


# ---------------------------------------------------------------------------
# Simulation 1: Fold comparison  (sim_folds.R)
# ---------------------------------------------------------------------------

def sim_folds(seed=1, n_set_default=None, n_set_hard=None, k_true=4,
              k_vec=None, p_set=None, q_set=None, fold_set=None, verbose=True):
    """
    Run fold-comparison simulation for a single seed.

    Parameters
    ----------
    seed : int
    n_set_default, n_set_hard : list[int]
    k_true : int
    k_vec : list[int]
    p_set, q_set : list[float]
    fold_set : list[int]
    verbose : bool

    Returns
    -------
    pd.DataFrame with columns: seed, n, k_true, p, q, fold, k_loss, k_mse.
    """
    if n_set_default is None:
        n_set_default = [100, 200, 300, 400, 500]
    if n_set_hard is None:
        n_set_hard = [200, 400, 600, 800, 1000]
    if k_vec is None:
        k_vec = list(range(2, 7))
    if p_set is None:
        p_set = [0.3, 0.3, 0.3]
    if q_set is None:
        q_set = [0.1, 0.15, 0.2]
    if fold_set is None:
        fold_set = [2, 3, 5, 10]

    rows = []
    for j, (p, q) in enumerate(zip(p_set, q_set)):
        n_set = n_set_hard if (p == 0.3 and q == 0.2) else n_set_default

        for n in n_set:
            for f in fold_set:
                rng = np.random.default_rng(10000 * seed + 200 * (j + 1) + 10 * (n // 100) + f)
                _, A = community_sim(k=k_true, n=n, n1=n // k_true, p=p, q=q, rng=rng)
                result = nscv_f_fold(A, k_vec=k_vec, restricted=True, f=f,
                                     method="affinity", p_est_type=3, rng=rng)
                rows.append({
                    "seed": seed, "n": n, "k_true": k_true,
                    "p": p, "q": q, "fold": f,
                    "k_loss": result["k_loss"], "k_mse": result["k_mse"],
                })
                if verbose:
                    print(f"  [seed={seed}] p={p}, q={q}, n={n}, f={f} => "
                          f"k_loss={result['k_loss']}, k_mse={result['k_mse']}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation 2: Community structure comparison  (sim_community.R)
# ---------------------------------------------------------------------------

def sim_community(seed=1, n=500, k_set=None, q_set=None, ratio=3,
                  f=10, k_candid_func=None, n1_func=None, verbose=True):
    """
    Run community-structure comparison simulation for a single seed.

    Parameters
    ----------
    seed : int
    n : int
    k_set : list[int]
    q_set : list[float]
    ratio : float
        p = ratio * q.
    f : int
    k_candid_func : callable
        Maps k -> list of candidate K values.
    n1_func : callable
        Maps k -> list of smallest-community sizes.
    verbose : bool

    Returns
    -------
    pd.DataFrame
    """
    if k_set is None:
        k_set = [2, 3, 6, 8]
    if q_set is None:
        q_set = [0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]

    def _default_k_candid(k):
        if k == 2:
            return list(range(1, 7))
        elif k == 3:
            return list(range(1, 7))
        elif k == 6:
            return list(range(2, 11))
        elif k == 8:
            return list(range(4, 13))
        return list(range(2, 11))

    def _default_n1(k):
        if k == 2:
            return [50, 150, 250]
        elif k == 3:
            return [30, 90, 150]
        elif k == 6:
            return [25, 50, 80]
        elif k == 8:
            return [20, 40, 60]
        return [50]

    if k_candid_func is None:
        k_candid_func = _default_k_candid
    if n1_func is None:
        n1_func = _default_n1

    rows = []
    for j, q in enumerate(q_set):
        p = ratio * q
        for k in k_set:
            k_vec = k_candid_func(k)
            for n_min in n1_func(k):
                rng = np.random.default_rng(
                    10000 * seed + 200 * (j + 1) + 10 * (n_min // 10) + k
                )
                _, A = community_sim(k=k, n=n, n1=n_min, p=p, q=q, rng=rng)
                result = nscv_f_fold(A, k_vec=k_vec, restricted=True, f=f,
                                     method="affinity", p_est_type=3, rng=rng)
                rows.append({
                    "seed": seed, "n": n, "k_true": k, "n1": n_min,
                    "p": p, "q": q, "fold": f,
                    "k_loss": result["k_loss"], "k_mse": result["k_mse"],
                })
                if verbose:
                    print(f"  [seed={seed}] q={q}, k={k}, n1={n_min} => "
                          f"k_loss={result['k_loss']}, k_mse={result['k_mse']}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation 3: Method comparison  (sim_compare.R)
# ---------------------------------------------------------------------------

def sim_compare(seed=1, n=500, k_set=None, rho_set=None, k_vec=None,
                f=10, n1_func=None, verbose=True):
    """
    Run method-comparison simulation (INCV vs NCV vs ECV).

    Parameters
    ----------
    seed : int
    n : int
    k_set : list[int]
    rho_set : list[float]
    k_vec : list[int]
    f : int
    n1_func : callable
    verbose : bool

    Returns
    -------
    pd.DataFrame with columns: seed, n1, k_true, rho, k_loss, k_mse, k_ncv, k_ecv.
    """
    if k_set is None:
        k_set = [3, 6]
    if rho_set is None:
        rho_set = [0.05, 0.1, 0.2, 0.3]
    if k_vec is None:
        k_vec = list(range(2, 10))

    def _default_n1(k):
        if k == 3:
            return [30, 90, 150]
        elif k == 6:
            return [25, 50, 80]
        return [50]

    if n1_func is None:
        n1_func = _default_n1

    rows = []
    for j, rho in enumerate(rho_set):
        p = rho * 3
        q = rho
        for k in k_set:
            for n_min in n1_func(k):
                rng = np.random.default_rng(
                    10000 * seed + 200 * (j + 1) + 10 * (n_min // 10) + k
                )
                _, A = community_sim(k=k, n=n, n1=n_min, p=p, q=q, rng=rng)

                # INCV
                incv_result = nscv_f_fold(A, k_vec=k_vec, restricted=True, f=f,
                                          method="affinity", p_est_type=3, rng=rng)

                # NCV
                try:
                    ncv_result = ncv_select(A, max_K=9, rng=rng)
                    k_ncv = ncv_result["l2_model"]
                except Exception:
                    k_ncv = "NA"

                # ECV
                try:
                    ecv_result = ecv_block(A, max_K=9, rng=rng)
                    k_ecv = ecv_result["l2_model"]
                except Exception:
                    k_ecv = "NA"

                rows.append({
                    "seed": seed, "n1": n_min, "k_true": k, "rho": rho,
                    "k_loss": incv_result["k_loss"],
                    "k_mse": incv_result["k_mse"],
                    "k_ncv": k_ncv,
                    "k_ecv": k_ecv,
                })
                if verbose:
                    print(f"  [seed={seed}] rho={rho}, k={k}, n1={n_min} => "
                          f"k_loss={incv_result['k_loss']}, "
                          f"k_ncv={k_ncv}, k_ecv={k_ecv}")

    return pd.DataFrame(rows)
