"""
Competitor cross-validation methods for network community detection.

Includes NCV (Node Cross-Validation) and ECV (Edge Cross-Validation)
translated from all_base.R (the official NETCROP code by Chakrabarty et al., 2025).
"""

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sum_fast(X):
    return np.sum(X)


def _fast_sbm_est(A, g, n=None, K=None):
    """Estimate B matrix for SBM given adjacency and labels."""
    if n is None:
        n = A.shape[0]
    if K is None:
        K = int(g.max())
    B = np.zeros((K, K))
    if K == 1:
        B[0, 0] = _sum_fast(A) / (n ** 2 - n)
        return B

    G = [np.where(g == k + 1)[0] for k in range(K)]
    nk = np.array([len(g_k) for g_k in G])

    for k in range(K):
        for l in range(k, K):
            B[k, l] = B[l, k] = _sum_fast(A[np.ix_(G[k], G[l])]) / (nk[k] * nk[l])

    diag_vals = np.diag(B) * nk / np.maximum(nk - 1, 1)
    np.fill_diagonal(B, diag_vals)
    B[~np.isfinite(B)] = 1e-6
    return B


def _auc_score(A_vec, P_vec):
    """Compute negative AUC (matches R code convention)."""
    A_vec = np.asarray(A_vec, dtype=float).ravel()
    P_vec = np.asarray(P_vec, dtype=float).ravel()
    n1 = int(np.sum(A_vec == 0))
    n2 = len(A_vec) - n1
    if n1 == 0 or n2 == 0:
        return 0.0
    order = np.argsort(P_vec)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(P_vec) + 1, dtype=float)
    U = np.sum(ranks[A_vec == 0]) - n1 * (n1 + 1) / 2
    return -(1 - U / n1 / n2)


# ---------------------------------------------------------------------------
# NCV – Node Cross-Validation (from all_base.R: NCV.for.blockmodel)
# ---------------------------------------------------------------------------

def _cv_evaluate_all(A, train_index, holdout_index, K, dc_est=1):
    """Evaluate SBM and DCSBM on a single train/holdout split."""
    n = A.shape[0]
    reorder = np.concatenate([train_index, holdout_index])
    A_new = A[np.ix_(reorder, reorder)]
    n_holdout = len(holdout_index)
    n_train = n - n_holdout

    A1 = A_new[:n_train, :]

    # SVD for SBM
    k_svd = min(K, min(A1.shape) - 1)
    if k_svd < 1:
        k_svd = 1
    U_sbm, S_sbm, Vt_sbm = svds(A1 + 0.001, k=k_svd)
    V_sbm = Vt_sbm.T[:, ::-1]

    # SVD for DCSBM
    U_dc, S_dc, Vt_dc = svds(A1.astype(float), k=k_svd)
    V_dc = Vt_dc.T[:, ::-1]

    if K == 1:
        V_norms = np.abs(V_dc[:, 0])
    else:
        V = V_dc[:, :K]
        V_norms = np.sqrt(np.sum(V ** 2, axis=1))

    iso_index = V_norms == 0
    Psi = V_norms / np.max(V_norms) if np.max(V_norms) > 0 else V_norms
    Psi_outer = np.outer(Psi, Psi)

    if K == 1:
        A0 = A_new[:n_train, :n_train]
        pb = np.sum(A0) / n_train ** 2
        pb = np.clip(pb, 1e-6, 1 - 1e-6)

        A2 = A_new[n_train:, n_train:]
        tri_idx = np.tril_indices(A2.shape[0], k=-1)
        A2_lower = A2[tri_idx]

        auc_val = _auc_score(A2_lower, np.full(len(A2_lower), pb))
        loglike = -np.sum(A2_lower * np.log(pb)) - np.sum((1 - A2_lower) * np.log(1 - pb))
        l2_val = np.sum((A2_lower - pb) ** 2)

        # DC version
        N1 = np.arange(n_train)
        N2 = np.arange(n_train, n)
        dc_pb_num = np.sum(A_new[np.ix_(N1, N1)]) / 2 + np.sum(A_new[np.ix_(N1, N2)]) + 1
        dc_pb_den = np.sum(Psi_outer[np.ix_(N1, N1)]) / 2 + np.sum(Psi_outer[np.ix_(N1, N2)]) - np.sum(np.diag(Psi_outer)) + 1
        dc_pb = dc_pb_num / dc_pb_den

        Psi_holdout = Psi[n_train:]
        dc_P_hat = np.outer(Psi_holdout, Psi_holdout) * dc_pb
        dc_P_hat = np.clip(dc_P_hat, 1e-6, 1 - 1e-6)

        dc_P_lower = dc_P_hat[tri_idx]
        dc_loglike = -np.sum(A2_lower * np.log(dc_P_lower)) - np.sum((1 - A2_lower) * np.log(1 - dc_P_lower))
        dc_auc = _auc_score(A2_lower, dc_P_lower)
        dc_l2 = np.sum((A2_lower - dc_P_lower) ** 2)

        return {
            "loglike": loglike, "l2": l2_val, "auc": auc_val,
            "dc_loglike": dc_loglike, "dc_l2": dc_l2, "dc_auc": dc_auc,
        }

    # K > 1
    V_km = V_sbm[:, :K] if V_sbm.shape[1] >= K else V_sbm
    km = KMeans(n_clusters=K, n_init=30, max_iter=30).fit(V_km)

    inv_norms = 1.0 / np.maximum(V_norms, 1e-10)
    inv_norms[iso_index] = 1
    V_normalized = V_dc[:, :K] * inv_norms[:, None]
    dc_km = KMeans(n_clusters=K, n_init=30, max_iter=30).fit(V_normalized)

    km_labels = km.labels_ + 1
    dc_labels = dc_km.labels_ + 1

    B = np.zeros((K, K))
    dc_B = np.zeros((K, K))

    tmp = []
    for ii in range(1, K + 1):
        N1 = np.intersect1d(np.arange(n_train), np.where(km_labels == ii)[0])
        N2 = np.intersect1d(np.arange(n_train, n), np.where(km_labels == ii)[0])
        dc_N1 = np.intersect1d(np.arange(n_train), np.where(dc_labels == ii)[0])
        dc_N2 = np.intersect1d(np.arange(n_train, n), np.where(dc_labels == ii)[0])
        tmp.append({"N1": N1, "N2": N2, "dc_N1": dc_N1, "dc_N2": dc_N2})

    for i in range(K - 1):
        for j in range(i + 1, K):
            B[i, j] = B[j, i] = (
                _sum_fast(A_new[np.ix_(tmp[i]["N1"], tmp[j]["N1"])]) +
                _sum_fast(A_new[np.ix_(tmp[i]["N1"], tmp[j]["N2"])]) +
                _sum_fast(A_new[np.ix_(tmp[j]["N1"], tmp[i]["N2"])]) + 1
            ) / (
                len(tmp[i]["N1"]) * len(tmp[j]["N1"]) +
                len(tmp[j]["N1"]) * len(tmp[i]["N2"]) +
                len(tmp[i]["N1"]) * len(tmp[j]["N2"]) + 1
            )

            dc_denom = (
                _sum_fast(Psi_outer[np.ix_(tmp[i]["dc_N1"], tmp[j]["dc_N1"])]) +
                _sum_fast(Psi_outer[np.ix_(tmp[i]["dc_N1"], tmp[j]["dc_N2"])]) +
                _sum_fast(Psi_outer[np.ix_(tmp[j]["dc_N1"], tmp[i]["dc_N2"])]) + 1
            )
            dc_B[i, j] = dc_B[j, i] = (
                _sum_fast(A_new[np.ix_(tmp[i]["dc_N1"], tmp[j]["dc_N1"])]) +
                _sum_fast(A_new[np.ix_(tmp[i]["dc_N1"], tmp[j]["dc_N2"])]) +
                _sum_fast(A_new[np.ix_(tmp[j]["dc_N1"], tmp[i]["dc_N2"])]) + 1
            ) / dc_denom

    Theta = np.zeros((n, K))
    dc_Theta = np.zeros((n, K))
    for i in range(K):
        B[i, i] = (
            _sum_fast(A_new[np.ix_(tmp[i]["N1"], tmp[i]["N1"])]) / 2 +
            _sum_fast(A_new[np.ix_(tmp[i]["N1"], tmp[i]["N2"])]) + 1
        ) / (
            len(tmp[i]["N1"]) * (len(tmp[i]["N1"]) - 1) / 2 +
            len(tmp[i]["N1"]) * len(tmp[i]["N2"]) + 1
        )
        Theta[km_labels == i + 1, i] = 1

        dc_B[i, i] = (
            _sum_fast(A_new[np.ix_(tmp[i]["dc_N1"], tmp[i]["dc_N1"])]) / 2 +
            _sum_fast(A_new[np.ix_(tmp[i]["dc_N1"], tmp[i]["dc_N2"])]) + 1
        ) / (
            _sum_fast(Psi_outer[np.ix_(tmp[i]["dc_N1"], tmp[i]["dc_N1"])]) / 2 +
            _sum_fast(Psi_outer[np.ix_(tmp[i]["dc_N1"], tmp[i]["dc_N2"])]) -
            np.sum(np.diag(Psi_outer)) + 1
        )
        dc_Theta[dc_labels == i + 1, i] = 1

    # SBM prediction on holdout
    P_hat_holdout = Theta[n_train:] @ B @ Theta[n_train:].T
    P_hat_holdout = np.clip(P_hat_holdout, 1e-6, 1 - 1e-6)
    A2 = A_new[n_train:, n_train:]
    tri_idx = np.tril_indices(A2.shape[0], k=-1)
    A2_lower = A2[tri_idx]
    P_lower = P_hat_holdout[tri_idx]

    loglike = -np.sum(A2_lower * np.log(P_lower)) - np.sum((1 - A2_lower) * np.log(1 - P_lower))
    auc_val = _auc_score(A2_lower, P_lower)
    l2_val = np.sum((A2_lower - P_lower) ** 2)

    # DCSBM prediction
    tmp_imt = dc_Theta[n_train:] * Psi[n_train:, None]
    dc_P_hat = tmp_imt @ dc_B @ tmp_imt.T
    dc_P_hat = np.clip(dc_P_hat, 1e-6, 1 - 1e-6)
    dc_P_lower = dc_P_hat[tri_idx]

    dc_loglike = -np.sum(A2_lower * np.log(dc_P_lower)) - np.sum((1 - A2_lower) * np.log(1 - dc_P_lower))
    dc_auc = _auc_score(A2_lower, dc_P_lower)
    dc_l2 = np.sum((A2_lower - dc_P_lower) ** 2)

    return {
        "loglike": loglike, "l2": l2_val, "auc": auc_val,
        "dc_loglike": dc_loglike, "dc_l2": dc_l2, "dc_auc": dc_auc,
    }


def ncv_select(A, max_K=6, cv=3, dc_est=1, rng=None):
    """
    Network Cross-Validation (NCV) for selecting K and model type.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Adjacency matrix.
    max_K : int
        Maximum number of communities to consider.
    cv : int
        Number of CV folds.
    dc_est : int
        Degree-corrected estimation method.
    rng : numpy.random.Generator or None

    Returns
    -------
    dict with keys: ``dev``, ``l2``, ``auc`` (for SBM and DCSBM), ``l2_model``, ``dev_model``, ``auc_model``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    avg_se = np.full(max_K, np.inf)
    avg_log = np.full(max_K, np.inf)
    avg_auc = np.full(max_K, np.inf)
    dc_avg_se = np.full(max_K, np.inf)
    dc_avg_log = np.full(max_K, np.inf)
    dc_avg_auc = np.full(max_K, np.inf)

    sample_index = rng.permutation(n)
    max_fold_num = int(np.ceil(n / cv))
    fold_index = np.tile(np.arange(1, cv + 1), max_fold_num)[:n]
    cv_index = fold_index[np.argsort(sample_index)]

    for KK in range(1, max_K + 1):
        l2_vals = np.zeros(cv)
        log_vals = np.zeros(cv)
        auc_vals = np.zeros(cv)
        dc_l2_vals = np.zeros(cv)
        dc_log_vals = np.zeros(cv)
        dc_auc_vals = np.zeros(cv)

        for k in range(1, cv + 1):
            holdout = np.where(cv_index == k)[0]
            train = np.where(cv_index != k)[0]
            result = _cv_evaluate_all(A, train, holdout, KK, dc_est)

            l2_vals[k - 1] = result["l2"]
            log_vals[k - 1] = result["loglike"]
            auc_vals[k - 1] = result["auc"]
            dc_l2_vals[k - 1] = result["dc_l2"]
            dc_log_vals[k - 1] = result["dc_loglike"]
            dc_auc_vals[k - 1] = result["dc_auc"]

        avg_se[KK - 1] = np.mean(l2_vals)
        avg_log[KK - 1] = np.mean(log_vals)
        avg_auc[KK - 1] = np.mean(auc_vals)
        dc_avg_se[KK - 1] = np.mean(dc_l2_vals)
        dc_avg_log[KK - 1] = np.mean(dc_log_vals)
        dc_avg_auc[KK - 1] = np.mean(dc_auc_vals)

    def _pick_model(sbm_vals, dc_vals):
        if np.min(sbm_vals) > np.min(dc_vals):
            return f"DCSBM-{np.argmin(dc_vals) + 1}"
        return f"SBM-{np.argmin(sbm_vals) + 1}"

    return {
        "dev": avg_log, "l2": avg_se, "auc": avg_auc,
        "dc_dev": dc_avg_log, "dc_l2": dc_avg_se, "dc_auc": dc_avg_auc,
        "l2_model": _pick_model(avg_se, dc_avg_se),
        "dev_model": _pick_model(avg_log, dc_avg_log),
        "auc_model": _pick_model(avg_auc, dc_avg_auc),
    }


# ---------------------------------------------------------------------------
# ECV – Edge Cross-Validation (from all_base.R)
# ---------------------------------------------------------------------------

def _iter_svd_core_fast(A_missing, Kmax, tau=0, p_sample=1.0):
    """Rank-k SVD approximations with thresholding for missing-data matrices."""
    A_work = A_missing.copy()
    A_work[np.isnan(A_work)] = 0
    A_work = A_work / p_sample

    k_svd = min(Kmax, min(A_work.shape) - 1)
    if k_svd < 1:
        k_svd = 1
    U, S, Vt = svds(A_work.astype(float), k=k_svd)
    # Sort descending
    idx = np.argsort(S)[::-1]
    U, S, Vt = U[:, idx], S[idx], Vt[idx, :]

    results = []
    A_approx = np.zeros_like(A_work, dtype=float)
    for K in range(1, Kmax + 1):
        if K <= len(S):
            A_approx = A_approx + S[K - 1] * np.outer(U[:, K - 1], Vt[K - 1, :])
        A_thr = A_approx.copy()
        A_thr[A_thr < tau] = tau
        A_thr[A_thr > 1] = 1
        results.append({
            "A_thr": A_thr,
            "V": Vt[:K, :].T,  # right singular vectors
            "d": S[:K],
        })
    return results


def ecv_block(A, max_K=6, B=3, holdout_p=0.1, tau=0, dc_est=2, rng=None):
    """
    Edge Cross-Validation (ECV) for SBM/DCSBM model selection.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Adjacency matrix.
    max_K : int
        Maximum number of communities.
    B : int
        Number of holdout replicates.
    holdout_p : float
        Fraction of edges held out.
    tau, dc_est : parameters for estimation.
    rng : numpy.random.Generator or None

    Returns
    -------
    dict with keys ``l2``, ``dev``, ``l2_model``, ``dev_model``, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    edge_indices = np.where(np.triu(np.ones((n, n), dtype=bool), k=1))
    edge_n = len(edge_indices[0])
    holdout_n = int(np.floor(holdout_p * edge_n))

    block_err_mat = np.zeros((B, max_K))
    loglike_mat = np.zeros((B, max_K))
    dc_block_err_mat = np.zeros((B, max_K))
    dc_loglike_mat = np.zeros((B, max_K))

    for b in range(B):
        holdout_sel = rng.choice(edge_n, holdout_n, replace=False)
        holdout_rows = edge_indices[0][holdout_sel]
        holdout_cols = edge_indices[1][holdout_sel]

        A_new = A.astype(float).copy()
        A_new[holdout_rows, holdout_cols] = np.nan
        A_new[holdout_cols, holdout_rows] = np.nan

        Omega_mask = np.isnan(A_new)
        Omega_idx = np.where(Omega_mask)

        svd_results = _iter_svd_core_fast(A_new, max_K, tau=tau, p_sample=1 - holdout_p)

        for k_idx in range(max_K):
            k = k_idx + 1
            A_thr = svd_results[k_idx]["A_thr"]

            if k == 1:
                A_temp = A_new.copy()
                A_temp[Omega_mask] = 0
                pb = (np.nansum(A_new) + 1) / (np.sum(~np.isnan(A_new)) - np.sum(~np.isnan(np.diag(A_new))) + 1)
                pb = np.clip(pb, 1e-6, 1 - 1e-6)
                A_Omega = A[Omega_idx]
                block_err_mat[b, k_idx] = np.sum((pb - A_Omega) ** 2)
                loglike_mat[b, k_idx] = -np.sum(A_Omega * np.log(pb)) - np.sum((1 - A_Omega) * np.log(1 - pb))
                dc_block_err_mat[b, k_idx] = block_err_mat[b, k_idx]
                dc_loglike_mat[b, k_idx] = loglike_mat[b, k_idx]
                continue

            # SBM estimate
            V_k = svd_results[k_idx]["V"][:, :k] if svd_results[k_idx]["V"].shape[1] >= k else svd_results[k_idx]["V"]
            km = KMeans(n_clusters=k, n_init=30, max_iter=30).fit(V_k)
            labels = km.labels_

            B_mat = np.zeros((k, k))
            Theta = np.zeros((n, k))
            for i in range(k):
                Ni = np.where(labels == i)[0]
                Theta[Ni, i] = 1
                for j in range(i, k):
                    Nj = np.where(labels == j)[0]
                    sub = A_new[np.ix_(Ni, Nj)]
                    valid = ~np.isnan(sub)
                    if i == j:
                        diag_valid = ~np.isnan(np.diag(A_new[np.ix_(Ni, Nj)])) if len(Ni) == len(Nj) else np.array([])
                        denom = np.sum(valid) - len(diag_valid[diag_valid]) + 1
                    else:
                        denom = np.sum(valid) + 1
                    B_mat[i, j] = B_mat[j, i] = (np.nansum(sub) + 1) / max(denom, 1)

            P_hat = Theta @ B_mat @ Theta.T
            np.fill_diagonal(P_hat, 0)
            P_Omega = np.clip(P_hat[Omega_idx], 1e-6, 1 - 1e-6)
            A_Omega = A[Omega_idx]

            block_err_mat[b, k_idx] = np.sum((P_Omega - A_Omega) ** 2)
            loglike_mat[b, k_idx] = -np.sum(A_Omega * np.log(P_Omega)) - np.sum((1 - A_Omega) * np.log(1 - P_Omega))

            # DCSBM estimate
            V_norms = np.sqrt(np.sum(V_k ** 2, axis=1))
            V_norms[V_norms == 0] = 1
            V_normalized = V_k / V_norms[:, None]
            dc_km = KMeans(n_clusters=k, n_init=30, max_iter=30).fit(V_normalized)
            dc_labels = dc_km.labels_

            dc_B = np.zeros((k, k))
            dc_Theta = np.zeros((n, k))
            A_new_na = A_new.copy()
            for i in range(k):
                Ni = np.where(dc_labels == i)[0]
                dc_Theta[Ni, i] = 1
                for j in range(k):
                    Nj = np.where(dc_labels == j)[0]
                    dc_B[i, j] = np.nansum(A_new_na[np.ix_(Ni, Nj)]) + 0.01

            partial_d = np.nansum(A_new_na, axis=1)
            partial_gd = dc_B.sum(axis=1)
            B_g = dc_Theta @ partial_gd
            phi = np.where(B_g > 0, partial_d / B_g, 0)

            dc_P_hat = (dc_Theta * phi[:, None]) @ dc_B @ (dc_Theta * phi[:, None]).T
            np.fill_diagonal(dc_P_hat, 0)
            dc_P_Omega = np.clip(dc_P_hat[Omega_idx], 1e-6, 1 - 1e-6)

            dc_block_err_mat[b, k_idx] = np.sum((dc_P_Omega - A_Omega) ** 2)
            dc_loglike_mat[b, k_idx] = -np.sum(A_Omega * np.log(dc_P_Omega)) - np.sum((1 - A_Omega) * np.log(1 - dc_P_Omega))

    avg_l2 = block_err_mat.mean(axis=0)
    avg_dev = loglike_mat.sum(axis=0)
    dc_avg_l2 = dc_block_err_mat.mean(axis=0)
    dc_avg_dev = dc_loglike_mat.sum(axis=0)

    def _pick(sbm, dc):
        if np.min(sbm) > np.min(dc):
            return f"DCSBM-{np.argmin(dc) + 1}"
        return f"SBM-{np.argmin(sbm) + 1}"

    return {
        "l2": avg_l2, "dev": avg_dev,
        "dc_l2": dc_avg_l2, "dc_dev": dc_avg_dev,
        "l2_model": _pick(avg_l2, dc_avg_l2),
        "dev_model": _pick(avg_dev, dc_avg_dev),
    }
