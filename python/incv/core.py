"""
Core INCV (Inductive Node-split Cross-Validation) functions for community detection.

Translated from the R implementation of "Inductive Node-split Cross-Validation in Networks".
"""

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def edge_index_map(u):
    """
    Map a 1-d index of upper-triangle edge pairs to (row, col) indices.

    For an n×n symmetric matrix with zero diagonal, the upper triangle is
    stored column-major.  Given a *1-based* index vector ``u`` the function
    returns ``(x, y)`` such that ``x < y`` (both 1-based).

    Parameters
    ----------
    u : np.ndarray
        1-based integer indices into the upper-triangle vector.

    Returns
    -------
    x, y : np.ndarray
        Row and column indices (1-based).
    """
    u = np.asarray(u, dtype=np.float64)
    v = (np.sqrt(1 + 8 * u) - 1) / 2
    f = np.floor(v)
    y = np.ceil(v).astype(int) + 1
    x = np.where(u == f * (f + 1) / 2, y - 1, (u - f * (f + 1) / 2)).astype(int)
    return x, y


def neglog(n, p):
    """Compute ``-n * log(p)`` safely, returning 0 when ``p <= 0``."""
    if p <= 0:
        return 0.0
    return -n * np.log(p)


# ---------------------------------------------------------------------------
# Community simulation
# ---------------------------------------------------------------------------

def community_sim(k=2, n=1000, n1=100, p=0.3, q=0.1, rng=None):
    """
    Simulate communities under a planted-partition / SBM model.

    Parameters
    ----------
    k : int
        Number of communities.
    n : int
        Number of nodes.
    n1 : int
        Size of the smallest community.
    p : float
        Within-community edge probability.
    q : float
        Between-community edge probability.
    rng : numpy.random.Generator or None
        Random number generator (for reproducibility).

    Returns
    -------
    membership : np.ndarray of shape (n,)
        Community labels (1-based, integers in 1..k).
    adjacency : np.ndarray of shape (n, n)
        Symmetric binary adjacency matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    sizes = np.zeros(k, dtype=int)
    sizes[0] = n1
    for i in range(1, k - 1):
        sizes[i] = int(np.floor((n - n1) / (k - 1)))
    sizes[k - 1] = n - sizes.sum()

    mem = np.concatenate([np.full(s, i + 1) for i, s in enumerate(sizes)])
    rng.shuffle(mem)

    conn = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        for j in range(i + 1, n):
            prob = p if mem[i] == mem[j] else q
            conn[i, j] = rng.binomial(1, prob)
            conn[j, i] = conn[i, j]

    return mem, conn


def community_sim_sbm(n, n1, eta=0.3, rho=0.1, K=3, rng=None):
    """
    Simulate communities from a general SBM with structured B matrix.

    The connectivity matrix B is defined as ``B[k1,k2] = eta^min(|k1-k2|, 3) * rho``.

    Parameters
    ----------
    n, n1, eta, rho, K : see ``community_sim`` for analogous parameters.
    rng : numpy.random.Generator or None

    Returns
    -------
    adjacency, membership, conn : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    B0 = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            B0[k1, k2] = eta ** min(abs(k1 - k2), 3)
    B0 *= rho

    sizes = np.zeros(K, dtype=int)
    sizes[0] = n1
    for i in range(1, K - 1):
        sizes[i] = int(np.floor((n - n1) / (K - 1)))
    sizes[K - 1] = n - sizes.sum()

    c0 = np.concatenate([np.full(s, i + 1) for i, s in enumerate(sizes)])
    rng.shuffle(c0)

    Member_mat = np.zeros((n, K))
    for i in range(n):
        Member_mat[i, c0[i] - 1] = 1

    P0 = Member_mat @ B0 @ Member_mat.T

    A = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        for j in range(i + 1, n):
            A[i, j] = rng.binomial(1, P0[i, j])
            A[j, i] = A[i, j]

    return A, c0, B0


# ---------------------------------------------------------------------------
# Spectral clustering for SBM
# ---------------------------------------------------------------------------

def sbm_spectral_clustering(A, k=2):
    """
    Spectral clustering for a Stochastic Block Model.

    Computes the top-*k* left singular vectors of *A* and applies k-means.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Adjacency matrix.
    k : int
        Number of clusters.

    Returns
    -------
    cluster : np.ndarray of shape (n,)
        Cluster labels (1-based).
    """
    n = A.shape[0]
    k_svd = min(k, n - 1)
    U, _, _ = svds(A.astype(float), k=k_svd)
    # svds returns singular values in ascending order; flip to descending
    U = U[:, ::-1]
    km = KMeans(n_clusters=k, n_init=25, max_iter=100, random_state=None)
    labels = km.fit_predict(U)
    return labels + 1  # 1-based


# ---------------------------------------------------------------------------
# SBM probability estimation
# ---------------------------------------------------------------------------

def sbm_prob(cluster, k, A, restricted=True):
    """
    Estimate within- and between-cluster connection probabilities.

    Parameters
    ----------
    cluster : np.ndarray of shape (n,)
        1-based cluster assignments.
    k : int
        Number of clusters.
    A : np.ndarray
        Adjacency matrix (or sub-matrix).
    restricted : bool
        If True, use a single within-probability *p* and between-probability *q*.

    Returns
    -------
    p_matrix : np.ndarray of shape (k, k)
    negloglike : float
    """
    n = A.shape[0]
    p_matrix = np.zeros((k, k))
    nll = 0.0

    # Extract upper triangle (column-major, matching R's c(A)[upper.tri(A)])
    edge_vector = A.T[np.triu_indices(n, k=1)]
    ones_idx = np.where(edge_vector == 1)[0] + 1  # 1-based
    zeros_idx = np.where(edge_vector == 0)[0] + 1  # 1-based

    if len(ones_idx) > 0:
        one_x, one_y = edge_index_map(ones_idx)
    else:
        one_x, one_y = np.array([], dtype=int), np.array([], dtype=int)

    if len(zeros_idx) > 0:
        zero_x, zero_y = edge_index_map(zeros_idx)
    else:
        zero_x, zero_y = np.array([], dtype=int), np.array([], dtype=int)

    if restricted:
        if len(one_x) > 0:
            within_connect = int(np.sum(cluster[one_x - 1] == cluster[one_y - 1]))
        else:
            within_connect = 0
        if len(zero_x) > 0:
            within_disconnect = int(np.sum(cluster[zero_x - 1] == cluster[zero_y - 1]))
        else:
            within_disconnect = 0
        between_connect = len(one_x) - within_connect
        between_disconnect = len(zero_x) - within_disconnect

        within_total = within_connect + within_disconnect
        between_total = between_connect + between_disconnect

        p_val = within_connect / within_total if within_total > 0 else 0
        q_val = between_connect / between_total if between_total > 0 else 0

        np.fill_diagonal(p_matrix, p_val)
        off_diag = ~np.eye(k, dtype=bool)
        p_matrix[off_diag] = q_val

        nll = (neglog(within_connect, p_val) +
               neglog(within_disconnect, 1 - p_val) +
               neglog(between_connect, q_val) +
               neglog(between_disconnect, 1 - q_val))
    else:
        for i in range(1, k + 1):
            for j in range(i, k + 1):
                if i == j:
                    connect = int(np.sum(
                        (cluster[one_x - 1] == i) & (cluster[one_y - 1] == i)
                    )) if len(one_x) > 0 else 0
                    disconnect = int(np.sum(
                        (cluster[zero_x - 1] == i) & (cluster[zero_y - 1] == i)
                    )) if len(zero_x) > 0 else 0
                    total = connect + disconnect
                    p_val = connect / total if total > 0 else 0
                    p_matrix[i - 1, i - 1] = p_val
                    nll += neglog(connect, p_val) + neglog(disconnect, 1 - p_val)
                else:
                    connect = 0
                    disconnect = 0
                    if len(one_x) > 0:
                        connect = int(np.sum(
                            (cluster[one_x - 1] == i) & (cluster[one_y - 1] == j)
                        ) + np.sum(
                            (cluster[one_x - 1] == j) & (cluster[one_y - 1] == i)
                        ))
                    if len(zero_x) > 0:
                        disconnect = int(np.sum(
                            (cluster[zero_x - 1] == i) & (cluster[zero_y - 1] == j)
                        ) + np.sum(
                            (cluster[zero_x - 1] == j) & (cluster[zero_y - 1] == i)
                        ))
                    total = connect + disconnect
                    q_val = connect / total if total > 0 else 0
                    p_matrix[i - 1, j - 1] = q_val
                    p_matrix[j - 1, i - 1] = q_val
                    nll += neglog(connect, q_val) + neglog(disconnect, 1 - q_val)

    return p_matrix, nll


# ---------------------------------------------------------------------------
# INCV – f-fold cross-validation
# ---------------------------------------------------------------------------

def nscv_f_fold(A, k_vec=None, restricted=True, f=10,
                method="affinity", p_est_type=3, rng=None):
    """
    Inductive Node-split Cross-Validation with f-fold splitting.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Adjacency matrix.
    k_vec : list[int] or None
        Candidate numbers of communities (default ``[2,3,4,5,6]``).
    restricted : bool
        See ``sbm_prob``.
    f : int
        Number of folds.
    method : {"affinity", "loss"}
        How test nodes are assigned to communities.
    p_est_type : {1, 2, 3}
        Which data partition is used to re-estimate *B*.
    rng : numpy.random.Generator or None

    Returns
    -------
    dict with keys:
        - ``k_loss`` : int – K selected by negative log-likelihood.
        - ``k_mse`` : int – K selected by MSE.
        - ``cv_loss`` : np.ndarray – mean NLL per candidate.
        - ``cv_mse`` : np.ndarray – mean MSE per candidate.
    """
    if k_vec is None:
        k_vec = list(range(2, 7))
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    num_k = len(k_vec)
    loss_f = np.zeros((f, num_k))
    mse_f = np.zeros((f, num_k))
    node = np.arange(n)

    # Create fold assignments
    f_group = np.tile(np.arange(1, f + 1), int(np.floor(n / f)))
    remainder = n % f
    if remainder != 0:
        f_group = np.concatenate([f_group, np.arange(1, remainder + 1)])
    rng.shuffle(f_group)

    for i in range(1, f + 1):
        training = node[f_group != i]
        testing = node[f_group == i]

        A11 = A[np.ix_(training, training)]
        A12 = A[np.ix_(training, testing)]
        A22 = A[np.ix_(testing, testing)]

        for j_idx, k in enumerate(k_vec):
            # Spectral clustering on training subgraph
            tr_cluster = sbm_spectral_clustering(A11, k)
            p_matrix, _ = sbm_prob(tr_cluster, k, A11, restricted)

            # Assign test nodes via A12
            n_test = len(testing)
            te_cluster = np.zeros(n_test, dtype=int)
            ni = np.array([np.sum(tr_cluster == t) for t in range(1, k + 1)])

            for s_idx in range(n_test):
                mi = np.array([np.sum(A12[:, s_idx][tr_cluster == t])
                               for t in range(1, k + 1)])
                affinity = mi / np.maximum(ni, 1)

                loss_vec = np.zeros(k)
                for t in range(1, k + 1):
                    loss_u = 0.0
                    for u in range(1, k + 1):
                        loss_u += (neglog(mi[u - 1], p_matrix[t - 1, u - 1]) +
                                   neglog(ni[u - 1] - mi[u - 1],
                                          1 - p_matrix[t - 1, u - 1]))
                    loss_vec[t - 1] = loss_u

                if method == "affinity":
                    group = np.argmax(affinity) + 1
                else:
                    group = np.argmin(loss_vec) + 1
                te_cluster[s_idx] = group

            # Re-estimate B based on p_est_type
            if p_est_type == 1:
                cluster = tr_cluster
                AA = A11
            elif p_est_type == 2:
                cluster = np.concatenate([tr_cluster, te_cluster])
                AA = np.hstack([A11, A12])
            else:  # p_est_type == 3
                cluster = te_cluster
                AA = A22

            p_matrix, _ = sbm_prob(cluster, k, AA, restricted)

            # Compute test loss on A22
            n_te = A22.shape[0]
            te_edge_vector = A22.T[np.triu_indices(n_te, k=1)]
            te_ones_idx = np.where(te_edge_vector == 1)[0] + 1
            te_zeros_idx = np.where(te_edge_vector == 0)[0] + 1

            if len(te_ones_idx) > 0:
                te_one_x, te_one_y = edge_index_map(te_ones_idx)
            else:
                te_one_x, te_one_y = np.array([], dtype=int), np.array([], dtype=int)
            if len(te_zeros_idx) > 0:
                te_zero_x, te_zero_y = edge_index_map(te_zeros_idx)
            else:
                te_zero_x, te_zero_y = np.array([], dtype=int), np.array([], dtype=int)

            te_nll = 0.0
            te_mse = 0.0
            for s in range(1, k + 1):
                for t in range(s, k + 1):
                    if s == t:
                        connect = int(np.sum(
                            (te_cluster[te_one_x - 1] == s) &
                            (te_cluster[te_one_y - 1] == s)
                        )) if len(te_one_x) > 0 else 0
                        disconnect = int(np.sum(
                            (te_cluster[te_zero_x - 1] == s) &
                            (te_cluster[te_zero_y - 1] == s)
                        )) if len(te_zero_x) > 0 else 0
                        prob = p_matrix[s - 1, s - 1]
                        te_nll += neglog(connect, prob) + neglog(disconnect, 1 - prob)
                        te_mse += connect * (1 - prob) ** 2 + disconnect * prob ** 2
                    else:
                        connect = 0
                        disconnect = 0
                        if len(te_one_x) > 0:
                            connect = int(np.sum(
                                (te_cluster[te_one_x - 1] == s) &
                                (te_cluster[te_one_y - 1] == t)
                            ) + np.sum(
                                (te_cluster[te_one_x - 1] == t) &
                                (te_cluster[te_one_y - 1] == s)
                            ))
                        if len(te_zero_x) > 0:
                            disconnect = int(np.sum(
                                (te_cluster[te_zero_x - 1] == s) &
                                (te_cluster[te_zero_y - 1] == t)
                            ) + np.sum(
                                (te_cluster[te_zero_x - 1] == t) &
                                (te_cluster[te_zero_y - 1] == s)
                            ))
                        prob = p_matrix[s - 1, t - 1]
                        te_nll += neglog(connect, prob) + neglog(disconnect, 1 - prob)
                        te_mse += connect * (1 - prob) ** 2 + disconnect * prob ** 2

            loss_f[i - 1, j_idx] = te_nll
            mse_f[i - 1, j_idx] = te_mse

    cv_loss = loss_f.mean(axis=0)
    cv_mse = mse_f.mean(axis=0)
    k_loss = k_vec[np.argmin(cv_loss)]
    k_mse = k_vec[np.argmin(cv_mse)]

    return {
        "k_loss": k_loss,
        "k_mse": k_mse,
        "cv_loss": cv_loss,
        "cv_mse": cv_mse,
    }


# ---------------------------------------------------------------------------
# INCV – random split
# ---------------------------------------------------------------------------

def nscv_random_split(A, k_vec=None, restricted=True, split=0.66,
                      ite=100, method="affinity", p_est_type=3, rng=None):
    """
    Inductive Node-split Cross-Validation with random train/test splits.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Adjacency matrix.
    k_vec : list[int] or None
        Candidate numbers of communities (default ``[2,3,4,5,6]``).
    split : float
        Fraction of nodes used for training.
    ite : int
        Number of random iterations.
    method, restricted, p_est_type : see ``nscv_f_fold``.

    Returns
    -------
    dict with keys ``k_chosen``, ``k_choice``, ``cv_loss``.
    """
    if k_vec is None:
        k_vec = list(range(2, 7))
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    num_k = len(k_vec)
    loss_f = np.zeros((ite, num_k))
    node = np.arange(n)

    for i in range(ite):
        n_train = int(np.ceil(n * split))
        training = np.sort(rng.choice(n, n_train, replace=False))
        testing = np.sort(np.setdiff1d(node, training))

        A11 = A[np.ix_(training, training)]
        A12 = A[np.ix_(training, testing)]
        A22 = A[np.ix_(testing, testing)]

        for j_idx, k in enumerate(k_vec):
            tr_cluster = sbm_spectral_clustering(A11, k)
            p_matrix, _ = sbm_prob(tr_cluster, k, A11, restricted)

            n_test = len(testing)
            te_cluster = np.zeros(n_test, dtype=int)
            ni = np.array([np.sum(tr_cluster == t) for t in range(1, k + 1)])

            for s_idx in range(n_test):
                mi = np.array([np.sum(A12[:, s_idx][tr_cluster == t])
                               for t in range(1, k + 1)])
                affinity = mi / np.maximum(ni, 1)

                loss_vec = np.zeros(k)
                for t in range(1, k + 1):
                    loss_u = 0.0
                    for u in range(1, k + 1):
                        loss_u += (neglog(mi[u - 1], p_matrix[t - 1, u - 1]) +
                                   neglog(ni[u - 1] - mi[u - 1],
                                          1 - p_matrix[t - 1, u - 1]))
                    loss_vec[t - 1] = loss_u

                if method == "affinity":
                    group = np.argmax(affinity) + 1
                else:
                    group = np.argmin(loss_vec) + 1
                te_cluster[s_idx] = group

            if p_est_type == 1:
                cluster = tr_cluster
                AA = A11
            elif p_est_type == 2:
                cluster = np.concatenate([tr_cluster, te_cluster])
                AA = np.hstack([A11, A12])
            else:
                cluster = te_cluster
                AA = A22

            p_matrix, _ = sbm_prob(cluster, k, AA, restricted)

            n_te = A22.shape[0]
            te_edge_vector = A22.T[np.triu_indices(n_te, k=1)]
            te_ones_idx = np.where(te_edge_vector == 1)[0] + 1
            te_zeros_idx = np.where(te_edge_vector == 0)[0] + 1

            if len(te_ones_idx) > 0:
                te_one_x, te_one_y = edge_index_map(te_ones_idx)
            else:
                te_one_x, te_one_y = np.array([], dtype=int), np.array([], dtype=int)
            if len(te_zeros_idx) > 0:
                te_zero_x, te_zero_y = edge_index_map(te_zeros_idx)
            else:
                te_zero_x, te_zero_y = np.array([], dtype=int), np.array([], dtype=int)

            te_nll = 0.0
            for s in range(1, k + 1):
                for t in range(s, k + 1):
                    if s == t:
                        connect = int(np.sum(
                            (te_cluster[te_one_x - 1] == s) &
                            (te_cluster[te_one_y - 1] == s)
                        )) if len(te_one_x) > 0 else 0
                        disconnect = int(np.sum(
                            (te_cluster[te_zero_x - 1] == s) &
                            (te_cluster[te_zero_y - 1] == s)
                        )) if len(te_zero_x) > 0 else 0
                        prob = p_matrix[s - 1, s - 1]
                        te_nll += neglog(connect, prob) + neglog(disconnect, 1 - prob)
                    else:
                        connect = 0
                        disconnect = 0
                        if len(te_one_x) > 0:
                            connect = int(np.sum(
                                (te_cluster[te_one_x - 1] == s) &
                                (te_cluster[te_one_y - 1] == t)
                            ) + np.sum(
                                (te_cluster[te_one_x - 1] == t) &
                                (te_cluster[te_one_y - 1] == s)
                            ))
                        if len(te_zero_x) > 0:
                            disconnect = int(np.sum(
                                (te_cluster[te_zero_x - 1] == s) &
                                (te_cluster[te_zero_y - 1] == t)
                            ) + np.sum(
                                (te_cluster[te_zero_x - 1] == t) &
                                (te_cluster[te_zero_y - 1] == s)
                            ))
                        prob = p_matrix[s - 1, t - 1]
                        te_nll += neglog(connect, prob) + neglog(disconnect, 1 - prob)

            loss_f[i, j_idx] = te_nll

    cv_loss = loss_f.mean(axis=0)
    k_chosen = k_vec[np.argmin(cv_loss)]

    return {
        "k_chosen": k_chosen,
        "k_choice": k_vec,
        "cv_loss": cv_loss,
    }
