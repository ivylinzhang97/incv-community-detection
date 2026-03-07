#' Find the best permutation label matching
#'
#' Matches community labels in \code{lab} to reference labels in \code{fixed}
#' via a greedy maximum-overlap permutation.
#'
#' @param lab Integer vector of labels to permute.
#' @param fixed Integer vector of reference labels.
#' @param n Number of nodes (default \code{length(lab)}).
#' @param K Number of communities (default \code{max(lab, fixed)}).
#' @return A K x K permutation matrix.
#' @export
best.perm.label.match <- function(lab, fixed,
                                  n = length(lab), K = max(lab, fixed)) {
  if (identical(lab, fixed)) return(diag(1, K))
  if (K == 1) return(matrix(1, 1, 1))
  if (K == 2) {
    if (sum(lab != fixed) <= n / 2) return(diag(1, 2))
    else return(matrix(c(0, 1, 1, 0), 2, 2, TRUE))
  }

  E <- matrix(0, K, K)
  C.lab <- as(Matrix::sparseMatrix(i = 1:n, j = lab, dims = c(n, K)), "dMatrix")
  C.fixed <- as(Matrix::sparseMatrix(i = 1:n, j = fixed, dims = c(n, K)), "dMatrix")
  M <- Matrix::crossprod(C.lab, C.fixed)

  while (max(M) != -1) {
    ind <- which(M == max(M), TRUE)[1, ]
    E[ind[2], ind[1]] <- 1
    M[ind[1], ] <- rep(-1, K)
    M[, ind[2]] <- rep(-1, K)
  }
  E
}

#' Apply label permutation to match reference
#'
#' @param lab Integer vector of labels to relabel.
#' @param fixed Integer vector of reference labels.
#' @param n Number of nodes.
#' @param K Number of communities.
#' @return Relabelled integer vector aligned to \code{fixed}.
#' @export
matched.lab <- function(lab, fixed,
                        n = length(lab), K = max(lab, fixed)) {
  E <- best.perm.label.match(lab, fixed, n = n, K = K)
  lmat <- Matrix::sparseMatrix(i = 1:n, j = lab, dims = c(n, K))
  as.vector(Matrix::tcrossprod(Matrix::tcrossprod(lmat, E), rbind(1:K)))
}

#' Fast SBM block probability estimation
#'
#' Estimates the K x K block probability matrix from an adjacency matrix
#' and community labels.
#'
#' @param A Adjacency matrix (n x n).
#' @param g Integer community label vector (length n).
#' @param n Number of nodes.
#' @param K Number of communities.
#' @return K x K estimated block probability matrix.
#' @export
fast.SBM.est <- function(A, g, n = nrow(A), K = max(g)) {
  B <- matrix(0, K, K)
  if (K == 1) {
    B[K, K] <- sumFast(A) / (n^2 - n)
    return(B)
  }
  G <- lapply(1:K, function(k) which(g == k))
  nk <- sapply(G, length)
  for (k in 1:K) {
    for (l in k:K) {
      B[k, l] <- B[l, k] <- sumFast(A[G[[k]], G[[l]]]) / (nk[k] * nk[l])
    }
  }
  diag(B) <- diag(B) * nk / (nk - 1)
  B[!is.finite(B)] <- 1e-6
  B
}

#' Fast DCBM parameter estimation
#'
#' Estimates the block sum matrix and degree heterogeneity parameters
#' for a Degree-Corrected Block Model.
#'
#' @param A Adjacency matrix.
#' @param g Community label vector.
#' @param n Number of nodes.
#' @param K Number of communities.
#' @param psi.omit Number of leading nodes to exclude from psi estimation
#'   (used in CROISSANT overlapping designs).
#' @param p.sample Sampling proportion for correction (default 1).
#' @return A list with:
#'   \item{Bsum}{K x K block sum matrix (divided by \code{p.sample}).}
#'   \item{psi}{Degree heterogeneity vector.}
#' @export
fast.DCBM.est <- function(A, g, n = nrow(A), K = max(g),
                          psi.omit = 0, p.sample = 1) {
  B.sum <- matrix(0, K, K)
  if (K == 1) {
    B.sum[K, K] <- sumFast(A) + 0.001
    if (psi.omit > 0) {
      psi <- as.numeric(rowSums(A[-(1:psi.omit), ]) / B.sum[K, K])
      return(list(Bsum = B.sum / p.sample, psi = psi))
    }
    psi <- as.numeric(rowSums(A) / B.sum[K, K]) + 0.0001
    return(list(Bsum = B.sum / p.sample, psi = psi))
  }
  G <- lapply(1:K, function(k) which(g == k))
  for (k in 1:K) {
    for (l in k:K) {
      B.sum[k, l] <- B.sum[l, k] <- sumFast(A[G[[k]], G[[l]]]) + 0.0001
    }
  }
  if (psi.omit > 0) {
    psi <- as.numeric(rowSums(A[-(1:psi.omit), ]) /
                        rowSums(B.sum)[g[-(1:psi.omit)]])
    return(list(Bsum = B.sum / p.sample, psi = psi))
  }
  psi <- as.numeric(rowSums(A) / rowSums(B.sum)[g])
  list(Bsum = B.sum / p.sample, psi = psi)
}

#' Eigenvector-based DCBM estimation
#'
#' Estimates DCBM parameters using the row-norms of the leading eigenvectors
#' for degree heterogeneity.
#'
#' @inheritParams fast.DCBM.est
#' @return A list with \code{Bsum} and \code{psi}.
#' @export
eigen.DCBM.est <- function(A, g, n = nrow(A), K = max(g),
                           psi.omit = 0, p.sample = 1) {
  U.hat <- irlba::irlba(A, nu = K, nv = K)$v
  psi.hat <- rowSums(U.hat^2)^0.5
  psi.outer <- tcrossprod(psi.hat)

  B.sum <- matrix(0, K, K)
  if (K == 1) {
    B.sum[K, K] <- sumFast(A) / sumFast(psi.outer)
    if (psi.omit > 0)
      return(list(Bsum = B.sum / p.sample, psi = psi.hat[-(1:psi.omit)]))
    return(list(Bsum = B.sum / p.sample, psi = psi.hat))
  }
  G <- lapply(1:K, function(k) which(g == k))
  for (k in 1:K) {
    for (l in k:K) {
      B.sum[k, l] <- B.sum[l, k] <-
        sumFast(A[G[[k]], G[[l]]]) / sumFast(psi.outer[G[[k]], G[[l]]])
    }
  }
  if (psi.omit > 0)
    return(list(Bsum = B.sum / p.sample, psi = psi.hat[-(1:psi.omit)]))
  list(Bsum = B.sum / p.sample, psi = psi.hat)
}
