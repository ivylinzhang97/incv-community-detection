#' Edge Cross-Validation for blockmodel selection
#'
#' Selects both the model type (SBM vs DCBM) and number of communities
#' by holding out random edges and evaluating predictive performance.
#'
#' @param A Adjacency matrix (n x n).
#' @param max.K Maximum number of communities to consider.
#' @param cv Number of cross-validation folds (NULL for holdout; default NULL).
#' @param B Number of holdout repetitions (default 3).
#' @param holdout.p Fraction of edges held out (default 0.1).
#' @param tau Regularisation parameter (default 0).
#' @param dc.est DCBM estimation type (1 or 2; default 2).
#' @param kappa Truncation parameter (default NULL).
#' @return A list with loss vectors and selected model strings:
#'   \item{l2}{SBM L2 losses by K.}
#'   \item{dev}{SBM deviance losses by K.}
#'   \item{auc}{SBM AUC losses by K.}
#'   \item{dc.l2, dc.dev, dc.auc}{DCBM losses.}
#'   \item{l2.model, dev.model, auc.model}{Selected model strings.}
#' @export
ECV.for.blockmodel <- function(A, max.K, cv = NULL, B = 3, holdout.p = 0.1,
                               tau = 0, dc.est = 2, kappa = NULL) {
  n <- nrow(A)
  edge.index <- which(upper.tri(A))
  edge.n <- length(edge.index)
  holdout.index.list <- list()

  if (is.null(cv)) {
    holdout.n <- floor(holdout.p * edge.n)
    for (j in 1:B)
      holdout.index.list[[j]] <- sample(x = edge.n, size = holdout.n)
  } else {
    sample.index <- sample.int(edge.n)
    max.fold.num <- ceiling(edge.n / cv)
    fold.index <- rep(1:cv, each = max.fold.num)[edge.n]
    cv.index <- fold.index[sample.index]
    B <- cv
    for (j in 1:B)
      holdout.index.list[[j]] <- which(cv.index == j)
  }

  result <- lapply(holdout.index.list, .holdout.evaluation.fast.all,
                   A = A, max.K = max.K, tau = tau, dc.est = dc.est,
                   p.sample = 1 - holdout.p, kappa = kappa)

  dc.block.err.mat <- dc.loglike.mat <- bin.dev.mat <- roc.auc.mat <-
    impute.err.mat <- block.err.mat <- loglike.mat <-
    matrix(0, nrow = B, ncol = max.K)
  sbm.auc.mat <- dc.auc.mat <- matrix(0, nrow = B, ncol = max.K)

  for (b in 1:B) {
    impute.err.mat[b, ] <- result[[b]]$impute.sq.err
    block.err.mat[b, ] <- result[[b]]$block.sq.err
    loglike.mat[b, ] <- result[[b]]$loglike
    roc.auc.mat[b, ] <- result[[b]]$roc.auc
    bin.dev.mat[b, ] <- result[[b]]$bin.dev
    dc.block.err.mat[b, ] <- result[[b]]$dc.block.sq.err
    dc.loglike.mat[b, ] <- result[[b]]$dc.loglike
    sbm.auc.mat[b, ] <- result[[b]]$sbm.auc
    dc.auc.mat[b, ] <- result[[b]]$dc.auc
  }

  output <- list(
    impute.err = colMeans(impute.err.mat),
    l2 = colMeans(block.err.mat),
    dev = colSums(loglike.mat),
    auc = colMeans(sbm.auc.mat),
    dc.l2 = colMeans(dc.block.err.mat),
    dc.dev = colSums(dc.loglike.mat),
    dc.auc = colMeans(dc.auc.mat),
    sse = colMeans(impute.err.mat),
    l2.mat = block.err.mat,
    dc.l2.mat = dc.block.err.mat,
    auc.mat = sbm.auc.mat,
    dc.auc.mat = dc.auc.mat,
    dev.mat = loglike.mat,
    dc.dev.mat = dc.loglike.mat)

  if (min(output$dev) > min(output$dc.dev))
    dev.model <- paste("DCSBM", which.min(output$dc.dev), sep = "-")
  else
    dev.model <- paste("SBM", which.min(output$dev), sep = "-")

  if (min(output$l2) > min(output$dc.l2))
    l2.model <- paste("DCSBM", which.min(output$dc.l2), sep = "-")
  else
    l2.model <- paste("SBM", which.min(output$l2), sep = "-")

  if (min(output$auc) > min(output$dc.auc))
    auc.model <- paste("DCSBM", which.min(output$dc.auc), sep = "-")
  else
    auc.model <- paste("SBM", which.min(output$auc), sep = "-")

  output$l2.model <- l2.model
  output$dev.model <- dev.model
  output$auc.model <- auc.model
  output
}

#' @keywords internal
.holdout.evaluation.fast.all <- function(holdout.index, A, max.K,
                                         soft = TRUE, tau = 0,
                                         dc.est = 1, fast = FALSE,
                                         p.sample = 1, kappa = NULL) {
  n <- nrow(A)
  edge.index <- which(upper.tri(A))
  edge.n <- length(edge.index)
  A.new <- matrix(0, n, n)
  A.new[upper.tri(A.new)] <- A[edge.index]
  A.new[edge.index[holdout.index]] <- NA
  A.new <- A.new + t(A.new)
  degrees <- colSums(A.new, na.rm = TRUE)

  Omega <- which(is.na(A.new))
  non.miss <- which(!is.na(A.new))
  SVD.result <- .iter.SVD.core.fast.all(A.new, max.K, fast = TRUE, p.sample = p.sample)

  dc.block.sq.err <- dc.loglike <- roc.auc <- bin.dev <-
    block.sq.err <- impute.sq.err <- loglike <- rep(0, max.K)
  sbm.auc <- dc.auc <- rep(0, max.K)

  for (k in 1:max.K) {
    tmp.est <- SVD.result[[k]]
    A.approx <- tmp.est$A.thr
    impute.sq.err[k] <- sum((A.approx[Omega] - A[Omega])^2)
    response <- A[edge.index[holdout.index]]
    predictors <- A.approx[edge.index[holdout.index]]
    roc.auc[k] <- 0
    trunc.predictors <- predictors
    trunc.predictors[predictors > (1 - 1e-6)] <- 1 - 1e-6
    trunc.predictors[predictors < 1e-6] <- 1e-6
    bin.dev[k] <- sum((response - trunc.predictors)^2)

    if (k == 1) {
      pb <- (sum(A.new, na.rm = TRUE) + 1) /
        (sum(!is.na(A.new)) - sum(!is.na(diag(A.new))) + 1)
      if (pb < 1e-6) pb <- 1e-6
      if (pb > 1 - 1e-6) pb <- 1 - 1e-6
      A.Omega <- A[Omega]
      block.sq.err[k] <- sum((pb - A[Omega])^2)
      loglike[k] <- -sum(A.Omega * log(pb)) - sum((1 - A.Omega) * log(1 - pb))
    }

    if (k == 1) U.approx <- matrix(tmp.est$SVD$v, ncol = k)
    else {
      U.approx <- tmp.est$SVD$v[, 1:k]
      if (tau > 0) {
        A.approx <- A.approx + tau * mean(colSums(A.approx)) / n
        d.approx <- colSums(A.approx)
        L.approx <- diag(1 / sqrt(d.approx)) %*% A.approx %*% diag(1 / sqrt(d.approx))
        A.approx.svd <- irlba::irlba(L.approx, nu = k, nv = k)
        U.approx <- A.approx.svd$v[, 1:k]
      }
    }

    km <- kmeans(U.approx, centers = k, nstart = 30, iter.max = 30)
    B <- matrix(0, k, k)
    Theta <- matrix(0, n, k)
    for (i in 1:k) {
      for (j in i:k) {
        N.i <- which(km$cluster == i)
        N.j <- which(km$cluster == j)
        if (i != j)
          B[i, j] <- B[j, i] <- (sum(A.new[N.i, N.j], na.rm = TRUE) + 1) /
            (sum(!is.na(A.new[N.i, N.j])) + 1)
        else
          B[i, j] <- B[j, i] <- (sum(A.new[N.i, N.j], na.rm = TRUE) + 1) /
            (sum(!is.na(A.new[N.i, N.j])) - sum(!is.na(diag(A.new[N.i, N.j]))) + 1)
      }
      Theta[N.i, i] <- 1
    }
    P.hat <- Theta %*% B %*% t(Theta)
    diag(P.hat) <- 0
    block.sq.err[k] <- sum((P.hat[Omega] - A[Omega])^2)
    P.hat.Omega <- P.hat[Omega]
    A.Omega <- A[Omega]
    P.hat.Omega[P.hat.Omega < 1e-6] <- 1e-6
    P.hat.Omega[P.hat.Omega > (1 - 1e-6)] <- 1 - 1e-6
    loglike[k] <- -sum(A.Omega * log(P.hat.Omega)) -
      sum((1 - A.Omega) * log(1 - P.hat.Omega))
    sbm.auc[k] <- AUC(A.Omega, P.hat.Omega)

    V <- U.approx
    if (k == 1) V.norms <- as.numeric(abs(V))
    else V.norms <- apply(V, 1, function(x) sqrt(sum(x^2)))

    iso.index <- which(V.norms == 0)
    Psi <- V.norms
    Psi <- Psi / max(V.norms)
    inv.V.norms <- 1 / V.norms
    inv.V.norms[iso.index] <- 1
    V.normalized <- diag(as.numeric(inv.V.norms)) %*% V

    if (k == 1) {
      if (dc.est > 1) {
        B <- sum(A.new, na.rm = TRUE) + 0.01
        partial.d <- colSums(A.new, na.rm = TRUE)
        partial.gd <- B
        phi <- as.numeric(partial.d / partial.gd)
        B <- B / p.sample
        P.hat <- t(t(matrix(B, n, n) * phi) * phi)
        diag(P.hat) <- 0
      }
      dc.block.sq.err[k] <- sum((pb - A[Omega])^2)
      P.hat.Omega <- P.hat[Omega]
      A.Omega <- A[Omega]
      P.hat.Omega[P.hat.Omega < 1e-6] <- 1e-6
      P.hat.Omega[P.hat.Omega > (1 - 1e-6)] <- 1 - 1e-6
      dc.loglike[k] <- -sum(A.Omega * log(P.hat.Omega)) -
        sum((1 - A.Omega) * log(1 - P.hat.Omega))
      dc.auc[k] <- AUC(A.Omega, P.hat.Omega)
    } else {
      km <- kmeans(V.normalized, centers = k, nstart = 30, iter.max = 30)
      if (dc.est > 1) {
        B <- matrix(0, k, k)
        Theta <- matrix(0, n, k)
        for (i in 1:k) {
          for (j in 1:k) {
            N.i <- which(km$cluster == i)
            N.j <- which(km$cluster == j)
            B[i, j] <- sum(A.new[N.i, N.j], na.rm = TRUE) + 0.01
          }
          Theta[N.i, i] <- 1
        }
        Theta <- Matrix::Matrix(Theta, sparse = TRUE)
        partial.d <- colSums(A.new, na.rm = TRUE)
        partial.gd <- colSums(B)
        B.g <- Theta %*% partial.gd
        phi <- as.numeric(partial.d / B.g)
        B <- B / p.sample
        tmp.int.mat <- Theta * phi
        P.hat <- as.matrix(tmp.int.mat %*% B %*% t(tmp.int.mat))
        diag(P.hat) <- 0
      }
      dc.block.sq.err[k] <- sum((P.hat[Omega] - A[Omega])^2)
      P.hat.Omega <- P.hat[Omega]
      A.Omega <- A[Omega]
      P.hat.Omega[P.hat.Omega < 1e-6] <- 1e-6
      P.hat.Omega[P.hat.Omega > (1 - 1e-6)] <- 1 - 1e-6
      dc.loglike[k] <- -sum(A.Omega * log(P.hat.Omega)) -
        sum((1 - A.Omega) * log(1 - P.hat.Omega))
      dc.auc[k] <- AUC(A.Omega, P.hat.Omega)
    }
  }

  list(impute.sq.err = impute.sq.err, block.sq.err = block.sq.err,
       loglike = loglike, roc.auc = roc.auc, no.edge = sum(degrees == 0),
       dc.block.sq.err = dc.block.sq.err, dc.loglike = dc.loglike,
       bin.dev = bin.dev, sbm.auc = sbm.auc, dc.auc = dc.auc)
}

#' @keywords internal
.iter.SVD.core.fast.all <- function(A, Kmax, tol = 1e-5, max.iter = 100,
                                    sparse = TRUE, init = NULL,
                                    verbose = FALSE, tau = 0, fast = FALSE,
                                    p.sample = 1) {
  if (sparse) A <- Matrix::Matrix(A, sparse = TRUE)
  A[which(is.na(A))] <- 0
  A <- A / p.sample
  svd.new <- irlba::irlba(A, nu = Kmax, nv = Kmax)
  result <- list()
  for (K in 1:Kmax) {
    if (K == 1) {
      A.new <- svd.new$d[1] *
        matrix(svd.new$u[, 1], ncol = 1) %*% t(matrix(svd.new$v[, 1], ncol = 1))
    } else {
      A.new <- A.new + svd.new$d[K] *
        matrix(svd.new$u[, K], ncol = 1) %*% t(matrix(svd.new$v[, K], ncol = 1))
    }
    A.new.thr <- A.new
    A.new.thr[A.new < 0 + tau] <- 0 + tau
    cap <- 1
    A.new.thr[A.new > cap] <- cap
    tmp.SVD <- list(u = svd.new$u[, 1:K], v = svd.new$v[, 1:K], d = svd.new$d[1:K])
    result[[K]] <- list(iter = NA, SVD = tmp.SVD, A = A.new,
                        err.seq = NA, A.thr = A.new.thr)
  }
  result
}

#' Edge Cross-Validation for RDPG rank selection
#'
#' Selects the embedding rank for an RDPG by holding out random edges.
#'
#' @param A Adjacency matrix.
#' @param max.K Maximum rank to consider.
#' @param B Number of holdout repetitions (default 3).
#' @param holdout.p Fraction of edges held out (default 0.1).
#' @param soft Logical (default FALSE).
#' @param fast Logical (default FALSE).
#' @return A list with:
#'   \item{rank.sse, rank.dev, rank.auc}{Selected ranks by each loss.}
#'   \item{sse, dev, auc}{Average loss vectors.}
#' @export
ECV.undirected.Rank <- function(A, max.K, B = 3, holdout.p = 0.1,
                                soft = FALSE, fast = FALSE) {
  n <- nrow(A)
  edge.index <- which(upper.tri(A))
  edge.n <- length(edge.index)
  holdout.index.list <- list()
  holdout.n <- floor(holdout.p * edge.n)

  for (j in 1:B)
    holdout.index.list[[j]] <- sample(x = edge.n, size = holdout.n)

  result <- lapply(holdout.index.list,
                   .missing.undirected.Rank.fast.all,
                   A = A, max.K = max.K, soft = soft, fast = fast,
                   p.sample = 1 - holdout.p)

  sse.mat <- roc.auc.mat <- dev.mat <- matrix(0, nrow = B, ncol = max.K)
  for (b in 1:B) {
    roc.auc.mat[b, ] <- result[[b]]$roc.auc
    sse.mat[b, ] <- result[[b]]$sse
    dev.mat[b, ] <- result[[b]]$dev
  }

  auc.seq <- colMeans(roc.auc.mat)
  sse.seq <- colMeans(sse.mat)
  dev.seq <- colMeans(dev.mat)

  list(rank.sse = which.min(sse.seq), sse = sse.seq,
       rank.dev = which.min(dev.seq), dev = dev.seq,
       rank.auc = which.min(auc.seq), auc = auc.seq)
}

#' @keywords internal
.missing.undirected.Rank.fast.all <- function(holdout.index, A, max.K,
                                              soft = FALSE, fast = FALSE,
                                              p.sample = 1) {
  n <- nrow(A)
  edge.index <- which(upper.tri(A))
  edge.n <- length(edge.index)
  A.new <- matrix(0, n, n)
  A.new[upper.tri(A.new)] <- A[edge.index]
  A.new[edge.index[holdout.index]] <- NA
  A.new <- A.new + t(A.new)
  diag(A.new) <- diag(A)

  Omega <- which(is.na(A.new))
  sse <- roc.auc <- dev <- rep(0, max.K)
  SVD.result <- .iter.SVD.core.fast.all(A.new, max.K, fast = TRUE,
                                        p.sample = p.sample)
  for (k in 1:max.K) {
    tmp.est <- SVD.result[[k]]
    A.approx <- tmp.est$A
    response <- A[Omega]
    predictors <- A.approx[Omega]
    roc.auc[k] <- AUC(response, predictors)
    sse[k] <- mean((response - predictors)^2)
    predictors[predictors < 1e-6] <- 1e-6
    predictors[predictors > 1 - 1e-6] <- 1 - 1e-6
    dev[k] <- bin.dev(matrix(response, ncol = k), matrix(predictors, ncol = k))
  }
  list(Omega = Omega, roc.auc = roc.auc, sse = sse, dev = dev)
}
