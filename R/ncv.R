#' Node Cross-Validation for blockmodel selection
#'
#' Selects both the model type (SBM vs DCBM) and number of communities
#' by holding out random nodes and evaluating predictive performance on the
#' held-out subgraph.
#'
#' @param A Adjacency matrix (n x n).
#' @param max.K Maximum number of communities to consider.
#' @param cv Number of node folds (default 3).
#' @param dc.est DCBM estimation type (1 or 2; default 1).
#' @return A list with:
#'   \item{dev, l2, auc}{SBM loss vectors by K.}
#'   \item{dc.dev, dc.l2, dc.auc}{DCBM loss vectors by K.}
#'   \item{l2.model, dev.model, auc.model}{Selected model strings.}
#' @export
NCV.for.blockmodel <- function(A, max.K, cv = 3, dc.est = 1) {
  dc.avg.se <- dc.avg.log <- dc.avg.auc <-
    avg.se <- avg.log <- avg.auc <- rep(0, max.K)
  dc.avg.se[1] <- dc.avg.log[1] <- dc.avg.auc[1] <-
    avg.se[1] <- avg.log[1] <- avg.auc[1] <- Inf
  dc.dev.mat <- dc.l2.mat <- dc.auc.mat <-
    sbm.dev.mat <- sbm.l2.mat <- sbm.auc.mat <- matrix(0, cv, max.K)

  n <- nrow(A)
  sample.index <- sample.int(n)
  max.fold.num <- ceiling(n / cv)
  fold.index <- rep(1:cv, each = max.fold.num)[1:n]
  cv.index <- fold.index[sample.index]

  for (KK in 1:max.K) {
    dc.l2 <- l2.cv <- dc.log.like <- log.like <- sbm.auc <- dc.auc <- rep(0, cv)
    for (k in 1:cv) {
      holdout.index <- which(cv.index == k)
      train.index <- which(cv.index != k)
      tmp.all <- .cv.evaluate.all(A, train.index, holdout.index, KK, dc.est)

      sbm.l2.mat[k, KK] <- l2.cv[k] <- tmp.all$l2
      sbm.dev.mat[k, KK] <- log.like[k] <- tmp.all$loglike
      sbm.auc.mat[k, KK] <- sbm.auc[k] <- tmp.all$auc
      dc.l2.mat[k, KK] <- dc.l2[k] <- tmp.all$dc.l2
      dc.dev.mat[k, KK] <- dc.log.like[k] <- tmp.all$dc.loglike
      dc.auc.mat[k, KK] <- dc.auc[k] <- tmp.all$dc.auc
    }
    avg.se[KK] <- mean(l2.cv)
    avg.log[KK] <- mean(log.like)
    avg.auc[KK] <- mean(sbm.auc)
    dc.avg.se[KK] <- mean(dc.l2)
    dc.avg.log[KK] <- mean(dc.log.like)
    dc.avg.auc[KK] <- mean(dc.auc)
  }

  if (min(avg.log) > min(dc.avg.log))
    dev.model <- paste("DCSBM", which.min(dc.avg.log), sep = "-")
  else
    dev.model <- paste("SBM", which.min(avg.log), sep = "-")

  if (min(avg.se) > min(dc.avg.se))
    l2.model <- paste("DCSBM", which.min(dc.avg.se), sep = "-")
  else
    l2.model <- paste("SBM", which.min(avg.se), sep = "-")

  if (min(avg.auc) > min(dc.avg.auc))
    auc.model <- paste("DCSBM", which.min(dc.avg.auc), sep = "-")
  else
    auc.model <- paste("SBM", which.min(avg.auc), sep = "-")

  list(dev = avg.log, l2 = avg.se, auc = avg.auc,
       dc.dev = dc.avg.log, dc.l2 = dc.avg.se, dc.auc = dc.avg.auc,
       sbm.l2.mat = sbm.l2.mat, sbm.dev.mat = sbm.dev.mat,
       sbm.auc.mat = sbm.auc.mat,
       dc.l2.mat = dc.l2.mat, dc.dev.mat = dc.dev.mat,
       dc.auc.mat = dc.auc.mat,
       l2.model = l2.model, dev.model = dev.model, auc.model = auc.model)
}

#' @keywords internal
.cv.evaluate.all <- function(A, train.index, holdout.index, K, dc.est = 1) {
  n <- nrow(A)
  A.new <- A[c(train.index, holdout.index), c(train.index, holdout.index)]
  n.holdout <- length(holdout.index)
  n.train <- n - n.holdout
  A1 <- A.new[1:n.train, ]
  A1.svd <- irlba::irlba(A1 + 0.001, nu = K, nv = K)
  dc.A1.svd <- irlba::irlba(A1, nu = K, nv = K)

  V <- dc.A1.svd$v[, 1:K]
  if (K == 1) V.norms <- abs(V)
  else V.norms <- sqrt(rowSums(V^2))

  iso.index <- which(V.norms == 0)
  Psi <- V.norms / max(V.norms)
  Psi.outer <- tcrossprod(Psi)
  inv.V.norms <- 1 / V.norms
  inv.V.norms[iso.index] <- 1
  V.normalized <- crossprod(diag(inv.V.norms), V)

  if (K == 1) {
    A0 <- A1[1:n.train, 1:n.train]
    pb <- sum(A0) / n.train^2
    if (pb < 1e-6) pb <- 1e-6
    if (pb > 1 - 1e-6) pb <- 1 - 1e-6
    A.2 <- A.new[(n.train + 1):n, (n.train + 1):n]
    sum.index <- lower.tri(A.2)
    auc <- AUC(A.2[sum.index], rep(pb, length(A.2[sum.index])))
    loglike <- -sum(A.2[sum.index] * log(pb)) -
      sum((1 - A.2[sum.index]) * log(1 - pb))
    l2.val <- sum((A.2[sum.index] - pb)^2)

    N.1i <- 1:n.train
    N.2i <- (n.train + 1):n
    dc.pb <- (sum(A.new[N.1i, N.1i]) / 2 + sum(A.new[N.1i, N.2i]) + 1) /
      (sum(Psi.outer[N.1i, N.1i]) / 2 + sum(Psi.outer[N.1i, N.2i]) -
         sum(diag(Psi.outer)) + 1)
    dc.P.hat.holdout <- tcrossprod(
      crossprod(diag(Psi[(n.train + 1):n]),
                matrix(1, ncol = n - n.train, nrow = n - n.train)),
      diag(Psi[(n.train + 1):n])) * dc.pb
    dc.P.hat.holdout[dc.P.hat.holdout < 1e-6] <- 1e-6
    dc.P.hat.holdout[dc.P.hat.holdout > (1 - 1e-6)] <- 1 - 1e-6
    dc.loglike <- -sum(A.2[sum.index] * log(dc.P.hat.holdout[sum.index])) -
      sum((1 - A.2[sum.index]) * log(1 - dc.P.hat.holdout[sum.index]))
    dc.auc <- AUC(A.2[sum.index], dc.P.hat.holdout[sum.index])
    dc.l2 <- sum((A.2[sum.index] - dc.P.hat.holdout[sum.index])^2)
    return(list(loglike = loglike, l2 = l2.val, auc = auc,
                dc.loglike = dc.loglike, dc.l2 = dc.l2, dc.auc = dc.auc,
                no.edge = NA, impute.err = NA))
  }

  V <- A1.svd$v
  km <- kmeans(V, centers = K, nstart = 30, iter.max = 30)
  dc.km <- kmeans(V.normalized, centers = K, nstart = 30, iter.max = 30)

  B.mat <- dc.B <- matrix(0, K, K)

  tmp <- lapply(1:K, function(ii) {
    list(
      N1 = intersect(1:n.train, which(km$cluster == ii)),
      N2 = intersect((n.train + 1):n, which(km$cluster == ii)),
      dc.N1 = intersect(1:n.train, which(dc.km$cluster == ii)),
      dc.N2 = intersect((n.train + 1):n, which(dc.km$cluster == ii)))
  })

  for (i in 1:(K - 1)) {
    for (j in (i + 1):K) {
      B.mat[i, j] <- B.mat[j, i] <- (
        sum(A.new[tmp[[i]]$N1, tmp[[j]]$N1]) +
          sum(A.new[tmp[[i]]$N1, tmp[[j]]$N2]) +
          sum(A.new[tmp[[j]]$N1, tmp[[i]]$N2]) + 1) / (
        length(tmp[[i]]$N1) * length(tmp[[j]]$N1) +
          length(tmp[[j]]$N1) * length(tmp[[i]]$N2) +
          length(tmp[[i]]$N1) * length(tmp[[j]]$N2) + 1)

      dc.B[i, j] <- dc.B[j, i] <- (
        sum(A.new[tmp[[i]]$dc.N1, tmp[[j]]$dc.N1]) +
          sum(A.new[tmp[[i]]$dc.N1, tmp[[j]]$dc.N2]) +
          sum(A.new[tmp[[j]]$dc.N1, tmp[[i]]$dc.N2]) + 1) / (
        sum(Psi.outer[tmp[[i]]$dc.N1, tmp[[j]]$dc.N1]) +
          sum(Psi.outer[tmp[[i]]$dc.N1, tmp[[j]]$dc.N2]) +
          sum(Psi.outer[tmp[[j]]$dc.N1, tmp[[i]]$dc.N2]) + 1)
    }
  }

  Theta <- matrix(0, n, K)
  dc.Theta <- matrix(0, n, K)
  for (i in 1:K) {
    B.mat[i, i] <- (
      sum(A.new[tmp[[i]]$N1, tmp[[i]]$N1]) / 2 +
        sum(A.new[tmp[[i]]$N1, tmp[[i]]$N2]) + 1) / (
      length(tmp[[i]]$N1) * (length(tmp[[i]]$N1) - 1) / 2 +
        length(tmp[[i]]$N1) * length(tmp[[i]]$N2) + 1)
    Theta[which(km$cluster == i), i] <- 1

    dc.B[i, i] <- (
      sum(A.new[tmp[[i]]$dc.N1, tmp[[i]]$dc.N1]) / 2 +
        sum(A.new[tmp[[i]]$dc.N1, tmp[[i]]$dc.N2]) + 1) / (
      sum(Psi.outer[tmp[[i]]$dc.N1, tmp[[i]]$dc.N1]) / 2 +
        sum(Psi.outer[tmp[[i]]$dc.N1, tmp[[i]]$dc.N2]) -
        sum(diag(Psi.outer)) + 1)
    dc.Theta[which(dc.km$cluster == i), i] <- 1
  }

  P.hat.holdout <- tcrossprod(
    tcrossprod(Theta[(n.train + 1):n, ], B.mat),
    Theta[(n.train + 1):n, ])
  P.hat.holdout[P.hat.holdout < 1e-6] <- 1e-6
  P.hat.holdout[P.hat.holdout > (1 - 1e-6)] <- 1 - 1e-6
  A.2 <- A.new[(n.train + 1):n, (n.train + 1):n]
  sum.index <- lower.tri(A.2)
  loglike <- -sum(A.2[sum.index] * log(P.hat.holdout[sum.index])) -
    sum((1 - A.2[sum.index]) * log(1 - P.hat.holdout[sum.index]))
  auc <- AUC(A.2[sum.index], P.hat.holdout[sum.index])
  l2.val <- sum((A.2[sum.index] - P.hat.holdout[sum.index])^2)

  tmp.imt.mat <- dc.Theta[(n.train + 1):n, ] * Psi[(n.train + 1):n]
  dc.P.hat.holdout <- tcrossprod(tcrossprod(tmp.imt.mat, dc.B), tmp.imt.mat)

  if (dc.est == 2) {
    dc.B2 <- matrix(0, K, K)
    dc.Theta2 <- matrix(0, n, K)
    A.new.na <- A.new
    A.new.na[(n.train + 1):n, (n.train + 1):n] <- NA
    for (i in 1:K) {
      for (j in 1:K) {
        N.i <- which(km$cluster == i)
        N.j <- which(km$cluster == j)
        dc.B2[i, j] <- sum(A.new.na[N.i, N.j], na.rm = TRUE) + 0.01
      }
      dc.Theta2[N.i, i] <- 1
    }
    partial.d <- colSums(A.new.na, na.rm = TRUE)
    partial.gd <- colSums(dc.B2)
    B.g <- dc.Theta2 %*% partial.gd
    phi <- as.numeric(partial.d / B.g)
    P.hat <- tcrossprod(
      crossprod(diag(phi),
                tcrossprod(tcrossprod(dc.Theta2, dc.B2), dc.Theta2)),
      diag(phi))
    diag(P.hat) <- 0
    dc.P.hat.holdout <- P.hat[(n.train + 1):n, (n.train + 1):n]
  }

  dc.P.hat.holdout[dc.P.hat.holdout < 1e-6] <- 1e-6
  dc.P.hat.holdout[dc.P.hat.holdout > (1 - 1e-6)] <- 1 - 1e-6
  dc.loglike <- -sum(A.2[sum.index] * log(dc.P.hat.holdout[sum.index])) -
    sum((1 - A.2[sum.index]) * log(1 - dc.P.hat.holdout[sum.index]))
  dc.auc <- AUC(A.2[sum.index], dc.P.hat.holdout[sum.index])
  dc.l2 <- sum((A.2[sum.index] - dc.P.hat.holdout[sum.index])^2)

  list(loglike = loglike, l2 = l2.val, auc = auc,
       dc.loglike = dc.loglike, dc.l2 = dc.l2, dc.auc = dc.auc,
       no.edge = sum(colSums(A.new[1:n.train, ]) == 0), impute.err = NA)
}
