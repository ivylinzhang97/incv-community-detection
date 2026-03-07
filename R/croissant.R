#' CROISSANT for blockmodel selection
#'
#' Cross-validated, Overlapping, In-Sample Selection of the Number of
#' communities and model Type. Jointly selects between SBM and DCBM and
#' the number of communities \code{K} using overlapping node subsamples.
#'
#' @param A Adjacency matrix (n x n, can be sparse).
#' @param K.CAND Candidate community numbers (integer vector, or a single
#'   integer interpreted as \code{1:K.CAND}).
#' @param s Number of non-overlapping subsamples.
#' @param o Size of the overlapping set.
#' @param R Number of independent repetitions.
#' @param tau Regularisation parameter for the adjacency (default 0).
#' @param laplace Logical; use graph Laplacian normalisation (default FALSE).
#' @param dc.est DCBM estimation method: 1 = eigenvector, 2 = fast (default 2).
#' @param loss Character vector of loss functions to evaluate.
#'   Supported: \code{"l2"}, \code{"bin.dev"}, \code{"AUC"}.
#' @param ncore Number of cores for parallel computation.
#' @return A list containing:
#'   \item{loss}{data.table of loss values by candidate model and K.}
#'   \item{Candidate Models}{"SBM and DCBM".}
#'   \item{Candidate Values}{The candidate K vector.}
#'   \item{*.model}{Selected model string (e.g. "SBM-3") for each loss.}
#' @export
croissant.blockmodel <- function(A, K.CAND, s, o, R,
                                 tau = 0, laplace = FALSE, dc.est = 2,
                                 loss = c("l2", "bin.dev", "AUC"),
                                 ncore = 1) {
  if (length(K.CAND) == 1) K.CAND <- 1:K.CAND
  K.max <- max(K.CAND)
  n <- nrow(A)
  m <- (n - o) / s
  L <- list()
  mod <- c("SBM", "DCBM")

  over <- lapply(1:R, function(ii) sample.int(n, o, FALSE))
  non.over <- lapply(1:R, function(ii) sample((1:n)[-over[[ii]]], n - o, replace = FALSE))
  raw.ind <- cbind(rep(1:R, each = s), rep(1:s, R))

  raw.out <- parallel::mclapply(1:nrow(raw.ind), function(ii) {
    q <- raw.ind[ii, 2]
    r <- raw.ind[ii, 1]
    sonn <- c(over[[r]], non.over[[r]][((q - 1) * m + 1):(q * m)])
    A.sonn <- A[sonn, sonn]
    deg <- rowSums(A.sonn)
    avg.deg <- mean(deg)
    A.sonn.tau <- A.sonn + tau * avg.deg / n
    d.sonn.tau <- Matrix::sparseMatrix(
      i = 1:(o + m), j = 1:(o + m),
      x = 1 / sqrt(deg + tau * avg.deg))
    L.sonn <- A.sonn.tau
    if (laplace) {
      L.sonn <- Matrix::tcrossprod(Matrix::crossprod(d.sonn.tau, A.sonn.tau), d.sonn.tau)
      L.sonn[is.na(L.sonn)] <- 0
    }
    eig.max <- irlba::partial_eigen(x = L.sonn, n = K.max, symmetric = TRUE)$vectors
    out.SBM <- list()
    out.DCBM <- list()
    for (k.cand in seq_along(K.CAND)) {
      if (K.CAND[k.cand] == 1) {
        out.SBM[[k.cand]] <- out.DCBM[[k.cand]] <- rep(1, o + m)
        next
      }
      work.K <- K.CAND[[k.cand]]
      out.SBM[[k.cand]] <- as.integer(cluster::pam(
        eig.max[, 1:work.K], work.K,
        metric = "euclidean", do.swap = FALSE,
        cluster.only = TRUE, pamonce = 6))
      rownorm <- sqrt(rowSums(eig.max[, 1:work.K]^2))
      rownorm[rownorm == 0] <- 1
      rn.eig <- eig.max[, 1:work.K] / rownorm
      out.DCBM[[k.cand]] <- as.integer(cluster::pam(
        rn.eig, work.K,
        metric = "euclidean", do.swap = FALSE,
        cluster.only = TRUE, pamonce = 6))
    }
    list(SBM = out.SBM, DCBM = out.DCBM)
  }, mc.cores = ncore)

  K.size <- length(K.CAND)

  est.out <- parallel::mclapply(1:(K.size * nrow(raw.ind)), function(ii) {
    k.cand <- ii %% K.size
    k.cand <- ifelse(k.cand == 0, K.size, k.cand)
    rot <- ceiling(ii / K.size)
    q <- raw.ind[rot, 2]
    r <- raw.ind[rot, 1]
    sonn <- c(over[[r]], non.over[[r]][((q - 1) * m + 1):(q * m)])
    A.sonn <- A[sonn, sonn]
    out.SBM.std <- raw.out[raw.ind[, 1] == r][[1]]$SBM[[k.cand]]
    out.DCBM.std <- raw.out[raw.ind[, 1] == r][[1]]$DCBM[[k.cand]]
    out.SBM <- raw.out[raw.ind[, 1] == r][[q]]$SBM[[k.cand]]
    out.DCBM <- raw.out[raw.ind[, 1] == r][[q]]$DCBM[[k.cand]]
    work.K <- K.CAND[[k.cand]]

    if (work.K == 1) {
      mat.SBM <- mat.DCBM <- rep(1, m)
      B.SBM <- fast.SBM.est(A.sonn, rep(1, o + m), o + m, 1)
      mat.SBM <- rep(1, m)
      if (dc.est == 2)
        tmp <- fast.DCBM.est(A.sonn, rep(1, o + m), o + m, 1, o, p.sample = 1)
      else
        tmp <- eigen.DCBM.est(A.sonn, rep(1, o + m), o + m, 1, o, p.sample = 1)
      B.DCBM <- tmp$Bsum
      psi.DCBM <- tmp$psi
      mat.DCBM <- rep(1, m)
      return(list(gSBM = mat.SBM, BSBM = B.SBM,
                  gDCBM = mat.DCBM, BDCBM = B.DCBM, psiDCBM = psi.DCBM))
    }

    E.SBM.kc <- best.perm.label.match(out.SBM[1:o], out.SBM.std[1:o], o, work.K)
    E.DCBM.kc <- best.perm.label.match(out.DCBM[1:o], out.DCBM.std[1:o], o, work.K)
    tmp.SBM <- Matrix::sparseMatrix(i = 1:(o + m), j = out.SBM, dims = c(o + m, work.K))
    tmp.DCBM <- Matrix::sparseMatrix(i = 1:(o + m), j = out.DCBM, dims = c(o + m, work.K))
    mat.SBM <- as.vector(Matrix::tcrossprod(Matrix::tcrossprod(tmp.SBM, E.SBM.kc), rbind(1:work.K)))
    mat.DCBM <- as.vector(Matrix::tcrossprod(Matrix::tcrossprod(tmp.DCBM, E.DCBM.kc), rbind(1:work.K)))
    B.SBM <- fast.SBM.est(A.sonn, mat.SBM, o + m, work.K)
    mat.SBM <- mat.SBM[-(1:o)]
    if (dc.est == 2)
      tmp <- fast.DCBM.est(A.sonn, mat.DCBM, o + m, work.K, o, p.sample = 1)
    else
      tmp <- eigen.DCBM.est(A.sonn, mat.DCBM, o + m, work.K, o, p.sample = 1)
    B.DCBM <- tmp$Bsum
    psi.DCBM <- tmp$psi
    mat.DCBM <- mat.DCBM[-(1:o)]

    list(gSBM = mat.SBM, BSBM = B.SBM,
         gDCBM = mat.DCBM, BDCBM = B.DCBM, psiDCBM = psi.DCBM)
  }, mc.cores = ncore)

  g.SBM <- g.DCBM <- B.SBM <- B.DCBM <- psi.DCBM <- list()
  raw.mat <- cbind(raw.ind[rep(1:nrow(raw.ind), each = K.size), ],
                   rep(1:K.size, nrow(raw.ind)))

  for (r in 1:R) {
    g.SBM[[r]] <- B.SBM[[r]] <- g.DCBM[[r]] <- B.DCBM[[r]] <- psi.DCBM[[r]] <- list()
    for (k.cand in seq_along(K.CAND)) {
      tmp.est <- est.out[which(raw.mat[, 3] == k.cand & raw.mat[, 1] == r)]
      B.SBM[[r]][[k.cand]] <- 0
      B.DCBM[[r]][[k.cand]] <- 0
      g.SBM[[r]][[k.cand]] <- g.DCBM[[r]][[k.cand]] <- psi.DCBM[[r]][[k.cand]] <- list()
      for (q in 1:s) {
        B.SBM[[r]][[k.cand]] <- B.SBM[[r]][[k.cand]] + tmp.est[[q]]$BSBM / s
        B.DCBM[[r]][[k.cand]] <- B.DCBM[[r]][[k.cand]] + tmp.est[[q]]$BDCBM / s
        g.SBM[[r]][[k.cand]][[q]] <- tmp.est[[q]]$gSBM
        g.DCBM[[r]][[k.cand]][[q]] <- tmp.est[[q]]$gDCBM
        psi.DCBM[[r]][[k.cand]][[q]] <- tmp.est[[q]]$psiDCBM
      }
    }
  }

  non.size <- s * (s - 1) / 2
  non.mat <- matrix(nrow = R * non.size * K.size, ncol = 4)
  cc <- 1
  for (r in 1:R)
    for (k.cand in seq_along(K.CAND))
      for (p in 1:(s - 1))
        for (q in (p + 1):s) {
          non.mat[cc, ] <- c(r, k.cand, p, q)
          cc <- cc + 1
        }

  L.all <- parallel::mclapply(1:nrow(non.mat), function(ii) {
    r <- non.mat[ii, 1]
    k.cand <- non.mat[ii, 2]
    p <- non.mat[ii, 3]
    q <- non.mat[ii, 4]
    p.non <- non.over[[r]][((p - 1) * m + 1):(p * m)]
    q.non <- non.over[[r]][((q - 1) * m + 1):(q * m)]
    A.non <- A[p.non, q.non]
    L.temp <- matrix(0, nrow = 2 * length(loss), ncol = 1)
    row.names(L.temp) <- paste(rep(mod, each = length(loss)), rep(loss, 2), sep = "_")
    colnames(L.temp) <- as.character(K.CAND[k.cand])
    P.SBM <- B.SBM[[r]][[k.cand]][g.SBM[[r]][[k.cand]][[p]],
                                    g.SBM[[r]][[k.cand]][[q]]]
    P.DCBM <- B.DCBM[[r]][[k.cand]][g.DCBM[[r]][[k.cand]][[p]],
                                      g.DCBM[[r]][[k.cand]][[q]]] *
      tcrossprod(psi.DCBM[[r]][[k.cand]][[p]], psi.DCBM[[r]][[k.cand]][[q]])
    P.DCBM[P.DCBM < 1e-6] <- 1e-6
    P.DCBM[P.DCBM > 1 - 1e-6] <- 1 - 1e-6
    for (mq in seq_along(mod)) {
      P.use <- if (mod[mq] == "SBM") P.SBM else P.DCBM
      for (lq in seq_along(loss)) {
        tmp.nm <- paste(mod[mq], loss[lq], sep = "_")
        L.temp[tmp.nm, 1] <- L.temp[tmp.nm, 1] +
          do.call(loss[lq], list(as.numeric(A.non), P.use)) / (s * (s - 1) * 0.5)
      }
    }
    L.temp
  }, mc.cores = ncore)

  for (r in 1:R) {
    L[[r]] <- do.call("cbind", lapply(1:K.size, function(kk) {
      Reduce("+", L.all[which(non.mat[, 2] == kk & non.mat[, 1] == r)])
    }))
    row.names(L[[r]]) <- paste(rep(mod, each = length(loss)), rep(loss, 2), sep = "_")
    colnames(L[[r]]) <- as.character(K.CAND)
  }

  obj <- data.table::data.table(
    `Candidate Model` = rep(mod, length(K.CAND)),
    `Candidate Value` = rep(K.CAND, each = 2))

  for (lq in seq_along(loss))
    for (r in 1:R)
      obj[[paste0(loss[lq], "-Rep=", r)]] <-
        as(rbind(L[[r]][paste0("SBM_", loss[lq]), ],
                 L[[r]][paste0("DCBM_", loss[lq]), ]), "vector")

  obj2 <- list()
  obj2[["Candidate Models"]] <- "SBM and DCBM"
  obj2[["Candidate Values"]] <- K.CAND

  for (lq in seq_along(loss)) {
    obj2[[paste0("Mod.K.hat.each.rep (", loss[lq], ")")]] <- sapply(1:R, function(r) {
      l.sbm <- min(L[[r]][paste0("SBM_", loss[lq]), ])
      l.dcbm <- min(L[[r]][paste0("DCBM_", loss[lq]), ])
      ifelse(l.dcbm < l.sbm,
             paste0("DCSBM-", K.CAND[which.min(L[[r]][paste0("DCBM_", loss[lq]), ])]),
             paste0("SBM-", K.CAND[which.min(L[[r]][paste0("SBM_", loss[lq]), ])]))
    })
    obj2[[paste0(loss[lq], ".model")]] <-
      modal(obj2[[paste0("Mod.K.hat.each.rep (", loss[lq], ")")]])
  }

  c(list(loss = obj), obj2)
}

#' CROISSANT for RDPG rank selection
#'
#' Selects the embedding rank for a Random Dot Product Graph model using
#' the CROISSANT overlapping node-subsample framework.
#'
#' @param A Adjacency matrix.
#' @param d.cand Candidate embedding ranks.
#' @param s Number of non-overlapping subsamples.
#' @param o Overlap size.
#' @param R Number of repetitions.
#' @param laplace Use Laplacian normalisation (default FALSE).
#' @param loss Loss functions to evaluate.
#' @param ncore Number of cores.
#' @return A list with loss table and selected rank per loss.
#' @export
croissant.rdpg <- function(A, d.cand, s, o, R,
                           laplace = FALSE,
                           loss = c("l2", "bin.dev", "AUC"),
                           ncore = 1) {
  n <- nrow(A)
  m <- (n - o) / s
  if (length(d.cand) == 1) d.cand <- 1:d.cand
  dmax <- max(d.cand)

  over <- lapply(1:R, function(ii) sample.int(n, o, FALSE))
  non.over <- lapply(1:R, function(ii) sample((1:n)[-over[[ii]]], n - o, replace = FALSE))
  raw.ind <- cbind(rep(1:R, each = s), rep(1:s, R))
  colnames(raw.ind) <- c("r", "s")

  raw.out <- parallel::mclapply(1:nrow(raw.ind), function(ii) {
    q <- raw.ind[ii, "s"]
    r <- raw.ind[ii, "r"]
    sonn <- c(over[[r]], non.over[[r]][((q - 1) * m + 1):(q * m)])
    A.sonn <- A[sonn, sonn]
    if (!laplace) {
      L.sonn <- A.sonn
    } else {
      degree <- rowSums(A.sonn)
      D <- Matrix::sparseMatrix(i = 1:(o + m), j = 1:(o + m), x = 1 / sqrt(degree))
      L.sonn <- Matrix::tcrossprod(Matrix::crossprod(D, A.sonn), D)
    }
    eig <- irlba::irlba(L.sonn, nv = dmax)
    U <- eig$v
    sigma.half <- diag(sqrt(abs(eig$d)))
    tcrossprod(U, sigma.half)
  }, mc.cores = ncore)

  match.out <- parallel::mclapply(1:nrow(raw.ind), function(ii) {
    r <- raw.ind[ii, "r"]
    q <- raw.ind[ii, "s"]
    if (q == 1) return(raw.out[[ii]][-(1:o), ])
    stand <- which(raw.ind[, "r"] == r & raw.ind[, "s"] == 1)
    proc.mat <- IMIFA::Procrustes(raw.out[[ii]][(1:o), ], raw.out[[stand]][(1:o), ])$R
    raw.out[[ii]][-(1:o), ] %*% proc.mat
  }, mc.cores = ncore)

  non.size <- s * (s - 1) / 2
  ld <- length(d.cand)
  non.mat <- matrix(nrow = R * non.size * ld, ncol = 4)
  cc <- 1
  for (r in 1:R)
    for (dd in seq_along(d.cand))
      for (p in 1:(s - 1))
        for (q in (p + 1):s) {
          non.mat[cc, ] <- c(r, dd, p, q)
          cc <- cc + 1
        }
  colnames(non.mat) <- c("r", "dd", "p", "q")

  L.all <- parallel::mclapply(1:nrow(non.mat), function(ii) {
    r <- non.mat[ii, "r"]
    dd <- non.mat[ii, "dd"]
    p <- non.mat[ii, "p"]
    q <- non.mat[ii, "q"]
    p.non <- non.over[[r]][((p - 1) * m + 1):(p * m)]
    q.non <- non.over[[r]][((q - 1) * m + 1):(q * m)]
    A.non <- A[p.non, q.non]
    L.temp <- matrix(0, nrow = length(loss), ncol = 1)
    row.names(L.temp) <- loss
    colnames(L.temp) <- as.character(d.cand[dd])
    ind1 <- which(raw.ind[, "r"] == r & raw.ind[, "s"] == p)
    ind2 <- which(raw.ind[, "r"] == r & raw.ind[, "s"] == q)
    P.hat <- tcrossprod(match.out[[ind1]][, 1:d.cand[dd]],
                        match.out[[ind2]][, 1:d.cand[dd]])
    P.hat[P.hat < 1e-6] <- 1e-6
    P.hat[P.hat > 1 - 1e-6] <- 1 - 1e-6
    for (lq in seq_along(loss)) {
      tmp.nm <- loss[lq]
      L.temp[tmp.nm, 1] <- L.temp[tmp.nm, 1] +
        do.call(loss[lq], list(A.non, P.hat)) / (s * (s - 1) * 0.5)
    }
    L.temp
  }, mc.cores = ncore)

  L <- list()
  for (r in 1:R) {
    L[[r]] <- do.call("cbind", lapply(1:ld, function(dd) {
      Reduce("+", L.all[which(non.mat[, "dd"] == dd & non.mat[, "r"] == r)])
    }))
    row.names(L[[r]]) <- loss
    colnames(L[[r]]) <- as.character(d.cand)
  }

  obj <- data.table::data.table(`Candidate Rank` = d.cand)
  for (lq in seq_along(loss))
    for (r in 1:R)
      obj[[paste0(loss[lq], "-Rep=", r)]] <-
        as(rbind(L[[r]][loss[lq], ]), "vector")

  obj2 <- list()
  obj2[["Candidate Rank"]] <- d.cand
  for (lq in seq_along(loss)) {
    obj2[[paste0("d.hat.each.rep (", loss[lq], ")")]] <- sapply(1:R, function(r) {
      d.cand[which.min(L[[r]][loss[lq], ])]
    })
    obj2[[paste0(loss[lq], ".model")]] <-
      modal(obj2[[paste0("d.hat.each.rep (", loss[lq], ")")]])
  }
  c(list(loss = obj), obj2)
}

#' CROISSANT for latent space model dimension selection
#'
#' Selects the latent dimension for a latent space network model using
#' the CROISSANT framework with MLE fitting.
#'
#' @inheritParams croissant.rdpg
#' @return A list with loss table and selected dimension per loss.
#' @export
croissant.latent <- function(A, d.cand, s, o, R,
                             loss = c("l2", "bin.dev", "AUC"),
                             ncore = 1) {
  if (!requireNamespace("latentnet", quietly = TRUE))
    stop("Package 'latentnet' is required for croissant.latent()")
  if (!requireNamespace("rdist", quietly = TRUE))
    stop("Package 'rdist' is required for croissant.latent()")

  n <- nrow(A)
  m <- (n - o) / s
  if (length(d.cand) == 1) d.cand <- 1:d.cand
  dmax <- max(d.cand)
  ld <- length(d.cand)

  over <- lapply(1:R, function(ii) sample.int(n, o, FALSE))
  non.over <- lapply(1:R, function(ii) sample((1:n)[-over[[ii]]], n - o, replace = FALSE))

  raw.ind <- matrix(nrow = R * s * ld, ncol = 3)
  cc <- 1
  for (r in 1:R)
    for (dd in seq_along(d.cand))
      for (q in 1:s) {
        raw.ind[cc, ] <- c(r, dd, q)
        cc <- cc + 1
      }
  colnames(raw.ind) <- c("r", "dd", "q")

  raw.out <- parallel::mclapply(1:nrow(raw.ind), function(ii) {
    q <- raw.ind[ii, "q"]
    r <- raw.ind[ii, "r"]
    dd <- raw.ind[ii, "dd"]
    sonn <- c(over[[r]], non.over[[r]][((q - 1) * m + 1):(q * m)])
    A.sonn <- A[sonn, sonn]
    net.sonn <- latentnet::as.network(A.sonn, matrix.type = "adjacency")
    out.lat <- latentnet::ergmm(net.sonn ~ latentnet::euclidean(d = d.cand[dd]),
                                tofit = "mle")
    list(Z.hat = out.lat$mle$Z, beta.hat = out.lat$mle$beta)
  }, mc.cores = ncore)

  match.out <- parallel::mclapply(1:nrow(raw.ind), function(ii) {
    r <- raw.ind[ii, "r"]
    q <- raw.ind[ii, "q"]
    dd <- raw.ind[ii, "dd"]
    if (q == 1)
      return(list(Z.rot = raw.out[[ii]]$Z.hat[-(1:o), ],
                  beta.hat = raw.out[[ii]]$beta.hat))
    stand <- which(raw.ind[, "r"] == r & raw.ind[, "q"] == 1 &
                     raw.ind[, "dd"] == dd)
    proc.par <- IMIFA::Procrustes(
      cbind(raw.out[[ii]]$Z.hat[(1:o), ]),
      cbind(raw.out[[stand]]$Z.hat[(1:o), ]),
      translate = TRUE, dilate = FALSE)
    Z.rot <- cbind(raw.out[[ii]]$Z.hat[-(1:o), ]) %*% proc.par$R +
      matrix(proc.par$t, nrow = m, ncol = d.cand[dd])
    list(Z.rot = Z.rot, beta.hat = raw.out[[ii]]$beta.hat)
  }, mc.cores = ncore)

  non.size <- s * (s - 1) / 2
  non.mat <- matrix(nrow = R * non.size * ld, ncol = 4)
  cc <- 1
  for (r in 1:R)
    for (dd in seq_along(d.cand))
      for (p in 1:(s - 1))
        for (q in (p + 1):s) {
          non.mat[cc, ] <- c(r, dd, p, q)
          cc <- cc + 1
        }
  colnames(non.mat) <- c("r", "dd", "p", "q")

  L.all <- parallel::mclapply(1:nrow(non.mat), function(ii) {
    r <- non.mat[ii, "r"]
    dd <- non.mat[ii, "dd"]
    p <- non.mat[ii, "p"]
    q <- non.mat[ii, "q"]
    p.non <- non.over[[r]][((p - 1) * m + 1):(p * m)]
    q.non <- non.over[[r]][((q - 1) * m + 1):(q * m)]
    A.non <- A[p.non, q.non]
    L.temp <- matrix(0, nrow = length(loss), ncol = 1)
    row.names(L.temp) <- loss
    colnames(L.temp) <- as.character(d.cand[dd])
    ind1 <- which(raw.ind[, "r"] == r & raw.ind[, "q"] == p &
                    raw.ind[, "dd"] == dd)
    ind2 <- which(raw.ind[, "r"] == r & raw.ind[, "q"] == q &
                    raw.ind[, "dd"] == d.cand[dd])
    Z1.hat <- match.out[[ind1]]$Z.rot
    Z2.hat <- match.out[[ind2]]$Z.rot
    beta.hat <- (match.out[[ind1]]$beta.hat + match.out[[ind2]]$beta.hat) / 2
    log.hat <- beta.hat - rdist::cdist(Z1.hat, Z2.hat)
    P.hat <- exp(log.hat) / (1 + exp(log.hat))
    for (lq in seq_along(loss)) {
      tmp.nm <- loss[lq]
      L.temp[tmp.nm, 1] <- L.temp[tmp.nm, 1] +
        do.call(loss[lq], list(A.non, P.hat)) / (s * (s - 1) * 0.5)
    }
    L.temp
  }, mc.cores = ncore)

  L <- list()
  for (r in 1:R) {
    L[[r]] <- do.call("cbind", lapply(1:ld, function(dd) {
      Reduce("+", L.all[which(non.mat[, "dd"] == dd & non.mat[, "r"] == r)])
    }))
    row.names(L[[r]]) <- loss
    colnames(L[[r]]) <- as.character(d.cand)
  }

  obj <- data.table::data.table(`Candidate Rank` = d.cand)
  for (lq in seq_along(loss))
    for (r in 1:R)
      obj[[paste0(loss[lq], "-Rep=", r)]] <-
        as(rbind(L[[r]][loss[lq], ]), "vector")

  obj2 <- list()
  obj2[["Candidate Rank"]] <- d.cand
  for (lq in seq_along(loss)) {
    obj2[[paste0("d.hat.each.rep (", loss[lq], ")")]] <- sapply(1:R, function(r) {
      d.cand[which.min(L[[r]][loss[lq], ])]
    })
    obj2[[paste0(loss[lq], ".model")]] <-
      modal(obj2[[paste0("d.hat.each.rep (", loss[lq], ")")]])
  }
  c(list(loss = obj), obj2)
}

#' CROISSANT for regularisation parameter tuning in spectral methods
#'
#' Selects the regularisation parameter tau for spectral clustering using
#' the CROISSANT framework.
#'
#' @param A Adjacency matrix.
#' @param K Fixed number of communities.
#' @param tau.cand Candidate regularisation parameter values.
#' @param DCBM Logical; if TRUE, use row-normalised eigenvectors (DCBM).
#' @param s,o,R CROISSANT design parameters.
#' @param laplace Use Laplacian normalisation.
#' @param dc.est DCBM estimation type (1 or 2).
#' @param loss Loss functions to evaluate.
#' @param ncore Number of cores.
#' @return A list with loss table and selected tau per loss.
#' @export
croissant.tune.regsp <- function(A, K, tau.cand,
                                 DCBM = FALSE,
                                 s, o, R,
                                 laplace = FALSE, dc.est = 2,
                                 loss = c("l2", "bin.dev", "AUC"),
                                 ncore = 1) {
  n <- nrow(A)
  m <- (n - o) / s
  L <- list()

  over <- lapply(1:R, function(ii) sample.int(n, o, FALSE))
  non.over <- lapply(1:R, function(ii) sample((1:n)[-over[[ii]]], n - o, replace = FALSE))
  raw.ind <- cbind(rep(1:R, each = s), rep(1:s, R))

  raw.out <- parallel::mclapply(1:nrow(raw.ind), function(ii) {
    q <- raw.ind[ii, 2]
    r <- raw.ind[ii, 1]
    sonn <- c(over[[r]], non.over[[r]][((q - 1) * m + 1):(q * m)])
    A.sonn <- A[sonn, sonn]
    deg <- rowSums(A.sonn)
    avg.deg <- mean(deg)
    out.BM <- list()
    for (tt in seq_along(tau.cand)) {
      A.sonn.tau <- A.sonn + tau.cand[tt] * avg.deg / n
      d.sonn.tau <- Matrix::sparseMatrix(
        i = 1:(o + m), j = 1:(o + m),
        x = 1 / sqrt(deg + tau.cand[tt] * avg.deg))
      L.sonn <- A.sonn.tau
      if (laplace) {
        L.sonn <- Matrix::tcrossprod(Matrix::crossprod(d.sonn.tau, A.sonn.tau), d.sonn.tau)
        L.sonn[is.na(L.sonn)] <- 0
      }
      eig.max <- irlba::partial_eigen(x = L.sonn, n = K, symmetric = TRUE)$vectors
      if (K == 1) {
        out.BM[[tt]] <- rep(1, o + m)
        next
      }
      rn.eig <- eig.max
      if (DCBM) {
        rownorm <- sqrt(rowSums(eig.max^2))
        rownorm[rownorm == 0] <- 1
        rn.eig <- eig.max / rownorm
      }
      out.BM[[tt]] <- as.integer(cluster::pam(
        rn.eig, K,
        metric = "euclidean", do.swap = FALSE,
        cluster.only = TRUE, pamonce = 6))
    }
    list(BM = out.BM)
  }, mc.cores = ncore)

  tau.size <- length(tau.cand)

  est.out <- parallel::mclapply(1:(tau.size * nrow(raw.ind)), function(ii) {
    tt <- ii %% tau.size
    tt <- ifelse(tt == 0, tau.size, tt)
    rot <- ceiling(ii / tau.size)
    q <- raw.ind[rot, 2]
    r <- raw.ind[rot, 1]
    sonn <- c(over[[r]], non.over[[r]][((q - 1) * m + 1):(q * m)])
    A.sonn <- A[sonn, sonn]
    out.BM.std <- raw.out[raw.ind[, 1] == r][[1]]$BM[[tt]]
    out.BM <- raw.out[raw.ind[, 1] == r][[q]]$BM[[tt]]

    if (K == 1) {
      mat.BM <- rep(1, m)
      if (!DCBM) {
        B.BM <- fast.SBM.est(A.sonn, rep(1, o + m), o + m, 1)
        return(list(gBM = mat.BM, BBM = B.BM, psiBM = rep(1, m)))
      }
      if (dc.est == 2)
        tmp <- fast.DCBM.est(A.sonn, rep(1, o + m), o + m, 1, o, p.sample = 1)
      else
        tmp <- eigen.DCBM.est(A.sonn, rep(1, o + m), o + m, 1, o, p.sample = 1)
      return(list(gBM = mat.BM, BBM = tmp$Bsum, psiBM = tmp$psi))
    }

    E.BM.kc <- best.perm.label.match(out.BM[1:o], out.BM.std[1:o], o, K)
    tmp.BM <- Matrix::sparseMatrix(i = 1:(o + m), j = out.BM, dims = c(o + m, K))
    mat.BM <- as.vector(Matrix::tcrossprod(Matrix::tcrossprod(tmp.BM, E.BM.kc), rbind(1:K)))

    if (!DCBM) {
      B.BM <- fast.SBM.est(A.sonn, mat.BM, o + m, K)
      return(list(gBM = mat.BM[-(1:o)], BBM = B.BM, psiBM = rep(1, m)))
    }
    if (dc.est == 2)
      tmp <- fast.DCBM.est(A.sonn, mat.BM, o + m, K, o, p.sample = 1)
    else
      tmp <- eigen.DCBM.est(A.sonn, mat.BM, o + m, K, o, p.sample = 1)
    list(gBM = mat.BM[-(1:o)], BBM = tmp$Bsum, psiBM = tmp$psi)
  }, mc.cores = ncore)

  g.BM <- B.BM <- psi.BM <- list()
  raw.mat <- cbind(raw.ind[rep(1:nrow(raw.ind), each = tau.size), ],
                   rep(1:tau.size, nrow(raw.ind)))

  for (r in 1:R) {
    g.BM[[r]] <- B.BM[[r]] <- psi.BM[[r]] <- list()
    for (tt in seq_along(tau.cand)) {
      tmp.est <- est.out[which(raw.mat[, 3] == tt & raw.mat[, 1] == r)]
      B.BM[[r]][[tt]] <- 0
      g.BM[[r]][[tt]] <- psi.BM[[r]][[tt]] <- list()
      for (q in 1:s) {
        B.BM[[r]][[tt]] <- B.BM[[r]][[tt]] + tmp.est[[q]]$BBM / s
        g.BM[[r]][[tt]][[q]] <- tmp.est[[q]]$gBM
        psi.BM[[r]][[tt]][[q]] <- tmp.est[[q]]$psiBM
      }
    }
  }

  non.size <- s * (s - 1) / 2
  non.mat <- matrix(nrow = R * non.size * tau.size, ncol = 4)
  cc <- 1
  for (r in 1:R)
    for (tt in seq_along(tau.cand))
      for (p in 1:(s - 1))
        for (q in (p + 1):s) {
          non.mat[cc, ] <- c(r, tt, p, q)
          cc <- cc + 1
        }

  L.all <- parallel::mclapply(1:nrow(non.mat), function(ii) {
    r <- non.mat[ii, 1]
    tt <- non.mat[ii, 2]
    p <- non.mat[ii, 3]
    q <- non.mat[ii, 4]
    p.non <- non.over[[r]][((p - 1) * m + 1):(p * m)]
    q.non <- non.over[[r]][((q - 1) * m + 1):(q * m)]
    A.non <- A[p.non, q.non]
    L.temp <- matrix(0, nrow = length(loss), ncol = 1)
    row.names(L.temp) <- loss
    colnames(L.temp) <- as.character(tau.cand[tt])
    P.BM <- B.BM[[r]][[tt]][g.BM[[r]][[tt]][[p]], g.BM[[r]][[tt]][[q]]] *
      tcrossprod(psi.BM[[r]][[tt]][[p]], psi.BM[[r]][[tt]][[q]])
    P.BM[P.BM < 1e-6] <- 1e-6
    P.BM[P.BM > 1 - 1e-6] <- 1 - 1e-6
    for (lq in seq_along(loss)) {
      tmp.nm <- loss[lq]
      L.temp[tmp.nm, 1] <- L.temp[tmp.nm, 1] +
        do.call(loss[lq], list(as.numeric(A.non), P.BM)) / (s * (s - 1) * 0.5)
    }
    L.temp
  }, mc.cores = ncore)

  for (r in 1:R) {
    L[[r]] <- do.call("cbind", lapply(1:tau.size, function(tt) {
      Reduce("+", L.all[which(non.mat[, 2] == tt & non.mat[, 1] == r)])
    }))
    row.names(L[[r]]) <- loss
    colnames(L[[r]]) <- as.character(tau.cand)
  }

  obj <- data.table::data.table(`Candidate Tau` = tau.cand)
  for (lq in seq_along(loss))
    for (r in 1:R)
      obj[[paste0(loss[lq], "-Rep=", r)]] <-
        as(rbind(L[[r]][loss[lq], ]), "vector")

  obj2 <- list()
  obj2[["Candidate Tau"]] <- tau.cand
  for (lq in seq_along(loss)) {
    obj2[[paste0("tau.hat.each.rep (", loss[lq], ")")]] <- sapply(1:R, function(r) {
      tau.cand[which.min(L[[r]][loss[lq], ])]
    })
    obj2[[paste0(loss[lq], ".model")]] <-
      mean(obj2[[paste0("tau.hat.each.rep (", loss[lq], ")")]])
  }
  c(list(loss = obj), obj2)
}
