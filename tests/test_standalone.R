###############################################################################
## Standalone Test — runs WITHOUT installing the package
##
## Requires only: Matrix, RSpectra, ClusterR, irlba, Rfast, data.table, cluster
##
## Usage:
##   cd INCVCommunityDetection
##   Rscript tests/test_standalone.R
###############################################################################

cat("Loading required packages...\n")
suppressPackageStartupMessages({
  library(Matrix)
  library(RSpectra)
  library(ClusterR)
  library(irlba)
  library(Rfast)
  library(data.table)
  library(cluster)
})

cat("Sourcing package R files...\n")
source("R/utils.R")
source("R/simulation.R")
source("R/spectral.R")
source("R/estimation.R")
source("R/incv.R")

pass <- 0L
fail <- 0L

check <- function(desc, expr) {
  result <- tryCatch(expr, error = function(e) e)
  if (inherits(result, "error")) {
    cat("[FAIL]", desc, "\n       Error:", conditionMessage(result), "\n")
    fail <<- fail + 1L
  } else {
    cat("[PASS]", desc, "\n")
    pass <<- pass + 1L
  }
}

cat("\n====== INCVCommunityDetection Standalone Tests ======\n\n")

# --- Utilities ---------------------------------------------------------------
check("edge.index.map: u=1 -> (1,2)", {
  r <- edge.index.map(1); stopifnot(r$x == 1, r$y == 2)
})

check("edge.index.map: vectorised", {
  r <- edge.index.map(1:10); stopifnot(length(r$x) == 10)
})

check("neglog(1, 0.5) == -log(0.5)", {
  stopifnot(abs(neglog(1, 0.5) + log(0.5)) < 1e-12)
})

check("neglog(n, 0) == 0", { stopifnot(neglog(10, 0) == 0) })

# --- Simulation --------------------------------------------------------------
set.seed(1)
check("community.sim: 2-community SBM", {
  net <- community.sim(k = 2, n = 60, n1 = 25, p = 0.5, q = 0.1)
  stopifnot(nrow(net$adjacency) == 60, isSymmetric(net$adjacency))
  stopifnot(all(diag(net$adjacency) == 0))
  stopifnot(length(unique(net$membership)) == 2)
})

check("community.sim: 4-community SBM", {
  net <- community.sim(k = 4, n = 120, n1 = 20, p = 0.4, q = 0.05)
  stopifnot(length(unique(net$membership)) == 4)
})

check("community.sim.sbm: distance-decay SBM", {
  net <- community.sim.sbm(n = 90, n1 = 30, K = 3)
  stopifnot(nrow(net$conn) == 3, nrow(net$adjacency) == 90)
})

# --- Spectral clustering -----------------------------------------------------
set.seed(42)
net <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)

check("SBM.spectral.clustering: k=3", {
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)
  stopifnot(length(cl$cluster) == 150, all(cl$cluster %in% 1:3))
})

check("SBM.prob: restricted", {
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)$cluster
  res <- SBM.prob(cl, 3, net$adjacency, restricted = TRUE)
  cat("   p =", round(res$p.matrix[1, 1], 3),
      ", q =", round(res$p.matrix[1, 2], 3), "\n")
  stopifnot(nrow(res$p.matrix) == 3, is.finite(res$negloglike))
})

check("SBM.prob: unrestricted", {
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)$cluster
  res <- SBM.prob(cl, 3, net$adjacency, restricted = FALSE)
  stopifnot(is.finite(res$negloglike))
})

# --- Estimation ---------------------------------------------------------------
check("fast.SBM.est", {
  cl <- SBM.spectral.clustering(net$adjacency, k = 3)$cluster
  B <- fast.SBM.est(net$adjacency, cl)
  stopifnot(nrow(B) == 3, all(is.finite(B)))
})

check("best.perm.label.match: identity", {
  lab <- c(1, 1, 2, 2, 3, 3)
  E <- best.perm.label.match(lab, lab)
  stopifnot(all(E == diag(3)))
})

check("matched.lab: permuted labels", {
  lab   <- c(2, 2, 1, 1, 3, 3)
  fixed <- c(1, 1, 2, 2, 3, 3)
  stopifnot(all(matched.lab(lab, fixed) == fixed))
})

# --- INCV f-fold --------------------------------------------------------------
set.seed(100)
net2 <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)

check("nscv.f.fold: runs and selects K", {
  res <- nscv.f.fold(net2$adjacency, k.vec = 2:5, f = 5)
  cat("   k.loss =", res$k.loss, ", k.mse =", res$k.mse, "\n")
  stopifnot(res$k.loss %in% 2:5, length(res$cv.loss) == 4)
})

check("nscv.f.fold: loss method", {
  res <- nscv.f.fold(net2$adjacency, k.vec = 2:4, f = 5, method = "loss")
  cat("   k.loss =", res$k.loss, "\n")
  stopifnot(res$k.loss %in% 2:4)
})

# --- INCV random split --------------------------------------------------------
check("nscv.random.split: runs and selects K", {
  res <- nscv.random.split(net2$adjacency, k.vec = 2:4, ite = 10)
  cat("   k.chosen =", res$k.chosen, "\n")
  stopifnot(res$k.chosen %in% 2:4, length(res$cv.loss) == 3)
})

# --- Summary ------------------------------------------------------------------
cat("\n====================================================\n")
cat(sprintf("  Results: %d passed, %d failed\n", pass, fail))
cat("====================================================\n")

if (fail > 0) {
  cat("*** Some tests FAILED ***\n")
  quit(status = 1)
} else {
  cat("*** All tests PASSED ***\n")
}
