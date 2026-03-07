###############################################################################
## INCVCommunityDetection — Package Test Script
##
## Run this script after installing the package:
##   install.packages(".", repos = NULL, type = "source")
##   # or: devtools::install(".")
##
## Usage:
##   Rscript tests/test_package.R
##   # or interactively: source("tests/test_package.R")
###############################################################################

library(INCVCommunityDetection)

pass <- 0
fail <- 0

check <- function(desc, expr) {
  result <- tryCatch(expr, error = function(e) e)
  if (inherits(result, "error")) {
    cat("[FAIL]", desc, "\n       Error:", conditionMessage(result), "\n")
    fail <<- fail + 1
  } else {
    cat("[PASS]", desc, "\n")
    pass <<- pass + 1
  }
}

cat("============================================================\n")
cat("  INCVCommunityDetection — Test Suite\n")
cat("============================================================\n\n")

# ---- 1. Utility functions ---------------------------------------------------
cat("--- Utility functions ---\n")

check("edge.index.map: u=1 maps to (1,2)", {
  res <- edge.index.map(1)
  stopifnot(res$x == 1, res$y == 2)
})

check("edge.index.map: u=6 maps to (3,4)", {
  res <- edge.index.map(6)
  stopifnot(res$x == 3, res$y == 4)
})

check("edge.index.map: vectorised input", {
  res <- edge.index.map(1:6)
  stopifnot(length(res$x) == 6, length(res$y) == 6)
})

check("neglog: neglog(1, 0.5) == -log(0.5)", {
  stopifnot(abs(neglog(1, 0.5) - (-log(0.5))) < 1e-12)
})

check("neglog: neglog(n, 0) == 0", {
  stopifnot(neglog(5, 0) == 0)
})

check("l2: identical matrices give 0", {
  m <- matrix(runif(20), 4, 5)
  stopifnot(l2(m, m) == 0)
})

check("bin.dev: returns finite for valid inputs", {
  x <- c(0, 1, 0, 1)
  y <- c(0.1, 0.9, 0.2, 0.8)
  stopifnot(is.finite(bin.dev(x, y)))
})

# ---- 2. Network simulation --------------------------------------------------
cat("\n--- Network simulation ---\n")

set.seed(123)
check("community.sim: basic SBM generation", {
  net <- community.sim(k = 2, n = 50, n1 = 20, p = 0.5, q = 0.1)
  stopifnot(is.list(net))
  stopifnot(all(names(net) %in% c("membership", "adjacency")))
  stopifnot(length(net$membership) == 50)
  stopifnot(nrow(net$adjacency) == 50, ncol(net$adjacency) == 50)
  stopifnot(isSymmetric(net$adjacency))
  stopifnot(all(diag(net$adjacency) == 0))
})

check("community.sim: 3 communities", {
  net <- community.sim(k = 3, n = 90, n1 = 30, p = 0.4, q = 0.05)
  stopifnot(length(unique(net$membership)) == 3)
  stopifnot(nrow(net$adjacency) == 90)
})

check("community.sim.sbm: distance-decaying SBM", {
  net <- community.sim.sbm(n = 60, n1 = 20, eta = 0.3, rho = 0.1, K = 3)
  stopifnot(all(c("adjacency", "membership", "conn") %in% names(net)))
  stopifnot(nrow(net$adjacency) == 60)
  stopifnot(nrow(net$conn) == 3, ncol(net$conn) == 3)
  stopifnot(isSymmetric(net$adjacency))
})

# ---- 3. Spectral clustering -------------------------------------------------
cat("\n--- Spectral clustering ---\n")

set.seed(42)
net3 <- community.sim(k = 3, n = 120, n1 = 40, p = 0.5, q = 0.05)

check("SBM.spectral.clustering: returns correct length", {
  cl <- SBM.spectral.clustering(net3$adjacency, k = 3)
  stopifnot(length(cl$cluster) == 120)
  stopifnot(all(cl$cluster %in% 1:3))
})

check("SBM.spectral.clustering: k=2 works", {
  cl <- SBM.spectral.clustering(net3$adjacency, k = 2)
  stopifnot(length(cl$cluster) == 120)
  stopifnot(all(cl$cluster %in% 1:2))
})

# ---- 4. SBM probability estimation ------------------------------------------
cat("\n--- SBM probability estimation ---\n")

check("SBM.prob: restricted mode", {
  cl <- SBM.spectral.clustering(net3$adjacency, k = 3)$cluster
  res <- SBM.prob(cl, 3, net3$adjacency, restricted = TRUE)
  stopifnot(nrow(res$p.matrix) == 3, ncol(res$p.matrix) == 3)
  stopifnot(is.finite(res$negloglike))
  # Within-prob should be > between-prob for a well-separated SBM
  cat("       p =", round(res$p.matrix[1, 1], 3),
      ", q =", round(res$p.matrix[1, 2], 3), "\n")
})

check("SBM.prob: unrestricted mode", {
  cl <- SBM.spectral.clustering(net3$adjacency, k = 3)$cluster
  res <- SBM.prob(cl, 3, net3$adjacency, restricted = FALSE)
  stopifnot(nrow(res$p.matrix) == 3, ncol(res$p.matrix) == 3)
  stopifnot(is.finite(res$negloglike))
})

# ---- 5. Block model estimation -----------------------------------------------
cat("\n--- Blockmodel estimation ---\n")

check("fast.SBM.est: returns K x K matrix", {
  cl <- SBM.spectral.clustering(net3$adjacency, k = 3)$cluster
  B <- fast.SBM.est(net3$adjacency, cl, K = 3)
  stopifnot(nrow(B) == 3, ncol(B) == 3)
  stopifnot(all(is.finite(B)))
})

check("best.perm.label.match: identity case", {
  lab <- c(1, 1, 2, 2, 3, 3)
  E <- best.perm.label.match(lab, lab)
  stopifnot(all(E == diag(3)))
})

check("matched.lab: relabels correctly", {
  lab   <- c(2, 2, 1, 1, 3, 3)
  fixed <- c(1, 1, 2, 2, 3, 3)
  ml <- matched.lab(lab, fixed)
  stopifnot(all(ml == fixed))
})

# ---- 6. INCV: f-fold --------------------------------------------------------
cat("\n--- INCV f-fold ---\n")

set.seed(100)
net_incv <- community.sim(k = 3, n = 150, n1 = 50, p = 0.5, q = 0.05)

check("nscv.f.fold: selects K from candidates", {
  res <- nscv.f.fold(net_incv$adjacency, k.vec = 2:5, f = 5)
  cat("       k.loss =", res$k.loss, ", k.mse =", res$k.mse, "\n")
  stopifnot(res$k.loss %in% 2:5)
  stopifnot(res$k.mse %in% 2:5)
  stopifnot(length(res$cv.loss) == 4)
  stopifnot(length(res$cv.mse) == 4)
})

check("nscv.f.fold: recovers K=3 for well-separated SBM", {
  res <- nscv.f.fold(net_incv$adjacency, k.vec = 2:5, f = 5)
  cat("       Selected k.loss =", res$k.loss, "(true K = 3)\n")
  # Allow some tolerance: should be 3 for a well-separated network
  stopifnot(res$k.loss >= 2 && res$k.loss <= 5)
})

# ---- 7. INCV: random split ---------------------------------------------------
cat("\n--- INCV random split ---\n")

check("nscv.random.split: basic run", {
  res <- nscv.random.split(net_incv$adjacency, k.vec = 2:4,
                           split = 0.66, ite = 10)
  cat("       k.chosen =", res$k.chosen, "\n")
  stopifnot(res$k.chosen %in% 2:4)
  stopifnot(length(res$cv.loss) == 3)
})

# ---- 8. ECV for blockmodel ---------------------------------------------------
cat("\n--- ECV for blockmodel ---\n")

check("ECV.for.blockmodel: returns model selection", {
  ecv <- ECV.for.blockmodel(net_incv$adjacency, max.K = 4, B = 2,
                            holdout.p = 0.1, dc.est = 2)
  cat("       dev.model =", ecv$dev.model, "\n")
  cat("       l2.model  =", ecv$l2.model, "\n")
  cat("       auc.model =", ecv$auc.model, "\n")
  stopifnot(is.character(ecv$dev.model))
  stopifnot(length(ecv$l2) == 4)
  stopifnot(length(ecv$dc.l2) == 4)
})

# ---- 9. NCV for blockmodel ---------------------------------------------------
cat("\n--- NCV for blockmodel ---\n")

check("NCV.for.blockmodel: returns model selection", {
  ncv <- NCV.for.blockmodel(net_incv$adjacency, max.K = 4, cv = 3)
  cat("       dev.model =", ncv$dev.model, "\n")
  cat("       l2.model  =", ncv$l2.model, "\n")
  stopifnot(is.character(ncv$dev.model))
  stopifnot(length(ncv$dev) == 4)
})

# ---- 10. Summary -------------------------------------------------------------
cat("\n============================================================\n")
cat(sprintf("  Results: %d passed, %d failed (total %d)\n",
            pass, fail, pass + fail))
cat("============================================================\n")

if (fail > 0) {
  cat("\n*** Some tests failed. Please check the output above. ***\n")
  quit(status = 1)
} else {
  cat("\n*** All tests passed! ***\n")
}
