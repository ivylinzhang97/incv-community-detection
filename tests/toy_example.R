###############################################################################
## INCVCommunityDetection — Toy Example
##
## A complete walkthrough for new users demonstrating how to simulate a
## network, select the number of communities with INCV, and compare with
## ECV and NCV.
##
## Usage:
##   library(INCVCommunityDetection)
##   source("tests/toy_example.R")
###############################################################################

library(INCVCommunityDetection)

# ── 1. Simulate a network with 3 communities ──────────────────────
set.seed(42)
net <- community.sim(
  k  = 3,      # 3 communities
  n  = 200,    # 200 nodes
  n1 = 60,     # smallest community has 60 nodes
  p  = 0.4,    # within-community connection probability
  q  = 0.05    # between-community connection probability
)

cat("True community sizes:\n")
print(table(net$membership))

# ── 2. Select K using INCV (f-fold) ───────────────────────────────
result <- nscv.f.fold(
  A        = net$adjacency,
  k.vec    = 2:7,
  f        = 10,
  method   = "affinity"
)

cat("\nINCV f-fold results:\n")
cat("  K selected by loss:", result$k.loss, "\n")
cat("  K selected by MSE: ", result$k.mse, "\n")

# Plot the CV loss curve
plot(2:7, result$cv.loss, type = "b", pch = 19, col = "steelblue",
     xlab = "Number of communities (K)",
     ylab = "CV Negative Log-Likelihood",
     main = "INCV: Selecting the number of communities")
abline(v = result$k.loss, lty = 2, col = "red")
legend("topright", legend = paste("Selected K =", result$k.loss),
       col = "red", lty = 2)

# ── 3. Select K using INCV (random split) ─────────────────────────
result2 <- nscv.random.split(
  A     = net$adjacency,
  k.vec = 2:7,
  split = 0.66,
  ite   = 30
)

cat("\nINCV random-split results:\n")
cat("  K selected (loss):", result2$k.loss, "\n")
cat("  K selected (MSE):", result2$k.mse, "\n")

# ── 4. Compare with ECV ───────────────────────────────────────────
ecv <- ECV.for.blockmodel(net$adjacency, max.K = 7, B = 3)
cat("\nECV results:\n")
cat("  Model (deviance):", ecv$dev.model, "\n")
cat("  Model (L2):      ", ecv$l2.model, "\n")

# ── 5. Compare with NCV ───────────────────────────────────────────
ncv <- NCV.for.blockmodel(net$adjacency, max.K = 7, cv = 3)
cat("\nNCV results:\n")
cat("  Model (deviance):", ncv$dev.model, "\n")
cat("  Model (L2):      ", ncv$l2.model, "\n")

# ── 6. Inspect the spectral clustering at the chosen K ────────────
cl <- SBM.spectral.clustering(net$adjacency, k = result$k.loss)
prob <- SBM.prob(cl$cluster, k = result$k.loss,
                 A = net$adjacency, restricted = TRUE)

cat("\nEstimated block probabilities:\n")
cat("  Within-community p: ", round(prob$p.matrix[1, 1], 3), "\n")
cat("  Between-community q:", round(prob$p.matrix[1, 2], 3), "\n")

# ── 7. Summary ────────────────────────────────────────────────────
cat("\n══ Summary ══════════════════════════════════════\n")
cat("  True K = 3\n")
cat("  INCV f-fold  selected K =", result$k.loss, "\n")
cat("  INCV random  selected K =", result2$k.loss, "\n")
cat("  ECV selected:", ecv$dev.model, "\n")
cat("  NCV selected:", ncv$dev.model, "\n")
