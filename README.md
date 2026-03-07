# INCVCommunityDetection

Inductive Node-Splitting Cross-Validation (INCV) for community detection in network data.

## Overview

This package implements **Inductive Node-Splitting Cross-Validation (INCV)** for selecting the number of communities in Stochastic Block Models (SBMs). It also provides competing methods — **CROISSANT**, **Edge Cross-Validation (ECV)**, and **Node Cross-Validation (NCV)** — for comprehensive model selection in network analysis.

### Key features

- **INCV (f-fold and random split)**: Node-level cross-validation that splits nodes into folds, fits spectral clustering on training nodes, infers held-out node communities, and evaluates via negative log-likelihood and MSE.
- **CROISSANT**: Overlapping subsample-based cross-validation for joint selection of model type (SBM vs DCBM) and number of communities.
- **ECV**: Edge holdout cross-validation for blockmodel selection.
- **NCV**: Node holdout cross-validation for blockmodel selection.
- **Network simulation**: Generators for SBM, DCBM, RDPG, and latent space models.
- **Multiple loss functions**: L2, binomial deviance, and AUC.

## Installation

```r
# From CRAN
install.packages("INCVCommunityDetection")

# Or install the development version from GitHub
devtools::install_github("ivylinzhang97/incv-community-detection")
```

## Quick start

### Simulate a network and select K with INCV

```r
library(INCVCommunityDetection)

set.seed(42)
net <- community.sim(k = 3, n = 300, n1 = 100, p = 0.3, q = 0.05)

# F-fold INCV
result <- nscv.f.fold(net$adjacency, k.vec = 2:6, f = 10)
cat("Selected K (loss):", result$k.loss, "\n")
cat("Selected K (MSE):", result$k.mse, "\n")

# Random-split INCV
result2 <- nscv.random.split(net$adjacency, k.vec = 2:6, ite = 50)
cat("Selected K:", result2$k.chosen, "\n")
```

### Compare with ECV and NCV

```r
# Edge Cross-Validation
ecv <- ECV.for.blockmodel(net$adjacency, max.K = 6, B = 5)
cat("ECV model (deviance):", ecv$dev.model, "\n")

# Node Cross-Validation
ncv <- NCV.for.blockmodel(net$adjacency, max.K = 6, cv = 3)
cat("NCV model (deviance):", ncv$dev.model, "\n")
```

### CROISSANT for joint SBM/DCBM selection

```r
cr <- croissant.blockmodel(net$adjacency, K.CAND = 1:6,
                           s = 3, o = 100, R = 2, ncore = 1)
cat("CROISSANT (L2):", cr$l2.model, "\n")
```

## Package structure

| File | Contents |
|------|----------|
| `R/incv.R` | `nscv.f.fold()`, `nscv.random.split()` — core INCV methods |
| `R/croissant.R` | `croissant.blockmodel()`, `croissant.rdpg()`, `croissant.latent()`, `croissant.tune.regsp()` |
| `R/ecv.R` | `ECV.for.blockmodel()`, `ECV.undirected.Rank()` |
| `R/ncv.R` | `NCV.for.blockmodel()` |
| `R/spectral.R` | `SBM.spectral.clustering()`, `SBM.prob()` |
| `R/estimation.R` | `fast.SBM.est()`, `fast.DCBM.est()`, `eigen.DCBM.est()`, label matching |
| `R/simulation.R` | `community.sim()`, `community.sim.sbm()`, `blockmodel.gen.fast()`, `sparse.RDPG.gen()`, `latent.gen()` |
| `R/utils.R` | `edge.index.map()`, `neglog()`, `l2()`, `bin.dev()`, `AUC()` |

## Dependencies

**Required**: Matrix, RSpectra, ClusterR, irlba, parallel, cluster, Rfast, data.table, IMIFA

**Optional** (for latent space methods): latentnet, rdist

## License

MIT
