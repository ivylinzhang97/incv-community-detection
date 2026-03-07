# INCV Community Detection

Inductive Node-Splitting Cross-Validation (INCV) for community detection in network data.

Available in both **R** and **Python**.

## Overview

This repository implements **Inductive Node-Splitting Cross-Validation (INCV)** for selecting the number of communities in Stochastic Block Models (SBMs). It also provides competing methods — **CROISSANT**, **Edge Cross-Validation (ECV)**, and **Node Cross-Validation (NCV)** — for comprehensive model selection in network analysis.

### Key features

- **INCV (f-fold and random split)**: Node-level cross-validation that splits nodes into folds, fits spectral clustering on training nodes, infers held-out node communities, and evaluates via negative log-likelihood and MSE.
- **CROISSANT**: Overlapping subsample-based cross-validation for joint selection of model type (SBM vs DCBM) and number of communities.
- **ECV**: Edge holdout cross-validation for blockmodel selection.
- **NCV**: Node holdout cross-validation for blockmodel selection.
- **Network simulation**: Generators for SBM, DCBM, RDPG, and latent space models.
- **Multiple loss functions**: L2, binomial deviance, and AUC.

---

## R Package

### Installation

```r
# From CRAN (once accepted)
install.packages("INCVCommunityDetection")

# Or install the development version from GitHub
devtools::install_github("ivylinzhang97/incv-community-detection")
```

### Quick start

```r
library(INCVCommunityDetection)

set.seed(42)
net <- community.sim(k = 3, n = 300, n1 = 100, p = 0.3, q = 0.05)

# F-fold INCV
result <- nscv.f.fold(net$adjacency, k.vec = 2:6, f = 10)
cat("Selected K (loss):", result$k.loss, "\n")
cat("Selected K (MSE):", result$k.mse, "\n")
```

### R package structure

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

---

## Python Package

### Installation

```bash
# From PyPI (once published)
pip install incv-community-detection

# Or install from GitHub
pip install "git+https://github.com/ivylinzhang97/incv-community-detection.git#subdirectory=python"

# With optional network plotting support
pip install "incv-community-detection[network]"
```

### Quick start

```python
import numpy as np
from incv import community_sim, nscv_f_fold

rng = np.random.default_rng(42)
membership, A = community_sim(k=3, n=300, n1=60, p=0.3, q=0.1, rng=rng)

# Run 10-fold INCV
result = nscv_f_fold(A, k_vec=list(range(2, 8)), f=10, rng=rng)
print(f"Selected K (NLL): {result['k_loss']}")
print(f"Selected K (MSE): {result['k_mse']}")
```

### Python package structure

| Module | Contents |
|--------|----------|
| `incv.core` | `community_sim()`, `community_sim_sbm()`, `sbm_spectral_clustering()`, `sbm_prob()`, `nscv_f_fold()`, `nscv_random_split()` |
| `incv.competitors` | `ecv_block()`, `ncv_select()` |
| `incv.simulations` | `sim_folds()`, `sim_community()`, `sim_compare()` |
| `incv.applications` | `load_trade_data()`, `load_senate_data()` |
| `incv.plotting` | `plot_cv_loss()`, `plot_network()`, `plot_fold_comparison()`, `plot_community_comparison()` |

---

## Dependencies

**R**: Matrix, RSpectra, ClusterR, irlba, parallel, cluster, Rfast, data.table, IMIFA; optional: latentnet, rdist

**Python**: numpy, scipy, scikit-learn, pandas, matplotlib; optional: networkx

## License

MIT
