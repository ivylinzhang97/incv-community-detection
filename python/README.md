# incv-community-detection

Inductive Node-Splitting Cross-Validation (INCV) for community detection in network data.

## Overview

This package implements **Inductive Node-Splitting Cross-Validation (INCV)** for selecting the number of communities in Stochastic Block Models (SBMs). It also provides competing methods — **Edge Cross-Validation (ECV)** and **Node Cross-Validation (NCV)** — for comprehensive model selection in network analysis.

### Key features

- **INCV (f-fold and random split)**: Node-level cross-validation that splits nodes into folds, fits spectral clustering on training nodes, infers held-out node communities, and evaluates via negative log-likelihood and MSE.
- **ECV**: Edge holdout cross-validation for blockmodel selection.
- **NCV**: Node holdout cross-validation for blockmodel selection.
- **Network simulation**: Generators for SBM and planted-partition models.
- **Real-data applications**: Data loaders for International Trade and 108th U.S. Senate datasets.
- **Visualization**: CV loss curve plots, network graphs with community coloring.

## Installation

```bash
# From PyPI
pip install incv-community-detection

# Or install from GitHub
pip install git+https://github.com/ivylinzhang97/incv-community-detection.git

# With optional network plotting support
pip install "incv-community-detection[network]"
```

## Quick start

### Simulate a network and select K with INCV

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

### Compare with ECV and NCV

```python
from incv import ecv_block, ncv_select

ecv = ecv_block(A, max_K=6, B=5, rng=rng)
print(f"ECV model: {ecv['l2_model']}")

ncv = ncv_select(A, max_K=6, cv=3, rng=rng)
print(f"NCV model: {ncv['l2_model']}")
```

### Plot CV loss curves

```python
from incv import plot_cv_loss

plot_cv_loss(
    list(range(2, 8)), result["cv_loss"], result["cv_mse"],
    k_best_loss=result["k_loss"], k_best_mse=result["k_mse"],
    save_path="cv_loss.png",
)
```

## Package structure

| Module | Contents |
|--------|----------|
| `incv.core` | `community_sim()`, `community_sim_sbm()`, `sbm_spectral_clustering()`, `sbm_prob()`, `nscv_f_fold()`, `nscv_random_split()` |
| `incv.competitors` | `ecv_block()`, `ncv_select()` |
| `incv.simulations` | `sim_folds()`, `sim_community()`, `sim_compare()` |
| `incv.applications` | `load_trade_data()`, `load_senate_data()` |
| `incv.plotting` | `plot_cv_loss()`, `plot_network()`, `plot_fold_comparison()`, `plot_community_comparison()` |

## Dependencies

**Required**: numpy, scipy, scikit-learn, pandas, matplotlib

**Optional** (for network plots): networkx

## License

MIT

## Citation

If you use this package in your research, please cite the paper on Inductive Node-Splitting Cross-Validation in Networks.
