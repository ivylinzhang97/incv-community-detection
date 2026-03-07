"""
Example: International Trade Network (Section 5.1)

Reproduces the INCV analysis of the international trade dataset.
"""

import numpy as np
from incv import (
    load_trade_data,
    nscv_f_fold,
    sbm_spectral_clustering,
    plot_cv_loss,
    plot_network,
)


def main():
    # Load data and build adjacency matrix
    A, W, attrs = load_trade_data(year=20, quantile=0.75)
    N = A.shape[0]
    sparsity = A.sum() / N ** 2
    print(f"Trade network: N={N}, sparsity={sparsity:.4f}")

    # Run 10-fold INCV
    k_vec = list(range(2, 11))
    np.random.seed(2026)
    rng = np.random.default_rng(2026)
    result = nscv_f_fold(A, k_vec=k_vec, restricted=True, f=10,
                         method="affinity", p_est_type=3, rng=rng)

    print(f"\nINCV results:")
    print(f"  K selected by NLL: {result['k_loss']}")
    print(f"  K selected by MSE: {result['k_mse']}")
    print(f"  CV Loss (NLL): {result['cv_loss']}")
    print(f"  CV MSE:         {result['cv_mse']}")

    # Plot CV loss curves
    fig, _ = plot_cv_loss(
        k_vec, result["cv_loss"], result["cv_mse"],
        k_best_loss=result["k_loss"], k_best_mse=result["k_mse"],
        title_prefix="International Trade INCV",
        save_path="trade_cv_loss.png",
    )
    print("\nSaved: trade_cv_loss.png")

    # Cluster with selected K and plot network
    labels = sbm_spectral_clustering(A, k=result["k_loss"])
    country_names = list(attrs["country"]) if attrs is not None else None

    try:
        fig, _ = plot_network(
            A, labels=labels, node_names=country_names,
            colors=["#87CEEB", "#9ACD32", "#FA8072", "#FFD700"],
            layout="spring", title="International Trade Network (INCV)",
            save_path="trade_network.png",
        )
        print("Saved: trade_network.png")
    except ImportError:
        print("(networkx not installed – skipping network plot)")


if __name__ == "__main__":
    main()
