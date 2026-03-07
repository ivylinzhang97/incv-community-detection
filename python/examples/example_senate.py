"""
Example: 108th U.S. Senate Co-sponsorship Network (Section 5.2)

Reproduces the INCV analysis of the senate co-sponsorship dataset.
"""

import numpy as np
from incv import (
    load_senate_data,
    nscv_f_fold,
    sbm_spectral_clustering,
    plot_cv_loss,
    plot_network,
)


def main():
    # Load data and build adjacency matrix
    A, W, attrs = load_senate_data(threshold=4)
    N = A.shape[0]
    sparsity = A.sum() / N ** 2
    print(f"Senate network: N={N}, sparsity={sparsity:.4f}")

    # Run 10-fold INCV
    k_vec = list(range(2, 11))
    rng = np.random.default_rng(33)
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
        title_prefix="108th Senate INCV",
        save_path="senate_cv_loss.png",
    )
    print("\nSaved: senate_cv_loss.png")

    # Cluster with K=2 and plot network
    rng2 = np.random.default_rng(12345)
    labels = sbm_spectral_clustering(A, k=2)
    senator_names = list(attrs["senate_member"]) if attrs is not None else None

    try:
        fig, _ = plot_network(
            A, labels=labels, node_names=senator_names,
            colors=["blue", "red"],
            layout="kamada_kawai",
            title="108th Senate Network (INCV, K=2)",
            save_path="senate_network.png",
        )
        print("Saved: senate_network.png")
    except ImportError:
        print("(networkx not installed – skipping network plot)")

    # Print cluster membership
    if senator_names is not None:
        for c in [1, 2]:
            members = [senator_names[i] for i in range(N) if labels[i] == c]
            print(f"\nCluster {c} ({len(members)} senators):")
            for m in members:
                print(f"  {m}")


if __name__ == "__main__":
    main()
