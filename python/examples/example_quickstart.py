"""
Example: Quick simulation to verify INCV works.

Generates a simple planted-partition network and runs INCV to recover K.
"""

import numpy as np
from incv import community_sim, nscv_f_fold, sbm_spectral_clustering, plot_cv_loss


def main():
    # Simulate a network with K=3 communities
    rng = np.random.default_rng(42)
    membership, A = community_sim(k=3, n=300, n1=60, p=0.3, q=0.1, rng=rng)
    print(f"Simulated network: n={A.shape[0]}, true K=3")
    print(f"  Sparsity: {A.sum() / A.shape[0]**2:.4f}")

    # Run 10-fold INCV
    k_vec = list(range(2, 8))
    result = nscv_f_fold(A, k_vec=k_vec, restricted=True, f=10,
                         method="affinity", p_est_type=3, rng=rng)

    print(f"\nINCV results:")
    print(f"  K selected by NLL: {result['k_loss']}")
    print(f"  K selected by MSE: {result['k_mse']}")
    for i, k in enumerate(k_vec):
        print(f"    K={k}: NLL={result['cv_loss'][i]:.4f}, MSE={result['cv_mse'][i]:.4f}")

    # Plot
    fig, _ = plot_cv_loss(
        k_vec, result["cv_loss"], result["cv_mse"],
        k_best_loss=result["k_loss"], k_best_mse=result["k_mse"],
        title_prefix="Simulation (true K=3)",
        save_path="sim_quickstart.png",
    )
    print("\nSaved: sim_quickstart.png")


if __name__ == "__main__":
    main()
