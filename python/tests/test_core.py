"""Tests for incv.core — simulation, spectral clustering, SBM prob, INCV."""

import numpy as np
import pytest
from incv import (
    community_sim,
    community_sim_sbm,
    sbm_spectral_clustering,
    sbm_prob,
    nscv_f_fold,
    nscv_random_split,
    edge_index_map,
    neglog,
)


class TestHelpers:
    def test_edge_index_map_basic(self):
        x, y = edge_index_map(np.array([1, 2, 3]))
        assert x.shape == (3,)
        assert y.shape == (3,)
        assert np.all(x < y)

    def test_neglog_zero_prob(self):
        assert neglog(10, 0) == 0.0

    def test_neglog_positive(self):
        val = neglog(5, 0.5)
        assert val == pytest.approx(5 * np.log(2))


class TestSimulation:
    def test_community_sim_shape(self):
        rng = np.random.default_rng(42)
        mem, A = community_sim(k=3, n=100, n1=20, p=0.3, q=0.1, rng=rng)
        assert mem.shape == (100,)
        assert A.shape == (100, 100)
        assert np.allclose(A, A.T)
        assert np.all(np.diag(A) == 0)

    def test_community_sim_labels(self):
        rng = np.random.default_rng(42)
        mem, _ = community_sim(k=4, n=200, n1=30, p=0.3, q=0.1, rng=rng)
        assert set(np.unique(mem)) == {1, 2, 3, 4}

    def test_community_sim_sbm(self):
        rng = np.random.default_rng(42)
        A, c0, B0 = community_sim_sbm(n=100, n1=20, eta=0.3, rho=0.1, K=3, rng=rng)
        assert A.shape == (100, 100)
        assert np.allclose(A, A.T)
        assert c0.shape == (100,)
        assert B0.shape == (3, 3)


class TestSpectralClustering:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        _, A = community_sim(k=3, n=100, n1=30, p=0.3, q=0.05, rng=rng)
        labels = sbm_spectral_clustering(A, k=3)
        assert labels.shape == (100,)
        assert set(np.unique(labels)).issubset({1, 2, 3})

    def test_k1(self):
        rng = np.random.default_rng(42)
        _, A = community_sim(k=2, n=50, n1=25, p=0.3, q=0.1, rng=rng)
        labels = sbm_spectral_clustering(A, k=1)
        assert np.all(labels == 1)


class TestSBMProb:
    def test_restricted_prob(self):
        rng = np.random.default_rng(42)
        _, A = community_sim(k=2, n=100, n1=50, p=0.5, q=0.1, rng=rng)
        labels = sbm_spectral_clustering(A, k=2)
        P, nll = sbm_prob(labels, 2, A, restricted=True)
        assert P.shape == (2, 2)
        assert nll >= 0
        assert P[0, 0] == P[1, 1]  # restricted: single p value
        assert P[0, 1] == P[1, 0]

    def test_unrestricted_prob(self):
        rng = np.random.default_rng(42)
        _, A = community_sim(k=2, n=100, n1=50, p=0.5, q=0.1, rng=rng)
        labels = sbm_spectral_clustering(A, k=2)
        P, nll = sbm_prob(labels, 2, A, restricted=False)
        assert P.shape == (2, 2)
        assert nll >= 0


class TestINCV:
    @pytest.fixture
    def network(self):
        rng = np.random.default_rng(42)
        _, A = community_sim(k=3, n=150, n1=40, p=0.3, q=0.05, rng=rng)
        return A

    def test_f_fold_returns_expected_keys(self, network):
        rng = np.random.default_rng(42)
        result = nscv_f_fold(network, k_vec=[2, 3, 4], f=5, rng=rng)
        assert "k_loss" in result
        assert "k_mse" in result
        assert "cv_loss" in result
        assert "cv_mse" in result
        assert len(result["cv_loss"]) == 3
        assert len(result["cv_mse"]) == 3

    def test_f_fold_selects_correct_k(self, network):
        rng = np.random.default_rng(42)
        result = nscv_f_fold(network, k_vec=[2, 3, 4, 5], f=10, rng=rng)
        assert result["k_loss"] in [2, 3, 4, 5]
        assert result["k_mse"] in [2, 3, 4, 5]

    def test_random_split_returns_expected_keys(self, network):
        rng = np.random.default_rng(42)
        result = nscv_random_split(network, k_vec=[2, 3, 4], ite=10, rng=rng)
        assert "k_chosen" in result
        assert "cv_loss" in result
        assert len(result["cv_loss"]) == 3

    def test_random_split_selects_correct_k(self, network):
        rng = np.random.default_rng(42)
        result = nscv_random_split(network, k_vec=[2, 3, 4, 5], ite=20, rng=rng)
        assert result["k_chosen"] in [2, 3, 4, 5]
