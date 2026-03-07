"""Tests for incv.simulations — simulation runners."""

import pytest
from incv.simulations import sim_folds, sim_community, sim_compare


class TestSimFolds:
    def test_returns_dataframe(self):
        df = sim_folds(
            seed=1,
            n_set_default=[100],
            n_set_hard=[200],
            p_set=[0.3],
            q_set=[0.1],
            fold_set=[5],
            verbose=False,
        )
        assert len(df) == 1
        assert "k_loss" in df.columns
        assert "k_mse" in df.columns


class TestSimCommunity:
    def test_returns_dataframe(self):
        df = sim_community(
            seed=1, n=100,
            k_set=[2],
            q_set=[0.1],
            n1_func=lambda k: [30],
            verbose=False,
        )
        assert len(df) == 1
        assert "k_loss" in df.columns


class TestSimCompare:
    def test_returns_dataframe(self):
        df = sim_compare(
            seed=1, n=100,
            k_set=[3],
            rho_set=[0.1],
            n1_func=lambda k: [30],
            verbose=False,
        )
        assert len(df) == 1
        assert "k_loss" in df.columns
        assert "k_ncv" in df.columns
        assert "k_ecv" in df.columns
