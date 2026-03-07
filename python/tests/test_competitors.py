"""Tests for incv.competitors — NCV and ECV methods."""

import numpy as np
import pytest
from incv import community_sim, ncv_select, ecv_block


@pytest.fixture
def network():
    rng = np.random.default_rng(42)
    _, A = community_sim(k=3, n=100, n1=30, p=0.3, q=0.05, rng=rng)
    return A


class TestNCV:
    def test_ncv_returns_expected_keys(self, network):
        rng = np.random.default_rng(42)
        result = ncv_select(network, max_K=4, cv=3, rng=rng)
        expected_keys = {"dev", "l2", "auc", "dc_dev", "dc_l2", "dc_auc",
                         "l2_model", "dev_model", "auc_model"}
        assert expected_keys.issubset(result.keys())

    def test_ncv_loss_arrays_length(self, network):
        rng = np.random.default_rng(42)
        result = ncv_select(network, max_K=5, cv=3, rng=rng)
        assert len(result["l2"]) == 5
        assert len(result["dev"]) == 5

    def test_ncv_model_string_format(self, network):
        rng = np.random.default_rng(42)
        result = ncv_select(network, max_K=4, cv=3, rng=rng)
        for key in ["l2_model", "dev_model", "auc_model"]:
            assert result[key].startswith("SBM-") or result[key].startswith("DCSBM-")


class TestECV:
    def test_ecv_returns_expected_keys(self, network):
        rng = np.random.default_rng(42)
        result = ecv_block(network, max_K=4, B=2, rng=rng)
        expected_keys = {"l2", "dev", "dc_l2", "dc_dev", "l2_model", "dev_model"}
        assert expected_keys.issubset(result.keys())

    def test_ecv_loss_arrays_length(self, network):
        rng = np.random.default_rng(42)
        result = ecv_block(network, max_K=5, B=2, rng=rng)
        assert len(result["l2"]) == 5
        assert len(result["dev"]) == 5

    def test_ecv_model_string_format(self, network):
        rng = np.random.default_rng(42)
        result = ecv_block(network, max_K=4, B=2, rng=rng)
        for key in ["l2_model", "dev_model"]:
            assert result[key].startswith("SBM-") or result[key].startswith("DCSBM-")
