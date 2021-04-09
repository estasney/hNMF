import pytest

from hNMF import model as testmodel
from sklearn.datasets import make_sparse_spd_matrix
from scipy.stats import beta
import unittest
import numpy as np


@pytest.fixture(scope='module')
def sample_data():
    n_samples = np.random.randint(100, 30000)
    features_coeff = np.random.choice(np.linspace(0.1, 3))
    n_features = int(n_samples * features_coeff)
    alpha = beta(10, 1).rvs()
    X = np.abs(make_sparse_spd_matrix(dim=n_samples, alpha=alpha, norm_diag=False,
                                      smallest_coef=0.1, largest_coef=0.7))
    k = np.random.randint(5, 50)
    return X, k


def test_hnmf_inits(sample_data):
    X, k = sample_data
    model = testmodel.HierarchicalNMF(k)


def test_hnmf_runs(sample_data):
    X, k = sample_data
    model = testmodel.HierarchicalNMF(k)
    model.fit(X)
