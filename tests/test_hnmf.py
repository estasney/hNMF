from hNMF import model as testmodel
from sklearn.datasets import make_sparse_spd_matrix
from scipy.stats import beta
import unittest
import numpy as np


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.n_samples = np.random.randint(100, 30000)
        self.features_coeff = np.random.choice(np.linspace(0.1, 3))
        self.n_features = int(self.n_samples * self.features_coeff)
        self.alpha = beta(10, 1).rvs()
        self.X = np.abs(make_sparse_spd_matrix(dim=self.n_samples, alpha=self.alpha, norm_diag=False,
                                               smallest_coef=0.1, largest_coef=0.7))
        self.k = np.random.randint(5, 50)

    def test_hnmf_inits(self):
        try:
            model = testmodel.HierarchicalNMF(k=self.k)
        except:
            raise AssertionError

    def test_hnmf_runs(self):
        try:
            model = testmodel.HierarchicalNMF(k=self.k)
            model.fit(self.X)
        except:
            raise AssertionError


if __name__ == '__main__':
    unittest.main()
