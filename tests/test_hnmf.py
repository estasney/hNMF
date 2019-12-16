from hnmf import model
import unittest
from sklearn.datasets import make_blobs


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.module = model.HierarchicalNMF
        self.model = None
        self.X = None
        self.output = None

    def test_hnmf_inits(self):
        try:
            self.model = self.module(k=10)
        except:
            raise AssertionError

    def test_hnmf_runs(self):
        try:
            model = self.module(k=10)
            X = make_blobs(n_samples=1000, n_features=5000)[0]
            model.fit(X)
        except:
            raise AssertionError



if __name__ == '__main__':
    unittest.main()
