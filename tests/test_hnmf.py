import numpy.typing as npt
import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from hnmf import HierarchicalNMF


@pytest.fixture(scope="module")
def sample_data() -> tuple[npt.NDArray, int, dict[int, str]]:
    n_features = 1000
    n_leaves = 20

    data, _ = fetch_20newsgroups(
        shuffle=True,
        random_state=1,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )

    # Use tf-idf features for NMF.
    tfidf = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english",
    )

    X = tfidf.fit_transform(data)
    id2feature = dict(enumerate(tfidf.get_feature_names_out()))
    return X, n_leaves, id2feature


def test_hnmf_can_fit(sample_data):
    """Given dataset
    Check that model can fit to leaves
    Model will not be able to completely fit if regularization is not called as intended
    """
    X, n_leaves, _id2feature = sample_data
    model = HierarchicalNMF(k=n_leaves)
    model.fit(X)
    assert model.n_leaves_ == n_leaves
