import numpy.typing as npt
import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from hnmf import HierarchicalNMF


@pytest.fixture(scope="module")
def sample_data() -> tuple[npt.NDArray, dict[int, str]]:
    n_features = 1000
    data, _ = fetch_20newsgroups(
        shuffle=True,
        random_state=1,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )

    tfidf = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english",
    )

    X = tfidf.fit_transform(data)
    id2feature = dict(enumerate(tfidf.get_feature_names_out()))
    return X, id2feature

@pytest.mark.parametrize("k", [20, 1000])
def test_hnmf_can_fit(sample_data, k: int):
    """
    Given dataset
    Check that model successfully fits the data
    Ensure that the number of leaves does not exceed k
    """
    X, _id2feature = sample_data
    model = HierarchicalNMF(k=k)
    model.fit(X)
    assert 2 <= model.n_leaves_ <= k
    assert model.n_samples_ == X.shape[0]
    assert model.n_features_ == X.shape[1]
    assert model.clusters_.shape[0] == model.n_nodes_
    assert model.is_leaf_.shape[0] == model.clusters_.shape[0]
    assert model.n_samples_ == X.shape[0]
