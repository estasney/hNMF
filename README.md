# hierarchical-nmf-python
* fork of https://github.com/rudvlf0413/hierarchical-nmf-python
* with familiar SKLearn interface

## Installation
```bash
pip install hnmf
```

## Usage
### 20 Newsgroups
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from hnmf.model import HierarchicalNMF

n_features = 1000
n_leaves = 20

data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)


# Use tf-idf features for NMF.
tfidf = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')

X = tfidf.fit_transform(data)
id2feature = {i: token for i, token in enumerate(tfidf.get_feature_names())}

# hNMF
model = HierarchicalNMF(k=n_leaves)
model.fit(X)
model.cluster_features(id2feature=id2feature)
```

## Reference
- Papers: [Fast rank-2 nonnegative matrix factorization for hierarchical document clustering](https://smallk.github.io/papers/hierNMF2.pdf)

- Original version of codes (matlab): https://github.com/dakuang/hiernmf2
