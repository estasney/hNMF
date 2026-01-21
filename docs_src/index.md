---
title: hNMF - Hierarchical Non-negative Matrix Factorization
---

# hNMF

hNMF implements Rank-2 NMF for Hierarchical Clustering as described in this [paper](https://smallk.github.io/papers/hierNMF2.pdf) and [repository](https://github.com/dakuang/hiernmf2).

hNMF is a fork of [hierarchical-nmf-python](https://github.com/rudvlf0413/hierarchical-nmf-python) with several modifications:

- Interface to hNMF is provided with a scikit-learn compatible BaseEstimator
- Improved performance timings
- Convenience methods for interpreting results

## Why hNMF?

Unlike flat NMF where you specify cluster count upfront, hNMF discovers it through successive splitting using a coherence threshold. In practice, this means you don't need to guess the number of topics or account for nuances in topic granularity. Instead, you can rapidly iterate to find a coherence level that yields meaningful topics for your dataset.

## Installation

```bash
uv add hnmf
```

## Getting Started

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from hnmf import HierarchicalNMF

# Load 20newsgroups dataset
data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

# Vectorize documents
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
X = tfidf.fit_transform(data)

# Fit hierarchical NMF
model = HierarchicalNMF(k=20)
model.fit(X)

# Examine top words across all leaf nodes
feature_names = tfidf.get_feature_names_out()
top_features = model.top_features_in_nodes(n=10, leaves_only=True)

for node_id, features in top_features.items():
    words = [feature_names[idx] for idx, weight in features]
    print(f"Node {node_id}: {words}")
```

The model automatically discovers topic hierarchies:

```
Node 3: ['game', 'team', 'year', 'games', 'season', 'players', 'play', 'hockey', 'win', 'good']
Node 7: ['don', 'know', 'think', 'let', 'want', 'say', 'read', 'mean', 'wrong', 'try']
Node 8: ['feel', 'later', 'unless', 'started', 'gets', 'easy', 'job', 'worth', 'past', 'goes']
Node 9: ['usually', 'short', 'level', 'thinking', 'extra', 'inside', 'needs', 'happens', 'field', 'near']
Node 11: ['drive', 'scsi', 'hard', 'disk', 'drives', 'controller', 'card', 'ide', 'floppy', 'new']
Node 13: ['edu', 'com', 'university', 'cs', 'mail', 'email', 'send', 'internet', 'address', 'ftp']
Node 15: ['thanks', 'does', 'mail', 'advance', 'looking', 'hi', 'info', 'anybody', 'information', 'help']
Node 17: ['windows', 'file', 'dos', 'files', 'window', 'program', 'running', 'version', 'ftp', 'drivers']
Node 19: ['key', 'chip', 'keys', 'bit', 'number', 'algorithm', 'phone', 'chips', 'serial', 'bits']
Node 21: ['new', '00', 'price', '20', '50', '15', '30', 'buy', 'asking', 'used']
Node 23: ['government', 'law', 'encryption', 'clipper', 'public', 'federal', 'clinton', 'rights', 'state', 'enforcement']
Node 25: ['gun', 'guns', 'control', 'crime', 'weapons', 'firearms', 'carry', '000', 'self', 'rate']
Node 27: ['said', 'away', 'mr', 'children', 'says', 'jim', 'news', 'dead', 'today', 'asked']
Node 29: ['state', 'states', 'order', 'country', 'land', 'united', 'live', 'population', 'based', 'california']
Node 31: ['space', 'shuttle', 'cost', 'data', 'satellite', 'available', 'technology', 'commercial', 'technical', 'project']
Node 33: ['monitor', 'video', 'color', 'vga', 'memory', 'cards', 'graphics', 'mode', 'display', '16']
Node 34: ['used', 'using', 'software', 'data', 'available', 'computer', 'code', 'set', 'line', 'modem']
Node 35: ['work', 'works', 'fine', 'installed', 'tried', 'working', 'line', 'sun', 'recently', 'set']
Node 36: ['fact', 'person', 'reason', 'agree', 'order', 'non', 'place', 'means', 'example', 'children']
Node 37: ['science', 'nasa', 'research', 'scientific', 'dr', 'values', 'process', 'organization', 'study', 'provide']
```

Some nodes capture clear topics (sports, hardware, government), while others appear less interpretable (nodes 7, 8, 9).



## Performance

The paper mentions that the hierarchical NMF process takes advantage of a *fast* 2-rank matrix decomposition, While this may be true in MATLAB, the original Python implementation was significantly bottlenecked when running the 2-rank decomposition.

## Citations

[1] Da Kuang, Haesun Park, *Fast rank-2 nonnegative matrix factorization for hierarchical document clustering*, The 19th ACM SIGKDD International Conference on Knowledge, Discovery, and Data Mining (KDD '13), pp. 739-747, 2013.