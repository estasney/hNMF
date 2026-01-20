# hNMF

hNMF implements Rank-2 NMF for Hierarchical Clustering as described in this [paper](https://smallk.github.io/papers/hierNMF2.pdf) and [repository](https://github.com/dakuang/hiernmf2).

hNMF is a fork of [hierarchical-nmf-python](https://github.com/rudvlf0413/hierarchical-nmf-python) with several modifications:

- Interface to hNMF is provided with a scikit-learn compatible BaseEstimator
- Improved performance timings
- Convenience methods for interpreting results

## Performance

The paper mentions that the hierarchical NMF process takes advantage of a *fast* 2-rank matrix decomposition, While this may be true in MATLAB, the original Python implementation was significantly bottlenecked when running the 2-rank decomposition.

## Citations

[1] Da Kuang, Haesun Park, *Fast rank-2 nonnegative matrix factorization for hierarchical document clustering*, The 19th ACM SIGKDD International Conference on Knowledge, Discovery, and Data Mining (KDD '13), pp. 739-747, 2013.