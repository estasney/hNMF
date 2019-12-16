.. hNMF documentation master file, created by
   sphinx-quickstart on Sat Dec 14 14:55:20 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================================
|project|
===================================================

|project| implements Rank-2 NMF for Hierarchical Clustering as described in this `paper`_ and `repository`_.
|project| is a fork of `hierarchical-nmf-python`_ with several modifications:

- Interface to hNMF is provided with a scikit-learn compatible BaseEstimator
- Improved performance timings
- Convenience methods for interpreting results

.. _paper: https://smallk.github.io/papers/hierNMF2.pdf
.. _repository: https://github.com/dakuang/hiernmf2
.. _hierarchical-nmf-python: https://github.com/rudvlf0413/hierarchical-nmf-python

Reference
=========

.. toctree::
   :caption: Reference
   :hidden:

   hnmf

:ref:`hNMF`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citations
=========

.. [rank-2] Da Kuang, Haesun Park,
Fast rank-2 nonnegative matrix factorization for hierarchical document clustering,
The 19th ACM SIGKDD International Conference on Knowledge, Discovery, and Data Mining (KDD '13), pp. 739-747, 2013.
