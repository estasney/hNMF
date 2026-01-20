import logging
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF

from hnmf.helpers import (
    trial_split_sklearn,
)
from hnmf.progress_tree import ProgressTree


@dataclass(frozen=True, slots=True)
class DiscriminatedSample:
    sample: Any
    node: int
    node_value: float
    others_value: float


logger = logging.getLogger(__name__)


class HierarchicalNMF(BaseEstimator):
    k: int
    unbalanced: float
    init: Literal[None, "random", "nndsvd", "nndsvda", "nndsvdar"]
    solver: Literal["cd", "mu"]
    beta_loss: Literal["FRO", 0, "KL", 1, "IS", 2]
    alpha_W: float
    alpha_H: Literal["same"] | float
    random_state: np.random.RandomState
    trial_allowance: int
    tol: float
    maxiter: int
    dtype: npt.DTypeLike
    n_samples_: int | None
    n_features_: int | None
    n_nodes_: int
    n_leaves_: int
    tree_: npt.NDArray | None
    splits_: npt.NDArray | None
    is_leaf_: npt.NDArray | None
    clusters_: npt.NDArray | None
    Ws_: npt.NDArray | None
    Hs_: npt.NDArray | None
    W_buffer_: npt.NDArray | None
    H_buffer_: npt.NDArray | None
    priorities_: npt.NDArray | None
    id2sample_: dict[int, str] | None
    id2feature_: dict[int, str] | None
    feature2id_: dict[str, int] | None

    def __init__(
        self,
        k: int,
        unbalanced: float = 0.1,
        init: Literal[None, "random", "nndsvd", "nndsvda", "nndsvdar"] = None,
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: Literal["FRO", 0, "KL", 1, "IS", 2] = 0,
        alpha_W: float = 0.0,
        alpha_H: Literal["same"] | float = "same",
        random_state: int = 42,
        trial_allowance: int = 100,
        tol: float = 1e-6,
        maxiter: int = 10000,
        dtype: npt.DTypeLike = np.float64,
    ):
        self.k = k
        self.unbalanced = unbalanced
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.random_state = np.random.RandomState(seed=random_state)
        self.trial_allowance = trial_allowance
        self.tol = tol
        self.maxiter = maxiter
        self.dtype = dtype

        self.n_samples_ = None
        self.n_features_ = None
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        self.tree_ = None
        self.splits_ = None
        self.is_leaf_ = None
        self.clusters_ = None
        self.Ws_ = None
        self.Hs_ = None
        self.W_buffer_ = None
        self.H_buffer_ = None
        self.priorities_ = None
        self.id2sample_ = None
        self.id2feature_ = None
        self.feature2id_ = None

    """
    Implements Hierarchical rank-2 NMF

    Parameters
    ----------

    k: int
        The number of desired leaf nodes
    unbalanced : float
        A threshold to determine if one of the two clusters is an outlier set. A smaller value means more tolerance for
        imbalance between two clusters. See parameter beta in Algorithm 3 in the reference paper.
    init : InitMethod
        The initialization method used to initially fill W and H
    solver : NMFSolver
        The solver used to minimize the distance function
    beta_loss : BetaLoss
        Beta divergence to be minimized
    alpha_W : float, defaults to 0.0
        Constant that multiplies the regularization terms of W. Set it to zero (default) to have no regularization on W.
        See `sklearn.decomposition.NMF`_ 
    alpha_H: float or 'same', defaults to 'same'
        Constant that multiplies the regularization terms of H. Set it to zero to have no regularization on H. If 'same'
         (default), it takes the same value as alpha_W.
        See `sklearn.decomposition.NMF`_
    random_state : int
        random seed
    trial_allowance : int
        Number of trials allowed for removing outliers and splitting a node again. See parameter T in Algorithm 3 in
        the reference paper.
    tol : float
        Tolerance parameter for stopping criterion in each run of NMF.
    maxiter : int
        Maximum number of iteration times in each run of NMF
    dtype : npt.DTypeLike
        Dtype used for numpy arrays 

    
    Attributes
    ----------
    tree_ : np.ndarray
        A 2-by-(k-1) matrix that encodes the tree structure. The two entries in the i-th column are the numberings of
        the two children of the node with numbering i. The root node has numbering 0, with its two children always
        having numbering 1 and numbering 2. Thus the root node is NOT included in the 'tree' variable.

    splits_ :
        An array of length k-1. It keeps track of the numberings of the nodes being split from the 1st split to the
        (k-1)-th split. (The first entry is always 0.)

    is_leaf_ :
        An array of length 2*(k-1). A "1" at index ``i`` means that the node with numbering ``i`` is a leaf node in the final
        tree generated, and "0" indicates non-leaf nodes in the final tree.

    clusters_ :
        Array with shape(n_nodes, n_features). A "1" at index ``i`` means that the sample with numbering ``c`` was
        included in this nodes subset

    
    Hs_ :
        Array with shape (n_nodes, n_features)

    Ws_ :
        Array with shape (n_nodes, n_samples)

    Notes
    -----

    ``W`` refers to the decomposed matrix. scikit-learn equivalent of::

        W = model.fit_transform(X)

    ``H`` refers to the factorization matrix. scikit-learn equivalent of::

        model.components_


    Adapted from [rank-2]_

    """

    def _init_fit(
        self, X: npt.NDArray, term_subset: npt.NDArray
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if not self.n_samples_:
            raise ValueError("n_samples_ not set before _init_fit called")

        nmf = NMF(
            n_components=2,
            random_state=self.random_state,
            tol=self.tol,
            max_iter=self.maxiter,
            init=self.init,
        )

        if len(term_subset) == self.n_samples_:
            W = nmf.fit_transform(X)
            H = nmf.components_
            return W, H

        W_tmp = nmf.fit_transform(X[term_subset, :])
        H = nmf.components_
        W = np.zeros((self.n_samples_, 2), dtype=self.dtype)
        W[term_subset, :] = W_tmp

        return W, H

    def fit(self, X: npt.NDArray) -> Self:
        """
        Fit `HierarchicalNMF` to data
        """
        shape: tuple[int, int] = X.shape
        n_samples, n_features = shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # TODO Expect different sized ranks
        clusters: list[npt.NDArray[np.int64] | None] = [None] * (2 * (self.k - 1))
        Ws = [None] * (2 * (self.k - 1))
        Hs = [None] * (2 * (self.k - 1))
        W_buffer = [None] * (2 * (self.k - 1))
        H_buffer = [None] * (2 * (self.k - 1))
        priorities = np.zeros(2 * (self.k - 1), dtype=self.dtype)
        is_leaf = np.zeros(2 * (self.k - 1), dtype=np.bool)  # No leaves at start
        tree = np.zeros((2, 2 * (self.k - 1)), dtype=np.int64)
        splits = -np.ones(self.k - 1, dtype=np.int64)

        # Where X has at least one non-zero
        term_subset = np.flatnonzero(np.sum(X, axis=1))

        W, H = self._init_fit(X, term_subset)

        result_used = 0

        with ProgressTree() as pt:
            for i in range(self.k - 1):
                if i == 0:
                    split_node = 0
                    new_nodes = [0, 1]
                    min_priority = 1e40
                    split_subset = np.arange(n_features)
                else:
                    leaves = np.where(is_leaf == 1)[0]
                    temp_priority = priorities[leaves]

                    if len(np.where(temp_priority > 0)[0]) > 0:
                        min_priority = np.min(temp_priority[temp_priority > 0])
                        split_node = np.argmax(temp_priority)
                    else:  # There are no more candidates stop early
                        min_priority = -1
                        split_node = 0

                    if temp_priority[split_node] < 0 or min_priority == -1:
                        logger.warning(
                            f"Cannot generate all {self.k} leaf clusters, stopping at {i} leaf clusters"
                        )

                        Ws = [i for i in Ws if i is not None]
                        W_buffer = [i for i in W_buffer if i is not None]

                        Hs = [i for i in Hs if i is not None]
                        H_buffer = [i for i in H_buffer if i is not None]

                        # Resize attributes
                        tree = tree[:, :result_used]
                        splits = splits[:result_used]
                        is_leaf = is_leaf[:result_used]
                        clusters = clusters[:result_used]
                        priorities = priorities[:result_used]

                        self.tree_ = tree.T
                        self.splits_ = splits
                        self.is_leaf_ = is_leaf
                        self.n_nodes_ = self.is_leaf_.shape[0]
                        self.n_leaves_ = int(np.count_nonzero(self.is_leaf_))
                        self.clusters_ = self._stack_clusters(clusters)
                        self.Ws_ = np.array(Ws)
                        self.Hs_ = np.array(Hs)
                        self.W_buffer_ = np.array(W_buffer)
                        self.H_buffer_ = self._stack_H_buffer(H_buffer)
                        self.priorities_ = priorities
                        return self

                    split_node = leaves[split_node]  # Attempt to split this node
                    is_leaf[split_node] = 0
                    W = W_buffer[split_node]
                    H = H_buffer[split_node]

                    # Find which features are clustered on this node
                    split_subset = clusters[split_node]
                    new_nodes = [result_used, result_used + 1]
                    tree[:, split_node] = new_nodes

                result_used += 2
                # For each row find where it is more greatly represented
                cluster_subset = np.argmax(H, axis=0)

                subset_0 = np.flatnonzero(cluster_subset == 0)
                subset_1 = np.flatnonzero(cluster_subset == 1)
                ls0 = len(subset_0)
                ls1 = len(subset_1)

                if i == 0:
                    pt.add_branch("Root", new_nodes[0], ls0)
                    pt.add_branch("Root", new_nodes[1], ls1)
                else:
                    pt.add_branch(split_node, new_nodes[0], ls0)
                    pt.add_branch(split_node, new_nodes[1], ls1)

                clusters[new_nodes[0]] = split_subset[subset_0]
                clusters[new_nodes[1]] = split_subset[subset_1]
                Ws[new_nodes[0]] = W[:, 0]
                Ws[new_nodes[1]] = W[:, 1]

                # These will not have shape of (2, n_features) because they are fitting a subset
                # Create zero filled array of shape (2, n_features)
                h_temp = np.zeros(shape=(2, self.n_features_), dtype=self.dtype)
                # Which features are present in H

                h_temp[0, split_subset] = H[0]
                h_temp[1, split_subset] = H[1]

                Hs[new_nodes[0]] = h_temp[0]
                Hs[new_nodes[1]] = h_temp[1]

                splits[i] = split_node
                is_leaf[new_nodes] = 1

                subset = clusters[new_nodes[0]]
                (
                    subset,
                    W_buffer_one,
                    H_buffer_one,
                    priority_one,
                ) = trial_split_sklearn(
                    min_priority=min_priority,
                    X=X,
                    subset=subset,
                    W_parent=W[:, 0],
                    random_state=self.random_state,
                    trial_allowance=self.trial_allowance,
                    unbalanced=self.unbalanced,
                    dtype=self.dtype,
                    tol=self.tol,
                    maxiter=self.maxiter,
                    init=self.init,
                    alpha_W=self.alpha_W,
                    alpha_H=self.alpha_H,
                )
                clusters[new_nodes[0]] = subset
                W_buffer[new_nodes[0]] = W_buffer_one
                H_buffer[new_nodes[0]] = H_buffer_one
                priorities[new_nodes[0]] = priority_one

                subset = clusters[new_nodes[1]]
                (
                    subset,
                    W_buffer_one,
                    H_buffer_one,
                    priority_one,
                ) = trial_split_sklearn(
                    min_priority=min_priority,
                    X=X,
                    subset=subset,
                    W_parent=W[:, 1],
                    random_state=self.random_state,
                    trial_allowance=self.trial_allowance,
                    unbalanced=self.unbalanced,
                    dtype=self.dtype,
                    tol=self.tol,
                    maxiter=self.maxiter,
                    init=self.init,
                    alpha_W=self.alpha_W,
                    alpha_H=self.alpha_H,
                )
                clusters[new_nodes[1]] = subset
                W_buffer[new_nodes[1]] = W_buffer_one
                H_buffer[new_nodes[1]] = H_buffer_one
                priorities[new_nodes[1]] = priority_one
        self.tree_ = tree.T
        self.splits_ = splits
        self.is_leaf_ = is_leaf
        self.clusters_ = self._stack_clusters(clusters)
        self.Ws_ = np.array(Ws)
        self.Hs_ = np.array(Hs)
        self.W_buffer_ = np.array(W_buffer)
        self.H_buffer_ = self._stack_H_buffer(H_buffer)
        self.priorities_ = priorities
        self.n_nodes_ = self.is_leaf_.shape[0]
        self.n_leaves_ = int(np.count_nonzero(self.is_leaf_))
        return self

    def _stack_clusters(self, clusters: list[npt.NDArray | None]) -> npt.NDArray:
        if not self.n_features_:
            raise ValueError("n_features_ not set before _stack_clusters called")
        result = np.zeros((len(clusters), self.n_features_), dtype=np.int64)
        for i, cluster in enumerate(clusters):
            result[i, cluster] = 1
        return result

    def _stack_H_buffer(self, buffer: list) -> npt.NDArray:
        if self.n_features_ is None:
            raise ValueError("n_features_ not set before _stack_H_buffer called")
        if self.clusters_ is None:
            raise ValueError("clusters_ not set before _stack_H_buffer called")

        result = np.zeros((len(buffer), 2, self.n_features_), dtype=self.dtype)
        for i, buff in enumerate(buffer):
            cluster_nz_idx = np.argwhere(self.clusters_[i]).flatten()
            result[i, 0, cluster_nz_idx] = buff[0, :]
            result[i, 1, cluster_nz_idx] = buff[1, :]
        return result

    def top_features_in_node(self, node: int, n: int = 10) -> list[tuple]:
        """
        For a given node, return the top n features and values
        """

        if self.Hs_ is None:
            raise ValueError("Model not fitted, Hs_ is None")

        node_i = self.Hs_[node]
        ranks = node_i.argsort()[::-1][:n]
        return [(i, node_i[i]) for i in ranks if node_i[i] > 0]

    def top_nodes_in_feature(
        self,
        feature_idx: int | str,
        n: int = 10,
        leaves_only: bool = True,
    ) -> list[tuple]:
        """
        Returns the top nodes for a specified feature
        """
        if self.Hs_ is None:
            raise ValueError("Model not fitted, Hs_ is None")

        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]
        node_weights = self.Hs_.T[feature_idx]
        ranks = node_weights.argsort()[::-1]
        if leaves_only:
            ranks = ranks[np.isin(ranks, node_leaf_idx)]

        ranks = ranks[:n]

        return [(i, node_weights[i]) for i in ranks if node_weights[i] > 0]

    def top_nodes_in_samples(self, n: int = 10, leaves_only: bool = True):
        """
        Returns the top nodes for each sample.
        """

        if self.Ws_ is None or self.n_nodes_ is None:
            raise ValueError("Model not fitted, Ws_ is None")

        # Idx of leaves
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]
        # Keep map of enumerated -> actual cluster
        if leaves_only:
            node_map = dict(enumerate(node_leaf_idx))
        else:
            node_map = dict(enumerate(range(self.n_nodes_)))

        # A dictionary of {sample : [top_nodes]}

        output = {}

        # Ws_ is shape n_nodes, n_samples
        # Transpose weights so it has samples as rows, nodes as columns

        weights = self.Ws_.T[node_leaf_idx].T if leaves_only else self.Ws_.T

        # The ellipsis indicates that the selection is done row wise
        sample_tops = weights.argsort()[:, ::-1][:, :n]

        # Create an array with samples as rows, top n weights as columns
        sample_top_weights = np.take_along_axis(weights, sample_tops, axis=1)

        for sample_idx, (node_ids, node_weights) in enumerate(
            zip(sample_tops, sample_top_weights, strict=True)
        ):
            tops = [
                (node_map[node_id], weight)
                for node_id, weight in zip(node_ids, node_weights, strict=True)
                if weight > 0
            ]
            tops.sort(key=itemgetter(1), reverse=True)
            output[sample_idx] = tops

        return output

    def top_samples_in_nodes(self, n: int = 10, leaves_only: bool = True):
        """
        Returns the top samples for each node
        """

        if self.Ws_ is None:
            raise ValueError("Model not fitted, Ws_ is None")

        # Idx of leaves
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        # A dictionary of {nodes : [sample]}

        output = {}

        # Ws_ is shape n_nodes, n_samples

        weights = self.Ws_

        # The ellipsis indicates that the selection is done row wise
        node_tops = weights.argsort()[:, ::-1][:, :n]

        # Create an array with samples as rows, top n weights as columns
        node_top_weights = np.take_along_axis(weights, node_tops, axis=1)

        for node_idx, (sample_ids, sample_weights) in enumerate(
            zip(node_tops, node_top_weights, strict=True)
        ):
            if leaves_only and node_idx not in node_leaf_idx:
                continue
            tops = [
                (sample_id, weight)
                for sample_id, weight in zip(sample_ids, sample_weights, strict=True)
                if weight > 0
            ]
            tops.sort(key=itemgetter(1), reverse=True)
            # Decode samples if available

            output[node_idx] = tops

        return output

    def top_discriminative_samples_in_node(
        self,
        node: int,
        n: int = 10,
        sign: Literal["positive", "negative", "abs"] = "abs",
    ) -> "list[DiscriminatedSample]":
        """
        Computes most discriminative samples (node vs rest)

        Parameters
        ----------
        node
        n
            The number of features to return
        sign
            One of `['positive', 'negative', 'abs']`.

        Returns
        --------
        list of dict with form::

            sample: Any
            node: int
            node_value: float
            others_value: float

        """

        if self.Ws_ is None:
            raise ValueError("Model not fitted, Ws_ is None")
        if sign not in ("positive", "negative", "abs"):
            raise ValueError("Sign must be one of 'positive', 'negative' or 'abs'")

        # Masks
        member_mask = np.array(node, dtype=np.int64)
        non_member_mask = np.array(
            [x for x in np.arange(0, self.n_nodes_) if x != node]
        )

        member_values = self.Ws_[member_mask].ravel()
        other_means = self.Ws_[non_member_mask].mean(axis=0)

        diffs = (
            np.abs(member_values - other_means)
            if sign == "positive"
            else member_values - other_means
            if sign == "positive"
            else other_means - member_values
        )

        diff_tops = diffs.argsort()[::-1][:n]

        return [
            DiscriminatedSample(
                sample=diff,
                node=node,
                node_value=member_values[diff],
                others_value=other_means[diff],
            )
            for diff in diff_tops
        ]

    def cluster_features(
        self,
        leaves_only: bool = True,
        include_outliers: bool = True,
    ) -> dict[int, list[int]]:
        """
        Returns the features assigned as a cluster to nodes

        Parameters
        ----------
        leaves_only
            Whether to return only leaf nodes
        include_outliers
            If True, features without a node assignment are returned under the key -1

        """

        if self.clusters_ is None:
            raise ValueError("Model not fitted, clusters_ is None")

        output = defaultdict(list)

        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        clusters = self.clusters_[node_leaf_idx] if leaves_only else self.clusters_

        assignments = np.argwhere(clusters)

        for cluster_idx, feature_idx in assignments:
            output[cluster_idx].append(feature_idx)

        if include_outliers:
            outliers = np.where(clusters.sum(axis=0) == 0)[0]
            output[-1] = outliers

        return dict(output)

    def cluster_assignments(
        self,
        leaves_only: bool = True,
        include_outliers: bool = True,
    ) -> dict[int, set[int]]:
        """
        Returns a mapping of features and their assigned cluster(s)

        Parameters
        ----------
        leaves_only
            Whether to return only leaf nodes
        include_outliers
            If True, include feature_idx keys that are not assigned a cluster.

        """

        if self.clusters_ is None:
            raise ValueError("Model not fitted, clusters_ is None")

        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        clusters = self.clusters_
        output = defaultdict(set)
        assignments = np.argwhere(clusters)
        if leaves_only:
            assignments = assignments[
                np.where(np.isin(assignments[:, 0], node_leaf_idx))[0]
            ]

        for cluster_idx, feature_idx in assignments:
            output[cluster_idx].add(feature_idx)

        if include_outliers:
            outliers = np.where(clusters.sum(axis=0) == 0)[0]
            for outlier in outliers:
                output[outlier] = set()

        return dict(output)
