import os
from os import PathLike

import networkx as nx
import numpy as np

from collections import defaultdict
from operator import itemgetter
from scipy.sparse.csr import csr_matrix
from typing import Optional, TYPE_CHECKING, Tuple
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF

from hnmf.progress_tree import ProgressTree

if TYPE_CHECKING:
    from typing import Union, Literal

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from hnmf.signatures import Sign

    Vectorizer = Union[dict[str, int], TfidfVectorizer, CountVectorizer]
    NDArray = np.ndarray
    SPArray = csr_matrix
    AnyArray = Union[np.ndarray, csr_matrix]
    NMFInitMethod = Literal[None, "random", "nndsvd", "nndsvda", "nndsvdar"]
    NMFSolver = Literal["cd", "mu"]
    NMFBetaLoss = Literal["FRO", 0, "KL", 1, "IS", 2]
    DType = Union[np.float32, np.float64]


import logging

from hnmf.signatures import DiscrimSample


from hnmf.helpers import (
    tree_to_nx,
    trial_split_sklearn,
    handle_enums,
)

logger = logging.getLogger(__name__)


class HierarchicalNMF(BaseEstimator):
    def __init__(
        self,
        k: int,
        unbalanced: float = 0.1,
        init: "NMFInitMethod" = None,
        solver: "NMFSolver" = "cd",
        beta_loss: "NMFBetaLoss" = 0,
        random_state: int = 42,
        trial_allowance: int = 100,
        tol: float = 1e-6,
        maxiter: int = 10000,
        dtype: "DType" = np.float64,
    ):
        self.k = k
        self.unbalanced = unbalanced
        self.init = handle_enums(init)
        self.solver = handle_enums(solver)
        self.beta_loss = handle_enums(beta_loss)
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
        self.graph_ = None
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
    random_state : int
        random seed
    trial_allowance : int
        Number of trials allowed for removing outliers and splitting a node again. See parameter T in Algorithm 3 in
        the reference paper.
    tol : float
        Tolerance parameter for stopping criterion in each run of NMF.
    maxiter : int
        Maximum number of iteration times in each run of NMF
    dtype : [np.float32, np.float64]
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

    graph_ :
        NetworkX DiGraph of nodes

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

    def _init_fit(self, X: "AnyArray", term_subset: "NDArray"):
        # TODO Flexible Rank
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

        else:
            W_tmp = nmf.fit_transform(X[term_subset, :])
            H = nmf.components_
            W = np.zeros((self.n_samples_, 2), dtype=self.dtype)
            W[term_subset, :] = W_tmp
            del W_tmp

        return W, H

    def fit(self, X: "AnyArray") -> "HierarchicalNMF":
        """
        Fit `HierarchicalNMF` to data

        Parameters
        ----------
        X : csr_matrix or ndarray

        Returns
        -------
        None
        """
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # TODO Expect different sized ranks
        clusters = [None] * (2 * (self.k - 1))
        Ws = [None] * (2 * (self.k - 1))
        Hs = [None] * (2 * (self.k - 1))
        W_buffer = [None] * (2 * (self.k - 1))
        H_buffer = [None] * (2 * (self.k - 1))
        priorities = np.zeros(2 * (self.k - 1), dtype=self.dtype)
        is_leaf = -np.ones(2 * (self.k - 1), dtype=np.int64)  # No leaves at start
        tree = np.zeros((2, 2 * (self.k - 1)), dtype=np.int64)
        splits = -np.ones(self.k - 1, dtype=np.int64)

        # Where X has at least one non-zero
        term_subset = np.where(np.sum(X, axis=1) != 0)[0]

        # W (n_samples|term_subset, 2)
        # H (2, n_features)
        W, H = self._init_fit(X, term_subset)

        result_used = 0

        with ProgressTree() as pt:
            for i in range((self.k - 1)):
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
                            "Cannot generate all {k} leaf clusters, stopping at {k_current} leaf clusters".format(
                                k=self.k, k_current=i
                            )
                        )

                        Ws = self._remove_empty(Ws)
                        W_buffer = self._remove_empty(W_buffer)

                        Hs = self._remove_empty(Hs)
                        H_buffer = self._remove_empty(H_buffer)

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
                        self.n_leaves_ = np.count_nonzero(self.is_leaf_)
                        self.clusters_ = self._stack_clusters(clusters)
                        self.Ws_ = np.array(Ws)
                        self.Hs_ = np.array(Hs)
                        self.W_buffer_ = np.array(W_buffer)
                        self.H_buffer_ = self._stack_H_buffer(H_buffer)
                        self.priorities_ = priorities
                        self.graph_ = tree_to_nx(tree.T)
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

                subset_0 = np.where(cluster_subset == 0)[0]
                subset_1 = np.where(cluster_subset == 1)[0]
                ls0 = len(subset_0)
                ls1 = len(subset_1)
                ls = ls0 + ls1
                if i == 0:

                    pt.add_branch("Root", new_nodes[0], ls0)
                    pt.add_branch("Root", new_nodes[1], ls1)
                else:
                    pt.add_branch(split_node, new_nodes[0], ls0)
                    pt.add_branch(split_node, new_nodes[1], ls1)

                del ls0, ls1, ls

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

                del h_temp

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
        self.graph_ = tree_to_nx(tree.T)
        self.n_nodes_ = self.is_leaf_.shape[0]
        self.n_leaves_ = np.count_nonzero(self.is_leaf_)
        return self

    def _handle_vectorizer(self, vectorizer: "Vectorizer", attr: str):
        target_attr = getattr(self, attr)
        if not vectorizer:
            if target_attr:
                return target_attr
            else:
                return None
        if isinstance(vectorizer, dict):
            setattr(self, attr, vectorizer)
            if attr == "id2feature_":
                reverse_idx = {v: k for k, v in vectorizer.items()}
                setattr(self, "feature2id_", reverse_idx)
        elif type(vectorizer) in [TfidfVectorizer, CountVectorizer]:
            setattr(
                self, attr, {i: v for i, v in enumerate(vectorizer.get_feature_names())}
            )
            if attr == "id2feature_":
                setattr(
                    self,
                    "feature2id_",
                    {v: i for i, v in enumerate(vectorizer.get_feature_names())},
                )
        else:
            raise AttributeError("Unexpected vectorizer received")
        return getattr(self, attr)

    def _handle_tops(
        self,
        arr: "Union[NDArray, list]",
        ranks: "Union[NDArray, list]",
        vec: "Optional[str]" = "id2feature_",
    ) -> "list[Tuple]":
        if vec:
            vec_ = getattr(self, vec, None)
        else:
            vec_ = None

        if vec_ is not None:
            return [(vec_[i], arr[i]) for i in ranks if arr[i] > 0]
        else:
            return [(i, arr[i]) for i in ranks if arr[i] > 0]

    def _handle_encoding(self, i: int, vec: "Optional[str]") -> "Union[int, str]":
        if vec:
            vec_ = getattr(self, vec, None)
        else:
            vec_ = getattr(self, vec, None)
        if vec_ is not None:
            return vec_[i]
        else:
            return i

    def _stack_clusters(self, clusters: list) -> "NDArray":
        stacked = []
        for cluster in clusters:
            x = np.zeros(self.n_features_)
            x[cluster] = 1
            stacked.append(x)
        return np.vstack(stacked).astype(np.int64)

    def _stack_H_buffer(self, buffer: list) -> "NDArray":
        # Returns components_ with shape (2*k-1, 2, n_features)
        stacked = []
        for i, buff in enumerate(buffer):
            x1 = np.zeros(self.n_features_)
            x2 = np.zeros(self.n_features_)
            cluster = self.clusters_[i]
            cluster_nz_idx = np.argwhere(cluster).flatten()
            x1[cluster_nz_idx] = buff[0, :]
            x2[cluster_nz_idx] = buff[1, :]
            stacked_buff = np.vstack([x1, x2])
            stacked.append(stacked_buff)
        return np.array(stacked)

    def _remove_empty(self, x) -> list:
        return [c for c in x if c is not None]

    def top_features_in_node(
        self, node: int, n: int = 10, id2feature: "Vectorizer" = None
    ) -> "list[Tuple]":
        """
        For a given node, return the top n features and values

        Parameters
        ----------
        node
            Index of node to return top items
        n
            Number of items to return
        id2feature
            Optional, if provided returns decoded features

        """

        self._handle_vectorizer(id2feature, "id2feature_")
        node = self.Hs_[node]
        ranks = node.argsort()[::-1][:n]
        tops = self._handle_tops(arr=node, ranks=ranks)
        return tops

    def top_features_in_nodes(
        self,
        n: int = 10,
        id2feature: "Vectorizer" = None,
        nodes: "Optional[list[int]]" = None,
        leaves_only: bool = False,
    ) -> "list[dict[str, list[tuple]]]":
        """
        Return the top n features and values from all nodes or nodes present in idx if idx is not None

        Parameters
        ----------
        n
            Number of items to return
        id2feature: Vectorizer
            Optional, if provided returns decoded items
        nodes: list[int]
            Optional, if provided, returns top items only for nodes listed
        leaves_only: bool
            If True and idx is None return top items for all leaf nodes

        """
        self._handle_vectorizer(id2feature, "id2feature_")
        node_idx = nodes
        if node_idx is not None:
            nodes = self.Hs_[node_idx]
        elif node_idx is None and leaves_only is True:
            node_idx = np.where(self.is_leaf_ == 1)[0]
            nodes = self.Hs_[node_idx]
        else:
            node_idx = np.arange(0, self.Hs_.shape[0])
            nodes = self.Hs_[node_idx]
        output = []
        for node_id, node in zip(node_idx, nodes):
            ranks = node.argsort()[::-1][:n]
            tops = self._handle_tops(arr=node, ranks=ranks)
            output.append({"node": node_id, "features": tops})
        return output

    def top_nodes_in_feature(
        self,
        feature: "Union[int, str]",
        n: int = 10,
        leaves_only: bool = True,
        id2feature: "Vectorizer" = None,
    ):
        """
        Returns the top nodes for a specified feature

        Parameters
        ----------
        feature: int, str
            The feature to return the top nodes for. If string, ``id2feature`` must not be ``None`` or
             ``HierarchicalNMF.id2feature_`` must not be ``None``
        n
            The number of nodes to return
        leaves_only
            Whether to filter nodes returned to leaf nodes
        id2feature
            Optional, if provided encodes ``feature`` if ``feature`` is passed as a string

        """

        self._handle_vectorizer(id2feature, "id2feature_")
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]
        if isinstance(feature, str):
            feature_idx = self.feature2id_[feature]
        else:
            feature_idx = feature

        node_weights = self.Hs_.T[feature_idx]
        ranks = node_weights.argsort()[::-1]
        if leaves_only:
            rank_is_leaf = np.isin(ranks, node_leaf_idx)
            ranks = ranks[rank_is_leaf]

        ranks = ranks[:n]
        tops = self._handle_tops(arr=node_weights, ranks=ranks, vec=None)
        return tops

    def top_nodes_in_features(
        self,
        features: "Union[list[int], list[str], NDArray]",
        n: int = 10,
        leaves_only: bool = True,
        id2feature: "Vectorizer" = None,
    ) -> "dict[Union[str, int], list[tuple]]":
        """
        Returns the top nodes for a specified feature

        Parameters
        ----------
        features
            The features to return the top nodes for. If features are strings, ``id2feature`` must not be ``None`` or
             ``HierarchicalNMF.id2feature_`` must not be ``None``
        n
            The number of nodes to return
        leaves_only
            Whether to filter nodes returned to leaf nodes

        id2feature
            Optional, if provided encodes ``feature`` if ``feature`` is passed as a string

        """

        self._handle_vectorizer(id2feature, "id2feature_")

        # Idx of leaves
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        output = {}

        # Encode features if needed
        feature_idx = [
            self.feature2id_[x] if isinstance(x, str) else x for x in features
        ]
        feature_names = [
            self.id2feature_[x] if isinstance(x, int) else x for x in feature_idx
        ]

        node_weights = self.Hs_.T[feature_idx]

        ranks = np.apply_along_axis(
            lambda arr: arr.argsort()[::-1], axis=1, arr=node_weights
        )

        if leaves_only:
            ranks = np.apply_along_axis(
                lambda x: x[np.isin(x, node_leaf_idx)][:n], axis=1, arr=ranks
            )
        else:
            ranks = np.apply_along_axis(lambda x: x[:n], axis=1, arr=ranks)

        for feature_name, node_weight, rank in zip(feature_names, node_weights, ranks):
            tops = self._handle_tops(arr=node_weight, ranks=rank, vec=None)
            output[feature_name] = tops

        return output

    def top_nodes_in_samples(
        self, n: int = 10, leaves_only: bool = True, id2sample: "Vectorizer" = None
    ):
        """

        Returns the top nodes for each sample.

        Parameters
        ----------
        n
            Number of items to return
        leaves_only
            Whether to filter top items to nodes identified as leaves
        id2sample
            Optional, if provided returns decoded samples. Should be of form idx : decoded_value


        """
        self._handle_vectorizer(id2sample, "id2sample_")

        # Idx of leaves
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]
        # Keep map of enumerated -> actual cluster
        if leaves_only:
            node_map = {i: v for i, v in enumerate(node_leaf_idx)}
        else:
            node_map = {i: v for i, v in np.arange(self.n_nodes_)}

        # A dictionary of {sample : [top_nodes]}

        output = {}

        # Ws_ is shape n_nodes, n_samples
        # Transpose weights so it has samples as rows, nodes as columns

        if leaves_only:
            weights = self.Ws_[node_leaf_idx].T
        else:
            weights = self.Ws_.T

        # The ellipsis indicates that the selection is done row wise
        sample_tops = weights.argsort()[:, ::-1][:, :n]

        # Create an array with samples as rows, top n weights as columns
        sample_top_weights = np.take_along_axis(weights, sample_tops, axis=1)

        for sample_idx, (node_ids, node_weights) in enumerate(
            zip(sample_tops, sample_top_weights)
        ):
            tops = [
                (node_map[node_id], weight)
                for node_id, weight in zip(node_ids, node_weights)
                if weight > 0
            ]
            tops.sort(key=itemgetter(1), reverse=True)
            # Decode samples if available
            feature_key = self._handle_encoding(i=sample_idx, vec="id2sample_")
            output[feature_key] = tops

        return output

    def top_samples_in_nodes(
        self, n: int = 10, leaves_only: bool = True, id2sample: "Vectorizer" = None
    ):
        """

        Returns the top samples for each node

        Parameters
        ----------
        n
            The number of samples to include for each node
        leaves_only
            Only include leaves
        id2sample
            Decodes samples

        Returns
        -------

        """

        self._handle_vectorizer(id2sample, "id2sample_")

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
            zip(node_tops, node_top_weights)
        ):
            if leaves_only and node_idx not in node_leaf_idx:
                continue
            tops = [
                (sample_id, weight)
                for sample_id, weight in zip(sample_ids, sample_weights)
                if weight > 0
            ]
            tops.sort(key=itemgetter(1), reverse=True)
            # Decode samples if available
            tops = [
                (self._handle_encoding(i=sample_id, vec="id2sample_"), weight)
                for (sample_id, weight) in tops
            ]
            output[node_idx] = tops

        return output

    def top_discriminative_samples_in_node(
        self,
        node: int,
        n: int = 10,
        sign: "Sign" = "abs",
        id2sample: "Vectorizer" = None,
    ) -> "list[DiscrimSample]":
        """
        Computes most discriminative samples (node vs rest)

        Parameters
        ----------
        node
        n
            The number of features to return
        sign
            One of `['positive', 'negative', 'abs']`.
        id2sample
            Decodes index of sample to something meaningful.

        Returns
        --------
        list of dict with form::

            sample: Any
            node: int
            node_value: float
            others_value: float

        """
        self._handle_vectorizer(id2sample, "id2sample_")

        output = []

        # Masks
        member_mask = np.array(node, dtype=np.int64)
        non_member_mask = np.array(
            [x for x in np.arange(0, self.n_nodes_) if x != node]
        )

        member_values = self.Ws_[member_mask].ravel()
        other_means = self.Ws_[non_member_mask].mean(axis=0)

        if sign == "abs":
            diffs = np.abs(member_values - other_means)
        elif sign == "positive":
            diffs = member_values - other_means
        else:
            diffs = other_means - member_values

        diff_tops = diffs.argsort()[::-1][:n]

        for diff in diff_tops:
            output.append(
                {
                    "sample": self._handle_encoding(i=diff, vec="id2sample_"),
                    "node": node,
                    "node_value": member_values[diff],
                    "others_value": other_means[diff],
                }
            )

        return output

    # TODO sample_similarity_by_node_weights

    def cluster_features(
        self,
        leaves_only: bool = True,
        id2feature: "Vectorizer" = None,
        include_outliers: bool = True,
    ) -> "dict[int, list[Union[str, int]]]":
        """
        Returns the features assigned as a cluster to nodes

        Parameters
        ----------
        leaves_only
            Whether to return only leaf nodes
        id2feature
            Decodes features
        include_outliers
            If True, features without a node assignment are returned under the key -1

        """
        self._handle_vectorizer(id2feature, "id2feature_")

        output = {}

        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        if leaves_only:
            clusters = self.clusters_[node_leaf_idx]
        else:
            clusters = self.clusters_

        assignments = np.argwhere(clusters)

        for cluster_idx, feature_idx in assignments:
            feature_name = self._handle_encoding(i=feature_idx, vec="id2feature_")
            cluster_features = output.get(cluster_idx, [])
            cluster_features.append(feature_name)
            output[cluster_idx] = cluster_features

        if include_outliers:
            outliers = np.where(clusters.sum(axis=0) == 0)[0]
            output[-1] = outliers

        return output

    def cluster_assignments(
        self,
        leaves_only: bool = True,
        id2feature: "Vectorizer" = None,
        include_outliers: bool = True,
    ) -> "dict[Union[int, str], Union[set[int], set[None]]]":
        """
        Returns a mapping of features and their assigned cluster(s)

        Parameters
        ----------
        leaves_only
            Whether to return only leaf nodes
        id2feature
            Decodes features
        include_outliers
            If True, include feature keys that are not assigned a cluster.

        """
        self._handle_vectorizer(id2feature, "id2feature_")

        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        clusters = self.clusters_
        output = {}
        assignments = np.argwhere(clusters)
        if leaves_only:
            assignments = assignments[
                np.where(np.in1d(assignments[:, 0], node_leaf_idx))[0]
            ]

        for cluster_idx, feature_idx in assignments:
            feature_name = self._handle_encoding(i=feature_idx, vec="id2feature_")
            feature_clusters = output.get(feature_name, set([]))
            feature_clusters.add(cluster_idx)
            output[feature_name] = feature_clusters

        if include_outliers:
            outliers = np.where(clusters.sum(axis=0) == 0)[0]
            for outlier in outliers:
                outlier_name = self._handle_encoding(i=outlier, vec="id2feature_")
                output[outlier_name] = {None}

        return output

    def enrich_tree(self, n: int, id2feature: "Vectorizer"):
        """
        Appends decoded top features to :py:attr:`self.graph_` Useful if it is desired to export for visualization

        Parameters
        ----------
        n
            The number of top items to add to each node
        id2feature
            Decodes features

        """
        g = self.graph_
        self._handle_vectorizer(id2feature, "id2feature_")
        nodes = self.Hs_
        node_id_incrementer = g.number_of_nodes() + 1
        id2nodeid = defaultdict()
        id2nodeid.default_factory = lambda: id2nodeid.__len__() + node_id_incrementer

        for node_id, node in enumerate(nodes):
            ranks = node.argsort()[::-1][:n]
            for rank in ranks:
                weight = node[rank]
                name = self._handle_encoding(rank, "id2feature_")
                word_node_id = id2nodeid[rank]
                g.add_node(word_node_id, name=name, is_word=True, id=str(word_node_id))
                g.add_edge(node_id, word_node_id, weight=weight)

        self.graph_ = g
        return self.graph_

    def json_graph(self, fp: "Optional[str, PathLike]" = None):
        """
        Export the graph to json

        Parameters
        ----------
        fp: str, PathLike
            Optional, if provided graph is written to this filepath
        """
        data = nx.readwrite.json_graph.node_link_data(self.graph_)
        # fix inconsistency where nodes have str ids and edges are integers
        for e in data["nodes"]:
            e["id"] = str(e["id"])
        for e in data["links"]:
            e["source"] = str(e["source"])
            e["target"] = str(e["target"])

        if not fp:
            return data
        else:
            import json

            with open(os.fspath(fp), "w+", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)
