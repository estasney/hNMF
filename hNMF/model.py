# coding: utf-8
from collections import defaultdict
from enum import Enum
from typing import Union, TypeVar, List, Tuple, Type, Dict

import networkx as nx
import numpy as np
from scipy.sparse.csr import csr_matrix

from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

from hNMF.helpers import (anls_entry_rank2_precompute, trial_split, nmfsh_comb_rank2, tree_to_nx,
                          trial_split_sklearn)

Vectorizer = TypeVar('Vectorizer', dict, TfidfVectorizer, CountVectorizer)
Array = TypeVar('Array', np.ndarray, csr_matrix)


class NMFMethod(Enum):
    SKLEARN = 0
    ORIGINAL = 1


class NMFInitMethod(Enum):
    """
    Class that

    """
    DEFAULT = None
    RANDOM = 'random'
    NNDSVD = 'nndsvd'
    NNDSVDA = 'nndsvda'
    NNDSVDAR = 'nndsvdar'


InitMethod = TypeVar('InitMethod', Type[NMFInitMethod], str, Type[None])


class HierarchicalNMF(BaseEstimator):
    """
    Implements Hierarchical rank-2 NMF

    Parameters
    ----------

    k:
        The number of desired leaf nodes
    random_state :
        random seed
    trial_allowance :
        Number of trials allowed for removing outliers and splitting a node again. See parameter T in Algorithm 3 in
        the reference paper.
    unbalanced :
        A threshold to determine if one of the two clusters is an outlier set. A smaller value means more tolerance for
        imbalance between two clusters. See parameter beta in Algorithm 3 in the reference paper.
    vec_norm :
        Indicates which norm to use for the normalization of W or H, e.g. vec_norm=2 means Euclidean norm; vec_norm=0
        means no normalization.
    normW :
        true if normalizing columns of W; false if normalizing rows of H.
    anls_alg :
        must implement NNLS
    tol :
        Tolerance parameter for stopping criterion in each run of NMF.
    maxiter :
        Maximum number of iteration times in each run of NMF
    dtype :
        Dtype used for numpy arrays
    nmf_method :
        Use NMF as implemented by Sklearn or Original Paper
    nmf_init_method :
        Only used when nmf_method is sklearn. Specifies nmf init procedure

    Attributes
    ----------
    tree_ :
        A 2-by-(k-1) matrix that encodes the tree structure. The two entries in the i-th column are the numberings of
        the two children of the node with numbering i. The root node has numbering 0, with its two children always
        having numbering 1 and numbering 2. Thus the root node is NOT included in the 'tree' variable.

    splits_ :
        An array of length k-1. It keeps track of the numberings of the nodes being split from the 1st split to the
        (k-1)-th split. (The first entry is always 0.)

    is_leaf_ :
        An array of length 2*(k-1). A "1" at index i means that the node with numbering i is a leaf node in the final
        tree generated, and "0" indicates non-leaf nodes in the final tree.

    clusters_ :
        A cell array of length 2*(k-1). The i-th element contains the subset of items at the node with numbering i.

    graph_ :
        NetworkX DiGraph of nodes

    Methods
    -------
    fit(X)
        Fit k number of leaves to input array

    Notes
    -----
    Adapted from [rank-2]_

    """

    def __init__(self, k: int,
                 random_state: int = 42,
                 trial_allowance: int = 100,
                 unbalanced: float = 0.1,
                 vec_norm: int = 2,
                 normW: bool = True,
                 anls_alg: callable = anls_entry_rank2_precompute,
                 tol: float = 1e-6,
                 maxiter: int = 10000,
                 dtype: Union[np.float32, np.float64] = np.float64,
                 nmf_init_method: InitMethod = NMFInitMethod.DEFAULT,
                 nmf_method: Union[type(NMFMethod.SKLEARN), type(NMFMethod.ORIGINAL)] = NMFMethod.SKLEARN):
        self.k = k
        self.random = np.random.RandomState(seed=random_state)
        self.trial_allowance = trial_allowance
        self.unbalanced = unbalanced
        self.vec_norm = vec_norm
        self.normW = normW
        self.anls_alg = anls_alg
        self.tol = tol
        self.maxiter = maxiter
        self.dtype = dtype
        self.nmf_init_method = nmf_init_method.value if type(nmf_init_method) == NMFInitMethod else nmf_init_method
        self.nmf_method = nmf_method
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

    def _init_2_rank(self, X, term_subset):
        if self.nmf_method == NMFMethod.SKLEARN:
            return self._init_fit_sklearn(X, term_subset)
        elif self.nmf_method == NMFMethod.ORIGINAL:
            return self._init_fit_original(X, term_subset)
        else:
            return self._init_fit_sklearn(X, term_subset)

    def _init_fit_sklearn(self, X, term_subset):
        nmf = NMF(n_components=2, random_state=self.random, tol=self.tol, max_iter=self.maxiter,
                  init=self.nmf_init_method)

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

    def _init_fit_original(self, X, term_subset):
        W = self.random.rand(len(term_subset), 2)
        H = self.random.rand(2, self.n_features_)

        # Compute the 2-rank NMF of W and H
        if len(term_subset) == self.n_samples_:
            # All documents have features
            W, H = nmfsh_comb_rank2(X, W, H, anls_alg=self.anls_alg, vec_norm=self.vec_norm, normW=self.normW,
                                    tol=self.tol,
                                    maxiter=self.maxiter,
                                    dtype=self.dtype)
        else:
            # Exclude documents without features
            W_tmp, H = nmfsh_comb_rank2(X[term_subset, :], W, H, anls_alg=self.anls_alg, vec_norm=self.vec_norm,
                                        normW=self.normW,
                                        tol=self.tol,
                                        maxiter=self.maxiter,
                                        dtype=self.dtype)

            W = np.zeros((self.n_samples_, 2), dtype=self.dtype)
            W[term_subset, :] = W_tmp
            del W_tmp
        return W, H

    def _trial_split(self, min_priority, X, subset, W_parent, random_state, trial_allowance,
                     unbalanced, dtype, anls_alg, vec_norm, normW, tol, maxiter, nmf_init_method):

        if self.nmf_method == NMFMethod.SKLEARN:
            subset, W_buffer_one, H_buffer_one, priority_one = trial_split_sklearn(min_priority=min_priority, X=X,
                                                                                   subset=subset, W_parent=W_parent,
                                                                                   random_state=random_state,
                                                                                   trial_allowance=trial_allowance,
                                                                                   unbalanced=unbalanced, dtype=dtype,
                                                                                   tol=tol, maxiter=maxiter,
                                                                                   nmf_init_method=nmf_init_method)
        else:
            subset, W_buffer_one, H_buffer_one, priority_one = trial_split(min_priority=min_priority,
                                                                           X=X, subset=subset,
                                                                           random_state=random_state,
                                                                           trial_allowance=trial_allowance,
                                                                           unbalanced=unbalanced,
                                                                           dtype=dtype,
                                                                           anls_alg=anls_alg,
                                                                           vec_norm=vec_norm,
                                                                           normW=normW,
                                                                           tol=tol, maxiter=maxiter, W_parent=W_parent)

        return subset, W_buffer_one, H_buffer_one, priority_one

    def _stack_clusters(self, clusters: list):
        stacked = []
        for cluster in clusters:
            x = np.zeros(self.n_features_)
            x[cluster] = 1
            stacked.append(x)
        return np.vstack(stacked)

    def _stack_H_buffer(self, buffer: list) -> np.ndarray:
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

    def _remove_empty(self, x):
        return [c for c in x if c is not None]

    def fit(self, X: Array):
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        clusters = [None] * (2 * (self.k - 1))
        Ws = [None] * (2 * (self.k - 1))
        Hs = [None] * (2 * (self.k - 1))
        W_buffer = [None] * (2 * (self.k - 1))
        H_buffer = [None] * (2 * (self.k - 1))
        priorities = np.zeros(2 * self.k - 1, dtype=self.dtype)
        is_leaf = -np.ones(2 * (self.k - 1), dtype=self.dtype)  # No leaves at start
        tree = np.zeros((2, 2 * (self.k - 1)), dtype=self.dtype)
        splits = -np.ones(self.k - 1, dtype=self.dtype)

        term_subset = np.where(np.sum(X, axis=1) != 0)[0]  # Where X has at least one non-zero
        W, H = self._init_2_rank(X, term_subset)

        result_used = 0
        pb = tqdm(desc="Finding Leaves", total=len(range(self.k - 1)))
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

                min_priority = np.min(temp_priority[temp_priority > 0])
                split_node = np.argmax(temp_priority)
                if temp_priority[split_node] < 0 or min_priority == -1:
                    tqdm.write("Cannot generate all {k} leaf clusters, stopping at {k_current} leaf clusters"
                               .format(k=self.k, k_current=i))
                    pb.close()

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
                    self.Ws_ = Ws
                    self.Hs_ = Hs
                    self.W_buffer_ = np.array(W_buffer)
                    self.H_buffer_ = self._stack_H_buffer(H_buffer)
                    self.priorities_ = priorities
                    self.graph_ = tree_to_nx(tree.T)
                    return self

                split_node = leaves[split_node]  # Attempt to split this node
                is_leaf[split_node] = 0
                W = W_buffer[split_node]
                H = H_buffer[split_node]
                split_subset = clusters[split_node]
                new_nodes = [result_used, result_used + 1]
                tree[:, split_node] = new_nodes

            result_used += 2
            # For each row find where it is more greatly represented
            cluster_subset = np.argmax(H, axis=0)
            clusters[new_nodes[0]] = split_subset[np.where(cluster_subset == 0)[0]]
            clusters[new_nodes[1]] = split_subset[np.where(cluster_subset == 1)[0]]
            Ws[new_nodes[0]] = W[:, 0]
            Ws[new_nodes[1]] = W[:, 1]
            Hs[new_nodes[0]] = H[:, 0]
            Hs[new_nodes[1]] = H[:, 1]
            splits[i] = split_node
            is_leaf[new_nodes] = 1

            subset = clusters[new_nodes[0]]
            subset, W_buffer_one, H_buffer_one, priority_one = self._trial_split(min_priority=min_priority,
                                                                                 X=X,
                                                                                 subset=subset,
                                                                                 W_parent=W[:, 0],
                                                                                 random_state=self.random,
                                                                                 trial_allowance=self.trial_allowance,
                                                                                 unbalanced=self.unbalanced,
                                                                                 dtype=self.dtype,
                                                                                 anls_alg=self.anls_alg,
                                                                                 vec_norm=self.vec_norm,
                                                                                 normW=self.normW,
                                                                                 tol=self.tol,
                                                                                 maxiter=self.maxiter,
                                                                                 nmf_init_method=self.nmf_init_method)
            clusters[new_nodes[0]] = subset
            W_buffer[new_nodes[0]] = W_buffer_one
            H_buffer[new_nodes[0]] = H_buffer_one
            priorities[new_nodes[0]] = priority_one

            subset = clusters[new_nodes[1]]
            subset, W_buffer_one, H_buffer_one, priority_one = self._trial_split(min_priority=min_priority,
                                                                                 X=X,
                                                                                 subset=subset,
                                                                                 W_parent=W[:, 1],
                                                                                 random_state=self.random,
                                                                                 trial_allowance=self.trial_allowance,
                                                                                 unbalanced=self.unbalanced,
                                                                                 dtype=self.dtype,
                                                                                 anls_alg=self.anls_alg,
                                                                                 vec_norm=self.vec_norm,
                                                                                 normW=self.normW,
                                                                                 tol=self.tol,
                                                                                 maxiter=self.maxiter,
                                                                                 nmf_init_method=self.nmf_init_method)
            clusters[new_nodes[1]] = subset
            W_buffer[new_nodes[1]] = W_buffer_one
            H_buffer[new_nodes[1]] = H_buffer_one
            priorities[new_nodes[1]] = priority_one

            pb.update(1)

        pb.close()
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

    def _handle_vectorizer(self, vectorizer: Vectorizer, attr: str):
        target_attr = getattr(self, attr)
        if not vectorizer:
            if target_attr:
                return target_attr
            else:
                return None
        if isinstance(vectorizer, dict):
            setattr(self, attr, vectorizer)
        elif type(vectorizer) in [TfidfVectorizer, CountVectorizer]:
            setattr(self, attr, {i: v for i, v in enumerate(vectorizer.get_feature_names())})
        else:
            raise AttributeError("Unexpected vectorizer received")
        return getattr(self, attr)

    def _handle_tops(self, arr: Union[np.ndarray, list], ranks: Union[np.ndarray, list],
                     vec: Union[str, Type[None]] = 'id2feature_') -> List[Tuple]:
        if vec:
            vec_ = getattr(self, vec, None)
        else:
            vec_ = None

        if vec_ is not None:
            return [(vec_[i], arr[i]) for i in ranks if arr[i] > 0]
        else:
            return [(i, arr[i]) for i in ranks if arr[i] > 0]

    def _handle_encoding(self, i: int, vec: Union[str, Type[None]]) -> Union[int, str]:
        if vec:
            vec_ = getattr(self, vec, None)
        else:
            vec_ = getattr(self, vec, None)
        if vec_ is not None:
            return vec_[i]
        else:
            return i

    def top_features_in_node(self, node: int, n: int = 10, id2feature: Vectorizer = None, merge=False) \
            -> Union[List[Tuple], List[List]]:
        """

        Parameters
        ----------

        For a given node, return the top n values

        node
            Index of node to return top items
        n
            Number of items to return
        id2feature
            Optional, if provided returns decoded features
        merge
            If False, returns top items for each column in node. If True, columns are averaged across rows and top items
            are computed from the result

        """

        self._handle_vectorizer(id2feature, 'id2feature_')
        node = self.H_buffer_[node]
        n1, n2 = node[0], node[1]
        if merge:
            m = n1 * n2
            t = m.argsort()[::-1][:n]
            tops = self._handle_tops(arr=m, ranks=t)
            return tops

        t1 = n1.argsort()[::-1][:n]
        t2 = n2.argsort()[::-1][:n]
        t1_tops = self._handle_tops(arr=n1, ranks=t1)
        t2_tops = self._handle_tops(arr=n2, ranks=t2)
        return [t1_tops, t2_tops]

    def top_features_in_nodes(self, n: int = 10, id2feature: Vectorizer = None, merge=False, idx=None) \
            -> List[Dict[str, List]]:
        """

        Parameters
        ----------

        Return the top n values from all nodes or nodes present in idx if idx is not None

        n
            Number of items to return
        id2feature
            Optional, if provided returns decoded items
        merge
            If False, returns top items for each column in node. If True, columns are averaged across rows and top items
            are computed from the result
        idx
            Optional, if provided, returns top items only for nodes specified in idx


        """

        self._handle_vectorizer(id2feature, 'id2feature_')
        if idx is not None:
            nodes = self.H_buffer_[idx]
        else:
            idx = np.arange(0, self.H_buffer_.shape[0])
            nodes = self.H_buffer_[idx]

        output = []
        for node_id, node in zip(idx, nodes):
            n1, n2 = node[0], node[1]
            if merge:
                m = n1 * n2
                t = m.argsort()[::-1][:n]
                tops = self._handle_tops(arr=m, ranks=t)
                output.append({'node': node_id, 'features': tops})
            else:
                t1 = n1.argsort()[::-1][:n]
                t2 = n2.argsort()[::-1][:n]
                t1_tops = self._handle_tops(arr=n1, ranks=t1)
                t2_tops = self._handle_tops(arr=n2, ranks=t2)
                output.append({"node": node_id, 'features': [t1_tops, t2_tops]})
        return output

    def top_features_in_leaves(self, n: int = 10, id2feature: Vectorizer = None, merge=False) \
            -> List[Dict[str, List]]:

        """
        Convenience method to return top items occurring from nodes identified as leaves

        Parameters
        ----------
         n
            Number of items to return
        id2feature
            Optional, if provided returns decoded items
        merge
            If False, returns top items for each column in node. If True, columns are averaged across rows and top items
            are computed from the result

        """

        leaf_idx = np.where(self.is_leaf_ == 1)[0]
        output = self.top_features_in_nodes(n, id2feature, merge, leaf_idx)
        return output

    def top_nodes_in_features(self, n: int, leaves_only: bool, id2feature: Vectorizer):
        """

        Returns the top items for W.

        Parameters
        ----------
        n
            Number of items to return
        leaves_only
            Whether to filter top items to nodes identified as leaves
        id2feature
            Optional, if provided returns decoded features

        Notes
        -----
        Rather than return the top features for each node, this returns the top nodes for each feature.

        """
        self._handle_vectorizer(id2feature, 'id2feature_')

        # Idx of leaves
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        # A dictionary of {feature : [top_nodes]}
        # output = {self._handle_encoding(i, is_sample=False): [] for i in range(self.n_features_)}
        output = {}

        # Array to hold feature weight from each reconstructed matrix
        feature_weights = np.zeros(shape=(self.n_nodes_, self.n_features_))

        for node_idx in range(len(self.W_buffer_)):

            # Don't bother reconstructing nodes we won't access
            if leaves_only and node_idx not in node_leaf_idx:
                continue

            # Reconstructed has a shape n_samples, n_features
            reconstructed = np.dot(self.W_buffer_[node_idx], self.H_buffer_[node_idx])

            # L2 Normalization
            # reconstructed = normalize(reconstructed, norm='l2', axis=1, return_norm=False)

            # Get the sum of weights for each feature across samples
            feature_node_weights = reconstructed.sum(axis=0)
            feature_weights[node_idx] = feature_node_weights

        # Transpose array so we can iterate per feature
        for feature_idx, feature_weight in enumerate(feature_weights.T):

            # Reindex here to prevent non-leaves from appearing in top items. This could happen if a feature has
            # non-zero weights < n

            if leaves_only:
                weights = feature_weight[node_leaf_idx]
            else:
                weights = feature_weight

            top_idx = weights.argsort()[::-1][:n]
            tops = self._handle_tops(arr=weights, ranks=top_idx, vec=None)

            # Decode
            feature_key = self._handle_encoding(i=feature_idx, vec='id2feature_')
            output[feature_key] = tops

        return output

    def top_nodes_in_samples(self, n: int, leaves_only: bool, id2sample: Vectorizer):
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
        self._handle_vectorizer(id2sample, 'id2sample_')

        # Idx of leaves
        node_leaf_idx = np.where(self.is_leaf_ == 1)[0]

        # A dictionary of {sample : [top_nodes]}

        output = {}

        # Ws_ is shape n_nodes, n_samples
        # Transpose weights so it has samples as rows, nodes as columns

        if leaves_only:
            weights = self.Ws_[node_leaf_idx].T
        else:
            weights = self.Ws_[node_leaf_idx].T

        sample_tops = weights.argsort()[::-1][:, :n]

        # Create an array with samples as rows, top n weights as columns
        sample_top_weights = np.take_along_axis(weights, sample_tops, axis=1)

        for sample_idx, (node_ids, node_weights) in enumerate(zip(sample_tops, sample_top_weights)):
            tops = [(node_id, weight) for node_id, weight in zip(node_ids, node_weights)]

            # Decode samples if available
            feature_key = self._handle_encoding(i=sample_idx, vec='id2sample_')
            output[feature_key] = tops

        return output

    def enrich_tree(self, n: int, id2feature: Vectorizer):
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
        self._handle_vectorizer(id2feature, 'id2feature_')
        nodes = self.H_buffer_
        node_id_incrementer = g.number_of_nodes() + 1
        id2nodeid = defaultdict()
        id2nodeid.default_factory = lambda: id2nodeid.__len__() + node_id_incrementer

        for node_id, node in enumerate(nodes):
            n1, n2 = node[0], node[1]
            t1 = n1.argsort()[::-1][:n]
            t2 = n2.argsort()[::-1][:n]

            for arr, ranks in [(n1, t1), (n2, t2)]:
                for rank in ranks:
                    weight = arr[rank]
                    name = self._handle_encoding(rank, True)
                    word_node_id = id2nodeid[rank]
                    g.add_node(word_node_id, name=name, is_word=True, id=str(word_node_id))
                    g.add_edge(node_id, word_node_id, weight=weight)

        self.graph_ = g
        return self.graph_

    def json_graph(self, fp: Union[str, Type[None]] = None):
        """
        Export the graph to json

        Parameters
        ----------
        fp
            Optional, if provided graph is written to this filepath


        """
        data = nx.readwrite.json_graph.node_link_data(self.graph_)
        # fix inconsistency where nodes have str ids and edges are integers
        for e in data['nodes']:
            e['id'] = str(e['id'])
        for e in data['links']:
            e['source'] = str(e['source'])
            e['target'] = str(e['target'])

        if not fp:
            return data
        else:
            import json
            with open(fp, "w+", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)
