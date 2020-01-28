import sys
import warnings
from enum import Enum
from typing import Union
import contextlib
import numpy as np
from numpy.linalg import matrix_rank, svd, norm
from numpy.random import mtrand
from scipy import sparse as sp
from sklearn.decomposition import non_negative_factorization
from tqdm.auto import tqdm
from tqdm.contrib import DummyTqdmFile

import logging

logger = logging.getLogger(__name__)


def anls_entry_rank2_precompute(left, right, H, dtype):
    eps = 1e-6
    n = right.shape[0]

    solve_either = np.zeros((n, 2), dtype=dtype)
    solve_either[:, 0] = right[:, 0] / left[0, 0]
    solve_either[:, 1] = right[:, 0] / left[1, 1]
    cosine_either = solve_either * np.sqrt(np.array([left[0, 0], left[1, 1]]))
    choose_first = cosine_either[:, 0] >= cosine_either[:, 1]
    solve_either[choose_first, 1] = 0
    solve_either[np.logical_not(choose_first), 0] = 0

    if np.abs(left[0, 0]) < eps and abs(left[0, 1]) < eps:
        logger.error('Error: The 2x2 matrix is close to singular or the input data matrix has tiny values')
    else:
        if np.abs(left[0, 0] >= np.abs(left[0, 1])):
            t = left[1, 0] / left[0, 0]
            a2 = left[0, 0] + t * left[1, 0]
            b2 = left[0, 1] + t * left[1, 1]
            d2 = left[1, 1] - t * left[0, 1]
            if np.abs(d2 / a2) < eps:
                logger.error('Error: The 2x2 matrix is close to singular')

            e2 = right[:, 0] + t * right[:, 1]
            f2 = right[:, 1] - t * right[:, 0]
        else:
            ct = left[0, 0] / left[1, 0]
            a2 = left[1, 0] + ct * left[0, 0]
            b2 = left[1, 1] + ct * left[0, 1]
            d2 = -left[0, 1] + ct * left[1, 1]
            if np.abs(d2 / a2) < eps:
                logger.error('Error: The 2x2 matrix is close to singular')

            e2 = right[:, 1] + ct * right[:, 0]
            f2 = -right[:, 0] + ct * right[:, 1]

        H[:, 1] = f2 * (1 / d2)
        H[:, 0] = (e2 - b2 * H[:, 1]) * (1 / a2)

    use_either = np.logical_not(np.all(H > 0, axis=1))
    H[use_either, :] = solve_either[use_either, :]

    return H


def hier8_neat(X, k, random_state=0, trial_allowance: int = 3, unbalanced: float = 0.1,
               vec_norm: Union[float, int] = 2.0, normW: bool = True, anls_alg: callable = anls_entry_rank2_precompute,
               tol: float = 1e-4, maxiter: int = 10000):
    """

    Parameters
    ----------

    X :

    k :
     int, The number of desired leaf nodes
    random_state :

    trial_allowance :
     Number of trials allowed for removing outliers and splitting a node again. See parameter T in Algorithm 3 in the reference paper.
    unbalanced :
     A threshold to determine if one of the two clusters is an outlier set. A smaller value means more tolerance for unbalance between two clusters. See parameter beta in Algorithm 3 in the reference paper.
    vec_norm :
     Indicates which norm to use for the normalization of W or H, e.g. vec_norm=2 means Euclidean norm; vec_norm=0 means no normalization.
    normW :
     true if normalizing columns of W; false if normalizing rows of H.
    anls_alg :
     The function handle to NNLS algorithm whose signature is the same as @anls_entry_rank2_precompute
    tol :
     Tolerance parameter for stopping criterion in each run of NMF.
    maxiter :
     Maximum number of iteration times in each run of NMF


    Returns
    -------

    From the output parameters, you can reconstruct the tree and "replay" the k-1 steps that generated it.

    For a binary tree with k leaf nodes, the total number of nodes (including leaf and non-leaf nodes)
     is 2*(k-1) plus the root node, because k-1 splits are performed and each split generates two new nodes.

    We only keep track of the 2*(k-1) non-root node in the output.

    tree: A 2-by-(k-1) matrix that encodes the tree structure. The two entries in the i-th column are the numberings
           of the two children of the node with numbering i.
           The root node has numbering 0, with its two children always having numbering 1 and numbering 2.
           Thus the root node is NOT included in the 'tree' variable.


    splits: An array of length k-1. It keeps track of the numberings of the nodes being split
             from the 1st split to the (k-1)-th split. (The first entry is always 0.)

    is_leaf: An array of length 2*(k-1). A "1" at index i means that the node with numbering i is a leaf node
              in the final tree generated, and "0" indicates non-leaf nodes in the final tree.

    clusters: A cell array of length 2*(k-1). The i-th element contains the subset of items
        at the node with numbering i.


    Ws: A cell array of length 2*(k-1).
         Its i-th element is the topic vector of the cluster at the node with numbering i.

    priorities: An array of length 2*(k-1).
                 Its i-th element is the modified NDCG scores at the node with numbering i (see the reference paper).

    Notes
    -----

     If you want to have the flat partitioning induced by the leaf nodes in the final tree,
     use this script:

     partitioning = zeros(1, n);  n is the total number of data points
     leaf_level = clusters(is_leaf == 1);
     for i = 1 : length(leaf_level)
         partitioning(leaf_level{i}) = i;
     end

     (Now the entries in 'partitioning' having value 0 indicate outliers that do not belong to any cluster.)

     Adapted from [rank-2]_
    """

    # Repack params
    params = {k: v for k, v in locals().items() if k not in ['X', 'k', 'random_state']}
    random_state = np.random.RandomState(seed=random_state)
    n_samples, n_features = X.shape
    clusters = [None] * (2 * (k - 1))
    Ws = [None] * (2 * (k - 1))
    W_buffer = [None] * (2 * (k - 1))
    H_buffer = [None] * (2 * (k - 1))
    priorities = np.zeros(2 * k - 1, dtype=np.float32)
    is_leaf = -np.ones(2 * (k - 1), dtype=np.float32)  # No leafs at start
    tree = np.zeros((2, 2 * (k - 1)), dtype=np.float32)
    splits = -np.ones(k - 1, dtype=np.float32)

    term_subset = np.where(np.sum(X, axis=1) != 0)[0]  # Select samples with at least 1 feature

    # Random initial guesses for W and H
    W = random_state.rand(len(term_subset), 2)
    H = random_state.rand(2, n_features)

    # Compute the 2-rank NMF of W and H
    if len(term_subset) == n_samples:
        W, H = nmfsh_comb_rank2(X, W, H, **params)
    else:
        W_tmp, H = nmfsh_comb_rank2(X[term_subset, :], W, H, **params)
        W = np.zeros((n_samples, 2), dtype=np.float32)
        W[term_subset, :] = W_tmp
        del W_tmp

    result_used = 0
    for i in range(k - 1):
        if i == 0:
            split_node = 0
            new_nodes = [0, 1]
            min_priority = 1e40
            split_subset = np.arange(n_features)
        else:
            leaves = np.where(is_leaf == 1)[0]
            temp_priority = priorities[leaves]
            min_priority = np.min(temp_priority[temp_priority > 0])
            split_node = np.argmax(temp_priority)
            if temp_priority[split_node] < 0:
                logger.info(f'Cannot generate all {k} leaf clusters')

                Ws = [W for W in Ws if W is not None]
                return tree, splits, is_leaf, clusters, Ws, priorities

            split_node = leaves[split_node]
            is_leaf[split_node] = 0
            W = W_buffer[split_node]
            H = H_buffer[split_node]
            split_subset = clusters[split_node]
            new_nodes = [result_used, result_used + 1]
            tree[:, split_node] = new_nodes

        result_used += 2
        cluster_subset = np.argmax(H, axis=0)
        clusters[new_nodes[0]] = split_subset[np.where(cluster_subset == 0)[0]]
        clusters[new_nodes[1]] = split_subset[np.where(cluster_subset == 1)[0]]
        Ws[new_nodes[0]] = W[:, 0]
        Ws[new_nodes[1]] = W[:, 1]
        splits[i] = split_node
        is_leaf[new_nodes] = 1

        subset = clusters[new_nodes[0]]
        subset, W_buffer_one, H_buffer_one, priority_one = trial_split(min_priority, X, subset, W[:, 0], random_state,
                                                                       **params)
        clusters[new_nodes[0]] = subset
        W_buffer[new_nodes[0]] = W_buffer_one
        H_buffer[new_nodes[0]] = H_buffer_one
        priorities[new_nodes[0]] = priority_one

        subset = clusters[new_nodes[1]]
        subset, W_buffer_one, H_buffer_one, priority_one = trial_split(min_priority, X, subset, W[:, 1], random_state,
                                                                       **params)
        clusters[new_nodes[1]] = subset
        W_buffer[new_nodes[1]] = W_buffer_one
        H_buffer[new_nodes[1]] = H_buffer_one
        priorities[new_nodes[1]] = priority_one

    return tree.T, splits, is_leaf, clusters, Ws, priorities


def trial_split_sklearn(min_priority: float, X, subset, W_parent, random_state, trial_allowance: int, unbalanced: float,
                        dtype: Union[np.float32, np.float64], tol, maxiter, init):
    m = X.shape[0]
    trial = 0
    subset_backup = subset
    while trial < trial_allowance:
        cluster_subset, W_buffer_one, H_buffer_one, priority_one = split_once_sklearn(X=X,
                                                                                      subset=subset,
                                                                                      W_parent=W_parent,
                                                                                      random_state=random_state,
                                                                                      dtype=dtype,
                                                                                      tol=tol,
                                                                                      maxiter=maxiter,
                                                                                      init=init)
        if priority_one < 0:
            break

        unique_cluster_subset = np.unique(cluster_subset)
        if len(unique_cluster_subset) != 2:
            logger.error('Invalid number of unique sub-clusters!')

        length_cluster1 = len(np.where(cluster_subset == unique_cluster_subset[0])[0])
        length_cluster2 = len(np.where(cluster_subset == unique_cluster_subset[1])[0])
        if min(length_cluster1, length_cluster2) < unbalanced * len(cluster_subset):
            logger.debug("Below imbalanced threshold: {}".format((unbalanced * len(cluster_subset))))
            idx_small = np.argmin(np.array([length_cluster1, length_cluster2]))
            subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[0]
            subset_small = subset[subset_small]
            _, _, _, priority_one_small = split_once_sklearn(X=X, subset=subset_small,
                                                             W_parent=W_buffer_one[:, idx_small],
                                                             random_state=random_state, dtype=dtype, maxiter=maxiter,
                                                             tol=tol, init=init)
            if priority_one_small < min_priority:
                trial += 1
                if trial < trial_allowance:
                    logger.debug("Dropped {} features...".format(len(subset_small)))
                    subset = np.setdiff1d(subset, subset_small)
            else:
                break
        else:
            break

    if trial == trial_allowance:
        logger.debug("Reached trial allowance, recycled {} features".format(len(subset_backup) - len(subset)))
        subset = subset_backup
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -2

    return subset, W_buffer_one, H_buffer_one, priority_one


def split_once_sklearn(X, subset, W_parent, random_state: mtrand.RandomState, dtype: Union[np.float32, np.float64],
                       tol, maxiter, init):
    m = X.shape[0]
    if len(subset) <= 3:
        cluster_subset = np.ones(len(subset), dtype=dtype)
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -1
    else:
        term_subset = np.where(np.sum(X[:, subset], axis=1) != 0)[0]
        X_subset = X[term_subset, :][:, subset]
        W = random_state.rand(len(term_subset), 2)
        H = random_state.rand(2, len(subset))
        W, H, n_iter_ = non_negative_factorization(X=X_subset, W=W, H=H, n_components=2,
                                                   init=init,
                                                   update_H=True,
                                                   solver='cd',
                                                   beta_loss=2,
                                                   tol=tol,
                                                   max_iter=maxiter,
                                                   alpha=0,
                                                   l1_ratio=0,
                                                   regularization='both',
                                                   random_state=random_state,
                                                   verbose=0,
                                                   shuffle=False
                                                   )
        cluster_subset = np.argmax(H, axis=0)
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        W_buffer_one[term_subset, :] = W
        H_buffer_one = H
        if len(np.unique(cluster_subset)) > 1:
            priority_one = compute_priority(W_parent, W_buffer_one, dtype=dtype)
        else:
            priority_one = -1
    return cluster_subset, W_buffer_one, H_buffer_one, priority_one


def trial_split(min_priority: float, X, subset, W_parent, random_state, trial_allowance: int, unbalanced: float,
                dtype: Union[np.float32, np.float64], anls_alg, vec_norm, normW, tol, maxiter):
    m = X.shape[0]
    trial = 0
    subset_backup = subset
    while trial < trial_allowance:
        cluster_subset, W_buffer_one, H_buffer_one, priority_one = split_once(X=X,
                                                                              subset=subset,
                                                                              W_parent=W_parent,
                                                                              random_state=random_state,
                                                                              dtype=dtype,
                                                                              anls_alg=anls_alg,
                                                                              vec_norm=vec_norm,
                                                                              normW=normW,
                                                                              tol=tol,
                                                                              maxiter=maxiter)
        if priority_one < 0:
            break

        unique_cluster_subset = np.unique(cluster_subset)
        if len(unique_cluster_subset) != 2:
            tqdm.write('Invalid number of unique sub-clusters!')

        length_cluster1 = len(np.where(cluster_subset == unique_cluster_subset[0])[0])
        length_cluster2 = len(np.where(cluster_subset == unique_cluster_subset[1])[0])
        if min(length_cluster1, length_cluster2) < unbalanced * len(cluster_subset):
            idx_small = np.argmin(np.array([length_cluster1, length_cluster2]))
            subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[0]
            subset_small = subset[subset_small]
            _, _, _, priority_one_small = split_once(X=X, subset=subset_small, W_parent=W_buffer_one[:, idx_small],
                                                     random_state=random_state, dtype=dtype, anls_alg=anls_alg,
                                                     vec_norm=vec_norm, normW=normW, maxiter=maxiter, tol=tol)
            if priority_one_small < min_priority:
                trial += 1
                if trial < trial_allowance:
                    tqdm.write("Dropped {} documents...".format(len(subset_small)))
                    subset = np.setdiff1d(subset, subset_small)
            else:
                break
        else:
            break

    if trial == trial_allowance:
        tqdm.write("Recycled {} documents...".format(len(subset_backup) - len(subset)))
        subset = subset_backup
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -2

    return subset, W_buffer_one, H_buffer_one, priority_one


def split_once(X, subset, W_parent, random_state: mtrand.RandomState, dtype: Union[np.float32, np.float64],
               anls_alg: callable, vec_norm, normW, tol, maxiter):
    m = X.shape[0]
    if len(subset) <= 3:
        cluster_subset = np.ones(len(subset), dtype=dtype)
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -1
    else:
        term_subset = np.where(np.sum(X[:, subset], axis=1) != 0)[0]
        X_subset = X[term_subset, :][:, subset]
        W = random_state.rand(len(term_subset), 2)
        H = random_state.rand(2, len(subset))
        W, H = nmfsh_comb_rank2(X_subset, W, H, anls_alg=anls_alg, vec_norm=vec_norm, normW=normW, tol=tol,
                                maxiter=maxiter, dtype=dtype)
        cluster_subset = np.argmax(H, axis=0)
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        W_buffer_one[term_subset, :] = W
        H_buffer_one = H
        if len(np.unique(cluster_subset)) > 1:
            priority_one = compute_priority(W_parent, W_buffer_one, dtype=dtype)
        else:
            priority_one = -1

    return cluster_subset, W_buffer_one, H_buffer_one, priority_one


def compute_priority(W_parent, W_child, dtype: Union[np.float32, np.float64]):
    n = len(W_parent)
    idx_parent = np.argsort(W_parent)[::-1]
    sorted_parent = W_parent[idx_parent]
    idx_child1 = np.argsort(W_child[:, 0])[::-1]
    idx_child2 = np.argsort(W_child[:, 1])[::-1]

    n_part = len(np.where(W_parent != 0)[0])
    if n_part <= 1:
        priority = -3
    else:
        weight = np.log(np.arange(n, 0, -1))
        first_zero = np.where(sorted_parent == 0)[0]
        if len(first_zero) > 0:
            weight[first_zero[0]:] = 1

        weight_part = np.zeros(n, dtype=dtype)
        weight_part[: n_part] = np.log(np.arange(n_part, 0, -1))
        idx1 = np.argsort(idx_child1)
        idx2 = np.argsort(idx_child2)
        max_pos = np.maximum(idx1, idx2)
        discount = np.log(n - max_pos[idx_parent] + 1)
        discount[discount == 0] = np.log(2)
        weight /= discount
        weight_part /= discount

        ndcg1 = NDCG_part(idx_parent, idx_child1, weight, weight_part)
        ndcg2 = NDCG_part(idx_parent, idx_child2, weight, weight_part)
        priority = ndcg1 * ndcg2

    return priority


def NDCG_part(ground, test, weight, weight_part):
    seq_idx = np.argsort(ground)
    weight_part = weight_part[seq_idx]

    n = len(test)
    uncum_score = weight_part[test]
    uncum_score[2:] /= np.log2(np.arange(2, n))
    cum_score = np.sum(uncum_score)

    ideal_score = np.sort(weight)[::-1]
    ideal_score[2:] /= np.log2(np.arange(2, n))
    cum_ideal_score = np.sum(ideal_score)

    score = cum_score / cum_ideal_score
    return score


def nmfsh_comb_rank2(A, Winit, Hinit, anls_alg: callable, vec_norm: float,
                     normW: bool, tol: float, maxiter: int,
                     dtype: Union[np.float32, np.float64]):
    """

    """
    eps = 1e-6
    m, n = A.shape
    W, H = Winit, Hinit

    if W.shape[1] != 2:
        warnings.warn("Error: Wrong size of W! Expected shape of (n, 2) but received W of shape ({}, {})"
                      .format(W.shape[0], W.shape[1]))

    if H.shape[0] != 2:
        warnings.warn("Error: Wrong size of H! Expected shape of (2, n) but received H of shape ({}, {})"
                      .format(H.shape[0], H.shape[1]))

    left = H.dot(H.T)
    right = A.dot(H.T)
    pb = tqdm(desc="Fitting 2-rank NMF of W and H", total=len(range(maxiter)), leave=False)
    for iter_ in range(maxiter):
        if matrix_rank(left) < 2:
            W = np.zeros((m, 2), dtype=dtype)
            H = np.zeros((2, n), dtype=dtype)
            if sp.issparse(A):
                U, S, V = svd(A.toarray(), full_matrices=False)
            else:
                U, S, V = svd(A, full_matrices=False)
            U, V = U[:, 0], V[0, :]
            if sum(U) < 0:
                U, V = -U, -V

            W[:, 0] = U
            H[0, :] = V
            pb.close()
            return W, H

        W = anls_alg(left, right, W, dtype=dtype)
        norms_W = norm(W, axis=0)
        if np.min(norms_W) < eps:
            tqdm.write('Error: Some column of W is essentially zero')

        W *= 1.0 / norms_W
        left = W.T.dot(W)
        right = A.T.dot(W)
        if matrix_rank(left) < 2:

            W = np.zeros((m, 2), dtype=dtype)
            H = np.zeros((2, n), dtype=dtype)
            if sp.issparse(A):
                U, S, V = svd(A.toarray(), full_matrices=False)
            else:
                U, S, V = svd(A, full_matrices=False)
            U, V = U[:, 0], V[0, :]
            if sum(U) < 0:
                U, V = -U, -V

            W[:, 0] = U
            H[0, :] = V
            pb.close()
            return W, H

        H = anls_alg(left, right, H.T, dtype=dtype).T
        gradH = left.dot(H) - right.T
        left = H.dot(H.T)
        right = A.dot(H.T)
        gradW = W.dot(left) - right

        if iter_ == 0:
            gradW_square = np.sum(np.power(gradW[np.logical_or(gradW <= 0, W > 0)], 2))
            gradH_square = np.sum(np.power(gradH[np.logical_or(gradH <= 0, H > 0)], 2))
            initgrad = np.sqrt(gradW_square + gradH_square)
            pb.update(1)
            continue
        else:
            gradW_square = np.sum(np.power(gradW[np.logical_or(gradW <= 0, W > 0)], 2))
            gradH_square = np.sum(np.power(gradH[np.logical_or(gradH <= 0, H > 0)], 2))
            projnorm = np.sqrt(gradW_square + gradH_square)
            pb.update(1)

        if projnorm < tol * initgrad:
            break

    if vec_norm != 0:
        if normW:
            norms = np.power(np.sum(np.power(W, vec_norm), axis=0), 1 / vec_norm)
            W /= norms
            H *= norms[:, None]
        else:
            norms = np.power(np.sum(np.power(H, vec_norm), axis=1), 1 / vec_norm)
            W *= norms[None, :]
            H /= norms
    pb.close()
    return W, H


def tree_to_nx(tree: np.ndarray, weights: np.ndarray = None):
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("Root", name="Root", is_word=False, id="Root")
    for parent_node, row in enumerate(tree, start=0):
        # Here the ith row refers to the ith node as a parent
        parent_id = str(int(parent_node))
        parent_idx = int(parent_node)
        parent_name = "Node {}".format(parent_id)
        if row.sum() > 0:
            for child in row:

                child_id = str(int(child))
                child_idx = int(child)
                child_name = "Node {}".format(child_id)

                if parent_idx not in g.nodes:
                    g.add_node(parent_idx, is_word=False, name=parent_name, id=parent_id)
                if child_idx not in g.nodes:
                    g.add_node(child_idx, is_word=False, name=child_name, id=child_id)
                g.add_edge(parent_idx, child_idx)
                if weights is not None:
                    child_weight = weights[child_idx]
                    g.nodes[child_idx]['weight'] = child_weight

    g.add_edge("Root", 0)
    g.add_edge("Root", 1)
    return g


def handle_enums(param):
    if isinstance(param, Enum):
        return param.value
    else:
        return param


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
