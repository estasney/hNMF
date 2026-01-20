import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy.linalg import matrix_rank, norm, svd
from numpy.random import mtrand
from scipy import sparse as sp
from sklearn.decomposition import non_negative_factorization

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

AnlsAlgorithm: TypeAlias = Callable[
    [
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.DTypeLike,
    ],
    npt.NDArray[np.float64],
]


def anls_entry_rank2_precompute(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float64]:
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
        logger.error(
            "Error: The 2x2 matrix is close to singular or the input data matrix has tiny values",
        )
    else:
        if np.abs(left[0, 0] >= np.abs(left[0, 1])):
            t = left[1, 0] / left[0, 0]
            a2 = left[0, 0] + t * left[1, 0]
            b2 = left[0, 1] + t * left[1, 1]
            d2 = left[1, 1] - t * left[0, 1]
            if np.abs(d2 / a2) < eps:
                logger.error("Error: The 2x2 matrix is close to singular")

            e2 = right[:, 0] + t * right[:, 1]
            f2 = right[:, 1] - t * right[:, 0]
        else:
            ct = left[0, 0] / left[1, 0]
            a2 = left[1, 0] + ct * left[0, 0]
            b2 = left[1, 1] + ct * left[0, 1]
            d2 = -left[0, 1] + ct * left[1, 1]
            if np.abs(d2 / a2) < eps:
                logger.error("Error: The 2x2 matrix is close to singular")

            e2 = right[:, 1] + ct * right[:, 0]
            f2 = -right[:, 0] + ct * right[:, 1]

        H[:, 1] = f2 * (1 / d2)
        H[:, 0] = (e2 - b2 * H[:, 1]) * (1 / a2)

    use_either = np.logical_not(np.all(H > 0, axis=1))
    H[use_either, :] = solve_either[use_either, :]

    return H


def trial_split_sklearn(
    min_priority: float,
    X: npt.NDArray,
    subset: npt.NDArray[np.int64],
    W_parent: npt.NDArray[np.float64],
    random_state: np.random.RandomState,
    trial_allowance: int,
    unbalanced: float,
    dtype: npt.DTypeLike,
    tol: float,
    maxiter: int,
    init: Literal[None, "random", "nndsvd", "nndsvda", "nndsvdar"],
    alpha_W: float,
    alpha_H: float | Literal["same"],
):
    m: int = X.shape[0]
    trial = 0
    subset_backup = subset
    W_buffer_one = np.zeros((m, 2), dtype=dtype)
    H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
    priority_one = -2.0
    while trial < trial_allowance:
        cluster_subset, W_buffer_one, H_buffer_one, priority_one = split_once_sklearn(
            X=X,
            subset=subset,
            W_parent=W_parent,
            random_state=random_state,
            dtype=dtype,
            tol=tol,
            maxiter=maxiter,
            init=init,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
        )
        if priority_one < 0:
            break

        unique_cluster_subset = np.unique(cluster_subset)
        if len(unique_cluster_subset) != 2:
            logger.error("Invalid number of unique sub-clusters!")

        length_cluster1 = len(np.where(cluster_subset == unique_cluster_subset[0])[0])
        length_cluster2 = len(np.where(cluster_subset == unique_cluster_subset[1])[0])
        if min(length_cluster1, length_cluster2) < unbalanced * len(cluster_subset):
            logger.debug(
                f"Below imbalanced threshold: {unbalanced * len(cluster_subset)}",
            )
            idx_small = np.argmin(np.array([length_cluster1, length_cluster2]))
            subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[
                0
            ]
            subset_small = subset[subset_small]
            _, _, _, priority_one_small = split_once_sklearn(
                X=X,
                subset=subset_small,
                W_parent=W_buffer_one[:, idx_small],
                random_state=random_state,
                dtype=dtype,
                tol=tol,
                maxiter=maxiter,
                init=init,
                alpha_W=0.0,
                alpha_H=0.0,
            )
            if priority_one_small < min_priority:
                trial += 1
                if trial < trial_allowance:
                    logger.debug(f"Dropped {len(subset_small)} features...")
                    subset = np.setdiff1d(subset, subset_small)
            else:
                break
        else:
            break

    if trial == trial_allowance:
        logger.debug(
            f"Reached trial allowance, recycled {len(subset_backup) - len(subset)} features",
        )
        subset = subset_backup
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -2

    return subset, W_buffer_one, H_buffer_one, priority_one


def split_once_sklearn(
    X: npt.NDArray,
    subset: npt.NDArray[np.int64],
    W_parent: npt.NDArray[np.float64],
    random_state: mtrand.RandomState,
    dtype: npt.DTypeLike,
    tol: float,
    maxiter: int,
    init: Literal[None, "random", "nndsvd", "nndsvda", "nndsvdar", "custom"],
    alpha_W: float,
    alpha_H: float | Literal["same"],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float
]:
    m = X.shape[0]
    if len(subset) <= 3:
        cluster_subset = np.ones(len(subset), dtype=dtype)
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -1
    else:
        term_subset = np.flatnonzero(np.sum(X[:, subset], axis=1))
        X_subset = X[term_subset, :][:, subset]
        W = random_state.rand(len(term_subset), 2)
        H = random_state.rand(2, len(subset))
        W, H, _n_iter = non_negative_factorization(
            X=X_subset,
            W=W,
            H=H,
            n_components=2,
            init=init,
            update_H=True,
            solver="cd",
            beta_loss=2,
            tol=tol,
            max_iter=maxiter,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=0.0,
            random_state=random_state,
            verbose=0,
            shuffle=False,
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


def trial_split(
    min_priority: float,
    X: npt.NDArray[np.float64],
    subset: npt.NDArray[np.int64],
    W_parent: npt.NDArray[np.float64],
    random_state: np.random.RandomState,
    trial_allowance: int,
    unbalanced: float,
    dtype: npt.DTypeLike,
    anls_alg: AnlsAlgorithm,
    vec_norm: float,
    normW: bool,
    tol: float,
    maxiter: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray, npt.NDArray, float]:
    m = X.shape[0]
    trial = 0
    subset_backup = subset
    W_buffer_one = np.zeros((m, 2), dtype=dtype)
    H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
    priority_one = -2.0
    while trial < trial_allowance:
        cluster_subset, W_buffer_one, H_buffer_one, priority_one = split_once(
            X=X,
            subset=subset,
            W_parent=W_parent,
            random_state=random_state,
            dtype=dtype,
            anls_alg=anls_alg,
            vec_norm=vec_norm,
            normW=normW,
            tol=tol,
            maxiter=maxiter,
        )
        if priority_one < 0:
            break

        unique_cluster_subset = np.unique(cluster_subset)
        if len(unique_cluster_subset) != 2:
            logger.warning("Invalid number of unique sub-clusters!")

        length_cluster1 = len(np.where(cluster_subset == unique_cluster_subset[0])[0])
        length_cluster2 = len(np.where(cluster_subset == unique_cluster_subset[1])[0])
        if min(length_cluster1, length_cluster2) < unbalanced * len(cluster_subset):
            idx_small = np.argmin(np.array([length_cluster1, length_cluster2]))
            subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[
                0
            ]
            subset_small = subset[subset_small]
            _, _, _, priority_one_small = split_once(
                X=X,
                subset=subset_small,
                W_parent=W_buffer_one[:, idx_small],
                random_state=random_state,
                dtype=dtype,
                anls_alg=anls_alg,
                vec_norm=vec_norm,
                normW=normW,
                maxiter=maxiter,
                tol=tol,
            )
            if priority_one_small < min_priority:
                trial += 1
                if trial < trial_allowance:
                    logger.info(f"Dropped {len(subset_small)} documents...")
                    subset = np.setdiff1d(subset, subset_small)
            else:
                break
        else:
            break

    if trial == trial_allowance:
        logger.info(f"Recycled {len(subset_backup) - len(subset)} documents...")
        subset = subset_backup
        W_buffer_one = np.zeros((m, 2), dtype=dtype)
        H_buffer_one = np.zeros((2, len(subset)), dtype=dtype)
        priority_one = -2

    return subset, W_buffer_one, H_buffer_one, priority_one


def split_once(
    X: npt.NDArray[np.float64],
    subset: npt.NDArray[np.int64],
    W_parent: npt.NDArray[np.float64],
    random_state: mtrand.RandomState,
    dtype: npt.DTypeLike,
    anls_alg: AnlsAlgorithm,
    vec_norm: float,
    normW: bool,
    tol: float,
    maxiter: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray, npt.NDArray, float]:
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
        W, H = nmfsh_comb_rank2(
            X_subset,
            W,
            H,
            anls_alg=anls_alg,
            vec_norm=vec_norm,
            normW=normW,
            tol=tol,
            maxiter=maxiter,
            dtype=dtype,
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


def compute_priority(
    W_parent: npt.NDArray[np.float64],
    W_child: npt.NDArray[np.float64],
    dtype: npt.DTypeLike,
) -> float:
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
            weight[first_zero[0] :] = 1

        weight_part = np.zeros(n, dtype=dtype)
        weight_part[:n_part] = np.log(np.arange(n_part, 0, -1))
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


def NDCG_part(
    ground: npt.NDArray[np.int64],
    test: npt.NDArray[np.int64],
    weight: npt.NDArray,
    weight_part: npt.NDArray,
) -> float:
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


def nmfsh_comb_rank2(
    A: npt.NDArray,
    Winit: npt.NDArray,
    Hinit: npt.NDArray,
    anls_alg: AnlsAlgorithm,
    vec_norm: float,
    normW: bool,
    tol: float,
    maxiter: int,
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """"""
    eps = 1e-6
    shape: tuple[int, int] = A.shape
    m, n = shape
    W, H = Winit, Hinit

    if W.shape[1] != 2:
        warnings.warn(
            f"Error: Wrong size of W! Expected shape of (n, 2) but received W of shape ({W.shape[0]}, {W.shape[1]})",
            stacklevel=2,
        )

    if H.shape[0] != 2:
        warnings.warn(
            f"Error: Wrong size of H! Expected shape of (2, n) but received H of shape ({H.shape[0]}, {H.shape[1]})",
            stacklevel=2,
        )

    left = H.dot(H.T)
    right = A.dot(H.T)
    for iter_ in range(maxiter):
        if matrix_rank(left) < 2:
            W = np.zeros((m, 2), dtype=dtype)
            H = np.zeros((2, n), dtype=dtype)
            if sp.issparse(A):
                U, _S, V = svd(A.toarray(), full_matrices=False)  # type: ignore[attr-defined]  # A can be sparse
            else:
                U, _S, V = svd(A, full_matrices=False)
            U, V = U[:, 0], V[0, :]
            if sum(U) < 0:
                U, V = -U, -V

            W[:, 0] = U
            H[0, :] = V

            return W, H

        W = anls_alg(left, right, W, dtype)
        norms_W = norm(W, axis=0)
        if np.min(norms_W) < eps:
            logger.warning("Error: Some column of W is essentially zero")

        W *= 1.0 / norms_W
        left = W.T.dot(W)
        right = A.T.dot(W)
        if matrix_rank(left) < 2:
            W = np.zeros((m, 2), dtype=dtype)
            H = np.zeros((2, n), dtype=dtype)
            if sp.issparse(A):
                U, _S, V = svd(A.toarray(), full_matrices=False)  # type: ignore[attr-defined]  # A can be sparse
            else:
                U, _S, V = svd(A, full_matrices=False)
            U, V = U[:, 0], V[0, :]
            if sum(U) < 0:
                U, V = -U, -V

            W[:, 0] = U
            H[0, :] = V

            return W, H

        H = anls_alg(left, right, H.T, dtype).T
        gradH = left.dot(H) - right.T
        left = H.dot(H.T)
        right = A.dot(H.T)
        gradW = W.dot(left) - right
        initgrad = 1
        if iter_ == 0:
            gradW_square = np.sum(np.power(gradW[np.logical_or(gradW <= 0, W > 0)], 2))
            gradH_square = np.sum(np.power(gradH[np.logical_or(gradH <= 0, H > 0)], 2))
            initgrad = np.sqrt(gradW_square + gradH_square)
            continue
        gradW_square = np.sum(np.power(gradW[np.logical_or(gradW <= 0, W > 0)], 2))
        gradH_square = np.sum(np.power(gradH[np.logical_or(gradH <= 0, H > 0)], 2))
        projnorm = np.sqrt(gradW_square + gradH_square)

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

    return W, H


def tree_to_nx(tree: npt.NDArray, weights: npt.NDArray | None = None) -> "nx.DiGraph":
    import networkx as nx

    g = nx.DiGraph()
    g.add_node("Root", name="Root", is_word=False, id="Root")
    for parent_node, row in enumerate(tree, start=0):
        # Here the ith row refers to the ith node as a parent
        parent_id = str(int(parent_node))
        parent_idx = int(parent_node)
        parent_name = f"Node {parent_id}"
        if row.sum() > 0:
            for child in row:
                child_id = str(int(child))
                child_idx = int(child)
                child_name = f"Node {child_id}"

                if parent_idx not in g.nodes:
                    g.add_node(
                        parent_idx,
                        is_word=False,
                        name=parent_name,
                        id=parent_id,
                    )
                if child_idx not in g.nodes:
                    g.add_node(child_idx, is_word=False, name=child_name, id=child_id)
                g.add_edge(parent_idx, child_idx)
                if weights is not None:
                    child_weight = weights[child_idx]
                    g.nodes[child_idx]["weight"] = child_weight

    g.add_edge("Root", 0)
    g.add_edge("Root", 1)
    return g
