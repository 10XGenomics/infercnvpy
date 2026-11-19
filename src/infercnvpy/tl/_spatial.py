"""Spatial tool: regularize labels with a Potts prior over a spatial neighbor graph.

It reads the graph from `adata.obsp`, e.g. `obsp["spatial_connectivities"]` from
:func:`squidpy.gr.spatial_neighbors_knn`; infercnvpy itself does not depend on squidpy. `_icm` runs sequential iterated
conditional modes sweeps. `_graph` and `_drop_cross_edges` also build the neighbor prior of
:func:`~infercnvpy.tl.copy_number`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from infercnvpy._logging import log, log_result
from infercnvpy._numba import njit
from infercnvpy._util import _is_dask, _obs_column, _params_for_uns


@njit
def _icm(indptr, indices, weights, unary, labels, beta, n_iter):  # pragma: no cover
    """Iterated conditional modes for a Potts model, in place on `labels` (-1 = unlabeled).

    Returns the sweeps run and whether the last one changed no label.

    Cells are visited in index order and each update is seen by the cells after it (Gauss-Seidel), so the result is
    deterministic. With symmetric `weights`, every update lowers the energy, so the sweeps stop at a local minimum.
    Ties keep the current label.
    """
    n, k = unary.shape
    e = np.empty(k)
    for sweep in range(n_iter):
        changed = 0
        for i in range(n):
            informative = False
            for c in range(k):
                e[c] = unary[i, c]
                if unary[i, c] != unary[i, 0]:
                    informative = True
            for p in range(indptr[i], indptr[i + 1]):
                lab = labels[indices[p]]
                if lab >= 0 and indices[p] != i:
                    # penalty beta * w for every class but the neighbor's, up to a constant per cell
                    e[lab] -= beta * weights[p]
                    informative = True
            current = labels[i]
            if not informative:
                continue
            best = current if current >= 0 else 0
            for c in range(k):
                if e[c] < e[best]:
                    best = c
            if best != current:
                labels[i] = best
                changed += 1
        if changed == 0:
            return sweep + 1, True
    return n_iter, False


def _graph(adata: AnnData, connectivity_key: str) -> sp.csr_matrix:
    if connectivity_key == "distances" or connectivity_key.endswith("_distances"):
        raise ValueError(
            f"`connectivity_key={connectivity_key!r}` looks like a distance matrix. The graph weights must be "
            "similarities or binary connectivities, where a larger weight means a closer neighbor, e.g. "
            f"{connectivity_key.removesuffix('distances') + 'connectivities'!r}."
        )
    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"`adata.obsp[{connectivity_key!r}]` not found. Compute a spatial graph first, "
            "e.g. with `squidpy.gr.spatial_neighbors_knn(adata)` "
            '(install with `pip install "infercnvpy[spatial]"`).'
        )
    G: Any = adata.obsp[connectivity_key]
    if _is_dask(G):  # `anndata.experimental.read_lazy`: the sparse cells x cells graph is small enough to load
        G = G.compute()
    G = sp.csr_matrix(G, copy=True) if sp.issparse(G) else sp.csr_matrix(np.asarray(G))
    if (G.data < 0).any():
        raise ValueError(f"`adata.obsp[{connectivity_key!r}]` must have non-negative weights.")
    G.data[G.indices == np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))] = 0  # a cell is not its own neighbor
    G.eliminate_zeros()  # nor are stored zeros
    return G


def _drop_cross_edges(G: sp.csr_matrix, ref: np.ndarray) -> None:
    """Drop, in place, the edges of `G` between the cells where `ref` is True and the other cells."""
    G.data *= ref[np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))] == ref[G.indices]
    G.eliminate_zeros()


def spatial_regions(
    adata: AnnData,
    *,
    key: str,
    connectivity_key: str = "spatial_connectivities",
    probabilities: str | None = None,
    beta: float = 2.0,
    n_iter: int = 20,
    key_added: str | None = None,
    inplace: bool = True,
) -> pd.Series | None:
    r"""Make per-cell labels spatially coherent with a Potts prior on a neighbor graph.

    Any categorical labels work, e.g. tumor/normal calls or CNV clusters from
    :func:`infercnvpy.tl.leiden`. The new labels are a local minimum of the energy of a hidden
    Markov random field with a Potts prior,

    .. math::

        E(l) = \sum_i U_i(l_i) + \beta \sum_{i < j} w_{ij} [l_i \neq l_j],

    where :math:`w_{ij}` are the graph weights, symmetrised and scaled so that the average
    cell's weights sum to 1. :math:`\beta` is the penalty for disagreeing with a whole
    neighborhood. The unary term :math:`U_i` is either

    - a label-flip penalty: 0 for the original label of the cell and 1 for any other label
      (default). For binary connectivities, two classes and a cell with the average number of
      neighbors, the cell changes its label when more than :math:`(1 + 1/\beta) / 2` of its
      neighbors disagree with it (75% for `beta=2`, 62.5% for `beta=4`).
    - the negative log of per-cell class probabilities (`probabilities`).

    Cells with a missing label (`NaN`) have no unary term and take the label their neighbors
    favor. The energy is minimized by iterated conditional modes (ICM, :cite:`Besag1986`):
    cells are visited in a fixed order and each takes the label with the lowest energy given
    its neighbors, until a sweep changes nothing or after `n_iter` sweeps. The result is
    deterministic and stays close to the input labels; it is not a new clustering.
    STARCH :cite:`Elyanow2021` and CalicoST :cite:`Ma2024` use the same kind of prior on clone labels.

    Parameters
    ----------
    adata
        Annotated data matrix.
    key
        Column of `adata.obs` with the labels, categorical or convertible to categorical.
    connectivity_key
        Key of the cells x cells graph in `adata.obsp`, e.g. from
        :func:`squidpy.gr.spatial_neighbors_knn`: connectivities or similarities, not distances.
    probabilities
        Key in `adata.obsm` with per-cell class probabilities or non-negative scores, cells x
        classes, in the order of the categories of `adata.obs[key]` (or a
        :class:`~pandas.DataFrame` with the categories as columns). Rows are normalized to sum
        to 1. Default: use the label-flip penalty.
    beta
        Strength of the spatial prior. `0` returns the input labels (or the most probable class).
    n_iter
        Maximum number of ICM sweeps.
    key_added
        Column of `adata.obs` for the result. Default: `f"{key}_spatial"`.
    inplace
        If `True`, store the result in adata, otherwise return it.

    Returns
    -------
    If `inplace=True`, returns `None` and sets the field below.
    Otherwise returns the labels as a categorical :class:`pandas.Series`.
    Sets the following fields:

    `adata.obs[key_added]` : :class:`pandas.Categorical`
        Spatially regularized labels, with the categories of `adata.obs[key]`.
    `adata.uns[key_added]["params"]` : :class:`dict`
        Parameters of this call and the number of sweeps run (`n_sweeps`).
    """
    if beta < 0:
        raise ValueError(f"`beta` must be non-negative, got {beta}.")
    key_added = f"{key}_spatial" if key_added is None else key_added
    labels_in = _obs_column(adata, key).astype("category")
    categories = labels_in.cat.categories
    labels = labels_in.cat.codes.to_numpy().astype(np.int64)
    k = len(categories)
    if probabilities is None:
        unary = np.ones((adata.n_obs, k))
        unary[labels >= 0, labels[labels >= 0]] = 0
        unary[labels < 0] = 0
    else:
        P: Any = adata.obsm[probabilities]
        P = P[list(categories)].to_numpy() if isinstance(P, pd.DataFrame) else np.asarray(P)
        if P.shape != (adata.n_obs, k):
            raise ValueError(f"`adata.obsm[{probabilities!r}]` has shape {P.shape}, expected {(adata.n_obs, k)}.")
        if (P < 0).any():
            raise ValueError(f"`adata.obsm[{probabilities!r}]` must be non-negative.")
        P = P / np.maximum(P.sum(axis=1, keepdims=True), np.finfo(float).tiny)
        unary = -np.log(np.maximum(P, 1e-12))
        missing = labels < 0
        labels[missing] = np.argmin(unary[missing], axis=1)

    G = _graph(adata, connectivity_key)
    if G.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(f"`adata.obsp[{connectivity_key!r}]` has shape {G.shape}, expected {(adata.n_obs,) * 2}.")
    W = ((G + G.T) / 2).tocsr()
    W.data *= adata.n_obs / max(W.data.sum(), np.finfo(float).tiny)  # the mean row sum is 1

    before = labels.copy()
    start = log.debug("spatial_regions: %d cells, %d classes, beta=%g", adata.n_obs, k, beta)
    n_sweeps, converged = 0, True
    if beta > 0:
        n_sweeps, converged = _icm(W.indptr, W.indices, W.data, unary, labels, float(beta), n_iter)
    n_changed = int(np.count_nonzero(labels != before))
    log_result(
        "spatial_regions",
        "changed %d of %d labels of %r in %d sweeps",
        *(n_changed, adata.n_obs, key, n_sweeps),
        time=start,
    )
    if not converged and n_iter > 0:
        log.info("spatial_regions: stopped after `n_iter=%d` sweeps before convergence", n_iter)

    result = pd.Series(pd.Categorical.from_codes(labels, categories=categories), index=adata.obs_names, name=key_added)
    if not inplace:
        return result
    adata.obs[key_added] = result
    adata.uns[key_added] = {
        "params": _params_for_uns(
            key=key,
            connectivity_key=connectivity_key,
            probabilities=probabilities,
            beta=beta,
            n_iter=n_iter,
            n_sweeps=n_sweeps,
        )
    }
    return None
