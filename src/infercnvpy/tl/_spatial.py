"""Spatial tools: smooth a CNV matrix over a spatial neighbor graph, and regularize labels with a Potts prior.

Both read the graph from `adata.obsp`, e.g. `obsp["spatial_connectivities"]` from :func:`squidpy.gr.spatial_neighbors_knn`;
infercnvpy itself does not depend on squidpy.

`_hop_csr` / `_hop_dense` compute one smoothing step ``out = S @ X`` with ``S = diag(a) + diag((1 - a) / d) @ A``
(`A` the graph without its diagonal, `d` its row sums, `a` the self weight per cell) without forming `S`: the row
normalization and the self weight are applied on the fly while each output row gathers its neighbors' rows, so a
CSR input is read in place and never densified. `_icm` runs sequential iterated conditional modes sweeps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from infercnvpy._logging import log, log_result
from infercnvpy._numba import njit, njit_parallel, numba_threads, prange
from infercnvpy._util import _as_matrix, _listed, _obs_column, _params_for_uns, _reference_cells


@njit
def _row_weights(g_indptr, g_indices, g_data, i, alpha):
    """Self weight `a` and neighbor scale `(1 - a) / d` of cell `i` (diagonal entries of the graph are ignored)."""
    d = 0.0
    deg = 0
    for e in range(g_indptr[i], g_indptr[i + 1]):
        if g_indices[e] != i:
            d += g_data[e]
            deg += 1
    if d <= 0:  # no neighbors: keep the cell's own values
        return 1.0, 0.0
    a = alpha if alpha >= 0 else 1.0 / (deg + 1)
    return a, (1.0 - a) / d


@njit_parallel
def _hop_csr(g_indptr, g_indices, g_data, alpha, indptr, indices, data, out):  # pragma: no cover
    """`out = S @ X` for the CSR matrix `X` (`indptr`, `indices`, `data`), see the module docstring."""
    for i in prange(out.shape[0]):
        a, scale = _row_weights(g_indptr, g_indices, g_data, i, alpha)
        row = out[i]
        row[:] = 0
        for k in range(indptr[i], indptr[i + 1]):
            row[indices[k]] += a * data[k]
        for e in range(g_indptr[i], g_indptr[i + 1]):
            j = g_indices[e]
            if j != i:
                w = scale * g_data[e]
                for k in range(indptr[j], indptr[j + 1]):
                    row[indices[k]] += w * data[k]


@njit_parallel
def _hop_dense(g_indptr, g_indices, g_data, alpha, X, out):  # pragma: no cover
    """`out = S @ X` for the dense matrix `X`."""
    n_cols = X.shape[1]
    for i in prange(out.shape[0]):
        a, scale = _row_weights(g_indptr, g_indices, g_data, i, alpha)
        row = out[i]
        for c in range(n_cols):
            row[c] = a * X[i, c]
        for e in range(g_indptr[i], g_indptr[i + 1]):
            j = g_indices[e]
            if j != i:
                w = scale * g_data[e]
                for c in range(n_cols):
                    row[c] += w * X[j, c]


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
    G = adata.obsp[connectivity_key]
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


def smooth_cnv(
    adata: AnnData,
    *,
    use_rep: str = "cnv",
    connectivity_key: str = "spatial_connectivities",
    key_added: str = "cnv_smoothed",
    alpha: float | None = None,
    n_hops: int = 1,
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    inplace: bool = True,
    n_jobs: int | None = None,
) -> np.ndarray | None:
    r"""Smooth a CNV matrix over a spatial neighbor graph.

    Each cell's CNV profile is replaced by a weighted mean of its own profile and those of its
    neighbors in the graph `adata.obsp[connectivity_key]`. One step is

    .. math::

        y_i = a_i x_i + (1 - a_i) \frac{\sum_j w_{ij} x_j}{\sum_j w_{ij}}

    with the graph weights :math:`w_{ij}` (the diagonal is ignored) and the self weight
    :math:`a_i` (`alpha`). With `n_hops > 1` the step is repeated on its own output, which
    widens the neighborhood to cells up to `n_hops` edges away. Cells without neighbors keep
    their values.

    On imaging-based spatial data (Xenium, CosMx, ...) a cell has a few hundred counts from a
    gene panel, so single-cell CNV profiles are noisy. Neighboring cells in a tumor usually
    share the same copy number state, and averaging over them recovers the signal. Related
    methods average before the CNV inference instead. InSituCNV :cite:`Jensen2025` smooths the
    normalized expression over a kNN graph of expression similarity and then runs infercnvpy,
    and SlideCNA :cite:`Zhang2025` pools the expression of neighboring beads. Han et al.
    :cite:`Han2026` found that aggregating cells on a grid improved CNV recovery on imaging
    platforms compared to native single-cell resolution. Smoothing also blurs the boundary
    between tumor and normal tissue by about `n_hops` cells.

    By default, :func:`infercnvpy.tl.infercnv` sets the values in its noise band to 0, and
    this function averages those zeros together with the values outside the band. To smooth the
    unfiltered values, run `tl.infercnv(..., dynamic_threshold=None)`.

    The graph usually comes from :func:`squidpy.gr.spatial_neighbors_knn`, e.g.
    `sq.gr.spatial_neighbors_knn(adata, n_neighs=10)`, or from
    :func:`squidpy.gr.spatial_neighbors_delaunay`. Any cells x cells graph with non-negative
    weights works, as long as the weights are similarities or binary connectivities (a larger
    weight means a closer neighbor). Distances, e.g. `obsp["spatial_distances"]`, give the
    farthest neighbors the largest weight and are rejected.

    A correctly annotated normal cell next to a tumor cell shares no copy number change with
    it, so smoothing over the edge between them dilutes the tumor cell and moves tumor signal
    into the normal cell. With `reference_key` and `reference_cat`, usually the same as in
    :func:`infercnvpy.tl.infercnv`, the edges between the reference cells and the other cells
    are dropped before smoothing. The edges within each of the two groups stay, so that
    reference cells away from the tumor are smoothed like the other cells. Cells at the border
    keep fewer neighbors, so with the default `alpha` their own profile weighs more. Cells left
    without neighbors keep their unsmoothed values. The mask has two groups and not one per
    cell type, so that tumor cells annotated as different types still share edges.

    The mask is only as good as the reference labels. Normal cells outside `reference_cat`
    (e.g. normal epithelium next to a carcinoma, or cells with `NaN`) still share edges with
    the tumor, and a tumor cell labeled as a reference type is smoothed with the reference
    cells. The mask also does not remove tumor signal that a normal cell carries in its own
    counts, e.g. from spillover or from a segmentation that merged it with a tumor cell. Before
    you set a threshold on the smoothed values, compare the reference cells near the tumor with
    those far from it in two ways. First, compare their unsmoothed values, which include the
    cells right at the border. Second, compare their smoothed values among cells with the same
    number of neighbors after the mask, because reference cells at the border keep fewer
    neighbors, so their smoothed values are noisier and score higher even without tumor signal.
    Take the threshold from the far reference cells only if the near ones score higher in one
    of these comparisons, because then they carry tumor signal in their own counts. Even with
    the mask, more of the normal cells at the border can pass a threshold after smoothing than
    without smoothing, because averaging them with their normal neighbors cuts the noise around
    the signal in their own counts. A call at the border therefore needs evidence per cell
    beyond the CNV profile, e.g. marker genes or morphology.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_rep
        Smooth `adata.obsm["X_{use_rep}"]`, e.g. the result of :func:`infercnvpy.tl.infercnv`.
        A numpy array or a scipy sparse matrix; a CSR matrix is read in place.
    connectivity_key
        Key of the cells x cells graph in `adata.obsp`: connectivities or similarities, not
        distances.
    key_added
        The result is stored in `adata.obsm["X_{key_added}"]` and `adata.uns[key_added]`.
    alpha
        Weight of the cell itself, between 0 and 1. Default: the cell counts as one more
        neighbor, i.e. `1 / (n_neighbors + 1)`, counting the neighbors left after the
        reference mask; with binary connectivities this is the plain mean over the cell and its
        neighbors.
    n_hops
        Number of smoothing steps.
    reference_key
        Column of `adata.obs` that annotates the normal (reference) cells. Give it together with
        `reference_cat` to drop the edges between the reference cells and the other cells.
        Default: smooth over the full graph.
    reference_cat
        One value or a list of values of `adata.obs[reference_key]` that mark the reference
        cells. Cells with any other value or `NaN` form the other group.
    inplace
        If `True`, store the result in adata, otherwise return it.
    n_jobs
        Number of threads. Default: all cores (numba's setting, see :func:`numba.set_num_threads`).

    Returns
    -------
    If `inplace=True`, returns `None` and sets the fields below.
    Otherwise returns the smoothed matrix.
    Sets the following fields:

    `adata.obsm["X_{key_added}"]` : :class:`numpy.ndarray`
        Smoothed CNV values (float32), cells x windows. Dense, as smoothing fills in the zeros
        of a sparse CNV matrix.
    `adata.uns[key_added]` : :class:`dict`
        The entries of `adata.uns[use_rep]` (e.g. `chr_pos`, so that
        :func:`infercnvpy.pl.chromosome_heatmap` works with `use_rep=key_added`), with `params`
        replaced by the parameters of this call.
    """
    if f"X_{use_rep}" not in adata.obsm:
        raise KeyError(f"`adata.obsm['X_{use_rep}']` not found. Did you run `tl.infercnv`?")
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError(f"`alpha` must be between 0 and 1, got {alpha}.")
    if n_hops < 1:
        raise ValueError(f"`n_hops` must be at least 1, got {n_hops}.")
    reference_cat = _listed(reference_cat)
    ref = _reference_cells(adata, reference_key, reference_cat)
    G = _graph(adata, connectivity_key)
    X = _as_matrix(adata.obsm[f"X_{use_rep}"], f"`obsm['X_{use_rep}']`")
    if G.shape != (X.shape[0], X.shape[0]):
        raise ValueError(f"`adata.obsp[{connectivity_key!r}]` has shape {G.shape}, expected {(X.shape[0],) * 2}.")
    if ref is not None:
        _drop_cross_edges(G, ref)

    start = log.debug("smooth_cnv: %d cells x %d windows, %d hops", X.shape[0], X.shape[1], n_hops)
    graph = (G.indptr, G.indices, G.data.astype(np.float64, copy=False), -1.0 if alpha is None else float(alpha))
    out = np.empty(X.shape, dtype=np.float32)
    with numba_threads(n_jobs):
        if sp.issparse(X):
            X = sp.csr_matrix(X)
            _hop_csr(*graph, X.indptr, X.indices, X.data, out)
        else:
            _hop_dense(*graph, np.ascontiguousarray(X), out)
        if n_hops > 1:
            buf = np.empty_like(out)
            for _ in range(n_hops - 1):
                _hop_dense(*graph, out, buf)
                out, buf = buf, out
    n_isolated = int(np.count_nonzero(np.diff(G.indptr) == 0))
    report = n_isolated > 0 or reference_key is not None  # with the mask, report the count even if it is 0
    log_result(
        "smooth_cnv",
        "smoothed %d cells over %d hops"
        + (" without the edges between reference and other cells" if reference_key is not None else "")
        + (", %d cells without neighbors kept their values" if report else ""),
        *([X.shape[0], n_hops] + ([n_isolated] if report else [])),
        time=start,
    )

    if not inplace:
        return out
    adata.obsm[f"X_{key_added}"] = out
    uns = {k: v for k, v in adata.uns.get(use_rep, {}).items() if k != "params"}
    uns["params"] = _params_for_uns(
        use_rep=use_rep,
        connectivity_key=connectivity_key,
        alpha=alpha,
        n_hops=n_hops,
        reference_key=reference_key,
        reference_cat=reference_cat,
    )
    adata.uns[key_added] = uns
    return None


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
