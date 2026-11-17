"""`mode="legacy"` of the chromosome heatmaps: the implementation before 0.7.0, through :func:`scanpy.pl.heatmap`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import TwoSlopeNorm

from infercnvpy._util import _as_df, _as_matrix

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap


def legacy_heatmap(
    adata: AnnData,
    *,
    groupby: str,
    use_rep: str,
    cmap: str | Colormap,
    figsize: tuple[float, float],
    **kwargs,
) -> dict[str, Axes]:
    """Every cell as a row of :func:`scanpy.pl.heatmap`, in the original order within each group."""
    X = _as_matrix(adata.obsm[f"X_{use_rep}"], f"`adata.obsm['X_{use_rep}']`")
    obs = _as_df(adata.obs)
    # an actual AnnData (not a view) that shares `uns`, so the colors scanpy picks are stored in `adata.uns`
    order = _stable_group_order(AnnData(X=X, obs=obs, uns=adata.uns), groupby, kwargs.get("dendrogram", False))
    tmp_adata = AnnData(X=X[order], obs=obs.iloc[order], uns=adata.uns)
    values = X if isinstance(X, np.ndarray) else X.data
    return _scanpy_heatmap(
        tmp_adata,
        adata,
        use_rep,
        np.nanmin(values),
        np.nanmax(values),
        0,
        groupby=groupby,
        cmap=cmap,
        figsize=figsize,
        **kwargs,
    )


def _stable_group_order(adata: AnnData, groupby: str, dendrogram: bool | str) -> np.ndarray:
    """The cells sorted by group with a stable sort, in the group order `scanpy.pl.heatmap` will use.

    `scanpy.pl.heatmap` sorts the cells with pandas' default (unstable) sort, so the order of the cells within a
    group depends on the platform. pandas does not reorder an index that is already sorted, so cells passed in this
    order keep their original order within each group. With a dendrogram, the groups are in its order; it is
    computed here, as `scanpy.pl.heatmap` would, if `adata.uns` does not have it yet.
    """
    categories = _as_df(adata.obs)[groupby].astype("category").cat.categories
    if dendrogram:
        key = dendrogram if isinstance(dendrogram, str) else f"dendrogram_{groupby}"
        if key not in adata.uns:
            sc.tl.dendrogram(adata, groupby, key_added=key)
        categories = pd.Index(adata.uns[key]["categories_ordered"])
    rank = pd.Categorical(_as_df(adata.obs)[groupby], categories=categories).codes.astype(np.int64)
    rank[rank < 0] = len(categories)  # cells without a group last, as pandas sorts NaN
    return np.argsort(rank, kind="stable")


def legacy_summary(
    adata: AnnData,
    *,
    groupby: str,
    use_rep: str,
    cmap: str | Colormap,
    figsize: tuple[float, float],
    **kwargs,
) -> dict[str, Axes]:
    """Every group mean as 10 rows of :func:`scanpy.pl.heatmap`."""
    # this dirty hack repeats each row 10 times, since scanpy
    # heatmap cannot really handle it if there's just one observation
    # per row. Scanpy matrixplot is not an option, since it plots each
    # gene individually.
    X = _as_matrix(adata.obsm[f"X_{use_rep}"], f"`adata.obsm['X_{use_rep}']`")
    groups = adata.obs[groupby].unique()
    tmp_obs = pd.DataFrame()
    tmp_obs[groupby] = np.hstack([np.repeat(x, 10) for x in groups])
    tmp_obs.index = tmp_obs.index.astype(str)

    def _get_group_mean(group):
        group_mean = np.mean(X[adata.obs[groupby].values == group, :], axis=0)
        if len(group_mean.shape) == 1:
            # derived from an array instead of sparse matrix -> 1 dim instead of 2
            group_mean = group_mean[np.newaxis, :]
        return group_mean

    tmp_adata = sc.AnnData(
        X=np.vstack([np.repeat(_get_group_mean(group), 10, axis=0) for group in groups]), obs=tmp_obs, uns=adata.uns
    )

    return _scanpy_heatmap(
        tmp_adata,
        adata,
        use_rep,
        np.min(tmp_adata.X),
        np.max(tmp_adata.X),
        -1,
        groupby=groupby,
        cmap=cmap,
        figsize=figsize,
        **kwargs,
    )


def _scanpy_heatmap(
    tmp_adata: AnnData,
    adata: AnnData,
    use_rep: str,
    lo: float,
    hi: float,
    ymin: float,
    *,
    groupby: str,
    cmap: str | Colormap,
    figsize: tuple[float, float],
    **kwargs,
) -> dict[str, Axes]:
    """Draw `tmp_adata` with :func:`scanpy.pl.heatmap`, with chromosome annotations.

    Without a `norm`, the color scale is centered at `vcenter` (0), from `vmin` to `vmax` (default `lo` and `hi`).
    """
    # re-sort, as saving & loading anndata destroys the order
    chr_pos_dict = dict(sorted(adata.uns[use_rep]["chr_pos"].items(), key=lambda x: x[1]))
    chr_pos = list(chr_pos_dict.values())
    vmin, vmax, vcenter = (kwargs.pop(k, None) for k in ("vmin", "vmax", "vcenter"))
    if kwargs.get("norm") is None:
        kwargs["norm"] = two_slope_norm(0 if vcenter is None else vcenter, lo, hi, vmin, vmax)
    kwargs.pop("show_gene_labels", None)  # there are too many windows to label
    return_ax_dic = sc.pl.heatmap(
        tmp_adata,
        var_names=tmp_adata.var.index.values,
        groupby=groupby,
        figsize=figsize,
        cmap=cmap,
        show_gene_labels=False,
        var_group_positions=list(zip(chr_pos, chr_pos[1:] + [tmp_adata.shape[1]], strict=False)),
        var_group_labels=list(chr_pos_dict.keys()),
        show=False,
        save=False,
        **kwargs,
    )
    return_ax_dic["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=ymin, ymax=tmp_adata.shape[0])
    return return_ax_dic


def two_slope_norm(vcenter: float, lo: float, hi: float, vmin: float | None, vmax: float | None) -> TwoSlopeNorm:
    """`TwoSlopeNorm` at `vcenter` from `vmin` to `vmax`, by default the data range `lo` to `hi` around `vcenter`.

    The data range is widened to hold `vcenter`, as `TwoSlopeNorm` needs `vmin < vcenter < vmax`, e.g. for a matrix of
    zeros.
    """
    lo, hi = min(lo, vcenter - 1e-9), max(hi, vcenter + 1e-9)
    return TwoSlopeNorm(vcenter, vmin=lo if vmin is None else vmin, vmax=hi if vmax is None else vmax)
