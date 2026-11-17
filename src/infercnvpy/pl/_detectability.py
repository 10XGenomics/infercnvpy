from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from infercnvpy.pl._chromosome_heatmap import _figure, _finish
from infercnvpy.tl._infercnv import _natural_sort

_METRICS = {
    "cells_gain": "cells to detect the gain",
    "cells_loss": "cells to detect the loss",
    "gain_ratio": "smallest detectable gain (copy ratio)",
    "loss_ratio": "mildest detectable loss (copy ratio)",
    "info": "information per cell",
    "null_var": "null variance",
}
_UNREACHABLE = "#bdbdbd"
_UNREACHABLE_LABEL = "not detectable or not measured"


def detectability(
    data: AnnData | pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    metric: Literal["cells_gain", "cells_loss", "gain_ratio", "loss_ratio", "info", "null_var"] = "cells_gain",
    key: str = "detectability",
    ideal: bool = False,
    kind: Literal["auto", "heatmap", "bar"] = "auto",
    min_genes: int = 5,
    cmap: str | Colormap = "Blues",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool | None = None,
) -> Figure | Axes | None:
    """Plot the result of :func:`infercnvpy.tl.detectability` per chromosome arm.

    One condition (a group at a depth) gives a bar chart, with a bar per arm for the achievable value and a black tick
    for the ideal counting bound, so that the gap between them is visible. Several conditions (groups, depths, or
    several panels passed as a dict of tables) give a heatmap with arms as columns and conditions as rows. Arms with
    fewer than `min_genes` panel genes are hatched, and values that no copy ratio or number of cells attains are
    gray.

    Parameters
    ----------
    data
        AnnData with the result in `adata.uns[key]`, or a table that :func:`infercnvpy.tl.detectability` returns with
        `inplace=False`, or a dict of such tables, one per panel for example (the keys label the rows).
    metric
        Column to plot: `cells_gain`, `cells_loss` (log scale), `gain_ratio`, `loss_ratio` (log scale),
        `info` or `null_var`.
    key
        Key of the result in `adata.uns`.
    ideal
        Plot the ideal counting bound (the `_ideal` columns) instead of the achievable value. Only for the heatmap,
        since the bar chart shows both.
    kind
        `"bar"`, `"heatmap"`, or `"auto"` (a bar chart for one condition, a heatmap for several).
    min_genes
        Arms with fewer panel genes are hatched.
    cmap
        Color map of the heatmap.
    figsize
        Size of the figure, if `ax` is not given. Default: from the number of arms and rows.
    ax
        Axes to draw into.
    show
        Show the plot. Default: `scanpy.settings.autoshow`.
    save
        Save the figure as `<figdir>/detectability<save>`, as scanpy does.
    return_fig
        Return the figure.

    Returns
    -------
    The figure if `return_fig` is true, the axes if `show` is false, otherwise `None`.

    See Also
    --------
    infercnvpy.tl.detectability : compute the table.

    Examples
    --------
    >>> cnv.tl.detectability(adata, groupby="cell_type", reference_key="cell_type", reference_cat=["T cell", "B cell"])
    >>> cnv.pl.detectability(adata, metric="cells_loss")

    Several panels side by side, from tables computed with `inplace=False`:

    >>> cnv.pl.detectability({"panel A": table_a, "panel B": table_b})
    """
    if metric not in _METRICS:
        raise ValueError(f"`metric` must be one of {', '.join(_METRICS)}, got {metric!r}.")
    if kind not in ("auto", "bar", "heatmap"):
        raise ValueError(f"`kind` must be 'auto', 'bar' or 'heatmap', got {kind!r}.")
    if ideal and metric in ("info", "null_var"):
        raise ValueError(f"`metric={metric!r}` has no ideal column; `ideal=True` is for cells and ratios.")
    table = _table(data, key)
    table["_row"] = _conditions(table)
    if table.duplicated(["_row", "arm"]).any():
        raise ValueError(
            "The table has several rows per condition and arm, for example from calls with different `at_n_cells` or "
            "`alpha`. Pass the tables as a dict, `{label: table}`, to label them."
        )
    n_rows = table["_row"].nunique()
    if kind == "auto":
        kind = "bar" if n_rows == 1 else "heatmap"
    if kind == "bar" and n_rows > 1:
        raise ValueError("A bar chart shows one condition; select one group and depth, or use `kind='heatmap'`.")
    if kind == "bar" and ideal:
        raise ValueError("The bar chart shows the ideal bound as ticks already; `ideal=True` is for the heatmap.")
    arms = _natural_sort(set(table["arm"].astype(str)))

    own = ax is None
    if figsize is None:
        figsize = (max(6.0, 0.25 * len(arms) + 2.5), 3.5 if kind == "bar" else 0.35 * n_rows + 2.2)
    fig, ax = _figure(ax, figsize)
    if own:
        fig.set_layout_engine("constrained")
    if kind == "bar":
        handles = _bars(ax, table, arms, metric, min_genes)
    else:
        handles = _heatmap(ax, table, arms, metric, ideal, min_genes, cmap)
    if own:
        fig.legend(handles=handles, loc="outside lower left", frameon=False, fontsize="small", ncols=len(handles))
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 0), frameon=False, fontsize="small")
    return _finish(fig, ax, show=show, save=save, return_fig=return_fig, name="detectability", own=own)


def _table(data, key: str) -> pd.DataFrame:
    """The detectability table(s) of `data` as one frame, with `panel` for a dict and index levels as columns."""
    if isinstance(data, AnnData):
        if key not in data.uns or "table" not in data.uns[key]:
            raise KeyError(f"`adata.uns[{key!r}]` has no detectability table. Run `tl.detectability` first.")
        data = data.uns[key]["table"]
    if isinstance(data, Mapping):
        data = pd.concat([_table(df, key).assign(panel=label) for label, df in data.items()], ignore_index=True)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be an AnnData, a DataFrame or a dict of DataFrames.")
    named = [n for n in data.index.names if n is not None and n not in data.columns]
    return data.reset_index(level=named or None, drop=not named).reset_index(drop=True)


def _conditions(table: pd.DataFrame) -> pd.Series:
    """A label per row from the columns that vary: panel, group and depth."""
    cols = [c for c in ("panel", "group", "depth") if c in table and table[c].nunique() > 1]
    if not cols:
        return pd.Series("", index=table.index)
    parts = [table[c].map(lambda d: f"{d:g} counts") if c == "depth" else table[c].astype(str) for c in cols]
    return pd.Series([" | ".join(p) for p in zip(*parts, strict=True)], index=table.index)


def _bars(ax: Axes, table: pd.DataFrame, arms: list[str], metric: str, min_genes: int) -> list:
    t = table.assign(arm=table["arm"].astype(str)).set_index("arm").reindex(arms)
    value = t[metric].to_numpy(dtype=float)
    ok = np.isfinite(value)
    few = t["n_genes"].to_numpy() < min_genes
    log = metric.startswith("cells") or metric.endswith("ratio")
    base = 1.0 if log else 0.0
    x = np.arange(len(arms))
    bars = ax.bar(x[ok], value[ok] - base, bottom=base, width=0.7, color="#3a7dc9", label="achievable")
    for b in np.asarray(bars.patches)[few[ok]]:
        b.set_hatch("///")
        b.set_facecolor("#9ec2ea")
    if f"{metric}_ideal" in t:
        ideal = t[f"{metric}_ideal"].to_numpy(dtype=float)
        good = np.isfinite(ideal)
        ax.scatter(x[good], ideal[good], marker="_", s=120, color="black", linewidths=2, label="ideal (counting only)")
    if log:
        ax.set_yscale("log")
    if (~ok).any():
        ax.scatter(x[~ok], np.full((~ok).sum(), ax.get_ylim()[1]), marker="x", color=_UNREACHABLE, clip_on=False,
                   label=_UNREACHABLE_LABEL)  # fmt: skip
    ax.axhline(base, color="black", linewidth=0.8)
    ax.set_xticks(x, arms, rotation=90)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel(_METRICS[metric])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = ax.get_legend_handles_labels()[0]
    if few[ok].any():
        handles.append(Patch(facecolor="#9ec2ea", hatch="///", label=f"fewer than {min_genes} genes"))
    return handles


def _heatmap(ax: Axes, table, arms: list[str], metric: str, ideal: bool, min_genes: int, cmap: str | Colormap) -> list:
    column = f"{metric}_ideal" if ideal and f"{metric}_ideal" in table else metric
    labels = list(dict.fromkeys(table["_row"]))
    wide = table.assign(arm=table["arm"].astype(str)).pivot(index="_row", columns="arm", values=[column, "n_genes"])
    grid = wide[column].reindex(index=labels, columns=arms).to_numpy(dtype=float)
    few = wide["n_genes"].reindex(index=labels, columns=arms).to_numpy(dtype=float) < min_genes
    ratio = metric.endswith("ratio")
    shown = np.ma.masked_invalid(np.abs(np.log2(grid)) if ratio else grid)
    norm = None
    if metric.startswith("cells") and shown.count():
        lo = max(float(shown.min()), 1.0)
        norm = mcolors.LogNorm(lo, max(float(shown.max()), 10 * lo))
    ax.set_facecolor(_UNREACHABLE)
    mesh = ax.pcolormesh(shown, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.5)
    for i, j in zip(*np.nonzero(few), strict=True):
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, hatch="///", edgecolor="#555555", linewidth=0))
    if grid.size <= 300:
        for i, j in zip(*np.nonzero(~np.ma.getmaskarray(shown)), strict=True):
            v = grid[i, j]
            text = f"{v:.2f}" if ratio else f"{v:.3g}" if v < 1e4 else f"{v:.0e}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=6, color=_ink(mesh, shown[i, j]))
    ax.set_xticks(np.arange(len(arms)) + 0.5, arms, rotation=90)
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    label = f"|log2|, {_METRICS[metric].removesuffix(' (copy ratio)')}" if ratio else _METRICS[metric]
    fig = ax.get_figure(root=True)
    assert fig is not None
    fig.colorbar(mesh, ax=ax, label=label + (" (ideal)" if column != metric else ""), shrink=0.9)
    return [
        Patch(facecolor=_UNREACHABLE, label=_UNREACHABLE_LABEL),
        Patch(facecolor="white", edgecolor="#555555", hatch="///", label=f"fewer than {min_genes} genes"),
    ]


def _ink(mesh, value: float) -> str:
    """Black or white text, whichever reads on the cell's color."""
    r, g, b, _ = mesh.cmap(mesh.norm(value))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
