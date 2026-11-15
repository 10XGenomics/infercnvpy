import warnings

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData, ImplicitModificationWarning
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

import infercnvpy as cnv
from infercnvpy.pl._chromosome_heatmap import _group_means
from infercnvpy.pl._heatmap_kernels import _rows_per_group, binned_sums, pixel_rows, sampled_rows

AX_KEYS = {"heatmap_ax", "groupby_ax", "gene_groups_ax", "colorbar_ax"}
both_plots = pytest.mark.parametrize("plot", [cnv.pl.chromosome_heatmap, cnv.pl.chromosome_heatmap_summary])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def adata_small() -> AnnData:
    """60 cells in uneven groups (one category unused, one cell without a group) x 12 windows on 3 chromosomes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 12)).astype(np.float32)
    X[np.abs(X) < 0.8] = 0
    groups = rng.choice(["b", "a", "c"], size=60, p=[0.6, 0.3, 0.1]).astype(object)
    groups[7] = None
    obs = pd.DataFrame({"group": pd.Categorical(groups, categories=["c", "b", "unused", "a"])})
    obs.index = obs.index.astype(str)
    return AnnData(
        obs=obs,
        obsm={"X_cnv": sp.csr_matrix(X)},
        uns={"cnv": {"chr_pos": {"chr2": 5, "chr1": 0, "chr3": 9}}, "group_colors": ["red", "green", "blue", "black"]},
    )


def _image(axes) -> np.ndarray:
    return np.asarray(axes["heatmap_ax"].images[0].get_array())


def _dask(X, block_rows=7):
    """A dask array with row blocks of `block_rows` (CSR blocks for sparse `X`)."""
    return da.from_array(sp.csr_matrix(X) if sp.issparse(X) else X, chunks=(block_rows, -1), asarray=False)


# --- data ---------------------------------------------------------------------------------------------------------


def test_rows_per_group():
    counts = np.array([500, 3, 0, 97, 1])
    np.testing.assert_array_equal(_rows_per_group(counts, 100), [83, 1, 0, 16, 1])  # proportional, >= 1 if not empty
    np.testing.assert_array_equal(_rows_per_group(counts, counts.sum()), counts)  # one row per cell


@pytest.mark.parametrize("fmt", [np.asarray, sp.csr_matrix, sp.csc_matrix, _dask, lambda X: _dask(sp.csr_matrix(X))])
@pytest.mark.parametrize("n_rows", [7, 20, 1000])
def test_pixel_rows_match_numpy(fmt, n_rows):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 9))
    X[np.abs(X) < 1] = 0
    codes = rng.choice([0, 1, 3, -1], size=120, p=[0.7, 0.2, 0.05, 0.05])  # group 2 empty, -1 no group
    order, bounds, rows = pixel_rows(codes, 4, n_rows)

    np.testing.assert_array_equal(np.sort(order), np.flatnonzero(codes >= 0))
    assert np.all(np.diff(codes[order]) >= 0)  # sorted by group
    assert np.all(np.diff(bounds) > 0)  # no empty pixel row
    assert rows[2] == 0
    assert np.all(rows[[0, 1, 3]] >= 1)
    assert rows.sum() == len(bounds) - 1
    starts = np.concatenate([[0], np.cumsum(np.bincount(codes[order], minlength=4))])
    assert set(bounds[:-1]) >= set(starts[:-1][rows > 0])  # each pixel row holds cells of one group

    runs = list(zip(bounds[:-1], bounds[1:], strict=True))
    sums = binned_sums(fmt(X), order, bounds)
    np.testing.assert_allclose(sums, np.stack([X[order[a:b]].sum(axis=0) for a, b in runs]))

    picks = order[(bounds[:-1] + bounds[1:]) // 2]
    picked, lo, hi = sampled_rows(fmt(X), picks)
    np.testing.assert_array_equal(picked, X[picks])
    stored = X if fmt is np.asarray or fmt is _dask else X[X != 0]  # sparse: stored values only, as before 0.7.0
    assert (lo, hi) == (stored.min(), stored.max())

    expected = pd.DataFrame(X[codes >= 0]).groupby(codes[codes >= 0]).mean().to_numpy()
    np.testing.assert_allclose(_group_means(fmt(X), codes, 4), expected)


# --- fast vs legacy -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [np.asarray, sp.csr_matrix])
@pytest.mark.parametrize("kwargs", [{}, {"standard_scale": "var"}, {"standard_scale": "obs", "log": True}])
@both_plots
def test_fast_draws_what_legacy_draws(adata_small, fmt, kwargs, plot):
    """With a row per cell, `mode="fast"` draws the cells of `mode="legacy"` in the same order and colours."""
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()  # legacy cannot draw cells without group
    adata_small.obsm["X_cnv"] = fmt(adata_small.obsm["X_cnv"].toarray())
    legacy = plot(adata_small, groupby="group", show=False, mode="legacy", **kwargs)
    fast = plot(adata_small, groupby="group", show=False, **kwargs)
    assert set(fast) == AX_KEYS
    legacy_values = _image(legacy)
    if plot is cnv.pl.chromosome_heatmap_summary:
        # legacy repeats each group 10 times, and sorts the groups by name instead of the order of the categories
        legacy_labels = [t.get_text() for t in legacy["groupby_ax"].get_yticklabels()]
        legacy_values = legacy_values[::10][[legacy_labels.index(c) for c in ["c", "b", "a"]]]
    np.testing.assert_allclose(_image(fast), legacy_values, rtol=1e-6, atol=1e-6)
    for key in ("vmin", "vcenter", "vmax"):
        assert getattr(fast["heatmap_ax"].images[0].norm, key) == pytest.approx(
            getattr(legacy["heatmap_ax"].images[0].norm, key)
        )
    assert [t.get_text() for t in fast["groupby_ax"].get_yticklabels()] == ["c", "b", "a"]


@pytest.mark.parametrize("dendrogram", [False, True])
def test_legacy_keeps_cell_order_within_groups(adata_small, dendrogram):
    """scanpy sorts with pandas' unstable default sort; legacy passes the cells already sorted, so it keeps them."""
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    adata_small.obsm["X_cnv"] = adata_small.obsm["X_cnv"].toarray()  # scanpy's dendrogram needs dense data here
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, mode="legacy", dendrogram=dendrogram)
    order = [t.get_text() for t in axes["groupby_ax"].get_yticklabels()]
    rank = pd.Categorical(adata_small.obs["group"], categories=order).codes
    np.testing.assert_array_equal(_image(axes), adata_small.obsm["X_cnv"][np.argsort(rank, kind="stable")])


def test_dendrogram_reuses_stored_one(adata_small):
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    adata_small.obsm["X_cnv"] = adata_small.obsm["X_cnv"].toarray()  # scanpy's dendrogram needs dense data here
    legacy = cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, mode="legacy", dendrogram=True)
    assert "dendrogram_group" in adata_small.uns  # scanpy stores it
    fast = cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, dendrogram=True)
    assert set(fast) == AX_KEYS | {"dendrogram_ax"}
    np.testing.assert_allclose(_image(fast), _image(legacy), rtol=1e-6)


def test_dendrogram_from_mean_profiles(adata_small):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", dendrogram=True, show=False)
    tmp = AnnData(adata_small.obsm["X_cnv"].toarray(), obs=adata_small.obs)
    tmp = tmp[tmp.obs["group"].notna()].copy()
    tmp.obs["group"] = tmp.obs["group"].cat.remove_unused_categories()
    expected = sc.tl.dendrogram(tmp, "group", var_names=tmp.var_names, inplace=False)["categories_ordered"]
    assert [t.get_text() for t in axes["groupby_ax"].get_yticklabels()] == expected
    # legacy clusters the group means in the order of their names, which can flip branches
    tmp.obsm["X_cnv"], tmp.uns["cnv"] = tmp.X, adata_small.uns["cnv"]
    legacy = cnv.pl.chromosome_heatmap_summary(tmp, groupby="group", dendrogram=True, show=False, mode="legacy")
    expected = [t.get_text() for t in legacy["groupby_ax"].get_yticklabels()]
    del adata_small.uns["dendrogram_group"]  # stored by the call above, and used by later ones
    summary = cnv.pl.chromosome_heatmap_summary(adata_small, groupby="group", dendrogram=True, show=False)
    assert [t.get_text() for t in summary["groupby_ax"].get_yticklabels()] == expected
    X, g = adata_small.obsm["X_cnv"].toarray(), adata_small.obs["group"]
    np.testing.assert_allclose(_image(summary), [X[(g == c).to_numpy()].mean(axis=0) for c in expected], rtol=1e-6)


def _many_windows(seed: int) -> AnnData:
    """400 cells in 8 groups x 300 windows: more windows than scanpy's `N_PCS`, so its dendrogram uses a PCA."""
    rng = np.random.default_rng(seed)
    profiles = rng.normal(size=(8, 300)) * (rng.random((8, 300)) < 0.3)
    labels = rng.integers(0, 8, 400)
    X = (profiles[labels] + rng.normal(scale=0.7, size=(400, 300))).astype(np.float32)
    groups = pd.Categorical([f"G{i}" for i in labels], categories=[f"G{i}" for i in (3, 0, 6, 1, 7, 4, 2, 5)])
    obs = pd.DataFrame({"g": groups}, index=[str(i) for i in range(400)])
    adata = AnnData(obs=obs, uns={"cnv": {"chr_pos": {"chr1": 0, "chr2": 150}}})
    adata.obsm["X_cnv"] = X
    return adata


@pytest.mark.parametrize("fmt", [np.asarray, sp.csr_matrix, _dask, lambda X: _dask(sp.csr_matrix(X), 50)])
@pytest.mark.parametrize("seed", [0, 1, 2])
@both_plots
def test_dendrogram_like_legacy_with_many_windows(fmt, seed, plot):
    legacy = plot(_many_windows(seed), groupby="g", dendrogram=True, show=False, mode="legacy")
    expected = [t.get_text() for t in legacy["groupby_ax"].get_yticklabels()]
    adata = _many_windows(seed)
    adata.obsm["X_cnv"] = fmt(adata.obsm["X_cnv"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # not scanpy's warning that it falls back to a PCA
        fast = plot(adata, groupby="g", dendrogram=True, show=False)
    assert [t.get_text() for t in fast["groupby_ax"].get_yticklabels()] == expected


@both_plots
def test_dendrogram_is_stored_and_reused(plot, monkeypatch):
    """The computed dendrogram is stored as `scanpy.tl.dendrogram` stores it, and later calls (also legacy) use it."""
    adata = _many_windows(0)
    first = plot(adata, groupby="g", dendrogram=True, show=False)
    order = [t.get_text() for t in first["groupby_ax"].get_yticklabels()]
    stored = adata.uns["dendrogram_g"]
    tmp = AnnData(
        np.random.default_rng(0).normal(size=(8, 3)),
        obs=pd.DataFrame({"g": adata.obs["g"].cat.categories.astype("category")}),
    )
    assert set(stored) == set(sc.tl.dendrogram(tmp, "g", inplace=False))
    assert stored["groupby"] == ["g"]
    assert stored["categories_ordered"] == order

    def fail(*args, **kwargs):
        raise AssertionError("recomputed")

    monkeypatch.setattr("infercnvpy.pl._chromosome_heatmap._dendrogram", fail)
    monkeypatch.setattr(sc.tl, "dendrogram", fail)
    for mode in ("fast", "legacy"):
        again = plot(adata, groupby="g", dendrogram=True, show=False, mode=mode)
        assert [t.get_text() for t in again["groupby_ax"].get_yticklabels()] == order


@pytest.mark.parametrize("kwargs", [{}, {"log": True}, {"aggregate": "mean"}])
def test_standard_scale_var_over_all_cells(adata_small, kwargs):
    """Each window is scaled with its minimum and maximum over all cells, not only the drawn ones."""
    if kwargs.get("log"):
        adata_small.obsm["X_cnv"] *= 0.2  # log1p needs values above -1
    axes = cnv.pl.chromosome_heatmap(
        adata_small, groupby="group", figsize=(4, 0.2), standard_scale="var", show=False, **kwargs
    )
    values = _image(axes)
    assert len(values) < 59
    raw = _image(cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(4, 0.2), show=False, **kwargs))
    X = adata_small.obsm["X_cnv"].toarray()
    X = np.log1p(X) if kwargs.get("log") else X
    lo, hi = X.min(axis=0), X.max(axis=0)
    np.testing.assert_allclose(values, (raw - lo) / (hi - lo), rtol=1e-6, atol=1e-7)


def test_legacy_stores_group_colors(adata_small):
    """scanpy gets an actual AnnData that shares `uns`, so the colours it picks land in the caller's `uns`."""
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    del adata_small.uns["group_colors"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ImplicitModificationWarning)
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, mode="legacy")
    assert "group_colors" in adata_small.uns


@both_plots
def test_unsupported_argument_falls_back_to_legacy(adata_small, plot):
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    with pytest.warns(UserWarning, match="mode='fast' does not support interpolation, using mode='legacy'"):
        axes = plot(adata_small, groupby="group", show=False, interpolation="none")
    assert "colorbar_ax" not in axes  # drawn by scanpy


@both_plots
def test_lazy_matrix_does_not_fall_back_to_legacy(adata_small, plot):
    """`mode="legacy"` loads the whole matrix as a dense table, so a lazy one raises instead of falling back."""
    adata_small.obsm["X_cnv"] = _dask(adata_small.obsm["X_cnv"])
    with pytest.raises(ValueError, match=r"does not support interpolation.*in memory"):
        plot(adata_small, groupby="group", show=False, interpolation="none")
    with pytest.raises(TypeError, match=r"\.compute\(\)"):
        plot(adata_small, groupby="group", show=False, mode="legacy")


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({}, "CNV (relative to reference)"),
        ({"standard_scale": "var"}, "CNV (relative to reference), scaled per window"),
    ],
)
def test_colorbar_label(adata_small, kwargs, label):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, **kwargs)
    assert axes["colorbar_ax"].get_ylabel() == label


def test_legacy_rejects_ax(adata_small):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="`ax` needs mode='fast'"):
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", ax=ax, mode="legacy")


def test_legacy_return_fig(adata_small):
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    assert isinstance(cnv.pl.chromosome_heatmap(adata_small, groupby="group", mode="legacy", return_fig=True), Figure)


# --- fast mode ----------------------------------------------------------------------------------------------------


def test_sample_picks_middle_cell(adata_small):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(4, 0.2), show=False)
    values = _image(axes)
    assert 3 <= len(values) < 59
    X = adata_small.obsm["X_cnv"].toarray()
    assert all(any(np.array_equal(row, cell) for cell in X) for row in values)  # whole cells, not means
    norm = axes["heatmap_ax"].images[0].norm
    assert (norm.vmin, norm.vmax) == (X[X != 0].min(), X[X != 0].max())  # colour scale of all cells


def test_mean(adata_small):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(4, 0.2), aggregate="mean", show=False)
    values = _image(axes)
    codes = adata_small.obs["group"].cat.codes.to_numpy().astype(np.int64)
    order, bounds, _ = pixel_rows(codes, 4, len(values))
    X = adata_small.obsm["X_cnv"].toarray()
    expected = [X[order[a:b]].mean(axis=0) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    np.testing.assert_allclose(values, expected, rtol=1e-6)
    norm = axes["heatmap_ax"].images[0].norm
    assert (norm.vmin, norm.vmax) == pytest.approx((np.min(expected), np.max(expected)))
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", aggregate="mean", vmin=-3, vmax=3, show=False)
    assert (axes["heatmap_ax"].images[0].norm.vmin, axes["heatmap_ax"].images[0].norm.vmax) == (-3, 3)


@pytest.mark.parametrize("aggregate", ["sample", "mean"])
@pytest.mark.parametrize("sparse", [False, True])
def test_dask_matches_in_memory(adata_small, aggregate, sparse):
    X = adata_small.obsm["X_cnv"]
    expected = _image(
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(4, 0.2), aggregate=aggregate, show=False)
    )
    adata_small.obsm["X_cnv"] = _dask(X if sparse else X.toarray())
    actual = _image(
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(4, 0.2), aggregate=aggregate, show=False)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    summary = _image(cnv.pl.chromosome_heatmap_summary(adata_small, groupby="group", show=False))
    adata_small.obsm["X_cnv"] = X
    np.testing.assert_allclose(
        summary, _image(cnv.pl.chromosome_heatmap_summary(adata_small, groupby="group", show=False))
    )


def test_swap_axes(adata_small):
    plain = cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False)
    swapped = cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(16, 10), swap_axes=True, show=False)
    assert set(swapped) == AX_KEYS
    np.testing.assert_array_equal(_image(swapped), _image(plain).T)  # 59 cells fit both ways
    assert [t.get_text() for t in swapped["groupby_ax"].get_xticklabels()] == ["c", "b", "a"]


def test_norm_options(adata_small):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", vmin=-5, vmax=2, vcenter=0.5, show=False)
    norm = axes["heatmap_ax"].images[0].norm
    assert (norm.vmin, norm.vcenter, norm.vmax) == (-5, 0.5, 2)
    custom = Normalize(-1, 1)
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", norm=custom, show_gene_labels=True, show=False)
    assert axes["heatmap_ax"].images[0].norm is custom


def test_var_group_rotation(adata_small):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", var_group_rotation=45, show=False)
    assert [t.get_rotation() for t in axes["gene_groups_ax"].texts] == [45] * 3
    assert [t.get_text() for t in axes["gene_groups_ax"].texts] == ["chr1", "chr2", "chr3"]


def test_group_colors(adata_small):
    axes = cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False)
    blocks = axes["groupby_ax"].images[0]
    assert [tuple(blocks.cmap(i)) for i in range(3)] == [
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.5019607843137255, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
    ]


@both_plots
def test_ax_and_return_fig(adata_small, plot):
    fig, (ax, other) = plt.subplots(1, 2)
    assert plot(adata_small, groupby="group", ax=ax, return_fig=True) is fig
    axes = plot(adata_small, groupby="group", ax=other, show=False)
    assert axes["heatmap_ax"] is other
    assert all(a.get_figure(root=True) is fig for a in axes.values())


def test_return_fig_without_ax(adata_small):
    fig = cnv.pl.chromosome_heatmap(adata_small, groupby="group", figsize=(5, 3), return_fig=True)
    assert isinstance(fig, Figure)
    assert tuple(fig.get_size_inches()) == (5, 3)


def test_save(adata_small, tmp_path):
    figdir = sc.settings.figdir
    sc.settings.figdir = tmp_path  # a property without deleter, so no monkeypatch
    try:
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, save=True)
        cnv.pl.chromosome_heatmap_summary(adata_small, groupby="group", show=False, save="_summary.png")
    finally:
        sc.settings.figdir = figdir
    ext = sc.settings.file_format_figs
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"heatmap.{ext}", "heatmap_summary.png"])


@pytest.mark.parametrize("mode", ["fast", "legacy"])
@both_plots
def test_save_like_scanpy(adata_small, tmp_path, plot, mode):
    """`save` names the file as `scanpy.pl.heatmap` does, and the figure is closed after saving unless returned."""
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    figdir, fmt = sc.settings.figdir, sc.settings.file_format_figs
    sc.settings.figdir, sc.settings.file_format_figs = tmp_path, "pdf"
    try:
        plot(adata_small, groupby="group", show=False, save=".png", mode=mode)
        plot(adata_small, groupby="group", show=False, save="", mode=mode)
        assert not plt.get_fignums()
        fig = plot(adata_small, groupby="group", save="_fig.svg", return_fig=True, mode=mode)
        assert plt.fignum_exists(fig.number)
        if mode == "fast":
            _, ax = plt.subplots()
            plot(adata_small, groupby="group", ax=ax, show=False, save="_ax")
            assert plt.fignum_exists(ax.get_figure(root=True).number)
    finally:
        sc.settings.figdir, sc.settings.file_format_figs = figdir, fmt
    expected = {"heatmap.png", "heatmap.pdf", "heatmap_fig.svg"} | ({"heatmap_ax.pdf"} if mode == "fast" else set())
    assert {p.name for p in tmp_path.iterdir()} == expected


@pytest.mark.parametrize("mode", ["fast", "legacy"])
def test_autosave(adata_small, tmp_path, mode):
    """With `scanpy.settings.autosave`, the heatmap is saved once (scanpy's own save does not leave a blank one)."""
    adata_small = adata_small[adata_small.obs["group"].notna()].copy()
    settings = sc.settings.figdir, sc.settings.file_format_figs, sc.settings.autosave
    sc.settings.figdir, sc.settings.file_format_figs, sc.settings.autosave = tmp_path, "png", True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cnv.pl.chromosome_heatmap(adata_small, groupby="group", show=False, mode=mode)
    finally:
        sc.settings.figdir, sc.settings.file_format_figs, sc.settings.autosave = settings
    assert [p.name for p in tmp_path.iterdir()] == ["heatmap.png"]
    assert (plt.imread(tmp_path / "heatmap.png")[..., :3] < 0.9).any()
    assert not plt.get_fignums()


def test_bad_arguments(adata_small):
    with pytest.raises(ValueError, match="tl.leiden"):
        cnv.pl.chromosome_heatmap(adata_small)
    with pytest.raises(ValueError, match="`mode`"):
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", mode="slow")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`aggregate`"):
        cnv.pl.chromosome_heatmap(adata_small, groupby="group", aggregate="max")  # type: ignore[arg-type]


# --- on infercnv results ------------------------------------------------------------------------------------------


def test_plot_chromosome_heatmap(adata_infercnv):
    cnv.pl.chromosome_heatmap(adata_infercnv, show=False)


def test_plot_chromosome_heatmap_summary(adata_infercnv):
    cnv.pl.chromosome_heatmap_summary(adata_infercnv, show=False)


# Grouped by the annotated cell types, not leiden clusters, so the baseline does not depend on the igraph version.
@both_plots
def test_chromosome_heatmap_image(_adata_infercnv, image_comparer, plot):
    plot(_adata_infercnv.copy(), groupby="cell_type", figsize=(8, 5), show=False)
    image_comparer(plot.__name__)


def _label_boxes(axes):
    """Window extents of the drawn group labels and chromosome labels, and the texts of the group labels."""
    fig = axes["heatmap_ax"].get_figure(root=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    group_ax = axes["groupby_ax"]
    ticks = [*group_ax.xaxis.get_major_ticks(), *group_ax.yaxis.get_major_ticks()]
    groups = [t.label1 for t in ticks if t.label1.get_visible() and t.label1.get_text()]
    chroms = [t for t in axes["gene_groups_ax"].texts if t.get_visible()]
    return [[t.get_window_extent(renderer) for t in texts] for texts in (groups, chroms)], [
        t.get_text() for t in groups
    ]


@both_plots
@pytest.mark.parametrize("swap_axes", [False, True])
def test_labels_do_not_overlap(plot, swap_axes):
    """Labels of groups and chromosomes too small for their text are left out, those of larger ones first."""
    sizes = {"large group": 150, "tiny 1": 1, "tiny 2": 2, "tiny 3": 1, "medium group": 80, "tiny 4": 1, "last": 60}
    groups = np.repeat(list(sizes), list(sizes.values()))
    windows = np.r_[np.repeat(np.arange(19), 20), np.arange(19, 24)]  # chr20 to chr24 have one window each
    adata = AnnData(
        obs=pd.DataFrame({"group": pd.Categorical(groups, categories=list(sizes))}, index=map(str, range(len(groups)))),
        obsm={"X_cnv": np.random.default_rng(0).normal(size=(len(groups), len(windows)))},
        uns={"cnv": {"chr_pos": {f"chr{c + 1}": int(np.argmax(windows == c)) for c in range(24)}}},
    )
    axes = plot(adata, groupby="group", figsize=(4, 3), swap_axes=swap_axes, show=False)
    boxes, shown = _label_boxes(axes)
    for bbs in boxes:
        for i, a in enumerate(bbs):
            for b in bbs[i + 1 :]:
                assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1)
    assert len(boxes[1]) < 24
    if plot is cnv.pl.chromosome_heatmap:
        assert {"large group", "medium group", "last"} <= set(shown) < set(sizes)


@both_plots
@pytest.mark.parametrize("swap_axes", [False, True])
def test_labels_of_normal_groups_all_drawn(adata_small, plot, swap_axes):
    boxes, shown = _label_boxes(plot(adata_small, groupby="group", swap_axes=swap_axes, show=False))
    assert shown == ["c", "b", "a"]
    assert len(boxes[1]) == 3
