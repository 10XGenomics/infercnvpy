import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from matplotlib import pyplot as plt

import infercnvpy as cnv
from infercnvpy.pl._spatial import _region_columns
from tests.conftest import knn_graph as _knn_graph


@pytest.fixture
def graph_and_cnv():
    rng = np.random.default_rng(0)
    n, m = 300, 40
    G = sp.random(n, n, density=0.03, random_state=1, format="csr") + sp.eye(n)  # weighted, with a diagonal
    G = sp.lil_matrix(G)
    G[5, :] = 0  # a cell without neighbors
    X = rng.normal(size=(n, m)) * (rng.random((n, m)) < 0.3)
    return sp.csr_matrix(G), X.astype(np.float32)


def test_spatial_cnv_on_infercnv_output(adata_infercnv):
    rng = np.random.default_rng(0)
    adata_infercnv.obsm["spatial"] = rng.random((adata_infercnv.n_obs, 2))
    cnv.pl.spatial_cnv(adata_infercnv, region=["chr1", "chr7:0-60_000_000"], color="cnv_leiden", show=False)


def test_window_positions(_adata_infercnv):
    uns = _adata_infercnv.uns["cnv"]
    start, end = uns["window_start"], uns["window_end"]
    assert len(start) == len(end) == _adata_infercnv.obsm["X_cnv"].shape[1]
    assert (end > start).all()
    bounds = [*sorted(uns["chr_pos"].values()), len(start)]
    for a, b in zip(bounds, bounds[1:], strict=False):
        assert (np.diff(start[a:b]) >= 0).all()


def _grid(side: int = 40, flip: float = 0.15, seed: int = 0):
    """Two halves of a grid (tumour on the left), with a fraction `flip` of the labels flipped."""
    rng = np.random.default_rng(seed)
    xy = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2).astype(float)
    truth = np.where(xy[:, 0] < side / 2, "tumour", "normal")
    noisy = truth.copy()
    flipped = rng.random(len(truth)) < flip
    noisy[flipped] = np.where(truth[flipped] == "tumour", "normal", "tumour")
    adata = AnnData(obs=pd.DataFrame({"call": pd.Categorical(noisy)}, index=[str(i) for i in range(len(xy))]))
    adata.obsm["spatial"] = xy
    adata.obsp["spatial_connectivities"] = _knn_graph(xy, k=8)
    return adata, truth


def test_spatial_regions_reduces_label_noise():
    adata, truth = _grid()
    before = np.mean(adata.obs["call"].to_numpy() == truth)
    cnv.tl.spatial_regions(adata, key="call", beta=4)
    after = np.mean(adata.obs["call_spatial"].to_numpy() == truth)
    assert before < 0.9
    assert after > 0.98
    assert list(adata.obs["call_spatial"].cat.categories) == list(adata.obs["call"].cat.categories)
    assert adata.uns["call_spatial"]["params"]["n_sweeps"] < 20


def test_spatial_regions_beta_zero_and_determinism():
    adata, _ = _grid()
    res0 = cnv.tl.spatial_regions(adata, key="call", beta=0, inplace=False)
    pd.testing.assert_series_equal(res0, adata.obs["call"], check_names=False)
    res1 = cnv.tl.spatial_regions(adata, key="call", beta=4, inplace=False)
    res2 = cnv.tl.spatial_regions(adata, key="call", beta=4, inplace=False)
    pd.testing.assert_series_equal(res1, res2)


def test_spatial_regions_probabilities_and_missing():
    adata, truth = _grid(flip=0.0)
    rng = np.random.default_rng(1)
    p_tumour = np.clip((truth == "tumour") * 0.4 + 0.3 + rng.normal(scale=0.2, size=len(truth)), 0.01, 0.99)
    adata.obsm["p"] = pd.DataFrame({"tumour": p_tumour, "normal": 1 - p_tumour}, index=adata.obs_names)
    calls = np.where(p_tumour > 0.5, "tumour", "normal").astype(object)
    calls[::7] = np.nan  # missing calls are filled from the probabilities and the neighbors
    adata.obs["call"] = pd.Categorical(calls)
    res = cnv.tl.spatial_regions(adata, key="call", probabilities="p", beta=4, inplace=False)
    assert res.notna().all()
    assert np.mean(res.to_numpy() == truth) > np.mean(calls == truth) + 0.1
    assert np.mean(res.to_numpy() == truth) > 0.97


def test_spatial_regions_fills_missing_labels():
    adata, truth = _grid(flip=0.0)
    calls = truth.astype(object)
    calls[::5] = np.nan
    adata.obs["call"] = pd.Categorical(calls)
    res = cnv.tl.spatial_regions(adata, key="call", inplace=False)
    assert res.notna().all()
    assert np.mean(res.to_numpy() == truth) > 0.99


def test_spatial_regions_leaves_graph_untouched():
    adata, _ = _grid()
    G = adata.obsp["spatial_connectivities"]
    G.setdiag(1.0)  # a graph with self loops stored, e.g. squidpy `set_diag=True`
    before = G.copy()
    cnv.tl.spatial_regions(adata, key="call", beta=4)
    after = adata.obsp["spatial_connectivities"]
    assert after.nnz == before.nnz
    assert len(after.data) == len(before.data)
    assert (after != before).nnz == 0


def test_spatial_regions_convergence_log(caplog):
    adata, _ = _grid()
    cnv.tl.spatial_regions(adata, key="call", beta=4)
    n_sweeps = adata.uns["call_spatial"]["params"]["n_sweeps"]
    with caplog.at_level(logging.INFO, logger="scverse.infercnvpy"):
        cnv.tl.spatial_regions(adata, key="call", beta=4, n_iter=n_sweeps)
    assert not any("before convergence" in r.getMessage() for r in caplog.records)
    with caplog.at_level(logging.INFO, logger="scverse.infercnvpy"):
        cnv.tl.spatial_regions(adata, key="call", beta=4, n_iter=n_sweeps - 1)
    assert any("before convergence" in r.getMessage() for r in caplog.records)


def test_graph_weights_must_not_be_distances(graph_and_cnv):
    G, X = graph_and_cnv
    adata = AnnData(obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]), obsm={"X_cnv": X})
    adata.obs["x"] = pd.Categorical(np.arange(X.shape[0]) % 2)
    adata.obsp["spatial_distances"] = G
    with pytest.raises(ValueError, match="similarities.*'spatial_connectivities'"):
        cnv.tl.spatial_regions(adata, key="x", connectivity_key="spatial_distances")
    adata.obsp["distances"] = G
    with pytest.raises(ValueError, match="'connectivities'"):
        cnv.tl.spatial_regions(adata, key="x", connectivity_key="distances")


def _with_windows(starts, ends):
    """Two cells x windows on chr1 at the given positions; each window's value is its column index."""
    adata = AnnData(obs=pd.DataFrame(index=["c0", "c1"]))
    adata.obsm["X_cnv"] = np.tile(np.arange(len(starts), dtype=np.float32), (2, 1))
    adata.uns["cnv"] = {"chr_pos": {"chr1": 0}, "window_start": np.array(starts), "window_end": np.array(ends)}
    return adata


def test_spatial_cnv_region_by_window_midpoint():
    """A range selects the windows whose midpoint lies in it, not every window that overlaps it."""
    adata = _with_windows([0, 100, 200, 300], [100, 200, 300, 400])  # midpoints 50, 150, 250, 350
    np.testing.assert_array_equal(_region_columns(adata, "cnv", "chr1:120-260"), [1, 2])
    np.testing.assert_array_equal(_region_columns(adata, "cnv", "chr1:150-150"), [1])
    with pytest.raises(ValueError, match="No windows"):
        _region_columns(adata, "cnv", "chr1:160-240")


@pytest.fixture
def adata_spatial() -> AnnData:
    """30 spots on a tissue four times as wide as high, with a categorical and a continuous `obs` column."""
    rng = np.random.default_rng(0)
    n = 30
    obs = pd.DataFrame(
        {"label": pd.Categorical(rng.choice(["x", "y"], n)), "score": rng.uniform(100, 200, n)},
        index=[str(i) for i in range(n)],
    )
    adata = AnnData(obs=obs, uns={"cnv": {"chr_pos": {"chr1": 0, "chr2": 2}}})
    adata.obsm["X_cnv"] = rng.normal(size=(n, 4)).astype(np.float32)
    adata.obsm["spatial"] = np.column_stack([rng.uniform(0, 4000, n), rng.uniform(0, 1000, n)])
    return adata


def test_spatial_cnv_panels(adata_spatial):
    """Region panels are found by position (whatever their title) and centred at 0; other panels are not."""
    axes = cnv.pl.spatial_cnv(
        adata_spatial, region=["chr1", "chr2"], color=["label", "score"], title=["A", "B", "C", "D"], show=False
    )
    assert [ax.get_title() for ax in axes] == ["A", "B", "C", "D"]
    for ax in axes[:2]:
        [points] = ax.collections
        assert points.norm.vcenter == 0
        assert points.colorbar.ax.get_ylabel() == "mean CNV (relative to reference)"
    [score] = axes[3].collections
    score_range = adata_spatial.obs["score"].min(), adata_spatial.obs["score"].max()
    assert (score.norm.vmin, score.norm.vmax) == pytest.approx(score_range)
    assert score.colorbar.ax.get_ylabel() == ""
    for ax in axes:  # as `scanpy.pl.spatial` draws without an image
        assert ax.get_aspect() == 1
        assert ax.yaxis_inverted()


def test_spatial_cnv_csr(adata_spatial):
    """A CSR CNV matrix (`tl.infercnv` with `dynamic_threshold`) gives the plotted values of the dense one."""
    [dense] = cnv.pl.spatial_cnv(adata_spatial, region="chr1", show=False).collections
    adata_spatial.obsm["X_cnv"] = sp.csr_matrix(adata_spatial.obsm["X_cnv"])
    [csr] = cnv.pl.spatial_cnv(adata_spatial, region="chr1", show=False).collections
    np.testing.assert_allclose(csr.get_array(), dense.get_array(), rtol=1e-6)


def test_spatial_cnv_saves_once(adata_spatial, tmp_path):
    settings = sc.settings.figdir, sc.settings.file_format_figs, sc.settings.autosave
    sc.settings.figdir, sc.settings.file_format_figs = tmp_path, "png"
    try:
        sc.settings.autosave = True
        cnv.pl.spatial_cnv(adata_spatial, region="chr1", show=False)
        sc.settings.autosave = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cnv.pl.spatial_cnv(adata_spatial, region="chr1", show=False, save="_cnv.png")
        fig = cnv.pl.spatial_cnv(adata_spatial, region="chr1", return_fig=True, save="_fig.png")
    finally:
        sc.settings.figdir, sc.settings.file_format_figs, sc.settings.autosave = settings
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spatial.png", "spatial_cnv.png", "spatial_fig.png"]
    for p in tmp_path.iterdir():
        assert (plt.imread(p)[..., :3] < 0.9).any(), p.name
    assert plt.get_fignums() == [fig.number]


def test_spatial_cnv_keeps_an_inverted_y_axis(adata_spatial):
    """Drawn into axes whose y axis already points down (an image drawn first), the axis stays that way."""
    ax = plt.figure().add_subplot()
    ax.invert_yaxis()
    cnv.pl.spatial_cnv(adata_spatial, region="chr1", ax=ax, show=False)
    assert ax.yaxis_inverted()
