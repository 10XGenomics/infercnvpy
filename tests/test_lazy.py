"""infercnv on lazy and on-disk data: dask arrays (dense, or sparse blocks as in lazily loaded spatial tables) and
backed h5ad. Adapted from the tests of scverse/infercnvpy#192 by Stephen Williams."""

import re
from contextlib import nullcontext

import anndata as ad
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

import infercnvpy as cnv
from infercnvpy.tl._infercnv import _get_reference
from tests.conftest import knn_graph

REF = {"reference_key": "cell_type", "reference_cat": ["Microglia/Macrophage", "Oligodendrocytes (non-malignant)"]}


@pytest.fixture(scope="module")
def expected(oligo):
    """In-memory result, with gene values."""
    return cnv.tl.infercnv(oligo, **REF, calculate_gene_values=True, inplace=False)


@pytest.fixture(scope="module")
def expected_dense(oligo):
    """In-memory result without a threshold (a dense array), with gene values."""
    return cnv.tl.infercnv(oligo, **REF, dynamic_threshold=None, calculate_gene_values=True, inplace=False)


def _with_X(oligo, X):
    return ad.AnnData(X=X, obs=oligo.obs.copy(), var=oligo.var.copy())


def _csr_blocks(oligo, block_rows, counter=None):
    """Dask array whose row blocks are CSR matrices loaded lazily, like a lazily loaded `spatialdata_io` table."""
    X = sp.csr_matrix(oligo.X)
    n, g = X.shape

    def load(start, stop):
        if counter is not None:
            counter.append((start, stop))
        return X[start:stop]

    meta = sp.csr_matrix((0, g), dtype=X.dtype)
    blocks = [
        da.from_delayed(
            dask.delayed(load)(s, min(s + block_rows, n)), shape=(min(block_rows, n - s), g), dtype=X.dtype, meta=meta
        )
        for s in range(0, n, block_rows)
    ]
    return da.concatenate(blocks, axis=0)


def _check(adata, expected, gene_values=True, **kw):
    """The result equals the in-memory one up to the precision of the reference (float64 sums in another order)."""
    chr_pos, res, gv = cnv.tl.infercnv(adata, **REF, calculate_gene_values=gene_values, inplace=False, **kw)
    if hasattr(res, "compute"):
        res, gv = dask.compute(res, gv)
    dense = isinstance(expected[1], np.ndarray)  # the result of `dynamic_threshold=None`
    assert isinstance(res, np.ndarray if dense else sp.csr_matrix)
    assert res.dtype == np.float32
    assert chr_pos == expected[0]
    want = expected[1] if dense else expected[1].toarray()
    np.testing.assert_allclose(res if dense else res.toarray(), want, rtol=1e-5, atol=1e-6)
    if gene_values:
        np.testing.assert_allclose(gv, expected[2], rtol=1e-5, atol=1e-6)
    return res


@pytest.mark.parametrize("block_rows", [40, 500])
def test_dask_dense(oligo, expected, block_rows):
    X = da.from_array(oligo.X.toarray(), chunks=(block_rows, 1000))  # also split along genes
    _check(_with_X(oligo, X), expected)


@pytest.mark.parametrize("block_rows", [10, 30, 97, 184])
def test_dask_sparse_blocks(oligo, expected, block_rows):
    _check(_with_X(oligo, _csr_blocks(oligo, block_rows)), expected)


def test_dask_layer(oligo, expected):
    a = _with_X(oligo, sp.csr_matrix(oligo.X.shape, dtype=np.float32))
    a.layers["raw"] = _csr_blocks(oligo, 50)
    _check(a, expected, layer="raw")


@pytest.mark.parametrize("fmt", ["csr", "dense"])
def test_backed_h5ad(oligo, expected, expected_dense, tmp_path, fmt):
    X = sp.csr_matrix(oligo.X) if fmt == "csr" else oligo.X.toarray()
    path = tmp_path / "a.h5ad"
    _with_X(oligo, X).write_h5ad(path)
    a = ad.read_h5ad(path, backed="r")
    assert a.isbacked
    _check(a, expected, chunksize=60)
    _check(a, expected_dense, chunksize=60, dynamic_threshold=None)
    cnv.tl.infercnv(a, **REF, chunksize=60)
    assert isinstance(a.obsm["X_cnv"], sp.csr_matrix)
    cnv.tl.infercnv(a, **REF, chunksize=60, dynamic_threshold=None)
    assert isinstance(a.obsm["X_cnv"], np.ndarray)


def test_backed_csc_is_read_once(oligo, expected, tmp_path, monkeypatch):
    """A row block of a backed CSC matrix would read all of it: the matrix is read into memory once instead."""
    import h5py

    path = tmp_path / "a.h5ad"
    _with_X(oligo, sp.csc_matrix(oligo.X)).write_h5ad(path)
    a = ad.read_h5ad(path, backed="r")
    read = []
    getitem = h5py.Dataset.__getitem__

    def counting(self, *args, **kwargs):
        res = getitem(self, *args, **kwargs)
        if self.name == "/X/data":
            read.append(np.size(res))
        return res

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counting)
    with pytest.warns(UserWarning, match="CSC"):
        _check(a, expected, chunksize=60)
    assert sum(read) == a.X.group["data"].shape[0]


@pytest.mark.parametrize("dynamic_threshold", [None, 1.5])
def test_reads_each_block_once_per_pass(oligo, expected, dynamic_threshold):
    """Reference means and the noise band (blocks with reference cells), and the result: one read of a block each."""
    loads = []
    a = _with_X(oligo, _csr_blocks(oligo, 46, counter=loads))
    cnv.tl.infercnv(a, **REF, dynamic_threshold=dynamic_threshold)
    n_blocks = -(-oligo.n_obs // 46)
    is_ref = oligo.obs["cell_type"].isin(REF["reference_cat"]).to_numpy()
    with_ref = sum(is_ref[s : s + 46].any() for s in range(0, oligo.n_obs, 46))
    assert 0 < with_ref < n_blocks
    assert len(loads) == n_blocks + with_ref * (1 if dynamic_threshold is None else 2)
    assert max(stop - start for start, stop in loads) <= 46, "more than one block was loaded at once"


def test_streaming_reference_matches_in_memory(oligo):
    lazy = _with_X(oligo, _csr_blocks(oligo, 25))
    for key, cats in ((None, None), (REF["reference_key"], REF["reference_cat"])):
        with pytest.warns(UserWarning) if key is None else nullcontext():
            got = _get_reference(lazy, key, cats, None, None)
        want = np.asarray(_get_reference(oligo, key, cats, None, None))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("scheduler", ["threads", "synchronous", "processes"])
def test_lazy_output(oligo, expected, scheduler):
    """With `lazy=True`, the result stays a dask array of CSR blocks and equals the eager result exactly."""
    X = _csr_blocks(oligo, 50)
    eager = cnv.tl.infercnv(_with_X(oligo, X), **REF, calculate_gene_values=True, inplace=False)
    a = _with_X(oligo, X)
    with dask.config.set(scheduler=scheduler):
        cnv.tl.infercnv(a, **REF, lazy=True, calculate_gene_values=True)
        assert isinstance(a.obsm["X_cnv"], da.Array)
        assert isinstance(a.layers["gene_values_cnv"], da.Array)
        res, gv = dask.compute(a.obsm["X_cnv"], a.layers["gene_values_cnv"])
    assert (sp.csr_matrix(res) != eager[1]).nnz == 0
    np.testing.assert_array_equal(gv, eager[2])
    _check(_with_X(oligo, X), expected, lazy=True)


def test_kernels_in_dask_tasks_run_serially(oligo, expected, monkeypatch):
    """Inside a dask task, every kernel runs its serial twin, also on the main thread (synchronous scheduler)."""
    from infercnvpy._numba import njit_parallel
    from infercnvpy.tl import _infercnv_kernels

    parallel_calls = []

    def spy(name, kernel):
        return lambda *args: (parallel_calls.append(name), kernel.serial(*args))[1]

    for name, kernel in vars(_infercnv_kernels).items():
        if isinstance(kernel, njit_parallel):
            monkeypatch.setattr(kernel, "parallel", spy(name, kernel))
    a = _with_X(oligo, _csr_blocks(oligo, 50))
    with dask.config.set(scheduler="synchronous"):
        _check(a, expected, lazy=True)
    assert parallel_calls == []


def test_lazy_is_opt_in(oligo):
    """Without `lazy=True`, the result of a dask array is in memory, whatever the active dask scheduler."""
    a = _with_X(oligo, _csr_blocks(oligo, 100))
    with dask.config.set(scheduler="threads"):
        cnv.tl.infercnv(a, **REF)
    assert isinstance(a.obsm["X_cnv"], sp.csr_matrix)


@pytest.mark.parametrize("dynamic_threshold", [None, 1.5])  # dense or CSR blocks
def test_tools_that_need_cnv_in_memory(oligo, dynamic_threshold):
    """Tools without block-wise reading tell the user to load a lazy CNV matrix; cnv_score reads it block by block."""
    a = _with_X(oligo, _csr_blocks(oligo, 100))
    cnv.tl.infercnv(a, **REF, lazy=True, dynamic_threshold=dynamic_threshold)
    eager = a.copy()
    eager.obsm["X_cnv"] = eager.obsm["X_cnv"].compute()
    a.obsm["spatial"] = np.random.default_rng(0).random((a.n_obs, 2))
    a.obsp["spatial_connectivities"] = sp.eye(a.n_obs, format="csr")
    calls = [
        lambda: cnv.tl.ithcna(a, groupby="cell_type"),
        lambda: cnv.pl.spatial_cnv(a, region="chr1", show=False),
        lambda: cnv.pl.chromosome_heatmap(a, groupby="cell_type", mode="legacy", show=False),
        lambda: cnv.pl.chromosome_heatmap_summary(a, groupby="cell_type", mode="legacy", show=False),
        lambda: cnv.tl.pca(a),
        lambda: cnv.tl.tsne(a),  # through the PCA it computes
        lambda: cnv.pp.neighbors(a),  # likewise
        lambda: cnv.pp.neighbors(a, use_rep="cnv"),
    ]
    for call in calls:
        with pytest.raises(
            TypeError, match=r"needs `(adata\.)?obsm\['X_cnv'\]` in memory, got a dask array.*\.compute\(\)"
        ):
            call()
    assert cnv.tl.cnv_score(a, groupby="cell_type", inplace=False) == pytest.approx(
        cnv.tl.cnv_score(eager, groupby="cell_type", inplace=False)
    )


def test_lazy_cnv_pca(oligo):
    """tsne and neighbors on a lazy PCA (as `read_lazy` gives it) ask to load the PCA, not the CNV matrix."""
    a = oligo.copy()
    a.obsm["X_cnv_pca"] = da.from_array(np.zeros((a.n_obs, 5), dtype=np.float32), chunks=(50, -1))
    for call in (lambda: cnv.tl.tsne(a), lambda: cnv.pp.neighbors(a)):
        with pytest.raises(TypeError, match=re.escape("`adata.obsm['X_cnv_pca'] = adata.obsm['X_cnv_pca'].compute()`")):
            call()


@pytest.mark.parametrize(("form", "layer"), [("dask", None), ("dask", "counts"), ("backed", None)])
def test_tools_that_need_x_in_memory(oligo, tmp_path, form, layer):
    """detectability and ithgex on a lazy or backed X (a backed AnnData has its layers in memory) name the matrix
    they need in the error."""
    counts = sp.csr_matrix(oligo.X.shape, dtype=np.float32)
    a = _with_X(oligo, counts)
    if form == "backed":
        a.write_h5ad(tmp_path / "a.h5ad")
        a = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
        field = "adata.X"
        hint = "got a backed (on-disk) matrix. Load the AnnData first with `adata = adata.to_memory()`."
    else:
        a.layers["counts"] = counts
        lazy = da.from_array(counts, chunks=(50, -1), asarray=False)
        a.X, a.layers["counts"] = (lazy, counts) if layer is None else (counts, lazy)
        field = "adata.X" if layer is None else f"adata.layers['{layer}']"
        hint = "got a dask array. Load it first with `.compute()` and store the result back in adata "
        hint += f"(e.g. `{field} = {field}.compute()`)"
    calls = {
        "the raw counts": lambda: cnv.tl.detectability(a, groupby="cell_type", layer=layer, **REF),
        "the expression matrix": lambda: cnv.tl.ithgex(a, groupby="cell_type", layer=layer),
    }
    for what, call in calls.items():
        with pytest.raises(TypeError, match=re.escape(f"needs {what} (`{field}`) in memory, {hint}")):
            call()


def test_lazy_without_threshold(oligo, expected):
    res = cnv.tl.infercnv(
        _with_X(oligo, _csr_blocks(oligo, 50)), **REF, dynamic_threshold=None, lazy=True, inplace=False
    )
    want = cnv.tl.infercnv(oligo, **REF, dynamic_threshold=None, inplace=False)[1]
    got = res[1].compute()
    assert isinstance(got, np.ndarray)
    assert isinstance(want, np.ndarray)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_lazily_normalised_table(oligo):
    """A dask array of CSR blocks, normalised lazily by scanpy, gives the eager result."""
    counts = sp.csr_matrix(oligo.X)

    def normalised(X):
        a = _with_X(oligo, X)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        return a

    lazy_input = _with_X(oligo, counts)
    lazy_input.X = da.from_array(counts, chunks=(40, -1), asarray=False)
    lazy = normalised(lazy_input.X)
    assert isinstance(lazy.X, da.Array)
    eager = normalised(counts)
    want = cnv.tl.infercnv(eager, **REF, inplace=False)
    _check(lazy, want, gene_values=False)


@pytest.fixture(scope="module")
def zarr_store(oligo, tmp_path_factory):
    path = tmp_path_factory.mktemp("lazy") / "oligo.zarr"
    b = oligo.copy()
    b.X = sp.csr_matrix(b.X)
    b.write_zarr(path)
    return path


def test_read_elem_lazy(oligo, expected, zarr_store):
    """`obs`/`var` in memory and `X` lazy via `read_elem_lazy`, as in the scanpy dask tutorial."""
    import zarr
    from anndata.experimental import read_elem_lazy

    f = zarr.open(zarr_store, mode="r")
    X = read_elem_lazy(f["X"], chunks=(50, oligo.n_vars))
    _check(ad.AnnData(X=X, obs=ad.io.read_elem(f["obs"]), var=ad.io.read_elem(f["var"])), expected)


def test_read_lazy(expected, expected_dense, zarr_store):
    """`anndata.experimental.read_lazy`: `obs` and `var` are lazy too."""
    a = ad.experimental.read_lazy(zarr_store)
    _check(a, expected)
    _check(a, expected_dense, dynamic_threshold=None)
    cnv.tl.infercnv(a, **REF)
    assert isinstance(a.obsm["X_cnv"], sp.csr_matrix)
    cnv.tl.infercnv(a, **REF, dynamic_threshold=None)
    assert isinstance(a.obsm["X_cnv"], np.ndarray)


def test_read_lazy_downstream(oligo, tmp_path):
    """Scores and heatmaps read the groups from the lazy `obs` of `anndata.experimental.read_lazy`."""
    b = oligo.copy()
    cnv.tl.infercnv(b, **REF)
    b.obs["cell_type"] = b.obs["cell_type"].astype("category")
    b.write_zarr(tmp_path / "cnv.zarr")
    a = ad.experimental.read_lazy(tmp_path / "cnv.zarr")
    assert not isinstance(a.obs, pd.DataFrame)
    # the lazy CNV matrix is summed in float64
    expected = cnv.tl.cnv_score(b, groupby="cell_type", inplace=False)
    assert cnv.tl.cnv_score(a, groupby="cell_type", inplace=False) == pytest.approx(expected, rel=1e-6)
    cnv.tl.cnv_score(a, groupby="cell_type")
    cnv.tl.cnv_score(b, groupby="cell_type")
    np.testing.assert_allclose(np.asarray(a.obs["cnv_score"]), b.obs["cnv_score"], rtol=1e-6)
    cnv.pl.chromosome_heatmap(a, groupby="cell_type", show=False)
    cnv.pl.chromosome_heatmap_summary(a, groupby="cell_type", show=False)
    a.obsm["X_cnv"] = a.obsm["X_cnv"].compute()  # ithcna needs the CNV matrix in memory
    lazy_ith = cnv.tl.ithcna(a, groupby="cell_type", inplace=False)
    ith = cnv.tl.ithcna(b, groupby="cell_type", inplace=False)
    assert list(lazy_ith) == list(ith)
    np.testing.assert_array_equal(list(lazy_ith.values()), list(ith.values()))


def test_read_lazy_spatial(oligo, tmp_path):
    """The spatial tools read the lazy graph, `obs` and colors of `read_lazy`; spatial_cnv needs `X_cnv` in memory."""
    b = oligo.copy()
    cnv.tl.infercnv(b, **REF)
    b.obs["cell_type"] = b.obs["cell_type"].astype("category")
    # colors that scanpy's default palette doesn't have, so the plot below shows that it read them
    b.uns["cell_type_colors"] = np.array([f"#{16 * i:02x}0000" for i in range(len(b.obs["cell_type"].cat.categories))])
    b.obsm["spatial"] = np.random.default_rng(0).random((b.n_obs, 2))
    b.obsp["spatial_connectivities"] = knn_graph(b.obsm["spatial"])
    b.write_zarr(tmp_path / "cnv.zarr")
    a = ad.experimental.read_lazy(tmp_path / "cnv.zarr")
    assert isinstance(a.obsp["spatial_connectivities"], da.Array)
    pd.testing.assert_series_equal(
        cnv.tl.spatial_regions(a, key="cell_type", inplace=False),
        cnv.tl.spatial_regions(b, key="cell_type", inplace=False),
        check_names=False,  # older anndata's read_lazy drops the index name
    )
    with pytest.raises(TypeError, match=r"needs `obsm\['X_cnv'\]` in memory"):
        cnv.pl.spatial_cnv(a, region="chr1", show=False)
    a.obsm["X_cnv"] = a.obsm["X_cnv"].compute()
    [[lazy, lazy_types], [eager, eager_types]] = (
        [ax.collections[0] for ax in cnv.pl.spatial_cnv(x, region="chr1", color="cell_type", show=False)]
        for x in (a, b)
    )
    np.testing.assert_array_equal(lazy.get_array(), eager.get_array())
    np.testing.assert_array_equal(lazy_types.get_facecolors(), eager_types.get_facecolors())
    assert (lazy_types.get_facecolors()[:, 1:3] == 0).all()  # the colors of `uns`, which have no green or blue


def test_scanpy_wrappers_on_lazy_and_backed(oligo, tmp_path):
    """umap and leiden ask to load a lazy graph, the embedding plots a `read_lazy` AnnData, and umap and tsne a backed
    one (which scanpy can't copy); the embedding plots and leiden work on a backed AnnData."""
    b = oligo.copy()
    cnv.tl.infercnv(b, **REF)
    cnv.pp.neighbors(b)
    cnv.tl.umap(b)
    cnv.tl.tsne(b)
    cnv.tl.leiden(b)
    b.write_zarr(tmp_path / "cnv.zarr")
    b.write_h5ad(tmp_path / "cnv.h5ad")
    a = ad.experimental.read_lazy(tmp_path / "cnv.zarr")
    graph = "cnv_neighbors_connectivities"
    for call in (lambda: cnv.tl.umap(a), lambda: cnv.tl.leiden(a)):
        with pytest.raises(TypeError, match=re.escape(f"`adata.obsp['{graph}'] = adata.obsp['{graph}'].compute()`")):
            call()
    a.obsp[graph] = a.obsp[graph].compute()
    cnv.tl.leiden(a)
    np.testing.assert_array_equal(np.asarray(a.obs["cnv_leiden"]).astype(str), b.obs["cnv_leiden"].astype(str))
    cnv.tl.umap(a)
    assert a.obsm["X_cnv_umap"].shape == (a.n_obs, 2)
    for call in (lambda: cnv.pl.umap(a, show=False), lambda: cnv.pl.tsne(a, show=False)):
        with pytest.raises(
            TypeError, match=re.escape("an AnnData from `anndata.experimental.read_lazy`. Load it first")
        ):
            call()
    backed = ad.read_h5ad(tmp_path / "cnv.h5ad", backed="r")
    for call in (lambda: cnv.tl.umap(backed), lambda: cnv.tl.tsne(backed)):
        with pytest.raises(
            TypeError, match=re.escape("got a backed AnnData. Load it first with `adata = adata.to_memory()`")
        ):
            call()
    cnv.tl.leiden(backed, key_added="leiden_backed")
    pd.testing.assert_series_equal(backed.obs["leiden_backed"], b.obs["cnv_leiden"], check_names=False)
    cnv.pl.umap(backed, color="cnv_leiden", show=False)
    cnv.pl.tsne(backed, color="cnv_leiden", show=False)


def test_read_lazy_gtf(testdata, tmp_path):
    """`io.genomic_position_from_gtf` loads the lazy `var` of `read_lazy`."""
    gtf = testdata / "chr21_gencode.gtf"
    genes = cnv.io.genomic_position_from_gtf(gtf)["gene_name"]
    b = ad.AnnData(obs=pd.DataFrame(index=["c0", "c1"]), var=pd.DataFrame(index=[*genes[:20], "not_in_gtf"]))
    b.write_zarr(tmp_path / "genes.zarr")
    a = ad.experimental.read_lazy(tmp_path / "genes.zarr")
    assert not isinstance(a.var, pd.DataFrame)
    pd.testing.assert_frame_equal(
        cnv.io.genomic_position_from_gtf(gtf, a, inplace=False), cnv.io.genomic_position_from_gtf(gtf, b, inplace=False)
    )


def test_unplaced_and_extra_features(oligo, expected):
    """Features without a position (e.g. control probes of a spatial panel) are skipped in lazy input too."""
    n_ctrl = 20
    ctrl = sp.csr_matrix(np.random.default_rng(0).poisson(0.5, (oligo.n_obs, n_ctrl)).astype(np.float32))
    X = sp.hstack([sp.csr_matrix(oligo.X), ctrl]).tocsr()
    var = pd.concat(
        [oligo.var[["chromosome", "start", "end"]], pd.DataFrame({"chromosome": [None] * n_ctrl, "start": np.nan})]
    )
    var.index = list(oligo.var_names) + [f"NegControlProbe_{i}" for i in range(n_ctrl)]
    a = ad.AnnData(X=da.from_array(X, chunks=(60, -1), asarray=False), obs=oligo.obs.copy(), var=var)
    chr_pos, res, gv = cnv.tl.infercnv(a, **REF, calculate_gene_values=True, inplace=False)
    np.testing.assert_allclose(res.toarray(), expected[1].toarray(), rtol=1e-5, atol=1e-6)
    assert np.isnan(gv[:, -n_ctrl:]).all()
