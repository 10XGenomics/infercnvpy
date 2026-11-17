"""`tl.copy_number`: kernels against references, and the whole method on small synthetic data."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from scipy.special import logsumexp

import infercnvpy as cnv
from infercnvpy._numba import numba_threads
from infercnvpy.tl import _copy_number as M
from infercnvpy.tl import _copy_number_kernels as K
from infercnvpy.tl._infercnv import _Rows
from infercnvpy.tl._infercnv_kernels import _group_sums_csr
from tests.conftest import CN_EVENTS, knn_graph
from tests.conftest import simulate_copy_number as _simulate


def _accuracy(adata, obs):
    return ((obs["copy_number"] == "aneuploid").to_numpy() == (adata.obs["truth"] == "tumour").to_numpy()).mean()


def _layout(adata, min_genes=200):
    excluded = M._excluded_genes(adata.var_names, None)
    return M._filter_cells_genes(adata, _Rows(adata.X, 5000), excluded, ("chrY",), min_genes, 0.05)


def _forward_backward_dense(y, v, stay, means, log_prior, nu):
    """Textbook forward-backward with explicit transition matrices, in log space."""
    S, G = len(means), len(y)
    z = (y[:, None] - means[None, :]) ** 2 / (nu * v[:, None])
    emis = -0.5 * np.log(v)[:, None] - 0.5 * (nu + 1) * np.log1p(z)
    prior = np.exp(log_prior)
    la = np.empty((G, S))
    la[0] = log_prior + emis[0]
    for i in range(1, G):
        T = stay[i] * np.eye(S) + (1 - stay[i]) * prior[None, :]
        la[i] = logsumexp(la[i - 1][:, None] + np.log(T), axis=0) + emis[i]
    lb = np.zeros((G, S))
    for i in range(G - 2, -1, -1):
        T = stay[i + 1] * np.eye(S) + (1 - stay[i + 1]) * prior[None, :]
        lb[i] = logsumexp(np.log(T) + (emis[i + 1] + lb[i + 1])[None, :], axis=1)
    return np.exp(la + lb - logsumexp(la[-1])) @ means


def test_hmm_kernel_matches_dense():
    rng = np.random.default_rng(0)
    G = 60
    means = np.log(np.array([0.5, 1.0, 1.5]))
    means[1] = 0.0
    log_prior = np.log(np.array([0.25, 0.5, 0.25]))
    y = np.r_[np.zeros(20), np.full(20, 0.4), np.zeros(20)] + rng.normal(0, 0.3, G)
    v = rng.uniform(0.05, 0.2, G)
    stay = np.full(G, 0.95)
    stay[0] = stay[30] = 0.0
    pm, pn, st = np.empty((1, G)), np.empty((1, G)), np.empty((1, G), dtype=np.int64)
    K.hmm_posteriors(y[None], v[None], stay, means, log_prior, 2.0, pm, pn, st)
    np.testing.assert_allclose(pm[0], _forward_backward_dense(y, v, stay, means, log_prior, 2.0), atol=1e-8)


def _nb_loglik(y, size, p, phi):
    """Negative binomial log likelihood up to clone-independent terms, in numpy."""
    m = size * p
    return (y * np.log(m) - (y + phi) * np.log(m + phi)).sum()


def test_count_kernels_match_references():
    rng = np.random.default_rng(1)
    X = sp.random(50, 30, density=0.3, format="csr", random_state=1, dtype=np.float32)
    X.data = np.round(X.data * 10)
    col_map = np.where(np.arange(30) % 7 == 3, -1, np.arange(30))  # genes 3, 10, 17, 24 dropped
    gene_win = np.repeat(np.arange(6), 5)
    out = np.empty((20, 6))
    K.window_counts(X.data, X.indices, X.indptr, np.arange(10, 30), col_map, gene_win, out, 3)
    Xk = X[10:30].toarray() * (col_map >= 0)
    win_of_col = np.where(col_map >= 0, gene_win[np.maximum(col_map, 0)], -1)
    np.testing.assert_allclose(out, np.stack([Xk[:, win_of_col == w].sum(1) for w in range(6)], axis=1))
    labels = rng.integers(0, 3, 20)
    slabs = np.zeros((4, 3, 30))
    _group_sums_csr(X.data, X.indices, X.indptr, np.arange(10, 30), labels, False, slabs, col_map)
    ref = np.zeros((3, 30))
    for lab in range(3):
        ref[lab, col_map[col_map >= 0]] = X[10:30][labels == lab].toarray()[:, col_map >= 0].sum(0)
    np.testing.assert_allclose(slabs.sum(0), ref)


@pytest.mark.parametrize(("sparse_gram", "dense_eigh"), [(0.0, 500), (1.0, 500), (1.0, 100)])
def test_expression_pcs_match_dense_pca(monkeypatch, sparse_gram, dense_eigh):
    """The PCA from the Gram matrix (sparse or dense rows; full or Lanczos eigensolver) is the PCA of the dense matrix."""
    monkeypatch.setattr(M, "_SPARSE_GRAM", sparse_gram)
    monkeypatch.setattr(M, "_DENSE_EIGH", dense_eigh)
    monkeypatch.setattr(M, "_N_HVG", 300)
    adata = _simulate(n_tumour=150, n_normal=150, depth=800)
    cells, lay, *_ = _layout(adata)
    data = M._Data(_Rows(adata.X, 70), cells, lay, 10)
    pcs = data.expression_pcs()
    X = adata.X[cells][:, lay.order].toarray().astype(np.float64)
    X = np.log1p(X * (1e4 / np.maximum(X.sum(axis=1), 1))[:, None])
    v = X.var(axis=0)
    hvg = np.sort(np.argsort(-v, kind="stable")[:300])
    np.testing.assert_array_equal(np.flatnonzero(data.hvg_of_gene >= 0), hvg)
    D = X[:, hvg] - X[:, hvg].mean(axis=0)
    V = np.linalg.svd(D, full_matrices=False)[2][:30].T
    cos = np.linalg.svd(V.T @ data.loadings, compute_uv=False)
    assert cos.min() > 1 - 1e-8  # the same subspace
    np.testing.assert_allclose(np.abs(pcs[:, :5]), np.abs(D @ V[:, :5]), atol=1e-6)
    peak = data.loadings[np.abs(data.loadings).argmax(axis=0), np.arange(30)]
    assert (peak > 0).all()  # sign convention


def test_kmeans_kernel_matches_numpy():
    rng = np.random.default_rng(0)
    X, C, k = rng.normal(size=(200, 4)), rng.normal(size=(6, 4)), 3  # two starts of 3 centres
    labels, dist = np.empty((200, 2), dtype=np.int64), np.empty((200, 2))
    sums, counts = np.zeros((5, 6, 4)), np.zeros((5, 6))
    K.kmeans_assign(X @ C.T, X, (C**2).sum(1), k, labels, dist, sums, counts)
    for s in range(2):
        d2 = ((X[:, None, :] - C[None, s * k : (s + 1) * k]) ** 2).sum(-1)
        np.testing.assert_array_equal(labels[:, s], d2.argmin(1))
        np.testing.assert_allclose(dist[:, s], d2.min(1), atol=1e-12)
        for j in range(k):
            np.testing.assert_allclose(sums.sum(0)[s * k + j], X[labels[:, s] == j].sum(0), atol=1e-12)
            assert counts.sum(0)[s * k + j] == (labels[:, s] == j).sum()


def test_kmeans_finds_separated_clusters_without_empty_ones():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 6, 3000)
    X = 10 * rng.normal(size=(6, 5))[truth] + rng.normal(size=(3000, 5))
    found = M._kmeans(X, 6, np.random.default_rng(1))
    assert len(np.unique(found)) == 6
    assert len(np.unique(truth * 6 + found)) == 6  # the same partition
    # three distinct points, repeated: in every start of these seeds one point is picked twice, so a cluster empties
    # and must restart at the point without a centre
    X, truth = np.repeat(np.eye(3), 50, axis=0), np.repeat(np.arange(3), 50)
    for seed in (0, 1, 3, 4):
        rng = np.random.default_rng(seed)
        assert all(len(np.unique(truth[rng.choice(150, 3, replace=False)])) < 3 for _ in range(2))
        found = M._kmeans(X, 3, np.random.default_rng(seed), n_starts=2)
        assert len(np.unique(found)) == 3 and len(np.unique(truth * 3 + found)) == 3


def test_kmeans_sums_do_not_depend_on_threads():
    """The centroid sums of the k-means pass are bit for bit the same with 1 and 2 threads (fixed chunks)."""
    rng = np.random.default_rng(0)
    X, C = rng.normal(size=(20_000, 8)), rng.normal(size=(20, 8))  # two starts of 10 centres
    out = []
    for threads in (1, 2):
        sums, counts = np.zeros((M._PCA_CHUNKS, 20, 8)), np.zeros((M._PCA_CHUNKS, 20))
        with numba_threads(threads):
            K.kmeans_assign(X @ C.T, X, (C**2).sum(1), 10, np.empty((20_000, 2), np.int64), np.empty((20_000, 2)), sums,
                            counts)  # fmt: skip
        out.append(sums.sum(axis=0))
    np.testing.assert_array_equal(*out)


@pytest.mark.parametrize("size", [None, 4, 7, 25, 1000])
def test_block_medians_match_array_split(size):
    adata = _simulate(n_tumour=20, n_normal=20, genes=97)
    lay = _layout(adata, min_genes=5)[1]
    x = np.random.default_rng(0).normal(size=lay.n_genes)
    vals, ch = M._block_medians(x, lay, size)
    size = M._block_size(lay) if size is None else size
    ref = [
        (np.median(x[part]), c)
        for c, (g0, g) in enumerate(zip(lay.g0s, lay.gs, strict=True))
        for part in np.array_split(np.arange(g0, g0 + g), max(1, g // size))
    ]
    np.testing.assert_array_equal(vals, [v for v, _ in ref])
    np.testing.assert_array_equal(ch, [c for _, c in ref])


def test_kmeans_split_shares_types_by_all_cells():
    """With a sample that holds every reference cell, the reference types follow the reference share of all cells."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    ref = np.arange(400) < 200  # half of the sample, but a tenth of all cells
    types, ref_type = M._kmeans_split(X, 10, np.random.default_rng(0), ref, 0.1)
    assert ref_type.sum() == 1 and (ref_type[types[ref]]).all() and not ref_type[types[~ref]].any()


def test_expression_pcs_do_not_depend_on_blocks(monkeypatch):
    """The PCA sums are the same, bit for bit, whatever the blocks of rows (sparse and dense Gram routes)."""
    adata = _simulate(n_tumour=150, n_normal=150, depth=800)
    cells, lay, *_ = _layout(adata)
    monkeypatch.setattr(M, "_GRAM_TILE", 64)
    for sparse in (0.0, 1.0):
        monkeypatch.setattr(M, "_SPARSE_GRAM", sparse)
        out = [M._Data(_Rows(adata.X, size), cells, lay, 10).expression_pcs() for size in (37, 5000)]
        np.testing.assert_array_equal(*out)


def test_copy_number_generator_seed():
    """A Generator works as `random_state` and gives the results of its integer seed."""
    adata = _simulate()
    a = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, random_state=7, n_restarts=2)[0]
    b = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, random_state=np.random.default_rng(7), n_restarts=2)[0]
    pd.testing.assert_frame_equal(a, b)


def test_copy_number_spatial_info_with_one_clone():
    """With a graph, `uns["spatial"]` is there even when the kept fit has only the diploid clone."""
    adata = _simulate(n_tumour=0, n_normal=300)
    adata.obsp["g"] = knn_graph(np.random.default_rng(0).random((adata.n_obs, 2)), 6)
    obs, _, uns, _ = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, connectivity_key="g", force_calls=True)
    assert len(uns["clones"]) == 1
    assert uns["spatial"] == {"neighbor_weight": 0.0, "sweeps": 0, "converged": True}


def test_progress_bar(capsys, monkeypatch):
    """The bar shows on stderr with `show_progress`, not when the guard labels the cells undetermined, and log
    messages during the fits get lines of their own, also with a handler that is no StreamHandler (as rich's)."""
    import logging
    import os
    import sys

    from scverse_misc.logging import config

    if os.environ.get("TQDM_DISABLE"):
        pytest.skip("TQDM_DISABLE is set")

    class Other(logging.Handler):  # writes to the current stderr, like rich's handler, but is no StreamHandler
        def emit(self, record):
            sys.stderr.write(self.format(record) + "\n")

    with M._progress(2, False) as bar:
        bar.update()
    cnv.tl.copy_number(_simulate(depth=60), min_genes=5, min_detection=0.01, n_jobs=2, inplace=False)
    assert "%|" not in capsys.readouterr().err
    config.verbosity = "debug"  # (re)installs the scverse handler, so ours goes in after
    owner = logging.getLogger("scverse")
    h = Other()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    monkeypatch.setattr(owner, "handlers", [h])
    obs = cnv.tl.copy_number(_simulate(), n_restarts=2, n_jobs=2, inplace=False)[0]
    assert owner.handlers == [h]  # restored after the fit
    assert _accuracy(_simulate(), obs) >= 0.98
    lines = [x for line in capsys.readouterr().err.splitlines() for x in line.split("\r")]
    assert any("copy_number restart 2/2" in x for x in lines)
    logged = [x for x in lines if "copy_number: restart 0, " in x]
    assert logged and not any("%|" in x for x in logged)


def test_copy_number_restarts():
    """The returned fit is the medoid of the restarts; restart r is the same fit for any `n_restarts` > r."""
    adata = _simulate()
    obs, _, uns, _ = cnv.tl.copy_number(adata, n_restarts=3, n_jobs=2, inplace=False)
    info = uns["restarts"]
    agree, chosen = info["agreement"], info["chosen"]
    assert agree.shape == (3, 3) and np.allclose(agree, agree.T) and np.allclose(np.diag(agree), 1.0)
    assert chosen == np.argmax(agree.sum(axis=1))
    an = (obs["copy_number"] == "aneuploid").to_numpy()
    assert an.sum() == info["n_aneuploid"][chosen]
    stability = obs["copy_number_stability"].to_numpy()
    assert set(np.round(stability * 3).astype(int)) <= {1, 2, 3}  # the chosen fit always agrees with itself
    np.testing.assert_allclose(stability.mean(), agree[chosen].mean())
    assert _accuracy(adata, obs) >= 0.98
    single = cnv.tl.copy_number(adata, n_restarts=1, n_jobs=2, inplace=False)
    assert (single[0]["copy_number"] == "aneuploid").sum() == info["n_aneuploid"][0]
    assert (single[0]["copy_number_stability"] == 1.0).all()
    np.testing.assert_array_equal(single[2]["restarts"]["agreement"], [[1.0]])


def test_log_rows_within_one_ulp():
    rng = np.random.default_rng(0)
    x = np.concatenate([np.exp(rng.uniform(-700, 700, 100_000)), [1.0, 2.0, 0.5, np.sqrt(2), 1e-310, 0.0, np.inf]])
    out = np.empty_like(x)
    K._log_rows(x, out, np.empty(len(x), dtype=np.int64), np.empty_like(x))
    with np.errstate(divide="ignore"):
        ref = np.log(x)
    finite = np.isfinite(ref) & (ref != 0)
    assert (np.abs(out - ref)[finite] <= np.spacing(np.abs(ref[finite]))).all()
    np.testing.assert_array_equal(out[~finite], ref[~finite])


def _loglik_reference(Y, size, types, share, phi, use, g, spill_ok, smax, report):
    """Log likelihood of each cell under each clone with `_nb_loglik` (numpy), as the reference of the kernel."""
    w = np.flatnonzero(use)
    out, spill = np.empty((len(Y), share.shape[1])), np.empty(len(Y))
    for r, t in enumerate(types):
        y = Y[r, w].astype(np.float64)
        spill[r] = K._spillover(y, share[t, 0, w], g[w]) if (spill_ok[t] or report) else 0.0
        s = min(spill[r], smax) if spill_ok[t] else 0.0
        for k in range(share.shape[1]):
            p = (1 - s) * share[t, k, w] + s * g[w] if k == 0 and s > 0 else share[t, k, w]
            out[r, k] = _nb_loglik(y, size[r], p, phi[w])
    return out, spill


@pytest.mark.parametrize(("half", "report"), [(False, False), (True, False), (False, True)])
def test_loglik_kernel_matches_numpy(half, report):
    """The kernel (fdlibm logarithms) gives the numpy log likelihoods within 1e-12 (relative), the same best clone and
    the same spillover fractions."""
    rng = np.random.default_rng(3)
    n, W, k, T = 60, 300, 5, 4
    share = rng.dirichlet(np.ones(W) * 5, (T, k))
    types = rng.integers(0, T, n)
    Y = np.stack([rng.poisson(800 * share[t, 0]) for t in types]).astype(np.float32)
    use = (np.arange(W) % 2 == 0) if half else np.ones(W, dtype=np.bool_)
    size = (Y.astype(np.float64) * use).sum(axis=1)
    phi, g = rng.uniform(0.5, 20, W), rng.dirichlet(np.ones(W))
    spill_ok = np.array([True, False, True, False])
    ll, spill = np.empty((n, k)), np.empty(n)
    K.nb_loglik_types(Y, size, types, share, np.log(share), phi, use, g, spill_ok, 0.05, report, ll, spill)
    ref, ref_spill = _loglik_reference(Y, size, types, share, phi, use, g, spill_ok, 0.05, report)
    np.testing.assert_allclose(ll, ref, rtol=1e-12)
    np.testing.assert_array_equal(ll.argmax(axis=1), ref.argmax(axis=1))
    np.testing.assert_array_equal(spill, ref_spill)


@pytest.mark.parametrize("n", [300, 3000, 1e6])
def test_spillover_is_the_likelihood_maximum(n):
    """The Newton steps reach the maximum also for a large `g / a` (a tumour-specific window)."""
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(2000))
    g = a.copy()
    spec = rng.choice(2000, 5, replace=False)
    a[spec], g[spec] = 1e-6, 0.02
    a, g = a / a.sum(), g / g.sum()
    y = rng.multinomial(int(n), 0.7 * a + 0.3 * g).astype(float)
    grid = np.linspace(0, 1, 2001)[:-1]
    best = grid[np.argmax([(y * np.log((1 - s) * a + s * g)).sum() for s in grid])]
    assert abs(K._spillover(y, a, g) - best) <= 1e-3
    assert K._spillover(1e6 * a, a, g) < 1e-3


@pytest.mark.parametrize(
    "kw",
    [{}, {"normal_depth": 1.0, "n_tumour": 600, "n_normal": 60}, {"n_tumour": 100, "n_normal": 500}],
)
def test_copy_number_synthetic(kw):
    adata = _simulate(**kw)
    cnv.tl.copy_number(adata, n_jobs=2)
    assert _accuracy(adata, adata.obs) >= 0.98
    tumour = adata.obs["truth"].to_numpy() == "tumour"
    uns = adata.uns["copy_number"]
    X = adata.obsm["X_copy_number"]
    assert X.shape == (adata.n_obs, len(uns["bins"]))
    # the tumour pseudobulk follows the simulated events: gains positive, the loss negative
    bins = uns["bins"]
    prof = X[tumour].mean(axis=0)
    for c, a, b, f in CN_EVENTS:
        sel = (bins["chromosome"] == f"chr{c + 1}").to_numpy() & (bins["start"] >= a * 200_000).to_numpy()
        sel &= (bins["end"] <= b * 200_000).to_numpy()
        assert np.sign(prof[sel].mean()) == np.sign(np.log(f))
    assert np.abs(X[~tumour].mean(axis=0)).max() < 0.05
    # label, clone and clone sizes agree
    an = (adata.obs["copy_number"] == "aneuploid").to_numpy()
    assert (an == adata.obs["copy_number_clone"].astype(str).str.startswith("T").to_numpy()).all()
    counts = adata.obs["copy_number_clone"].value_counts()
    np.testing.assert_array_equal(uns["clones"]["n_cells"], counts.reindex(uns["clones"].index).to_numpy())


_DETECTION_BIAS = pytest.mark.xfail(
    strict=True,
    reason="open: where a tumour is the majority at low depth, its losses fall below the detection filter more often "
    "than its gains and the tumour looks flattest. Scoring genes detected per expression cluster fixed this but "
    "inverted real panel cuts and droplet lines (cluster-specific genes enter the flatness), so it was not kept.",
)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"n_tumour": 600, "n_normal": 600, "depth": 400, "seed": 2}, marks=_DETECTION_BIAS),
        {"n_tumour": 600, "n_normal": 9},  # a rare normal population seeds the baseline
    ],
)
def test_copy_number_baseline_not_on_the_tumour(kw):
    adata = _simulate(**kw)
    assert _accuracy(adata, cnv.tl.copy_number(adata, n_jobs=2, inplace=False, force_calls=True)[0]) >= 0.95


def test_copy_number_warns_on_uncertain_baseline(caplog):
    """Without a clear flat split and without marker-rich groups the baseline is uncertain: a warning and the seed
    diagnostics in `uns`."""
    kw = {"n_tumour": 600, "n_normal": 200, "depth": 400, "normal_depth": 1.0, "seed": 4}
    with caplog.at_level("WARNING"):
        uns = cnv.tl.copy_number(_simulate(**kw), n_jobs=2, inplace=False, force_calls=True)[2]
    seed = uns["seed"]
    assert seed["clear_gap"] == M._FLAT_GAP
    assert ("baseline is uncertain" in caplog.text) == (seed["method"] == "flattest group")


@pytest.mark.parametrize("shuffled", [False, True])
def test_marker_counts_keep_undetected_markers(shuffled):
    """The marker fraction counts every marker, also one that the detection filter removed (a rare normal
    population keeps its markers); without such markers it is the share of the markers among the model's genes.
    With the genes of `var` not in genomic order, the markers outside the model's genes are still the right ones."""
    adata = _simulate(n_tumour=385, n_normal=15)
    normal = (adata.obs["truth"] == "normal").to_numpy()
    X = adata.X.tolil()
    X[:, 0] = normal[:, None].astype(np.float32) * 5  # a marker of the 15 normal cells: under 5% of all cells
    adata.X = X.tocsr()
    names = adata.var_names[[0, 5, 900, 1700]]  # g0 (filtered out) and three markers that pass
    if shuffled:
        adata = adata[:, np.random.default_rng(0).permutation(adata.n_vars)].copy()
    cells, lay, var, candidates = _layout(adata)
    data = M._Data(_Rows(adata.X, 5000), cells, lay, 10)
    clusters = normal[cells].astype(np.int64)
    Yc = data.pooled(clusters)[0]
    idx = adata.var_names.get_indexer(names)
    is_marker = np.zeros(adata.n_vars, dtype=bool)
    is_marker[idx] = True
    assert lay.col_map[idx[0]] < 0 and (lay.col_map[idx[1:]] >= 0).all()
    marker, total = M._marker_counts(data, clusters, Yc, M._Layout(var, is_marker & candidates))
    np.testing.assert_allclose(marker, Yc[:, lay.col_map[idx[1:]]].sum(axis=1) + [0, 75])
    np.testing.assert_allclose(total, Yc.sum(axis=1) + [0, 75])


def test_copy_number_small_panel_keeps_labels():
    """On a ~1,000-gene panel (45 genes per chromosome) the genomic statistics still have blocks to compare.

    With fixed 25-gene blocks, every chromosome had one block, so the baseline seed fell on an arbitrary group.
    """
    adata = _simulate(n_tumour=600, n_normal=400, depth=1000, n_chrom=22, genes=45)
    obs = cnv.tl.copy_number(adata, n_jobs=2, min_genes=10, min_detection=0.01, force_calls=True, inplace=False)[0]
    assert _accuracy(adata, obs) >= 0.95


def test_em_runs_after_baseline_and_uses_its_dosage(monkeypatch):
    """Every EM run follows a baseline update, also the last single-iteration run, and the reported dosage weights
    are those of the final likelihood."""
    calls = []
    for name in ("set_baseline", "run", "loglik"):
        orig = getattr(M._EM, name)

        def spy(self, *args, _orig=orig, _name=name, **kw):
            calls.append((_name, self.dosage.copy()))
            return _orig(self, *args, **kw)

        monkeypatch.setattr(M._EM, name, spy)
    uns = cnv.tl.copy_number(_simulate(), n_restarts=1, n_jobs=2, inplace=False)[2]  # the calls of one fit
    names = [c for c, _ in calls if c != "loglik"]
    assert names[0] == "set_baseline" and all(
        a == "set_baseline" for a, b in zip(names, names[1:], strict=False) if b == "run"
    )
    last_full = [d for c, d in calls if c == "loglik"][-3]  # the final loglik, then the two halves of the temperature
    np.testing.assert_allclose(uns["genes"]["dosage"].to_numpy(), last_full)
    assert (last_full != 1).any()


def test_copy_number_reference_and_not_inplace():
    adata = _simulate()
    obs, X, uns, gv = cnv.tl.copy_number(
        adata, reference_key="truth", reference_cat="normal", inplace=False, calculate_gene_values=True
    )
    assert gv.shape == adata.shape
    tumour_gv = np.nanmean(gv[(adata.obs["truth"] == "tumour").to_numpy()], axis=0)
    assert np.nanmean(tumour_gv[:200]) > 0.3  # chr1 0-200: gain (genes removed by the filters are NaN)
    assert uns["seed"]["method"] == "reference"
    assert _accuracy(adata, obs) >= 0.98
    assert "copy_number" not in adata.obs


def _reference_types_data(depth=8000, clone_in_reference_type=False, few_events=False):
    """Epithelial cells (type 0, half of them with copy number changes) and six reference types (1 to 6).

    Three reference types share an "immune" smooth expression axis, three share a "stromal" axis with the epithelial
    cells. With `clone_in_reference_type`, the epithelial cells have the expression of reference type 4, as normal
    epithelium next to its tumour.
    """
    rng = np.random.default_rng(0)
    n_chrom, genes, block = 12, 300, 40
    G = n_chrom * genes
    mu = rng.lognormal(-1, 1.2, G)

    def smooth(sd, n=1):
        return np.repeat(rng.normal(0, sd, (n, G // block + 1)), block, axis=1)[:, :G]

    # 5% of genes 4x up or down per type, plus smooth expression axes
    prog = np.log(np.where(rng.random((7, G)) < 0.05, rng.choice([0.25, 4.0], (7, G)), 1.0)) + smooth(0.1, 7)
    immune, stromal = smooth(0.3)[0], smooth(0.3)[0]
    prog[1:4] += immune
    prog[4:] += 0.6 * stromal
    prog[0] += stromal
    if clone_in_reference_type:
        prog[0] = prog[4]
    cn = np.ones(G)  # chromosome arms cycle through loss, neutral and gain, or (`few_events`) two arm events
    for arm in range(2 * n_chrom):
        cn[arm * genes // 2 : (arm + 1) * genes // 2] = (0.5, 1.0, 1.5)[arm % 3] if not few_events else 1.0
    if few_events:
        cn[: genes // 2], cn[5 * genes // 2 : 3 * genes] = 1.5, 0.5
    ctype = np.r_[np.zeros(600, int), np.repeat(np.arange(1, 7), 250)]
    clone = np.r_[np.ones(300, bool), np.zeros(len(ctype) - 300, bool)]
    lam = mu[None, :] * np.exp(prog[ctype]) * np.where(clone[:, None], cn[None, :], 1.0)
    size = np.where(ctype == 0, depth, depth / 2) * rng.lognormal(0, 0.2, len(ctype))
    lam *= (size / lam.sum(axis=1))[:, None]
    pos = np.tile(np.arange(genes) * (80_000_000 // genes) + 10_000, n_chrom)
    var = pd.DataFrame(
        {"chromosome": np.repeat([f"chr{i + 1}" for i in range(n_chrom)], genes), "start": pos, "end": pos + 5_000},
        index=[f"g{i}" for i in range(G)],
    )
    obs = pd.DataFrame({"type": [f"t{t}" for t in ctype]}, index=[f"c{i}" for i in range(len(ctype))])
    obs.loc[(ctype == 0) & ~clone, "type"] = "normal epithelium"
    return AnnData(sp.csr_matrix(rng.poisson(lam).astype(np.float32)), obs=obs, var=var), clone, ctype


def test_copy_number_several_reference_types_stay_diploid():
    """Reference cell types whose expression differs smoothly from the pooled reference stay diploid.

    The diploid epithelial cells, outside the reference, form a clone of their own expression, and the stromal
    reference types stay out of it at high depth.
    """
    adata, clone, ctype = _reference_types_data()
    refs = [f"t{t}" for t in range(1, 7)]
    obs, *_ = cnv.tl.copy_number(adata, reference_key="type", reference_cat=refs, inplace=False)
    called = (obs["copy_number"] == "aneuploid").to_numpy()
    assert called[clone].mean() > 0.95
    assert called[ctype > 0].mean() < 0.02


def test_copy_number_reference_shares_types_with_the_clone():
    """Normal cells of the clone's own cell type in the reference stay diploid, and the clone is called.

    The clone has two arm events and shares its expression with the reference epithelium, so k-means on all cells puts
    both into the same expression types. These mixed types pulled reference cells into clones (here 12% of the
    reference epithelium and 11% of the clone missed).
    """
    adata, clone, ctype = _reference_types_data(depth=500, clone_in_reference_type=True, few_events=True)
    refs = [f"t{t}" for t in range(1, 7)] + ["normal epithelium"]
    obs, *_ = cnv.tl.copy_number(adata, reference_key="type", reference_cat=refs, inplace=False)
    called = (obs["copy_number"] == "aneuploid").to_numpy()
    assert called[clone].mean() > 0.95
    assert called[~clone].mean() < 0.02


def test_type_priors_keep_reference_types_diploid():
    """A nearly flat clone that matches the profile of every type takes the prior of normal types, but not of
    reference types: their cells are known to be diploid."""
    _, lay, _, _ = _layout(_simulate())
    em = object.__new__(M._EM)
    em.data = type("D", (), {"lay": lay})()
    em.n_types = 2
    em.ref_type = np.array([False, True])
    flat = 0.1 * np.sin(np.arange(lay.n_genes) / 200.0)  # a weak smooth profile
    em.type_blocks = np.array([M._block_medians(flat, lay)[0]] * 2)
    pri = em.type_priors(np.array([np.zeros(lay.n_genes), flat]))
    assert pri[0, 0] < 0.05  # a type fitted on the clone: the clone takes the prior
    assert pri[1, 0] > 0.95  # a reference type keeps the diploid prior


def test_copy_number_reference_arguments():
    adata = _simulate()
    with pytest.raises(ValueError, match="both"):
        cnv.tl.copy_number(adata, reference_key="truth")
    with pytest.raises(ValueError, match="both"):
        cnv.tl.copy_number(adata, reference_cat="normal")
    with pytest.raises(ValueError, match="'nromal'"):
        cnv.tl.copy_number(adata, reference_key="truth", reference_cat=["normal", "nromal"])
    with pytest.raises(KeyError, match="column"):
        cnv.tl.copy_number(adata, reference_key="missing", reference_cat="normal")


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"window_size": 0}, "window_size"),
        ({"bin_size": 0}, "bin_size"),
        ({"chunksize": 0}, "chunksize"),
        ({"n_clusters": 0}, "n_clusters"),
        ({"max_fit_cells": 1}, "max_fit_cells"),
        ({"n_restarts": 0}, "n_restarts"),
        ({"min_detection": 1.5}, "min_detection"),
        ({"max_neighbor_weight": -0.1}, "max_neighbor_weight"),
        ({"max_neighbor_weight": 1.0}, "max_neighbor_weight"),
    ],
)
def test_copy_number_argument_checks(kw, match):
    with pytest.raises(ValueError, match=match):
        cnv.tl.copy_number(_simulate(n_tumour=20, n_normal=10), **kw)


@pytest.mark.parametrize("reference", [False, True])
def test_copy_number_fit_on_sample(reference):
    """With `max_fit_cells`, the model is fitted on a sample (always with the reference cells) and all cells are
    assigned in one pass."""
    if reference:  # 10 reference cells: a sample of 100 cells would hold about 2 of them without the rule
        adata = _simulate(n_tumour=390, n_normal=10)
        kw = {"reference_key": "truth", "reference_cat": "normal", "max_fit_cells": 100}
    else:
        adata, kw = _simulate(n_tumour=300, n_normal=200), {"max_fit_cells": 250}
    obs = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, **kw)[0]
    assert _accuracy(adata, obs) >= 0.98
    assert obs["copy_number_clone"].notna().all()


def test_copy_number_fit_on_sample_reads_counts_once_for_all_restarts(monkeypatch):
    """With `max_fit_cells`, the passes over all cells' counts don't grow with `n_restarts`."""
    adata = _simulate(n_tumour=300, n_normal=200)
    passes = []
    orig = M._row_blocks

    def counting(source, rows):
        passes[-1] += len(rows) > 250  # a pass over all cells, not over the sample of 250
        return orig(source, rows)

    monkeypatch.setattr(M, "_row_blocks", counting)
    for n_restarts in (1, 3):
        passes.append(0)
        cnv.tl.copy_number(adata, n_jobs=2, inplace=False, max_fit_cells=250, n_restarts=n_restarts)
    assert passes[0] == passes[1]


def test_copy_number_filtered_cells_and_writing(tmp_path):
    """Cells removed by the filters get a neutral profile and the clone "not.defined", so the downstream tools work;
    the outputs (with dict-of-tuple markers) can be written."""
    adata = _simulate()
    X = adata.X.tolil()
    X[:3, 50:] = 0  # three cells with fewer than 200 detected genes
    adata.X = X.tocsr()
    cnv.tl.copy_number(adata, n_jobs=2, normal_markers={"immune": ("g1", "g2")})
    filtered = (adata.obs["copy_number"] == "not.defined").to_numpy()
    assert filtered.sum() == 3
    assert (adata.obs["copy_number_clone"][filtered] == "not.defined").all()
    assert not np.isnan(adata.obsm["X_copy_number"]).any()
    assert (adata.obsm["X_copy_number"][filtered] == 0).all()
    cnv.tl.pca(adata, use_rep="copy_number")
    cnv.pl.chromosome_heatmap(adata, groupby="copy_number_clone", use_rep="copy_number", mode="legacy", show=False)
    adata.write_h5ad(tmp_path / "cn.h5ad")
    adata.write_zarr(tmp_path / "cn.zarr")


def test_copy_number_dense_and_strings():
    """Dense counts give the same calls as sparse ones; single strings are one chromosome or gene."""
    adata = _simulate()
    sparse = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, chunksize=50)[0]
    adata.X = adata.X.toarray()
    dense = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, chunksize=50, exclude_chromosomes="chr2")[0]
    pd.testing.assert_series_equal(sparse["copy_number"], dense["copy_number"])
    uns = cnv.tl.copy_number(adata, n_jobs=2, inplace=False, exclude_chromosomes="chr2", exclude_genes="g0")[2]
    assert "chr2" not in set(uns["bins"]["chromosome"]) and "g0" not in uns["genes"].index


def test_copy_number_nan_positions():
    """A missing end is the start (as in tl.infercnv); genes without a start are left out."""
    adata = _simulate()
    for col, gene in (("end", 5), ("start", 6)):
        adata.var[col] = adata.var[col].astype(float)
        adata.var.iloc[gene, adata.var.columns.get_loc(col)] = np.nan
    uns = cnv.tl.copy_number(adata, n_jobs=2, inplace=False)[2]
    genes = uns["genes"]
    assert "g6" not in genes.index
    assert genes.loc["g5", "end"] == genes.loc["g5", "start"]


def test_bins_pad_to_hg38_only_for_hg38():
    adata = _simulate()
    lay = _layout(adata)[1]
    bins = lay.bins(1_000_000)[0]
    assert bins.groupby("chromosome")["end"].max()["chr1"] == M.HG38_LENGTHS["chr1"]
    adata.var.loc[adata.var["chromosome"] == "chr6", "start"] += 100_000_000  # beyond GRCh38 chr6: not GRCh38
    adata.var["end"] = adata.var["start"] + 5_000
    lay = _layout(adata)[1]
    ends = lay.bins(1_000_000)[0].groupby("chromosome")["end"].max()
    assert ends["chr1"] == lay.end[lay.g0s[0] + lay.gs[0] - 1]


def test_smooth_amplitude_ignores_noise():
    lay = _layout(_simulate())[1]
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.5, lay.n_genes)
    step = np.where(np.arange(lay.n_genes) < lay.n_genes // 2, 0.3, -0.3)
    assert M._smooth_amplitude(noise, lay) < 0.05
    assert M._smooth_amplitude(noise + step, lay) > 0.2


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"n_chrom": 22, "genes": 45}, "genes"),  # a ~1,000-gene panel
        ({"depth": 60}, "counts"),  # 2,400 genes, but 60 counts per cell
    ],
)
def test_copy_number_undetermined_on_too_little_information(kw, reason, caplog):
    adata = _simulate(**kw)
    with caplog.at_level("WARNING"):
        obs, X, uns, _ = cnv.tl.copy_number(adata, min_genes=5, min_detection=0.01, n_jobs=2, inplace=False)
    assert "undetermined" in caplog.text and "tl.infercnv" in caplog.text
    assert (obs["copy_number"] == "undetermined").all() and (obs["copy_number_clone"] == "undetermined").all()
    assert obs[[f"copy_number_{c}" for c in ("score", "llr", "spillover", "stability")]].isna().all().all()
    assert (X == 0).all() and X.shape == (adata.n_obs, len(uns["bins"]))
    info = uns["undetermined"]
    assert info["n_genes"] < M._MIN_PANEL_GENES if reason == "genes" else info["median_counts"] < M._MIN_MEDIAN_COUNTS
    forced = cnv.tl.copy_number(adata, min_genes=5, min_detection=0.01, n_jobs=2, inplace=False, force_calls=True)[0]
    assert (forced["copy_number"] == "aneuploid").any() and (forced["copy_number"] == "diploid").any()


def test_copy_number_warns_when_all_diploid(caplog):
    with caplog.at_level("WARNING"):
        obs = cnv.tl.copy_number(_simulate(n_tumour=0, n_normal=300), n_jobs=2, inplace=False)[0]
    assert (obs["copy_number"] == "diploid").all()
    assert "reference cells" in caplog.text


def _tissue(seed, depth=100, n=2000, n_chrom=6, genes=400):
    """A disc of tumour epithelium in normal epithelium, with reference (immune) cells everywhere (35%)."""
    rng = np.random.default_rng(seed)
    G = n_chrom * genes
    mu = rng.lognormal(-1, 1.2, G)
    cn = np.ones(G)
    for c, a, b, f in CN_EVENTS:
        cn[c * genes + a * genes // 400 : c * genes + b * genes // 400] = f
    xy = rng.random((n, 2))
    ref = rng.random(n) < 0.35
    dist = np.hypot(xy[:, 0] - 0.35, xy[:, 1] - 0.5) - 0.25
    tumour = ~ref & (dist < 0)
    programs = np.where(rng.random((2, G)) < 0.05, rng.choice([0.25, 4.0], (2, G)), 1.0)
    lam = mu[None] * programs[ref.astype(int)] * np.where(tumour[:, None], cn[None], 1.0)
    lam *= (depth * rng.lognormal(0, 0.3, n) / lam.sum(axis=1))[:, None]
    pos = np.tile(np.arange(genes) * (80_000_000 // genes) + 10_000, n_chrom)
    var = pd.DataFrame(
        {"chromosome": np.repeat([f"chr{i + 1}" for i in range(n_chrom)], genes), "start": pos, "end": pos + 5_000},
        index=[f"g{i}" for i in range(G)],
    )
    obs = pd.DataFrame({"type": np.where(ref, "immune", "epithelial")}, index=[f"c{i}" for i in range(n)])
    adata = AnnData(sp.csr_matrix(rng.poisson(lam).astype(np.float32)), obs=obs, var=var)
    adata.obsp["spatial_connectivities"] = knn_graph(xy, 10)
    return adata, tumour, ref, np.abs(dist) < 0.05


@pytest.mark.parametrize("seed", [0, 1])
def test_copy_number_neighbor_prior_tissue(seed):
    """With reference cells, the neighbor prior raises accuracy (also at the tumour edge), calls no reference cell,
    lowers no reference cell's probability of being diploid, and is not overconfident."""
    adata, tumour, ref, edge = _tissue(seed)
    kw = {
        "min_genes": 5,
        "n_jobs": 2,
        "inplace": False,
        "force_calls": True,
        "reference_key": "type",
        "reference_cat": "immune",
    }
    plain = cnv.tl.copy_number(adata, **kw)
    spatial = cnv.tl.copy_number(adata, connectivity_key="spatial_connectivities", **kw)

    def correct(obs):
        return (obs["copy_number"] == "aneuploid").to_numpy() == tumour

    assert correct(spatial[0]).mean() >= 0.98
    assert correct(spatial[0]).mean() >= correct(plain[0]).mean() + 0.02
    assert correct(spatial[0])[edge].mean() >= correct(plain[0])[edge].mean()
    s_plain, s = plain[0]["copy_number_score"].to_numpy(), spatial[0]["copy_number_score"].to_numpy()
    assert (s[ref] <= s_plain[ref] + 1e-12).all()  # reference cells never move towards aneuploid
    assert not (spatial[0]["copy_number"].to_numpy()[ref] == "aneuploid").any()
    assert ((s - tumour) ** 2).mean() <= ((s_plain - tumour) ** 2).mean()  # Brier score
    assert tumour[s > 0.95].mean() >= 0.99  # not overconfident: P(tumour | score > 0.95)
    # the llr stays the cell's own; labels follow the spatial posterior
    np.testing.assert_allclose(spatial[0]["copy_number_llr"], plain[0]["copy_number_llr"])
    assert ((s > 0.5) == (spatial[0]["copy_number"] == "aneuploid")).all()
    info = spatial[2]["spatial"]
    assert info["converged"] and 0 < info["neighbor_weight"] <= 0.8


@pytest.mark.parametrize("seed", [0, 1])
def test_copy_number_neighbor_prior_without_reference(seed):
    """Without a reference the neighbor prior uses the whole graph and still improves the calls."""
    adata, tumour, *_ = _tissue(seed)
    kw = {"min_genes": 5, "n_jobs": 2, "inplace": False, "force_calls": True}
    plain = cnv.tl.copy_number(adata, **kw)[0]
    spatial = cnv.tl.copy_number(adata, connectivity_key="spatial_connectivities", **kw)[0]
    acc = [((o["copy_number"] == "aneuploid").to_numpy() == tumour).mean() for o in (plain, spatial)]
    assert acc[1] >= 0.95 and acc[1] >= acc[0] + 0.02


def test_copy_number_without_graph_edges_is_unchanged():
    """Graphs without (real) edges change nothing; a missing key raises."""
    adata = _simulate(n_tumour=300, n_normal=200, depth=300, seed=3)
    adata.obsp["empty_connectivities"] = sp.csr_matrix((adata.n_obs, adata.n_obs))
    knn = knn_graph(np.random.default_rng(0).random((adata.n_obs, 2)), 10).tocoo()
    n = adata.n_obs
    # stored zeros (as left by zeroing cross-type entries) are not neighbors, nor are self-loops
    zeros = sp.csr_matrix(
        (np.r_[np.zeros(knn.nnz), np.ones(n)], (np.r_[knn.row, np.arange(n)], np.r_[knn.col, np.arange(n)])), (n, n)
    )
    assert (zeros.data == 0).sum() == knn.nnz
    adata.obsp["zero_connectivities"] = zeros
    kw = {"min_genes": 5, "n_jobs": 2, "inplace": False}
    plain = cnv.tl.copy_number(adata, **kw)
    for key in ("empty_connectivities", "zero_connectivities"):
        same = cnv.tl.copy_number(adata, connectivity_key=key, **kw)
        pd.testing.assert_frame_equal(plain[0], same[0], check_exact=True)
        np.testing.assert_array_equal(plain[1], same[1])
        assert same[2]["spatial"] == {"neighbor_weight": 0.0, "sweeps": 0, "converged": True}
        assert same[2]["params"]["connectivity_key"] == key
    assert "spatial" not in plain[2]
    with pytest.raises(KeyError, match="not found"):
        cnv.tl.copy_number(adata, connectivity_key="missing")


def test_neighbor_prior_fixed_point_and_weight():
    """The sweeps reach the fixed point q_i ~ L_i ((1 - lam) pi_i + lam m_i); the weight is about 0 when neighbors do
    not predict a cell's clone and large when they do; reference cells only lose aneuploid probability."""
    rng = np.random.default_rng(0)
    n, k = 3000, 3
    xy = rng.random((n, 2))
    G = knn_graph(xy, 8)
    log_prior = np.log(np.full((n, k), 1.0 / k))

    def run(clone, ref=None):
        lt = rng.normal(0, 1.0, (n, k))
        lt[np.arange(n), clone] += 1.0  # weak evidence for the true clone
        resp = np.exp(lt + log_prior)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp, lt, *M._neighbor_prior(G, np.arange(n), resp, lt, log_prior, ref, 0.95)

    coherent = np.minimum((xy[:, 0] * k).astype(int), k - 1)  # three stripes
    resp, lt, q, info = run(coherent)
    assert info["converged"] and info["neighbor_weight"] > 0.5
    lam = info["neighbor_weight"]
    m = np.asarray(G @ q) / np.asarray(G.sum(axis=1))
    fp = np.exp(lt - lt.max(axis=1, keepdims=True)) * ((1 - lam) / k + lam * m)
    np.testing.assert_allclose(q, fp / fp.sum(axis=1, keepdims=True), atol=1e-4)
    assert (q.argmax(axis=1) == coherent).mean() > (resp.argmax(axis=1) == coherent).mean() + 0.2
    _, _, _, info = run(rng.integers(0, k, n))  # clones placed at random
    assert info["neighbor_weight"] < 0.1
    ref = rng.random(n) < 0.3
    resp, _, q, _ = run(coherent, ref)
    assert (q[ref, 1:].sum(axis=1) <= resp[ref, 1:].sum(axis=1) + 1e-12).all()
    # a reference cell whose neighbors are all non-reference has no edges left: it keeps its per-cell posterior
    G = G.tolil()
    for i in np.flatnonzero(ref)[:20]:
        G[i, :], G[:, i] = 0, 0
        G[i, np.flatnonzero(~ref)[:5]] = G[np.flatnonzero(~ref)[:5], i] = 1
    G = G.tocsr()
    resp, _, q, info = run(coherent, ref)
    assert info["neighbor_weight"] > 0
    np.testing.assert_array_equal(q[np.flatnonzero(ref)[:20]], resp[np.flatnonzero(ref)[:20]])


@pytest.mark.parametrize("cap", [0.0, 0.3, 0.6])
def test_copy_number_max_neighbor_weight(cap):
    """`max_neighbor_weight=0` gives the per-cell results; otherwise the learnt weight stays at or below the cap."""
    adata, *_ = _tissue(0)
    kw = {"min_genes": 5, "n_jobs": 2, "inplace": False, "force_calls": True, "reference_key": "type"}
    kw["reference_cat"] = "immune"
    spatial = cnv.tl.copy_number(adata, connectivity_key="spatial_connectivities", max_neighbor_weight=cap, **kw)
    info = spatial[2]["spatial"]
    assert spatial[2]["params"]["max_neighbor_weight"] == cap
    if cap == 0:
        plain = cnv.tl.copy_number(adata, **kw)
        pd.testing.assert_frame_equal(plain[0], spatial[0], check_exact=True)
        np.testing.assert_array_equal(plain[1], spatial[1])
        assert info == {"neighbor_weight": 0.0, "sweeps": 0, "converged": True}
    else:
        assert 0 < info["neighbor_weight"] <= cap
        assert info["converged"]


def test_parameters_can_be_written(tmp_path):
    """A Generator seed is stored as None, so that the AnnData can be written."""
    adata = _simulate(depth=60)
    cnv.tl.copy_number(adata, min_genes=5, min_detection=0.01, random_state=np.random.default_rng(0))
    assert adata.uns["copy_number"]["params"]["random_state"] is None
    adata.write_h5ad(tmp_path / "a.h5ad")


def test_iterators_as_arguments():
    """An iterator argument is read once, so the exclusion it holds is applied."""
    adata = _simulate(depth=60)
    kw = {"min_genes": 5, "min_detection": 0.01, "inplace": False}
    as_list = cnv.tl.copy_number(adata, exclude_chromosomes=["chr1"], exclude_genes=["g500"], **kw)[2]
    as_iter = cnv.tl.copy_number(adata, exclude_chromosomes=iter(["chr1"]), exclude_genes=map(str, ["g500"]), **kw)[2]
    assert as_iter["undetermined"] == as_list["undetermined"]
    assert as_iter["params"] == as_list["params"]
    chr_pos = cnv.tl.infercnv(adata, exclude_chromosomes=(c for c in ["chr1"]), window_size=50, inplace=False)[0]
    assert "chr1" not in chr_pos


def test_overlapping_progress_bars_leave_the_handlers():
    """Progress bars of runs in other threads end in any order; the wrapper applies the filters added to it."""
    import logging

    from infercnvpy._logging import log

    logger = log.logger
    while not logger.handlers and logger.propagate and logger.parent is not None:
        logger = logger.parent
    before = list(logger.handlers)
    a, b = M._progress(1, True), M._progress(1, True)
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    b.__exit__(None, None, None)
    assert logger.handlers == before

    seen = []

    class Record(logging.Handler):
        def emit(self, record):
            seen.append(record)

    wrapper = M._ClearBars(Record())
    wrapper.addFilter(lambda record: False)
    wrapper.handle(logging.makeLogRecord({"msg": "x", "levelno": logging.WARNING}))
    assert not seen


def test_no_aneuploid_warning_with_reference(caplog):
    """Without aneuploid cells, the warning for a run with reference cells doesn't ask for reference cells."""
    adata = _simulate(n_tumour=0, n_normal=300)
    cnv.tl.copy_number(adata, reference_key="truth", reference_cat="normal", n_restarts=1, n_jobs=2, inplace=False)
    [msg] = [r.getMessage() for r in caplog.records if "no cell is called aneuploid" in r.getMessage()]
    assert "comes from the sample itself" not in msg


def test_short_chromosomes_have_both_window_parities():
    """Cross-fitting scores each window with the profile fitted on the other parity, so a chromosome with fewer than
    `2 window_size` genes needs windows of both parities to enter the likelihood."""
    adata = _simulate(n_tumour=20, n_normal=20, n_chrom=3, genes=15)
    cells, lay, _, _ = M._filter_cells_genes(adata, _Rows(adata.X, 5000), np.zeros(45, bool), None, 0, 0)
    data = M._Data(_Rows(adata.X, 5000), cells, lay, 10)
    parity = data.win_parity[data.gene_win]
    for g0, g in zip(lay.g0s, lay.gs, strict=True):
        assert set(parity[g0 : g0 + g]) == {0, 1}


def test_copy_number_does_not_depend_on_n_jobs():
    adata = _simulate(120, 80, n_chrom=4, genes=300)
    a, b = (cnv.tl.copy_number(adata, n_restarts=1, n_jobs=n, inplace=False) for n in (1, 2))
    pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
    np.testing.assert_array_equal(a[1], b[1])
