"""`tl.copy_number`: kernels against references, and the whole method on small synthetic data."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.special import logsumexp

import infercnvpy as cnv
from infercnvpy.tl import _copy_number as M
from infercnvpy.tl import _copy_number_kernels as K
from infercnvpy.tl._infercnv import _Rows
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
    K.pooled_counts(X.data, X.indices, X.indptr, np.arange(10, 30), col_map, labels, slabs)
    ref = np.zeros((3, 30))
    for lab in range(3):
        ref[lab, col_map[col_map >= 0]] = X[10:30][labels == lab].toarray()[:, col_map >= 0].sum(0)
    np.testing.assert_allclose(slabs.sum(0), ref)
    # per-type likelihood
    Y = rng.poisson(5, (20, 6)).astype(float)
    size = Y.sum(1)
    share = rng.dirichlet(np.ones(6), (2, 3))
    phi = np.full(6, 3.0)
    types = rng.integers(0, 2, 20)
    use = np.ones(6, dtype=np.bool_)
    g = rng.dirichlet(np.ones(6))
    ll, spill = np.empty((20, 3)), np.empty(20)
    args = (Y, size, types, share, np.log(share), phi, use, g)
    K.nb_loglik_types(*args, np.zeros(2, dtype=np.bool_), 0.5, False, ll, spill)
    for r in range(20):
        for k in range(3):
            np.testing.assert_allclose(ll[r, k], _nb_loglik(Y[r], size[r], share[types[r], k], phi))
    # with spill_ok, the diploid likelihood uses the mixture (capped at 0.5), the aneuploid ones do not change
    ll2 = np.empty((20, 3))
    K.nb_loglik_types(*args, np.ones(2, dtype=np.bool_), 0.5, False, ll2, spill)
    np.testing.assert_allclose(ll2[:, 1:], ll[:, 1:])
    s = np.minimum(spill, 0.5)
    for r in range(20):
        mix = (1 - s[r]) * share[types[r], 0] + s[r] * g
        np.testing.assert_allclose(ll2[r, 0], _nb_loglik(Y[r], size[r], mix, phi))


@pytest.mark.parametrize("n", [300, 3000, 1e6])
def test_spillover_is_the_likelihood_maximum(n):
    """Large `g / a` (a tumour-specific window) made the Newton steps stop near 0."""
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(2000))
    g = a.copy()
    spec = rng.choice(2000, 5, replace=False)
    a[spec], g[spec] = 1e-6, 0.02
    a, g = a / a.sum(), g / g.sum()
    use = np.ones(2000, dtype=np.bool_)
    y = rng.multinomial(int(n), 0.7 * a + 0.3 * g).astype(float)
    grid = np.linspace(0, 1, 2001)[:-1]
    best = grid[np.argmax([(y * np.log((1 - s) * a + s * g)).sum() for s in grid])]
    assert abs(K._spillover(y, a, g, use) - best) <= 1e-3
    assert K._spillover(1e6 * a, a, g, use) < 1e-3


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
        pytest.param(
            {"n_tumour": 600, "n_normal": 200, "depth": 400, "normal_depth": 1.0, "seed": 4}, marks=_DETECTION_BIAS
        ),
        pytest.param({"n_tumour": 600, "n_normal": 600, "depth": 400, "seed": 2}, marks=_DETECTION_BIAS),
        {"n_tumour": 600, "n_normal": 9},  # a rare normal population was too small to seed the baseline
    ],
)
def test_copy_number_baseline_not_on_the_tumour(kw):
    adata = _simulate(**kw)
    assert _accuracy(adata, cnv.tl.copy_number(adata, n_jobs=2, inplace=False, force_calls=True)[0]) >= 0.95


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"n_tumour": 600, "n_normal": 200, "depth": 400, "normal_depth": 1.0, "seed": 4},
        {"n_tumour": 600, "n_normal": 15, "depth": 400, "seed": 1},
    ],
)
def test_copy_number_warns_on_uncertain_baseline(kw, caplog):
    """Without a clear flat split and without marker-rich groups the baseline is uncertain: a warning and the seed
    diagnostics in `uns`."""
    with caplog.at_level("WARNING"):
        uns = cnv.tl.copy_number(_simulate(**kw), n_jobs=2, inplace=False, force_calls=True)[2]
    seed = uns["seed"]
    assert seed["method"] == uns["baseline"] and seed["clear_gap"] == M._FLAT_GAP
    assert ("baseline is uncertain" in caplog.text) == (seed["method"] == "flattest group")


def test_marker_counts_keep_undetected_markers():
    """The marker fraction counts every marker, also one that the detection filter removed (a rare normal
    population keeps its markers); without such markers it is the share of the markers among the model's genes."""
    adata = _simulate(n_tumour=385, n_normal=15)
    normal = (adata.obs["truth"] == "normal").to_numpy()
    X = adata.X.tolil()
    X[:, 0] = normal[:, None].astype(np.float32) * 5  # a marker of the 15 normal cells: under 5% of all cells
    adata.X = X.tocsr()
    cells, lay, var, candidates = _layout(adata)
    data = M._Data(_Rows(adata.X, 5000), cells, lay, 10)
    clusters = normal[cells].astype(np.int64)
    Yc = data.pooled(clusters)[0]
    is_marker = np.zeros(adata.n_vars, dtype=bool)
    is_marker[[0, 5]] = True
    assert lay.col_map[0] < 0 and lay.col_map[5] >= 0
    marker, total = M._marker_counts(data, clusters, Yc, var, candidates, is_marker)
    np.testing.assert_allclose(marker, Yc[:, lay.col_map[5]] + [0, 75])
    np.testing.assert_allclose(total, Yc.sum(axis=1) + [0, 75])


def test_copy_number_small_panel_keeps_labels():
    """On a ~1,000-gene panel (45 genes per chromosome) the genomic statistics still have blocks to compare.

    With fixed 25-gene blocks, every chromosome had one block, so the baseline seed fell on an arbitrary group.
    """
    adata = _simulate(n_tumour=600, n_normal=400, depth=1000, n_chrom=22, genes=45)
    obs = cnv.tl.copy_number(adata, n_jobs=2, min_genes=10, min_detection=0.01, force_calls=True, inplace=False)[0]
    assert _accuracy(adata, obs) >= 0.95


def test_em_runs_after_baseline_and_uses_its_dosage(monkeypatch):
    """Every EM run follows a baseline update (the last single-iteration run skipped it), and the reported dosage
    weights are those of the final likelihood (they were learnt after it)."""
    calls = []
    for name in ("set_baseline", "run", "loglik"):
        orig = getattr(M._EM, name)

        def spy(self, *args, _orig=orig, _name=name, **kw):
            calls.append((_name, self.dosage.copy()))
            return _orig(self, *args, **kw)

        monkeypatch.setattr(M._EM, name, spy)
    uns = cnv.tl.copy_number(_simulate(), n_jobs=2, inplace=False)[2]
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
    assert uns["baseline"] == "reference"
    assert _accuracy(adata, obs) >= 0.98
    assert "copy_number" not in adata.obs


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
        ({"min_detection": 1.5}, "min_detection"),
        ({"n_hops": 0}, "n_hops"),
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
    assert obs[["copy_number_score", "copy_number_llr", "copy_number_spillover"]].isna().all().all()
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


@pytest.mark.parametrize("seed", [3, 4])
def test_copy_number_spatial_smoothing(seed):
    """Tumour cells in one half of the tissue: smoothing the posteriors over 10 neighbours keeps the calls, does not
    lower them away from the boundary, labels follow the smoothed posterior, and graphs without (real) edges change
    nothing."""
    adata = _simulate(n_tumour=600, n_normal=600, depth=200, normal_depth=1.0, seed=seed)
    rng = np.random.default_rng(seed)
    tumour = (adata.obs["truth"] == "tumour").to_numpy()
    xy = rng.random((adata.n_obs, 2))
    xy[:, 0] = np.where(tumour, xy[:, 0] * 0.5, 0.5 + xy[:, 0] * 0.5)
    interior = np.abs(xy[:, 0] - 0.5) > 0.1
    G = knn_graph(xy, 10)
    adata.obsp["spatial_connectivities"] = G
    adata.obsp["empty_connectivities"] = sp.csr_matrix((adata.n_obs, adata.n_obs))
    zeros = G.copy()
    zeros.data[:] = 0.0  # stored zeros (as left by zeroing cross-type entries) are not neighbours
    adata.obsp["zero_connectivities"] = zeros + sp.eye(adata.n_obs, format="csr")  # nor are self-loops
    kw = {"min_genes": 5, "n_jobs": 2, "inplace": False}
    plain = cnv.tl.copy_number(adata, **kw)
    smooth = cnv.tl.copy_number(adata, connectivity_key="spatial_connectivities", **kw)
    assert _accuracy(adata, smooth[0]) >= 0.98

    def correct(obs):
        return (obs["copy_number"] == "aneuploid").to_numpy() == tumour

    assert correct(smooth[0])[interior].mean() >= correct(plain[0])[interior].mean()
    assert ((smooth[0]["copy_number_score"] > 0.5) == (smooth[0]["copy_number"] == "aneuploid")).all()
    for key in ("empty_connectivities", "zero_connectivities"):
        same = cnv.tl.copy_number(adata, connectivity_key=key, **kw)
        pd.testing.assert_frame_equal(plain[0], same[0])
        np.testing.assert_allclose(plain[1], same[1], rtol=1e-6)
    assert smooth[2]["params"]["connectivity_key"] == "spatial_connectivities"
    with pytest.raises(KeyError, match="not found"):
        cnv.tl.copy_number(adata, connectivity_key="missing")
