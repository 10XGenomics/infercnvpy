import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData, read_h5ad
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm

import infercnvpy as cnv
from infercnvpy.tl._detectability import _HG38_CENTROMERES

CHROMS = ["chr1", "chr2", "chr3", "chr4"]
CENTROMERES = {c: _HG38_CENTROMERES[c] for c in CHROMS}
PHI = 0.5  # NB dispersion of every gene
ALPHA = 0.01
REPLICATES = 1000


class Sim:
    """NB counts of 15 genes on each arm of chr1 to chr4, with optional copy ratios and arm programs."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        rows = []
        for c in CHROMS:
            cen = _HG38_CENTROMERES[c]
            for side, (lo, hi) in (("p", (1e6, cen - 1e6)), ("q", (cen + 1e6, cen + 40e6))):
                rows += [(c, int(s), c + side) for s in np.sort(self.rng.uniform(lo, hi, 15))]
        self.var = pd.DataFrame(rows, columns=["chromosome", "start", "arm"], index=[f"g{i}" for i in range(len(rows))])
        share = self.rng.lognormal(0, 1, len(rows))
        self.share = share / share.sum()

    def arm(self, name):
        return (self.var["arm"] == name).to_numpy()

    def counts(self, n, depth, *, ratio=None, program=None, fold=None):
        """`ratio` and `fold` ({arm: r}) multiply an arm's mean; `program` ({arm: sd}) by a random log-normal per cell."""
        m = depth * self.rng.lognormal(0, 0.3, n)[:, None] * self.share[None, :]
        for arm, sd in (program or {}).items():
            m[:, self.arm(arm)] *= np.exp(sd * self.rng.standard_normal(n) - sd**2 / 2)[:, None]
        for arm, r in {**(ratio or {}), **(fold or {})}.items():
            m[:, self.arm(arm)] *= r
        return self.rng.poisson(self.rng.gamma(1 / PHI, m * PHI))

    def adata(self, X, **obs):
        obs_df = pd.DataFrame(obs, index=[f"c{i}" for i in range(len(X))])
        return AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs_df, var=self.var.drop(columns="arm"))

    def z_test(self, X, n, arm, v):
        """Score test of `arm` on clones of `n` cells (rows in order), with the true shares and the null variance `v`."""
        a = self.arm(arm)
        s = (X.sum(1) - X[:, a].sum(1)) / (1 - self.share[a].sum())  # leave-arm-out size, as the model's
        m = s[:, None] * self.share[a][None, :]
        w = 1 / (1 + m * PHI)
        U = ((X[:, a] - m) * w).sum(1).reshape(-1, n).sum(1)
        I = (m * w).sum(1).reshape(-1, n).sum(1)
        return U / np.sqrt(v * I)

    def power(self, depth, n, arm, r, v, **kwargs):
        z = self.z_test(self.counts(REPLICATES * n, depth, ratio={arm: r}, **kwargs), n, arm, v)
        return np.mean(z > norm.ppf(1 - ALPHA)) if r > 1 else np.mean(z < norm.ppf(ALPHA))


@pytest.fixture
def sim():
    return Sim()


def _run(adata, **kwargs):
    kwargs = {"inplace": False, "alpha": ALPHA, "at_n_cells": 50, "centromeres": CENTROMERES} | kwargs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Measuring the null variance on all cells")
        return cnv.tl.detectability(adata, **kwargs).set_index("arm", drop=False)


@pytest.mark.parametrize("depth", [10, 30, 100])
def test_power_matches_simulation(sim, depth):
    """A z-test on simulated clones attains the predicted power, at the cells needed and at the smallest ratio."""
    row = _run(sim.adata(sim.counts(3000, depth))).loc["chr1p"]
    v = row["null_var"]
    assert 1.02 < v < 1.4  # counting noise and the noise of the size factor only
    null = sim.z_test(sim.counts(REPLICATES * 50, depth), 50, "chr1p", v)
    assert np.mean(null > norm.ppf(1 - ALPHA)) < 2.5 * ALPHA
    for r, n in (
        (1.5, round(row["cells_gain"])),
        (0.5, round(row["cells_loss"])),
        (row["gain_ratio"], 50),
        (row["loss_ratio"], 50),
    ):
        assert sim.power(depth, n, "chr1p", r, v) == pytest.approx(0.8, abs=0.06), (r, n)


def test_null_variance_from_arm_program(sim):
    """A program that moves an arm's genes together inflates its null variance; the ideal bound over-promises."""
    program = {"chr1p": 0.3}
    df = _run(sim.adata(sim.counts(3000, 100, program=program)))
    row = df.loc["chr1p"]
    assert row["null_var"] > 1.5
    assert df.loc[df.arm != "chr1p", "null_var"].max() < 1.4
    assert row["cells_gain"] > 1.4 * row["cells_gain_ideal"]
    v = row["null_var"]
    achievable = sim.power(100, round(row["cells_gain"]), "chr1p", 1.5, v, program=program)
    ideal = sim.power(100, round(row["cells_gain_ideal"]), "chr1p", 1.5, v, program=program)
    assert achievable == pytest.approx(0.8, abs=0.06)
    assert ideal < 0.72


def test_cell_type_programs(sim):
    """Without `groupby`, a type difference on an arm is null variance; with it, it is part of the expectation."""
    X = np.vstack([sim.counts(1500, 50), sim.counts(1500, 50, fold={"chr2p": 4.0})])
    adata = sim.adata(X, cell_type=["A"] * 1500 + ["B"] * 1500)
    pooled = _run(adata)
    typed = _run(adata, groupby="cell_type")
    assert pooled.loc["chr2p", "null_var"] > 2
    assert (typed.loc["chr2p", "null_var"] < 1.4).all()
    b = typed[typed.group == "B"].loc["chr2p"]
    a = typed[typed.group == "A"].loc["chr2p"]
    assert b["info"] > 2 * a["info"]  # more counts on the arm in type B
    assert b["cells_gain"] < a["cells_gain"]


def test_reference_cells(sim):
    """Copy number changes of non-reference cells don't enter the null variance."""
    X = np.vstack([sim.counts(1500, 50), sim.counts(1500, 50, ratio={"chr1p": 2.0})])
    adata = sim.adata(X, status=["normal"] * 1500 + ["tumor"] * 1500)
    with pytest.warns(UserWarning, match="null variance on all cells"):
        everyone = cnv.tl.detectability(adata, inplace=False, centromeres=CENTROMERES).set_index("arm")
    ref = _run(adata, reference_key="status", reference_cat="normal")
    assert everyone.loc["chr1p", "null_var"] > 1.5
    assert ref.loc["chr1p", "null_var"] < 1.4


def test_inplace_and_obs(sim):
    adata = sim.adata(sim.counts(500, 30), status=["normal"] * 500)
    kwargs = {"reference_key": "status", "reference_cat": "normal", "centromeres": CENTROMERES}
    assert cnv.tl.detectability(adata, key_added="det", random_state=3, **kwargs) is None
    table = adata.uns["det"]["table"]
    assert list(table["arm"].astype(str)) == ["chr1p", "chr1q", "chr2p", "chr2q", "chr3p", "chr3q", "chr4p", "chr4q"]
    assert table["n_genes"].eq(15).all()
    params = adata.uns["det"]["params"]
    assert params["n_reference"] == 500
    assert params["centromeres"] == CENTROMERES
    assert params["exclude_chromosomes"] == ["chrY"]
    assert params["random_state"] == 3
    info = adata.obs["det_information"].to_numpy()
    assert np.all(info >= 0)
    assert info.sum() == pytest.approx(500 * table["info"].sum())


@pytest.mark.parametrize("centromeres", ["hg38", _HG38_CENTROMERES])
def test_default_arms(sim, centromeres):
    """GRCh38 (by name or passed): all arms but chrY and the acrocentric p arms, with 0 genes where the panel has none."""
    df = _run(sim.adata(sim.counts(300, 30)), centromeres=centromeres)
    assert len(df) == 41
    assert {"chrXp", "chrXq"} <= set(df.index)
    assert "chr13p" not in df.index
    empty = df.loc["chr22q"]
    assert empty["n_genes"] == 0
    assert np.isinf(empty["cells_gain"])
    assert np.isnan(empty["gain_ratio"])


def test_genes(sim):
    """`genes` cuts the panel out of adata, warns about missing genes, and stores the result in adata."""
    adata = sim.adata(sim.counts(1000, 50), status=["normal"] * 1000)
    genes = list(adata.var_names[::2])
    kwargs = {"reference_key": "status", "reference_cat": "normal"}
    with pytest.warns(UserWarning, match="1 of 61 genes"):
        from_list = _run(adata, genes=iter([*genes, "NOT_A_GENE"]), **kwargs)  # any iterable
    from_data = _run(adata[:, genes].copy(), **kwargs)
    pd.testing.assert_frame_equal(from_list, from_data)
    # the panel has fewer counts per cell than the transcriptome it is cut from
    assert from_list["group_counts"].iloc[0] < np.median(adata.X.sum(axis=1).A1)
    with pytest.warns(UserWarning, match="1 of 61 genes"):
        cnv.tl.detectability(adata, genes=[*genes, "NOT_A_GENE"], centromeres=CENTROMERES, key_added="plan", **kwargs)
    assert adata.uns["plan"]["params"]["genes"] == genes
    assert adata.obs["plan_information"].shape == (adata.n_obs,)


def test_depths(sim):
    adata = sim.adata(sim.counts(3000, 40))
    observed = np.median(adata.X.sum(axis=1).A1)
    df = _run(adata, depth=[observed / 2, observed, observed * 2])
    low, mid, high = (g for _, g in df.groupby("depth"))
    assert low["depth_factor"].iloc[0] == pytest.approx(0.5)
    assert high["depth_factor"].iloc[0] == pytest.approx(2.0)
    assert list(df.columns[:3]) == ["group", "depth", "depth_factor"]
    # the parameters travel with a returned table
    assert df.attrs["params"]["depth"] == pytest.approx([observed / 2, observed, observed * 2])
    assert df.attrs["params"]["observed_depth"] == pytest.approx(observed)
    assert set(low["null_var_source"]) == set(mid["null_var_source"]) == {"measured"}
    assert set(high["null_var_source"]) == {"extrapolated"}
    assert (low["cells_gain_ideal"] / mid["cells_gain_ideal"]).between(1.3, 2.3).all()
    assert (mid["cells_gain_ideal"] / high["cells_gain_ideal"]).between(1.3, 2.3).all()
    # each depth has its own random stream: thinning to a depth gives the same rows alone or in a list
    alone = _run(adata, depth=observed / 2)
    pd.testing.assert_frame_equal(alone, low)


def test_extrapolated_null_variance(sim):
    """Extrapolating to a deeper experiment matches the null variance measured on deeper data.

    The size-factor part of the excess stays about flat with depth, and only the program part grows.
    """
    program = {"chr1p": 0.3}
    shallow = sim.adata(sim.counts(3000, 20, program=program))
    deep = sim.adata(sim.counts(3000, 200, program=program))
    target = np.median(deep.X.sum(axis=1).A1)
    with pytest.warns(UserWarning, match="times the observed median"):
        extrapolated = _run(shallow, depth=target)
    assert extrapolated["depth_factor"].iloc[0] > 2
    measured = _run(deep)
    assert set(extrapolated["null_var_source"]) == {"extrapolated"}
    ratio = extrapolated["cells_gain"] / measured["cells_gain"]
    assert ratio.between(0.8, 1.25).all(), ratio
    assert extrapolated.loc["chr1p", "null_var"] > 1.5  # the program part grows with depth


def test_dispersion_without_reference_counts(sim):
    """Genes the reference doesn't express are not taken as Poisson, and their arm is not measured."""
    X = np.vstack([sim.counts(1000, 50, fold={"chr2p": 0.0}), sim.counts(1000, 50)])
    adata = sim.adata(X, cell_type=["A"] * 1000 + ["B"] * 1000)
    with pytest.warns(UserWarning, match="no information on the arms chr2p"):
        df = _run(adata, groupby="cell_type", reference_key="cell_type", reference_cat="A")
    b = df[df.group == "B"].loc["chr2p"]
    assert np.isnan(b["null_var"])
    assert np.isnan(b["cells_gain"])
    assert np.isfinite(b["cells_gain_ideal"])
    # the ideal bound of chr2p uses the median dispersion of the other genes (0.5), not 0
    from_phi = _run(adata[adata.obs.cell_type == "B"].copy())
    assert b["cells_gain_ideal"] == pytest.approx(from_phi.loc["chr2p", "cells_gain_ideal"], rel=0.15)


def test_small_reference_groups(sim):
    """With each cell left out of its own expectation, small groups don't shrink the dispersion."""
    X = sim.counts(3000, 30)
    one = _run(sim.adata(X))
    small = _run(sim.adata(X, cell_type=np.repeat(np.arange(300), 10).astype(str)), groupby="cell_type")
    per_arm = small.reset_index(drop=True).groupby("arm", observed=True)
    assert np.all((per_arm["cells_gain"].median() / one["cells_gain"].to_numpy()).between(0.9, 1.2))
    assert np.all((per_arm["null_var"].first() / one["null_var"].to_numpy()).between(0.9, 1.15))


def test_single_arm_panel(sim):
    """A panel on one arm has no counts to measure a cell's size with, so it has no information."""
    adata = sim.adata(sim.counts(500, 30))[:, sim.arm("chr1p")].copy()
    with pytest.warns(UserWarning, match="no information on the arms chr1p"):
        df = _run(adata, centromeres={"chr1": _HG38_CENTROMERES["chr1"]})
    row = df.loc["chr1p"]
    assert row["info"] == 0
    assert np.isinf(row["cells_gain"])
    assert np.isnan(row["gain_ratio"])


def test_non_canonical_counts(sim):
    """Duplicate CSR entries are summed, as in the dense matrix."""
    X = sp.csr_matrix(sim.counts(300, 30).astype(np.float32))
    half = np.floor(X.data / 2)
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    order = np.argsort(np.r_[rows, rows], kind="stable")  # each entry twice in its row, as half and rest
    dup = sp.csr_matrix(
        (np.r_[half, X.data - half][order], np.r_[X.indices, X.indices][order], 2 * X.indptr), shape=X.shape
    )
    assert not dup.has_canonical_format
    a, b = sim.adata(X.toarray()), sim.adata(X.toarray())
    b.X = dup
    pd.testing.assert_frame_equal(_run(a), _run(b))


def test_group_order(sim):
    """Groups come in the order of the categories of `obs[groupby]`, unused categories dropped."""
    adata = sim.adata(sim.counts(200, 30), cell_type=["A"] * 100 + ["B"] * 100)
    adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"], categories=["B", "C", "A"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        df = _run(adata, groupby="cell_type")
    assert list(dict.fromkeys(df["group"])) == ["B", "A"]
    assert list(df["group"].cat.categories) == ["B", "A"]


def test_exclude_chromosomes_string_and_inplace_false(sim):
    adata = sim.adata(sim.counts(200, 30))
    df = _run(adata, exclude_chromosomes="chr4")
    assert "chr4p" not in df.index
    assert "chr3p" in df.index
    assert "detectability" not in adata.uns
    assert "detectability_information" not in adata.obs


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"at_n_cells": 0}, "at_n_cells"),
        ({"gain": 0.9}, "gain"),
        ({"loss": 1.0}, "loss"),
        ({"alpha": 0}, "alpha"),
        ({"power": 1}, "power"),
        ({"power": 0.4}, "power"),
        ({"depth": -5}, "depth"),
        ({"depth": np.inf}, "depth"),
        ({"depth": []}, "depth"),
        ({"centromeres": "hg19"}, "centromeres"),
        ({"reference_key": "status"}, "both"),
        ({"reference_key": "status", "reference_cat": "x"}, "No cell has"),
        ({"reference_key": "status", "reference_cat": []}, "selects no cells"),
        ({"exclude_chromosomes": CHROMS}, "No gene"),
        ({"reference_key": "nope", "reference_cat": "x"}, "nope"),
        ({"groupby": "nope"}, "groupby"),
        ({"layer": "counts"}, "layer"),
        ({"genes": "g0"}, "not a string"),
        ({"genes": ["x"]}, "None of the `genes`"),
    ],
)
def test_argument_errors(sim, kwargs, match):
    adata = sim.adata(sim.counts(50, 30), status=["normal"] * 50)
    with pytest.raises((ValueError, KeyError, TypeError), match=match):
        _run(adata, **kwargs)


def test_input_errors(sim):
    adata = sim.adata(sim.counts(50, 30), status=["normal"] * 50)
    with pytest.raises(ValueError, match="raw counts"):
        _run(AnnData(adata.X * 0.5, var=adata.var))
    with pytest.raises(ValueError, match="chromosome"):
        _run(AnnData(adata.X))
    with pytest.raises(ValueError, match="needs cells and genes"):
        _run(adata[:0].copy())
    with pytest.raises(TypeError, match="pass a gene list as `genes`"):
        _run(["g0", "g1"])
    dup = adata.copy()
    dup.var_names = ["g0"] * 2 + list(dup.var_names[2:])
    with pytest.raises(ValueError, match="more than once"):
        _run(dup, genes=["g0", "g5"])


def test_positions_beyond_hg38(sim):
    """Gene starts beyond the GRCh38 chromosome lengths point at another genome build."""
    adata = sim.adata(sim.counts(100, 30))
    adata.var.loc[adata.var_names[-3:], "start"] = 260_000_000  # chr4 is 190 Mb long
    with pytest.warns(UserWarning, match="3 genes start beyond the end"):
        _run(adata, centromeres="hg38")


def test_h5ad_round_trip(sim, tmp_path):
    adata = sim.adata(sim.counts(300, 30), status=["normal"] * 300)
    kwargs = {"reference_key": "status", "reference_cat": "normal", "centromeres": CENTROMERES, "depth": [10, 20]}
    cnv.tl.detectability(adata, **kwargs)
    adata.write_h5ad(tmp_path / "a.h5ad")
    back = read_h5ad(tmp_path / "a.h5ad")
    pd.testing.assert_frame_equal(
        back.uns["detectability"]["table"], adata.uns["detectability"]["table"], check_categorical=False
    )
    assert back.uns["detectability"]["params"]["depth"].tolist() == [10.0, 20.0]
    cnv.pl.detectability(back, show=False)


@pytest.fixture
def table(sim):
    X = np.vstack([sim.counts(300, 30), sim.counts(300, 30, fold={"chr2p": 3.0})])
    return _run(sim.adata(X, cell_type=["A"] * 300 + ["B"] * 300), groupby="cell_type", centromeres="hg38")


@pytest.mark.parametrize("metric", ["cells_gain", "loss_ratio", "null_var"])
def test_plot_heatmap(table, metric):
    ax = cnv.pl.detectability(table, metric=metric, show=False)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]
    assert len(ax.get_xticklabels()) == 41
    # the legend sits in the figure margin, not on the tick labels
    assert ax.get_legend() is None
    assert len(ax.get_figure().legends) == 1


def test_plot_bar_and_panels(table):
    one = table[table.group == "A"]
    ax = cnv.pl.detectability(one, show=False)
    assert ax.get_yscale() == "log"
    assert len(ax.patches) == 8  # one bar per arm with genes
    ax = cnv.pl.detectability({"panel 1": one, "panel 2": one}, metric="gain_ratio", ideal=True, show=False)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["panel 1", "panel 2"]
    fig = cnv.pl.detectability(one, figsize=(5, 2), cmap=plt.get_cmap("viridis"), return_fig=True)
    assert isinstance(fig, Figure)
    assert tuple(fig.get_size_inches()) == (5, 2)
    _, ax = plt.subplots()
    assert cnv.pl.detectability(one, ax=ax, show=False) is ax
    assert ax.get_legend() is not None  # on the given axes, not on a figure the function doesn't own


def test_plot_indexed_and_reordered(table):
    """An index that holds a column is kept, and arms come in genome order whatever the input order."""
    one = table[table.group == "A"]
    shuffled = one.iloc[::-1].set_index("arm", drop=True)
    ax = cnv.pl.detectability({"x": shuffled, "y": one}, show=False)
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks[:3] == ["chr1p", "chr1q", "chr2p"]


def test_plot_constant_cells(table):
    """A heatmap whose cells all need the same number of cells doesn't fail on the color scale."""
    flat = table.assign(cells_gain=np.where(np.isfinite(table["cells_gain"]), 20.0, np.inf))
    cnv.pl.detectability(flat, show=False)


def test_plot_errors(table):
    with pytest.raises(ValueError, match="metric"):
        cnv.pl.detectability(table, metric="nope")
    with pytest.raises(ValueError, match="one condition"):
        cnv.pl.detectability(table, kind="bar")
    with pytest.raises(ValueError, match="ideal=True"):
        cnv.pl.detectability(table[table.group == "A"], kind="bar", ideal=True)
    with pytest.raises(ValueError, match="no ideal column"):
        cnv.pl.detectability(table, metric="null_var", ideal=True)
    with pytest.raises(KeyError, match="Run `tl.detectability`"):
        cnv.pl.detectability(AnnData(np.zeros((2, 2))))
    # rows of two calls that differ only in parameters not in the table are not merged silently
    with pytest.raises(ValueError, match="several rows per condition"):
        cnv.pl.detectability(pd.concat([table, table]))


def test_threads_and_input_formats(sim):
    """The result doesn't depend on the number of threads, the matrix format, the dtype, or X against a layer."""
    X = sim.counts(600, 30)
    a = sim.adata(X, status=["normal"] * 300 + ["tumor"] * 300)
    kwargs = {"groupby": "status", "reference_key": "status", "reference_cat": "normal", "depth": [10, 20]}
    expected = _run(a, n_jobs=1, **kwargs)
    pd.testing.assert_frame_equal(_run(a, n_jobs=4, **kwargs), expected, check_exact=True)
    for matrix in (X.astype(np.int64), sp.csr_matrix(X.astype(np.int64)), X.astype(np.float32)):
        b = a.copy()
        b.X = matrix
        pd.testing.assert_frame_equal(_run(b, **kwargs), expected, rtol=1e-6)
    c = a.copy()
    c.layers["counts"] = c.X.copy()
    c.X = sp.csr_matrix(np.log1p(c.X.toarray()))
    pd.testing.assert_frame_equal(_run(c, layer="counts", **kwargs), expected, rtol=1e-6)


def test_power_one_half(sim):
    """At power 0.5 the quadratic of `_min_ratio` is a square, which still has its root."""
    table = _run(sim.adata(sim.counts(1000, 50)), power=0.5)
    finite = np.isfinite(table["cells_gain"])
    assert finite.any()
    assert table.loc[finite, ["gain_ratio", "loss_ratio", "gain_ratio_ideal", "loss_ratio_ideal"]].notna().all().all()


def test_thinning_leaves_the_cell_out_of_its_group(sim):
    """At a lower depth, a share group of one reference cell gets less information, not more."""
    X = sim.counts(501, 50)
    adata = sim.adata(X, cell_type=["A"] * 500 + ["S"])
    observed = float(np.median(X.sum(axis=1)))
    kw = {"groupby": "cell_type", "reference_key": "cell_type", "reference_cat": ["A", "S"]}
    table = _run(adata, depth=[observed, 0.9 * observed], **kw)
    s = table.reset_index(drop=True).query("group == 'S'")
    info = s.pivot_table(index="arm", columns="depth", values="info", observed=True)
    assert (info.iloc[:, 0] <= info.iloc[:, 1] + 1e-12).all()


def test_depth_just_below_the_observed_one(sim):
    """A depth within rounding of the observed median counts as the observed one."""
    adata = sim.adata(sim.counts(300, 50))
    observed = float(np.median(np.asarray(adata.X.sum(axis=1)).ravel()))
    table = _run(adata, depth=observed * (1 - 1.0005e-5))
    assert (table["null_var_source"] == "measured").all()
