import logging

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import infercnvpy as cnv

LOGGER = "scverse.infercnvpy"


def test_ithgex(adata_ithgex):
    res = cnv.tl.ithgex(adata_ithgex, groupby="group", inplace=False)
    assert res["A"] == 0
    assert res["B"] == pytest.approx(1.2628, abs=0.001)


def test_ithgex_use_raw(adata_ithgex):
    """`use_raw=False` (the default) uses `.X` even when `.raw` is set; `use_raw=True` uses `.raw`."""
    expected_x = cnv.tl.ithgex(adata_ithgex, groupby="group", inplace=False)
    adata_ithgex.raw = adata_ithgex.copy()
    adata_ithgex.X = np.random.default_rng(0).normal(size=adata_ithgex.shape)
    assert cnv.tl.ithgex(adata_ithgex, groupby="group", inplace=False) != expected_x
    assert cnv.tl.ithgex(adata_ithgex, groupby="group", use_raw=True, inplace=False) == expected_x


def test_ithcna(adata_ithgex):
    res = cnv.tl.ithcna(adata_ithgex, groupby="group", inplace=False)
    assert res["A"] == pytest.approx(1.053, abs=0.001)
    assert res["B"] == 0


def test_cnv_score(adata_ithgex):
    res = cnv.tl.cnv_score(adata_ithgex, groupby="group", inplace=False)
    assert res["A"] == pytest.approx(2.25, abs=0.001)
    assert res["B"] == pytest.approx(2.5, abs=0.001)


def test_cnv_score_records_params(adata_ithgex):
    cnv.tl.cnv_score(adata_ithgex, groupby="group", key_added="score")
    assert adata_ithgex.uns["score"] == {"params": {"groupby": "group", "use_rep": "cnv"}}


@pytest.mark.parametrize("func", [cnv.tl.ithgex, cnv.tl.ithcna])
def test_ith_single_cell_group(adata_ithgex, func):
    """A group with one cell has no cell pairs: its score is NaN, the other groups are unaffected."""
    adata_ithgex.obs["group"] = list("AAAAABBC")
    res = func(adata_ithgex, groupby="group", inplace=False)
    assert np.isnan(res["C"])
    assert not np.isnan(res["A"])

    func(adata_ithgex, groupby="group", key_added="ith")
    ith = adata_ithgex.obs["ith"]
    assert np.isnan(ith["c8"])
    assert not ith[["c1", "c6"]].isna().any()


@pytest.mark.parametrize("block_rows", [3, 8])
def test_cnv_score_dask(adata_ithgex, block_rows):
    """A lazy CNV matrix (`infercnv(..., lazy=True)`) gives the scores of the in-memory one."""
    adata_ithgex.obs["group"] = list("AABACCBB")
    expected = cnv.tl.cnv_score(adata_ithgex, groupby="group", inplace=False)
    X = adata_ithgex.obsm["X_cnv"]
    adata_ithgex.obsm["X_cnv"] = da.from_array(X.toarray() if sp.issparse(X) else X, chunks=(block_rows, -1))
    res = cnv.tl.cnv_score(adata_ithgex, groupby="group", inplace=False)
    assert res.keys() == expected.keys()
    for k, v in expected.items():
        assert res[k] == pytest.approx(v)


def test_ith_reports_cells_with_constant_profile(adata_ithgex, caplog):
    """Cells without a defined correlation are left out and logged per group, with a warning above 20% of a group."""
    X = adata_ithgex.obsm["X_cnv"]
    X = np.array(X.toarray() if sp.issparse(X) else X, dtype=float)
    X[0] = 0  # one of five cells of A
    X[5:7] = 0  # two of three cells of B
    adata_ithgex.obsm["X_cnv"] = X
    with caplog.at_level(logging.INFO, logger=LOGGER):
        cnv.tl.ithcna(adata_ithgex, groupby="group", inplace=False)
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == LOGGER]
    assert (
        logging.INFO,
        "ithcna: left out 1 of 5 cells of group A with a constant profile (no defined correlation)",
    ) in msgs
    assert (
        logging.INFO,
        "ithcna: left out 2 of 3 cells of group B with a constant profile (no defined correlation)",
    ) in msgs
    [warning] = [m for level, m in msgs if level == logging.WARNING]
    assert (
        warning == "ithcna: more than 20% of the cells of groups B (2 of 3) have a constant profile and were left out"
    )


@pytest.mark.parametrize("func", [cnv.tl.ithgex, cnv.tl.ithcna])
def test_ith_exclude_self_pairs(adata_ithgex, func):
    default = func(adata_ithgex, groupby="group", inplace=False)
    assert func(adata_ithgex, groupby="group", inplace=False, exclude_self_pairs=False) == default
    without = func(adata_ithgex, groupby="group", inplace=False, exclude_self_pairs=True)
    assert without.keys() == default.keys()


def test_ithgex_use_raw_none_is_deprecated(adata_ithgex):
    expected = cnv.tl.ithgex(adata_ithgex, groupby="group", inplace=False)
    adata_ithgex.raw = adata_ithgex.copy()
    adata_ithgex.X = np.random.default_rng(0).normal(size=adata_ithgex.shape)
    with pytest.warns(FutureWarning, match="use_raw=None"):
        assert cnv.tl.ithgex(adata_ithgex, groupby="group", use_raw=None, inplace=False) != expected


def test_ithcna_use_rep_full_key_is_deprecated(adata_ithgex):
    expected = cnv.tl.ithcna(adata_ithgex, groupby="group", use_rep="cnv", inplace=False)
    with pytest.warns(FutureWarning, match="use_rep='cnv'"):
        assert cnv.tl.ithcna(adata_ithgex, groupby="group", use_rep="X_cnv", inplace=False) == expected


def test_ithcna_use_rep_any_obsm_key_is_deprecated(adata_ithgex):
    """As in 0.6, `use_rep` can still name an obsm key without the `X_` prefix."""
    expected = cnv.tl.ithcna(adata_ithgex, groupby="group", inplace=False)
    adata_ithgex.obsm["custom"] = adata_ithgex.obsm["X_cnv"]
    with pytest.warns(FutureWarning, match="use_rep='custom'"):
        assert cnv.tl.ithcna(adata_ithgex, groupby="group", use_rep="custom", inplace=False) == expected


def test_dataframe_cnv_matrix(adata_ithgex):
    """A CNV matrix stored as a DataFrame in obsm (as 0.6 accepted) works as its values."""
    funcs = (cnv.tl.ithcna, cnv.tl.cnv_score)
    expected = [f(adata_ithgex, groupby="group", inplace=False) for f in funcs]
    X = adata_ithgex.obsm["X_cnv"]
    adata_ithgex.obsm["X_cnv"] = pd.DataFrame(X.toarray() if sp.issparse(X) else X, index=adata_ithgex.obs_names)
    for f, want in zip(funcs, expected, strict=True):
        assert f(adata_ithgex, groupby="group", inplace=False) == pytest.approx(want)


@pytest.mark.parametrize("func", [cnv.tl.cnv_score, cnv.tl.ithgex, cnv.tl.ithcna])
def test_positional_groupby_is_deprecated(adata_ithgex, func):
    expected = func(adata_ithgex, groupby="group", inplace=False)
    with pytest.warns(FutureWarning, match="Passing `groupby` to .* positionally is deprecated"):
        assert func(adata_ithgex, "group", inplace=False) == expected
    with pytest.raises(TypeError, match="multiple values for groupby"):
        func(adata_ithgex, "group", groupby="group")
    with pytest.raises(TypeError, match="positional arguments"):
        func(adata_ithgex, "group", "extra")


def test_ith_logs_only_sampled_groups(adata_ithgex, caplog):
    """A group counts as sampled by its cells left after the constant ones are dropped."""
    X = adata_ithgex.obsm["X_cnv"]
    X = np.array(X.toarray() if sp.issparse(X) else X, dtype=float)
    X[0] = 0  # A keeps 4 of 5 cells: 16 matrix entries
    adata_ithgex.obsm["X_cnv"] = X

    def sampled(n_pairs):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            cnv.tl.ithcna(adata_ithgex, groupby="group", n_pairs=n_pairs, inplace=False)
        return [r.getMessage() for r in caplog.records if r.name == LOGGER and "sampled" in r.getMessage()]

    assert sampled(20) == []
    assert sampled(10) == ["ithcna: estimated from 10 sampled cell pairs for groups A"]


@pytest.mark.parametrize("func", [cnv.tl.ithgex, cnv.tl.ithcna])
def test_ith_leaves_out_unused_categories(adata_ithgex, func):
    expected = func(adata_ithgex, groupby="group", inplace=False)
    adata_ithgex.obs["group"] = adata_ithgex.obs["group"].astype("category").cat.add_categories("unused")
    assert func(adata_ithgex, groupby="group", inplace=False) == expected
    func(adata_ithgex, groupby="group", key_added="ith")
    np.testing.assert_array_equal(adata_ithgex.obs["ith"], adata_ithgex.obs["group"].map(expected).astype(float))


def test_cnv_score_obs_key_warns_at_the_caller(adata_ithgex):
    with pytest.warns(FutureWarning, match="obs_key") as record:
        cnv.tl.cnv_score(adata_ithgex, obs_key="group", inplace=False)
    assert record[0].filename == __file__
