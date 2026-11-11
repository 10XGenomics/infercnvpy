from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import infercnvpy as cnv

DATA = Path(__file__).parent / "data" / "scevan"
CELLS = [f"cell{i}-1" for i in range(1, 7)]


@pytest.fixture
def adata():
    return AnnData(X=np.zeros((7, 3)), obs=pd.DataFrame(index=[*CELLS, "extra-1"]))


def test_read_scevan(adata):
    cnv.io.read_scevan(adata, DATA, DATA / "sample_res_table.csv", subclones=False)
    assert list(adata.obs_names) == CELLS
    assert adata.obsm["X_scevan"].shape == (6, 12)
    assert adata.uns["scevan"]["chr_pos"] == {"chr1": 0, "chr2": 4, "chr3": 8}
    assert list(adata.obs["scevan_class"]) == ["tumor"] * 4 + ["normal"] * 2


def test_read_scevan_values(adata):
    cnv.io.read_scevan(adata, DATA, subclones=False)
    # values must match the R matrix (genes x cells, transposed to cells x genes)
    assert np.isfinite(adata.obsm["X_scevan"]).all()
    assert adata.obsm["X_scevan"].dtype.kind == "f"


def test_read_scevan_subclones(adata):
    cnv.io.read_scevan(adata, DATA, subclones=False)
    base = adata.obsm["X_scevan"].copy()

    adata2 = AnnData(X=np.zeros((7, 3)), obs=pd.DataFrame(index=[*CELLS, "extra-1"]))
    cnv.io.read_scevan(adata2, DATA, subclones=True)
    sub = adata2.obsm["X_scevan"]
    assert sub.shape == base.shape
    # first four cells are in the subclone matrix (fixture = CNA + 0.5), the rest are unchanged
    np.testing.assert_allclose(sub[:4], base[:4] + 0.5)
    np.testing.assert_allclose(sub[4:], base[4:])


def test_read_scevan_not_inplace(adata):
    res = cnv.io.read_scevan(adata, DATA, subclones=False, inplace=False)
    assert res is not None
    assert "X_scevan" in res.obsm
    assert "X_scevan" not in adata.obsm
