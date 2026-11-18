import numpy as np
import pytest
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

import infercnvpy as cnv
from infercnvpy.datasets import _DATASETS, DATASET_ENV_VAR


def test_oligodendroglioma():
    adata = cnv.datasets.oligodendroglioma()
    assert adata.shape == (184, 9450)
    assert {"chromosome", "start", "end"} <= set(adata.var.columns)


@pytest.mark.parametrize("name", _DATASETS)
def test_registry_entry(name):
    (file,) = _DATASETS[name].files
    assert file.sha256 is not None
    # files come only from exampledata.scverse.org, without fallback downloads
    assert not file.fallback_urls


@pytest.mark.internet
def test_maynard2020_3k(tmp_path, monkeypatch):
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path))
    adata = cnv.datasets.maynard2020_3k()
    assert adata.shape == tuple(_DATASETS["maynard2020_3k"].metadata["shape"])
    assert (tmp_path / "maynard2020_3k.h5ad").is_file()


@pytest.mark.internet
def test_wu2026_atera_breast(tmp_path, monkeypatch):
    """The schema that the Atera tutorial relies on: keys, columns and dtypes, no values."""
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path))
    adata = cnv.datasets.wu2026_atera_breast()
    assert adata.shape == tuple(_DATASETS["wu2026_atera_breast"].metadata["shape"])
    assert (tmp_path / "wu2026_atera_breast.h5ad").is_file()
    assert np.issubdtype(adata.X.dtype, np.integer)
    assert set(adata.obs.columns) == {
        "cell_type", "pred_state", "wgs_region", "wgs_clone_group", "wgs_region_ensemble_frac"
    }  # fmt: skip
    assert set(adata.var.columns) == {"gene_ids", "chromosome", "start", "end"}
    assert set(adata.obsm) == {"spatial"}
    assert set(adata.uns) == {"source", "wgs_truth", "window_um"}
    assert {"arms", "arms_by_wgs_run", "description"} <= set(adata.uns["wgs_truth"])
    assert {"attribution", "tissue", "preview_notice"} <= set(adata.uns["source"]["dataset"])
    assert {"Normal Luminal", "Normal Basal", "Ductal Luminal"} <= set(adata.obs["pred_state"].cat.categories)
    arms = adata.uns["wgs_truth"]["arms"]
    dtypes = {
        "clone_group": is_string_dtype, "arm": is_string_dtype, "kind": is_string_dtype, "log2": is_float_dtype,
        "log2_centred_on_N": is_float_dtype, "log2_min_region": is_float_dtype, "log2_max_region": is_float_dtype,
        "n_regions": is_integer_dtype, "wgs_runs": is_string_dtype,
    }  # fmt: skip
    assert list(arms.columns) == list(dtypes)
    for column, check in dtypes.items():
        assert check(arms[column]), column
    assert set(arms["kind"]) <= {"arm", "focal"}


def test_fetch_leaves_scanpy_datasetdir_alone(tmp_path, monkeypatch):
    """A file named like a dataset in scanpy's `datasetdir` is the user's: it is neither used nor overwritten."""
    import pooch
    import scanpy as sc

    import infercnvpy.datasets

    monkeypatch.delenv(DATASET_ENV_VAR, raising=False)
    (tmp_path / "oligodendroglioma.h5ad").write_text("mine")
    used = []
    monkeypatch.setattr(infercnvpy.datasets, "fetch", lambda entry, cache_dir, **kw: used.append(cache_dir))
    datasetdir, sc.settings.datasetdir = sc.settings.datasetdir, tmp_path
    try:
        cnv.datasets.oligodendroglioma()
    finally:
        sc.settings.datasetdir = datasetdir
    assert used == [pooch.os_cache("infercnvpy")]
    assert (tmp_path / "oligodendroglioma.h5ad").read_text() == "mine"
