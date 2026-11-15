import pytest

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


@pytest.mark.parametrize("cached", [True, False])
def test_maynard2020_3k_reuses_the_file_of_infercnvpy_0_6(tmp_path, monkeypatch, cached):
    """infercnvpy 0.6 downloaded it into scanpy's `datasetdir`: a file there is used instead of a new download."""
    import pooch
    import scanpy as sc

    import infercnvpy.datasets

    monkeypatch.delenv(DATASET_ENV_VAR, raising=False)
    if cached:
        (tmp_path / "maynard2020_3k.h5ad").touch()
    used = []
    monkeypatch.setattr(infercnvpy.datasets, "fetch", lambda entry, cache_dir, **kw: used.append(cache_dir))
    datasetdir = sc.settings.datasetdir
    sc.settings.datasetdir = tmp_path
    try:
        cnv.datasets.maynard2020_3k()
    finally:
        sc.settings.datasetdir = datasetdir
    assert used == [tmp_path if cached else pooch.os_cache("infercnvpy")]
