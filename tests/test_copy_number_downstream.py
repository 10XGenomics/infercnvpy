"""The outputs of `tl.copy_number` work with the downstream tools that read `tl.infercnv` outputs."""

import numpy as np
import pytest

import infercnvpy as cnv
from tests.conftest import knn_graph, simulate_copy_number


@pytest.fixture(scope="module")
def adata_cn():
    adata = simulate_copy_number()
    adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 100, (adata.n_obs, 2))
    adata.obsp["spatial_connectivities"] = knn_graph(adata.obsm["spatial"])
    cnv.tl.copy_number(adata, n_jobs=2)
    return adata


@pytest.mark.parametrize("mode", ["fast", "legacy"])
def test_chromosome_heatmaps(adata_cn, mode):
    cnv.pl.chromosome_heatmap(adata_cn, groupby="copy_number_clone", use_rep="copy_number", mode=mode, show=False)
    cnv.pl.chromosome_heatmap_summary(
        adata_cn, groupby="copy_number_clone", use_rep="copy_number", mode=mode, show=False
    )


def test_embedding_clustering_and_scores(adata_cn):
    a = adata_cn.copy()
    cnv.tl.pca(a, use_rep="copy_number", key_added="cn_pca")
    cnv.pp.neighbors(a, use_rep="cn_pca", key_added="cn_neighbors")
    cnv.tl.leiden(a, neighbors_key="cn_neighbors", key_added="cn_leiden")
    cnv.tl.umap(a, neighbors_key="cn_neighbors", key_added="cn_umap")
    assert a.obsm["X_cn_umap"].shape == (a.n_obs, 2)
    scores = cnv.tl.cnv_score(a, groupby="copy_number_clone", use_rep="copy_number", inplace=False)
    assert scores["T1"] > scores["N0"]
    ith = cnv.tl.ithcna(a, groupby="truth", use_rep="copy_number", inplace=False)
    assert ith["tumour"] >= 0


def test_spatial(adata_cn):
    a = adata_cn.copy()
    cnv.tl.smooth_cnv(a, use_rep="copy_number", key_added="cn_smoothed")
    assert a.obsm["X_cn_smoothed"].shape == a.obsm["X_copy_number"].shape
    assert a.uns["cn_smoothed"]["chr_pos"] == a.uns["copy_number"]["chr_pos"]
    cnv.tl.spatial_regions(a, key="copy_number_clone", key_added="cn_regions")
    assert a.obs["cn_regions"].notna().all()
    cnv.pl.spatial_cnv(a, region="chr1:0-40000000", use_rep="copy_number", show=False)
