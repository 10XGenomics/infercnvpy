"""Example datasets."""

import os
from pathlib import Path
from textwrap import dedent, indent

import pooch
from anndata import AnnData, read_h5ad
from scverse_misc.datasets import DatasetEntry, DownloadCB, fetch, parse_registry, register_loader

DATASET_ENV_VAR = "INFERCNVPY_DATA_DIR"

_BASE_URL, _DATASETS = parse_registry(Path(__file__).parent / "registry.yaml")


# scverse-misc keys loaders by type name across all packages in the process, so ours uses the package name.
@register_loader("infercnvpy")
def _load_h5ad(entry: DatasetEntry, target: Path, download: DownloadCB, /, **kwargs: object) -> AnnData:
    # Download into the cache dir itself rather than `target` (`<cache dir>/<type>`),
    # so files cached by earlier infercnvpy versions are reused.
    return read_h5ad(download(entry.file(suffix=".h5ad"), dest=target.parent))


def _fetch(name: str) -> AnnData:
    cache_dir = os.environ.get(DATASET_ENV_VAR) or pooch.os_cache("infercnvpy")
    return fetch(_DATASETS[name], cache_dir, base_url=_BASE_URL)


_POOCH_INFO = dedent(
    f"""\
    .. note::
        infercnvpy datasets are managed through `Pooch <https://github.com/fatiando/pooch>`_.

        By default, the dataset will be downloaded into your operating system's default
        cache directory (See :func:`pooch.os_cache` for more details). If it has already been
        downloaded, it will be retrieved from the cache.

        You can override the default cache dir by setting the `{DATASET_ENV_VAR}` environment variable
        to a path of your preference.
    """
)


def oligodendroglioma() -> AnnData:
    """The original inferCNV example dataset.

    Derived from :cite:`Tirosh2016`.

    {pooch_info}
    """
    return _fetch("oligodendroglioma")


def maynard2020_3k() -> AnnData:
    """Return the dataset from :cite:`Maynard2020` as AnnData object, downsampled to 3000 cells.

    In brief, this data set was processed as follows:
        * raw data downloaded from ENA
        * gene expression quantified using Salmon and the nf-core/rnaseq pipeline.
        * basic quality control (min_counts=20k, max_counts=5M, min_genes=1k, max_mitochondrial_fraction=0.2)
        * filtered to 6000 HVG using `sc.pp.highly_variable_genes(..., flavor="seurat_v3")`
        * raw counts processed using scVI, providing sample information as batch key.
        * cell types manually annotated based on marker genes and leiden clustering and subclustering.
        * downsampled to 3000 cells.

    `adata.X` contains the `log1p` transformed, cpm-normalized raw counts.
    The `scVI` latent representation is stored in `adata.obsm["X_scVI"]`.
    A UMAP for the 3000 cells is precomputed.

    {pooch_info}
    """
    return _fetch("maynard2020_3k")


def wu2026_atera_breast() -> AnnData:
    """A region of a 10x Atera whole transcriptome section of breast cancer, with arm-level copy number from WGS.

    Cells from a cropped region of a section of an FFPE ductal carcinoma in situ (DCIS) measured with 10x Atera in
    situ gene expression. :cite:`Wu2026` analyzed the same section, and they sequenced the DNA (whole genome
    sequencing, WGS) of laser-dissected regions of consecutive sections from the same block.

    * `adata.X` holds the raw counts of the vendor segmentation (integers). `adata.var` has `gene_ids` and the gene
      positions (GRCh38) in `chromosome`, `start` and `end`, and `adata.obsm["spatial"]` holds the cell centroids in
      µm.
    * `adata.obs` has `cell_type` (cell types from expression clustering), `pred_state` (epithelial states
      transferred from the annotated cells of :cite:`Wu2026`), `wgs_region` (the dissected region whose outline
      contains the cell), `wgs_clone_group` (the clone group A, C or N of that region) and `wgs_region_ensemble_frac`
      (the share of perturbed placements of the outlines that also put the cell inside).
    * `adata.uns["wgs_truth"]["arms"]` holds the log2 copy ratio of each clone group and arm, with the columns
      `clone_group`, `arm`, `kind` (`"arm"` or `"focal"`), `log2`, `log2_centred_on_N`, `log2_min_region`,
      `log2_max_region`, `n_regions` and `wgs_runs`. `adata.uns["wgs_truth"]["arms_by_wgs_run"]` holds the same per
      WGS run, and `adata.uns["wgs_truth"]["description"]` describes the pipeline and its limits.
      `adata.uns["source"]` holds the sources and the attribution, and `adata.uns["window_um"]` the bounds of the
      region.

    The infercnvpy authors computed the arm table from the WGS reads in the SRA (PRJNA1509685) with CNVkit. The region
    outlines were placed on the section by hand from a figure of the preprint, so the mapping holds at the level of
    clone groups only, and group N contains some tumor. The script `docs/notebooks/data/build_wu2026_atera_breast.py`
    in the infercnvpy repository shows how the file was built.

    The data are a preview, which 10x generated with development versions of the panel, protocols and software and may
    replace later. Source: "Preview Data: Atera In Situ Gene Expression, FFPE Human Breast Cancer. Atera Onboard
    Analysis (development version), 10x Genomics (2026, June 2)",
    https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer, licensed under CC BY 4.0
    (https://creativecommons.org/licenses/by/4.0/). The subset, the cell labels, the region labels and the arm table
    are changes by the infercnvpy authors, also licensed under CC BY 4.0. The arm table comes from the WGS reads of SRA
    PRJNA1509685, and the region labels from outlines traced from a figure of :cite:`Wu2026`. When you use the data,
    cite 10x Genomics as above and :cite:`Wu2026`.

    {pooch_info}
    """
    return _fetch("wu2026_atera_breast")


for _func in (oligodendroglioma, maynard2020_3k, wu2026_atera_breast):
    _func.__doc__ = (_func.__doc__ or "").format(pooch_info=indent(_POOCH_INFO, " " * 4).lstrip())
