from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy.queries
from anndata import AnnData

from infercnvpy._logging import log_result
from infercnvpy._util import _as_df, _splice_doc

_MERGE_RULES = """\
When adata is given, the gene positions from the annotation (the BioMart table or the GTF file)
replace the `chromosome`, `start` and `end` columns that `adata.var` already has. For any other
column name, an existing `adata.var` column wins over an annotation column of the same name.
Existing positions are not kept as a fallback. Genes without an entry in the annotation get NaN
positions, and so do genes whose id appears more than once in the annotation.
`adata.var["chromosome"].isna().sum()` counts these genes after the call, or count them on the
returned data frame with `inplace=False`. The log also reports the number of affected gene ids,
as a warning when half or more are affected and at info level otherwise.
:func:`infercnvpy.tl.infercnv` skips genes without a position.
"""


@_splice_doc(merge_rules=_MERGE_RULES)
def genomic_position_from_biomart(
    adata: AnnData | None = None,
    *,
    adata_gene_id: str | None = None,
    biomart_gene_id="ensembl_gene_id",
    species: str = "hsapiens",
    inplace: bool = True,
    **kwargs,
) -> pd.DataFrame | None:
    """
    Get genomic gene positions from ENSEMBL Biomart.

    Needs the optional dependency pybiomart. Install it with `pip install "infercnvpy[biomart]"` or
    `conda install -c bioconda pybiomart`, or read the gene positions from a GTF file with
    :func:`~infercnvpy.io.genomic_position_from_gtf` instead.

    {merge_rules}

    Parameters
    ----------
    adata
        Adds the genomic positions to `adata.var`. If adata is None, returns
        a data frame with the genomic positions instead, without the genes whose id
        appears more than once in the annotation.
    adata_gene_id
        Column in `adata.var` that contains (Ensembl) gene IDs. If not specified,
        use `adata.var_names`.
    biomart_gene_id
        The biomart column to use as gene identifier. Typically this would be `ensembl_gene_id` or `hgnc_symbol`,
        but could be different for other species.
    inplace
        If True, add the annotations to `adata.var`. If False, return a copy of `adata.var`
        with the annotations and leave adata unchanged. Ignored when adata is None.
    **kwargs
        passed on to :func:`scanpy.queries.biomart_annotations`
    """
    try:
        import pybiomart  # noqa: F401  # scanpy.queries imports it only when it queries
    except ImportError:
        raise ImportError(
            "genomic_position_from_biomart needs the optional dependency pybiomart. Install it with "
            '`pip install "infercnvpy[biomart]"` or `conda install -c bioconda pybiomart`, or read the gene positions '
            "from a GTF file with `genomic_position_from_gtf`."
        ) from None
    biomart_annot = (
        scanpy.queries.biomart_annotations(
            species,
            [
                biomart_gene_id,
                "start_position",
                "end_position",
                "chromosome_name",
            ],
            **kwargs,
        )
        .rename(
            columns={
                "start_position": "start",
                "end_position": "end",
                "chromosome_name": "chromosome",
            }
        )
        # use chr prefix for chromosome
        .assign(chromosome=lambda x: "chr" + x["chromosome"])
    )
    if adata is None:
        # same rule as when matching to adata: genes with more than one position are dropped
        return biomart_annot.loc[~biomart_annot[biomart_gene_id].duplicated(keep=False), :]

    gene_ids_adata = (adata.var_names if adata_gene_id is None else adata.var[adata_gene_id]).values
    biomart_annot = _drop_duplicates_and_report(
        "genomic_position_from_biomart",
        biomart_annot,
        biomart_gene_id,
        gene_ids_adata,
        source="BioMart",
        remedy=f"check that the gene ids in adata are {biomart_gene_id} values (see `adata_gene_id`)",
    )

    var_annotated = _merge_var(adata, biomart_annot, adata_gene_id, biomart_gene_id)
    if inplace:
        adata.var = var_annotated
        return None
    return var_annotated


@_splice_doc(merge_rules=_MERGE_RULES)
def genomic_position_from_gtf(
    gtf_file: Path | str,
    adata: AnnData | None = None,
    *,
    gtf_gene_id: Literal["gene_id", "gene_name"] = "gene_name",
    adata_gene_id: str | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Get genomic gene positions from a GTF file.

    The GTF file needs to match the genome annotation used for your single cell dataset.

    {merge_rules}

    .. warning::
        Currently only tested with GENCODE GTFs.

    Parameters
    ----------
    gtf_file
        Path to the GTF file
    adata
        Adds the genomic positions to `adata.var`. If adata is None, returns
        a data frame with the genomic positions instead, without the genes whose id
        appears more than once in the annotation.
    gtf_gene_id
        Use this GTF column to match it to adata
    adata_gene_id
        Match this column to the gene ids from the GTF file. Default: use
        adata.var_names.
    inplace
        If True, add the annotations to `adata.var`. If False, return a copy of `adata.var`
        with the annotations and leave adata unchanged. Ignored when adata is None.
    """
    try:
        import gtfparse
    except ImportError:
        raise ImportError(
            'genomic_position_from_gtf needs the optional dependency gtfparse. Install it with `pip install "infercnvpy[gtf]"`.'
        ) from None
    gtf = gtfparse.read_gtf(gtf_file, usecols=["seqname", "feature", "start", "end", "gene_id", "gene_name"])
    if hasattr(gtf, "to_pandas"):  # gtfparse 2 returns a polars DataFrame
        gtf = gtf.to_pandas()
    gtf = (
        gtf.loc[
            gtf["feature"] == "gene",
            ["seqname", "start", "end", "gene_id", "gene_name"],
        ]
        .drop_duplicates()
        .rename(columns={"seqname": "chromosome"})
    )
    # remove ensembl versions
    gtf["gene_id"] = gtf["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    if adata is None:
        # same rule as when matching to adata: genes with more than one position are dropped
        return _add_chr_prefix(gtf.loc[~gtf[gtf_gene_id].duplicated(keep=False), :])

    gene_ids_adata = (adata.var_names if adata_gene_id is None else adata.var[adata_gene_id]).values
    gtf = _drop_duplicates_and_report(
        "genomic_position_from_gtf",
        gtf.loc[gtf[gtf_gene_id].isin(gene_ids_adata), :],
        gtf_gene_id,
        gene_ids_adata,
        source="the GTF file",
        remedy="check `gtf_gene_id` and `adata_gene_id`, and that the GTF file matches the genome build of the data",
    )

    var_annotated = _add_chr_prefix(_merge_var(adata, gtf, adata_gene_id, gtf_gene_id))
    if inplace:
        adata.var = var_annotated
        return None
    return var_annotated


def _merge_var(adata: AnnData, annot: pd.DataFrame, adata_gene_id: str | None, annot_gene_id: str) -> pd.DataFrame:
    """`adata.var` with the columns of `annot` whose `annot_gene_id` matches `adata_gene_id` (default: var_names)."""
    TMP_INDEX_NAME, TMP_KEY = "adata_var_index", "annot_gene_id"
    left_on = TMP_INDEX_NAME if adata_gene_id is None else adata_gene_id
    tmp_var = _as_df(adata.var)
    # positions from an earlier run are replaced; other columns of var are kept over those of annot
    tmp_var = tmp_var.drop(columns=[c for c in ("chromosome", "start", "end") if c in tmp_var.columns])
    annot = annot.rename(columns={annot_gene_id: TMP_KEY})
    annot = annot.drop(columns=[c for c in annot.columns if c in tmp_var.columns or c == left_on])
    orig_index_name = tmp_var.index.name
    var_annotated = (
        tmp_var.rename_axis(TMP_INDEX_NAME)
        .reset_index()
        .merge(annot, how="left", left_on=left_on, right_on=TMP_KEY, validate="one_to_one")
    )
    if annot_gene_id in var_annotated.columns:
        var_annotated = var_annotated.drop(columns=TMP_KEY)
    else:
        var_annotated = var_annotated.rename(columns={TMP_KEY: annot_gene_id})
    var_annotated.set_index(TMP_INDEX_NAME, inplace=True)
    var_annotated.index.name = orig_index_name
    return var_annotated


def _drop_duplicates_and_report(
    func: str, annot: pd.DataFrame, id_col: str, gene_ids_adata: Iterable, *, source: str, remedy: str
) -> pd.DataFrame:
    """Drop annotation rows whose gene id is not unique, and log how many genes in adata stay unannotated."""
    duplicated = annot[id_col].duplicated(keep=False)
    genes = set(gene_ids_adata)
    n_duplicated = len(genes & set(annot.loc[duplicated, id_col]))
    n_missing = len(genes - set(annot[id_col]))
    msg = "annotated %d of %d genes"
    args: list[object] = [len(genes) - n_missing - n_duplicated, len(genes)]
    if n_missing:
        msg += ", %d with no entry in %s"
        args += [n_missing, source]
    if n_duplicated:
        msg += ", %d with more than one entry for their %s"
        args += [n_duplicated, id_col]
    log_result(func, msg, *args, dropped=n_missing + n_duplicated, total=len(genes), remedy=remedy)
    return annot.loc[~duplicated, :]


def _add_chr_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'chr' prefix to the chromosome names if none has one (non-GENCODE GTFs)."""
    if np.all(~df["chromosome"].dropna().str.startswith("chr")):
        # gtfparse 3 returns categoricals; object dtype keeps NaN for genes without a position
        df["chromosome"] = "chr" + df["chromosome"].astype(object)
    return df
