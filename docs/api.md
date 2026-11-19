# API

Import infercnvpy together with scanpy as

```python
import scanpy as sc
import infercnvpy as cnv
```

For consistency, the infercnvpy API tries to follow the {doc}`scanpy API <scanpy:api/index>` as closely as possible.

(api-io)=

## Input/Output: `io`

```{eval-rst}
.. module:: infercnvpy.io
.. currentmodule:: infercnvpy

.. autosummary::
    :toctree: generated

    io.genomic_position_from_biomart
    io.genomic_position_from_gtf
    io.read_scevan
```

## Preprocessing: `pp`

```{eval-rst}
.. module:: infercnvpy.pp
.. currentmodule:: infercnvpy

.. autosummary::
    :toctree: generated

    pp.neighbors
```

## Tools: `tl`

Tools add an interpretable annotation to the {class}`~anndata.AnnData` object,
which usually can be visualized by a corresponding plotting function.

The tools for embeddings and clustering mirror the scanpy API.
However, while the scanpy tools operate on transcriptomics data,
the infercnvpy equivalents operate on CNV data.

```{eval-rst}
.. module:: infercnvpy.tl
.. currentmodule:: infercnvpy
```

### Copy number inference

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.infercnv
    tl.copy_number
```

Which function to use:

- {func}`~infercnvpy.tl.infercnv` computes a continuous CNV signal (smoothed expression relative to a reference).
  Use it to explore that signal, for results comparable to the R inferCNV package, and for small gene panels. It
  needs reference (normal) cells.
- {func}`~infercnvpy.tl.copy_number` calls copy number. It returns calibrated aneuploid and diploid labels, clones and
  segments, from raw counts, with or without reference cells. Use it for droplet and imaging UMI data, not for
  Smart-seq2. Providing reference cells improves the detection of small clones on dense panels. On imaging data,
  `connectivity_key` lets each cell's spatial neighbors inform its call. With fewer than 1,000 usable genes (small
  panels such as Xenium 480) or a median below 150 counts per cell, it labels the cells `"undetermined"` instead of
  calling them. There, run {func}`~infercnvpy.tl.detectability` to see which arms the panel can resolve, and then
  `tl.infercnv` with reference cells, and read groups of cells rather than single cells, as in the
  {doc}`Xenium tutorial <notebooks/spatial_xenium>`.

`tl.infercnv` and `tl.copy_number` write their profiles to `adata.obsm` and `adata.uns` in the same layout, so the
plots, scores, embeddings and spatial tools below accept the output of either (`use_rep="cnv"` or
`use_rep="copy_number"`).

### CNV scores

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.cnv_score
    tl.ithcna
    tl.ithgex
```

### Detectability

{func}`~infercnvpy.tl.detectability` estimates how many cells of one type with the same change a test needs to detect
a gain or loss of each chromosome arm, so you can choose a panel and depth before an experiment, or see afterwards on
which arms there is enough power for a missing call to be informative.

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.detectability
```

### Spatial

Tools for spatial data. They read a cells x cells neighbor graph from `adata.obsp`,
e.g. `obsp["spatial_connectivities"]` from {func}`squidpy.gr.spatial_neighbors_knn`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.spatial_regions
```

### Embeddings

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.pca
    tl.umap
    tl.tsne
```

### Clustering

```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.leiden
```

## Plotting: `pl`

```{eval-rst}
.. module:: infercnvpy.pl
.. currentmodule:: infercnvpy
```

### InferCNV

```{eval-rst}
.. autosummary::
    :toctree: generated

    pl.chromosome_heatmap
    pl.chromosome_heatmap_summary
```

### Detectability

```{eval-rst}
.. autosummary::
    :toctree: generated

    pl.detectability
```

### Spatial

```{eval-rst}
.. autosummary::
    :toctree: generated

    pl.spatial_cnv
```

### Embeddings

```{eval-rst}
.. autosummary::
    :toctree: generated

    pl.umap
    pl.tsne
```

## Datasets: `datasets`

```{eval-rst}
.. module:: infercnvpy.datasets
.. currentmodule:: infercnvpy

.. autosummary::
    :toctree: generated

    datasets.maynard2020_3k
    datasets.oligodendroglioma
    datasets.wu2026_atera_breast
```
