(large-data)=

# Large and lazy datasets

{func}`infercnvpy.tl.infercnv` processes the cells in blocks and never needs the whole expression matrix in memory
as a dense array. The expression matrix can be

- an in-memory `numpy` array or `scipy` sparse matrix,
- a [dask](https://www.dask.org/) array, also with sparse `scipy` blocks, e.g. a matrix loaded lazily with
  {func}`anndata.experimental.read_elem_lazy` or {func}`anndata.experimental.read_lazy`, or a table from
  [spatialdata-io](https://spatialdata.scverse.org/projects/io/),
- the matrix of a backed AnnData (`anndata.read_h5ad(..., backed="r")`), stored as CSR or dense. Reading a block
  of rows of a CSC matrix reads all of it, so a backed CSC matrix is read into memory once, as CSR, with a warning.
  Convert it before writing the file (`adata.X = adata.X.tocsr()`) to read it block by block.

Install the optional dependency with `pip install "infercnvpy[dask]"` to use dask arrays.

## How the data is read

infercnv reads the data up to four times, one block of cells at a time:

1. With `min_mean`, the mean expression of every gene is computed in one pass over all blocks.
2. For a dask array or a backed matrix, the reference expression (the mean of all cells, or of every reference
   category) is computed in one pass over the blocks that hold reference cells (all blocks for the mean of all
   cells). In-memory data needs no extra pass.
3. The noise band of `dynamic_threshold` is computed from the blocks that hold reference cells, or from all blocks
   if there are no reference cells or `noise_reference="all"`. This pass is skipped when all cells fit in one block.
4. The CNV matrix is computed block by block.

In memory and backed, a block has `chunksize` cells. A dask array keeps its own blocks, and `chunksize` is ignored.
The block size does not change the result, except that the reference means of lazy data are sums over blocks, so
their last digits can differ from the in-memory means. Peak memory is the result plus about `chunksize` x windows
float32 values for the block being processed (and, for lazy data, the blocks that were read ahead).

## Threads and dask schedulers

By default, the next blocks are read in background threads (with the synchronous scheduler for a dask array) while
the current block is processed with `n_jobs` threads on the main thread. The result, `adata.obsm["X_cnv"]`, is in
memory. With `dynamic_threshold=None`, it is a dense array with 4 bytes for every value. Otherwise it is a CSR matrix
with 8 bytes (12 above 2^31 values) for each value that the noise filter keeps (`adata.obsm["X_cnv"].nnz` values).

With `lazy=True`, the result for a dask array stays a dask array of CSR blocks (dense blocks with
`dynamic_threshold=None`), computed with
{meth}`dask.array.Array.map_blocks`. The noise band is computed first, with one dask computation. `lazy=True` has to
be passed explicitly, also when a {class}`distributed.Client` is active. In this mode the dask scheduler decides how
many blocks run at the same time, each in a single thread, and `n_jobs` is ignored. infercnvpy never starts numba
threads inside a dask task, with any scheduler: the scheduler's threads or processes already use the cores, and
numba's default threading layer on macOS aborts the process when it is used from several threads at once.

:::{note}
infercnv expects log-normalized expression (see {ref}`input-data`). For a dask array, {func}`scanpy.pp.normalize_total`
and {func}`scanpy.pp.log1p` stay lazy. A backed AnnData cannot be normalized in place by scanpy, so normalize the data
before you write the file.
:::

## Example: a lazily loaded table

Spatial technologies often produce tables with a lazy dask `X` of CSR blocks that are indexed by Ensembl ID. Next to
the genes, such a table can contain control probes or codewords without a genomic position, which you remove before
normalization.

```python
import anndata as ad
import scanpy as sc
import zarr
from anndata.experimental import read_elem_lazy

import infercnvpy as cnv

# `obs` and `var` in memory, `X` as a dask array (or use the table of a SpatialData object)
f = zarr.open("table.zarr", mode="r")
table = ad.AnnData(X=read_elem_lazy(f["X"]), obs=ad.io.read_elem(f["obs"]), var=ad.io.read_elem(f["var"]))

# genomic positions: the table is indexed by Ensembl ID, so match the GTF's `gene_id`
cnv.io.genomic_position_from_gtf("gencode.v44.annotation.gtf", table, gtf_gene_id="gene_id")

# keep gene features only, and normalise lazily
genes = table[:, (table.var["feature_type"] == "Gene Expression").to_numpy()].copy()
sc.pp.normalize_total(genes, target_sum=1e4)
sc.pp.log1p(genes)

# `cell_type` and the categories stand for an annotation of normal cells in `obs`
cnv.tl.infercnv(genes, reference_key="cell_type", reference_cat=["Fibroblast", "Immune"])
```

Genes without a position, on chromosomes whose name doesn't start with `chr`, or on `chrM` (and on `chrX` and `chrY`
by default) are skipped, and a chromosome with at most `window_size` genes becomes a single window. Targeted panels with
few genes per chromosome carry much less CNV signal than whole-transcriptome data, so consider a smaller `window_size`.

## Plotting

{func}`infercnvpy.pl.chromosome_heatmap` draws one row per pixel of the figure, whatever the number of cells: the
cell in the middle of the cells that fall into that row (as the full plot looks when downscaled), or with
`aggregate="mean"` their mean. It reads the CNV matrix once (twice with `dendrogram=True`), block by block if `adata.obsm["X_cnv"]` is a dask array
(the lazy result of `infercnv`) or a backed matrix, and keeps only the drawn rows in memory.
{func}`~infercnvpy.pl.chromosome_heatmap_summary` computes the group means in the same way. `mode="legacy"` draws
every cell with {func}`scanpy.pl.heatmap`, which needs the whole matrix in memory as a dense table.

## Functions that need their input in memory

A lazy `adata.obsm["X_cnv"]` (from `lazy=True`, or read with {func}`anndata.experimental.read_lazy`) works with these functions, which read it block
by block:

- {func}`infercnvpy.pl.chromosome_heatmap` and {func}`~infercnvpy.pl.chromosome_heatmap_summary` with `mode="fast"`,
- {func}`infercnvpy.tl.cnv_score`.

{func}`infercnvpy.tl.copy_number` also reads a dask or backed expression matrix block by block, and makes the same
calls as in memory. On an AnnData from {func}`~anndata.experimental.read_lazy`, {func}`infercnvpy.tl.spatial_regions`
and {func}`infercnvpy.io.genomic_position_from_gtf` give the same result as in memory.

The other functions raise an error unless their input is in memory:

- the CNV matrix `adata.obsm["X_cnv"]` for {func}`infercnvpy.tl.ithcna`, {func}`infercnvpy.tl.smooth_cnv`,
  {func}`infercnvpy.pl.spatial_cnv`, {func}`infercnvpy.tl.pca` and the heatmaps with `mode="legacy"`,
- the representation that {func}`infercnvpy.pp.neighbors` and {func}`infercnvpy.tl.tsne` read (`use_rep`, by default
  the PCA `adata.obsm["X_cnv_pca"]`), and the CNV matrix when they compute the PCA,
- the expression matrix (`adata.X`, or the layer you pass) for {func}`infercnvpy.tl.ithgex`, and the raw counts
  (`adata.X`, or the layer you pass) for {func}`infercnvpy.tl.detectability`,
- the neighborhood graph of {func}`infercnvpy.pp.neighbors` (in `adata.obsp`) for {func}`infercnvpy.tl.umap` and
  {func}`infercnvpy.tl.leiden`,
- the whole AnnData for {func}`infercnvpy.pl.umap` and {func}`infercnvpy.pl.tsne` on an AnnData from
  {func}`anndata.experimental.read_lazy`,
- the whole AnnData for {func}`infercnvpy.tl.umap` and {func}`infercnvpy.tl.tsne` on a backed AnnData, because scanpy
  can't copy a backed AnnData.

A backed AnnData keeps `obsm` and `obsp` in memory, so on a backed AnnData only {func}`infercnvpy.tl.ithgex`,
{func}`infercnvpy.tl.detectability`, {func}`infercnvpy.tl.umap` and {func}`infercnvpy.tl.tsne` need it loaded.

{func}`~infercnvpy.tl.smooth_cnv` also returns a dense float32 matrix (cells x windows). Load the input before you call
these functions:

```python
adata.obsm["X_cnv"] = adata.obsm["X_cnv"].compute()  # a dask array
adata.X = adata.X.compute()  # a dask expression matrix, or the same for the layer you pass
graph = "cnv_neighbors_connectivities"  # the key of the graph that the error names
adata.obsp[graph] = adata.obsp[graph].compute()
adata = adata.to_memory()  # an AnnData from read_lazy or a backed file, including adata.X
```

`adata.to_memory()` also loads the expression matrix, which can be the largest part of the data. Load single fields
with `.compute()` where the error names one.
