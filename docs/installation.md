# Installation

infercnvpy needs Python 3.13 or newer.
We recommend [uv][], which installs Python for you and keeps each project's dependencies in their own virtual environment.

## PyPI

Install the latest release from [PyPI][]:

:::::{tab-set}
::::{tab-item} Pip
:sync: pip

```bash
pip install infercnvpy
```

::::

::::{tab-item} uv
:sync: uv

```bash
uv init  # only if you are not in a uv project yet
uv add infercnvpy
```

::::
:::::

## Development version

Install the latest development version from GitHub:

:::::{tab-set}
::::{tab-item} Pip
:sync: pip

```bash
pip install git+https://github.com/scverse/infercnvpy.git
```

::::

::::{tab-item} uv
:sync: uv

```bash
uv init  # only if you are not in a uv project yet
uv add git+https://github.com/scverse/infercnvpy.git
```

::::
:::::

## Optional dependencies

- `infercnvpy[biomart]` installs [pybiomart][] for {func}`infercnvpy.io.genomic_position_from_biomart`, which queries
  Ensembl BioMart over the internet.
  With conda, install pybiomart from bioconda (`conda install -c bioconda pybiomart`).
  You don't need it if you read the gene positions from a GTF file with {func}`infercnvpy.io.genomic_position_from_gtf`.
  BioMart returns the positions of the current Ensembl release, GRCh38 (hg38) for human.
  For data aligned to another genome assembly, such as GRCh37 (hg19), read the positions from the GTF file of the
  reference genome that the data were aligned to, or pass `host="grch37.ensembl.org"`.
- `infercnvpy[dask]` installs dask and the lazy readers of anndata (`anndata[dask,lazy]`) for {doc}`large and lazy datasets <large_data>`.
- `infercnvpy[gtf]` installs [gtfparse][] for {func}`infercnvpy.io.genomic_position_from_gtf`.
- `infercnvpy[scevan]` installs [rdata][] for {func}`infercnvpy.io.read_scevan`.
- `infercnvpy[spatial]` installs [squidpy][] for the spatial neighbor graph that {func}`infercnvpy.tl.spatial_regions`,
  {func}`infercnvpy.pl.spatial_cnv` and the `connectivity_key` of {func}`infercnvpy.tl.copy_number` read from
  `adata.obsp`.
  The functions themselves work with any graph in `adata.obsp`, so squidpy is only needed to build it.

To work on infercnvpy itself, see {doc}`contributing`.

[uv]: https://docs.astral.sh/uv/
[pypi]: https://pypi.org/project/infercnvpy
[gtfparse]: https://github.com/openvax/gtfparse
[pybiomart]: https://github.com/jrderuiter/pybiomart
[rdata]: https://github.com/vnmabus/rdata
[squidpy]: https://squidpy.readthedocs.io
