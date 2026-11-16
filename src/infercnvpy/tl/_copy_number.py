"""Copy number profiles, clones and aneuploid/diploid labels from raw counts with a clone-level count model.

The method is infercnvpy's own; see the docstring of `copy_number` for the steps. Implementation notes:

- the HMM is a forward-backward pass per profile in numba (`_copy_number_kernels.hmm_posteriors`), with Student-t
  emissions on gene-level log ratios and transitions that depend on the genomic distance between genes;
- per-cell work (window counts, pooled counts, likelihoods, dispersion moments) are fused numba kernels over row blocks
  of the count matrix, so memory is linear in the number of cells: one float32 matrix of cells x windows of genes;
- constants were chosen on the GSE148673 samples and the cnv-bench datasets (Schmid et al. 2025); the method targets
  droplet (10x) and imaging data with UMI counts, not full-length plate protocols.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.cluster import hierarchy
from scipy.ndimage import median_filter
from scipy.optimize import nnls
from scipy.spatial.distance import squareform

from infercnvpy._logging import log, log_result
from infercnvpy._numba import n_kernel_chunks, numba_threads
from infercnvpy._util import _as_matrix, _params_for_uns
from infercnvpy.tl import _copy_number_kernels as K
from infercnvpy.tl._infercnv import _categories, _natural_sort, _Rows
from infercnvpy.tl._spatial import _graph, _smooth

#: Cell-cycle genes of Tirosh et al. 2016 (S and G2/M phase), with the current symbols of renamed genes added.
CELL_CYCLE_GENES = (
    # S phase
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
    "UHRF1", "MLF1IP", "CENPU", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
    "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
    "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
    # G2/M phase
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67",
    "TMPO", "CENPF", "TACC3", "FAM64A", "PIMREG", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11",
    "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "JPT1", "CDC20", "TTK", "CDC25C", "KIF2C",
    "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR",
    "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
)  # fmt: skip

#: GRCh38 chromosome lengths, so that bins are the same for every dataset
HG38_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
    "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
    "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}  # fmt: skip

_KMEANS_ITER = 20

LABELS = ("aneuploid", "diploid", "not.defined", "undetermined")
UNDETERMINED = LABELS[3]

#: Markers of immune, endothelial and stromal cells, which are rarely tumour cells
NORMAL_MARKERS = (
    "PTPRC", "CD3E", "CD3D", "CD2", "CD7", "NKG7", "GNLY", "MS4A1", "CD79A", "CD19", "CD14", "LYZ", "CD68", "CSF1R",
    "FCGR3A", "TYROBP", "FCER1G", "C1QA", "C1QB", "CPA3", "TPSAB1", "PECAM1", "VWF", "CDH5", "CLDN5", "KDR", "COL1A1",
    "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "RGS5",
)  # fmt: skip

#: copy ratios of the HMM states (expression log ratios are their logarithms)
STATE_RATIOS = (0.35, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.6)
_P_NEUTRAL = 0.5  # prior probability of the neutral state
_NU = 1.5  # degrees of freedom of the t emissions
_N_HVG = 2000
_N_PCS = 30
_N_SUB = 20000  # cells for the expression PCA
_BLOCK = 25  # genes per block of the genomic statistics (flatness, smooth amplitude, mode); fewer on small panels
_MIN_CLONE = 5
_EXACT_SVD = 10_000  # cells up to which the expression PCA uses an SVD (above: eigendecomposition)
_DOSAGE_PRIOR = 1.0  # strength of the shrinkage of dosage weights towards 1, in median gene precisions
_MIN_PRIOR = 0.01  # smallest prior clone probability of a cell type
_AMP_TUMOUR = 0.1  # smooth amplitude (log ratio) of an aneuploid clone's profile, see `_smooth_amplitude`
_N_PASSES = 4
_RESEED_GAP = 0.05  # how much flatter than the diploid clone another clone must be to restart the passes from it
_FLAT_GAP = 0.15  # difference of the cluster means of group flatness that counts as a clear split
_MARKER_FOLD = 5.0
_MARKER_MIN = 1e-3  # share of counts in the markers above which a group can be marker-rich
_MAX_SPILL = 0.5  # largest spillover fraction of tumour transcripts in a normal cell's diploid expectation
_MIN_SEED = 3  # cells of the smallest group that can be the first diploid group (a rare normal population)
_MIN_RESEED = 10  # cells of the smallest clone that can become the diploid clone later
_MIN_SEG_GENES = 20  # genes of the shortest run that counts towards the burden; fewer on small panels
_DOSAGE = "final"  # "final": dosage weights of each fit enter its final likelihood; "drop": no dosage weights
_MIN_EFFECT = 0.1  # smallest posterior mean log ratio that counts as a copy number change
_MIN_PANEL_GENES = 1000  # genes passing the filters below which the calls are undetermined (small panels)
_MIN_MEDIAN_COUNTS = 150  # median counts per cell in those genes below which the calls are undetermined


def copy_number(
    adata: AnnData,
    *,
    layer: str | None = None,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    exclude_chromosomes: Sequence[str] | None = ("chrY",),
    exclude_genes: Sequence[str] | None = None,
    min_genes: int = 200,
    min_detection: float = 0.05,
    window_size: int = 10,
    n_clusters: int | None = None,
    segment_length: float = 30e6,
    min_burden: float = 0.03,
    normal_markers: Sequence[str] | Mapping[str, Sequence[str]] | None = NORMAL_MARKERS,
    n_iter: int = 10,
    calculate_gene_values: bool = False,
    max_fit_cells: int | None = None,
    connectivity_key: str | None = None,
    n_hops: int = 1,
    force_calls: bool = False,
    bin_size: int = 1_000_000,
    chunksize: int = 5000,
    random_state: int = 0,
    key_added: str = "copy_number",
    inplace: bool = True,
    n_jobs: int | None = None,
) -> None | tuple[pd.DataFrame, np.ndarray, dict, np.ndarray | None]:
    """Infer copy number profiles, clones and aneuploid/diploid labels with a clone-level count model.

    The expected count of a gene in a cell is the cell's size factor times a per-gene baseline (pooled diploid
    cells) times the expression deviation of the cell's type times the copy ratio of the cell's clone, with negative
    binomial noise. The steps are:

    1. keep cells with at least `min_genes` detected genes and genes detected in at least `min_detection` of them
       (on chromosomes named like "chr1"); sum each cell's counts in windows of `window_size` genes;
    2. cluster cells by expression (k-means on principal components of the most variable genes); these clusters
       are the expression types. A copy number HMM on the difference of the pooled profiles of every pair of types
       groups the types that differ in expression only;
    3. take the first baseline from the groups whose pooled profile is flat along the genome, and from groups that
       express `normal_markers` far above the rest (or from `reference_cat`);
    4. the expression deviation of a type is the gene-level part of its pooled profile that the HMM does not
       explain, so it holds the cell type but no copy number signal. A clone's profile is an HMM (8 copy-ratio states,
       transitions by genomic distance, `segment_length`) on the gene-level log ratio of its pooled counts to their
       expectation, corrected for the cell's capture of highly expressed genes and centred on its most common level;
    5. assign cells to clones by EM on the negative binomial likelihood of their window counts. Profiles are
       cross-fitted (windows of one parity are scored with the profile fitted on the other parity), clones without
       differences merge, and clones whose likelihood gain does not pay for their segments are dropped;
    6. repeat: a clearly flatter clone becomes diploid (unless normal markers chose the baseline), aneuploid clones
       without a smooth profile (amplitude below 0.1) merge into the diploid clone, which re-defines the baseline;
    7. learn per-gene dosage weights from the clone contrasts and per-type clone priors from a non-negative fit of
       each type's profile on the clone profiles; the per-cell posterior uses these priors and log likelihoods scaled
       by a temperature calibrated between the two halves of the windows.

    The method is made for UMI counts from droplet (10x) and imaging platforms. On full-length plate data
    (Smart-seq2) its profiles are poor.

    .. warning::
        Without reference cells, the diploid baseline comes from the sample itself. When tumour cells far outnumber
        normal cells, especially at low depth (a few hundred counts per cell), the baseline can land on a tumour group,
        and then the labels are inverted: tumour cells are called diploid and normal cells aneuploid. Pass known normal
        cells with `reference_key` and `reference_cat` in such samples. `adata.uns["copy_number"]["seed"]` shows how
        the baseline was found, and a warning flags the least certain case.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts and genomic positions (`chromosome`, `start` and `end` in `adata.var`).
    layer
        Layer with the raw counts. Default: `adata.X`.
    reference_key
        Column of `adata.obs` with annotations of normal cells. With `reference_cat`, these cells define the baseline.
        Providing reference cells improves the detection of small clones on dense panels.
    reference_cat
        One or more values of `adata.obs[reference_key]` that annotate normal cells.
    exclude_chromosomes
        Chromosomes to leave out.
    exclude_genes
        Genes to leave out. Default: HLA genes and the S and G2/M phase genes of :cite:`Tirosh2016melanoma`.
    min_genes
        Minimum number of detected genes of a cell.
    min_detection
        Minimum fraction of cells in which a gene is detected.
    window_size
        Genes per window of the per-cell likelihood.
    n_clusters
        Number of expression clusters. Default: one per 50 cells, between 8 and 40.
    segment_length
        Expected distance between copy number changes, in bases (sets the HMM transition probabilities).
    min_burden
        Fraction of genes in non-neutral states below which two profiles count as the same copy number.
    normal_markers
        Marker genes of cell types that are rarely tumour cells (a list, or a dict of lists per cell type), matched
        against `adata.var_names`. Expression groups whose share of counts in these genes is far above that of the
        marker-poor groups (and that are flatter than the median group) join the first diploid baseline, next to the
        groups that are flattest along the genome, and keep the baseline from moving to a flatter clone later.
        Marker-rich expression types also keep their own smooth deviation from the baseline in their diploid
        expectation, so that a normal type whose expression differs smoothly along the genome is not taken for
        aneuploid. This is prior knowledge, like the marker step of SCEVAN :cite:`DeFalco2021`; it matters in samples
        with few normal cells. Default: 33 immune, endothelial and stromal markers (`PTPRC`, `CD3E`, `CD14`, `PECAM1`,
        `COL1A1`, ...). `None` uses the genomic flatness alone.
    n_iter
        Maximum number of EM iterations.
    max_fit_cells
        Fit the model (expression types, clones, baseline) on a random sample of this many cells and then assign all
        cells with it in one pass over the counts; reference cells are always in the sample. Faster and leaner for very
        large datasets; results can differ from the default, which fits on all cells. Use tens of thousands of cells: with a few thousand or fewer, the
        diploid baseline and the clones can change from those of the full fit.
    connectivity_key
        Smooth the results over a spatial neighbour graph, the cells x cells graph `adata.obsp[connectivity_key]`
        (e.g. `"spatial_connectivities"` from :func:`squidpy.gr.spatial_neighbors_knn`), as
        :func:`~infercnvpy.tl.smooth_cnv` does for a CNV matrix. Each cell's posterior clone probabilities and llr
        become the mean over the cell and its neighbours (weighted by the graph); labels, clones and profiles follow
        from the smoothed posteriors with the same rules. On imaging data, where single cells hold few counts and
        neighbours usually share their clone, this lowers the errors inside clones and the false calls in normal
        tissue, at the cost of blurring the boundary between clones by about `n_hops` neighbours. About 10 neighbours
        (`squidpy.gr.spatial_neighbors_knn(adata, n_neighs=10)`) kept boundaries best on simulated clones; 100
        blurred them more than no smoothing. Across cell types, smoothing also spreads tumour calls into the normal
        cells next to a tumour. To avoid that, set the graph's entries between cells of different types to 0; this
        suits tumours that follow a cell type, while the full graph suits clones that span types.
        Default `None`: no smoothing.
    n_hops
        Number of smoothing steps over the graph of `connectivity_key`; each step reaches one more edge away.
    force_calls
        Call cells even where the data hold too little copy number information. With fewer than 1,000 genes passing
        the filters (small imaging panels) or a median below 150 counts per cell in those genes, the calls are not
        reliable, so by default copy_number logs a warning and labels the cells `"undetermined"` without fitting the
        model. For such data, use :func:`~infercnvpy.tl.infercnv` with reference cells, smoothed with
        :func:`~infercnvpy.tl.smooth_cnv`.
    calculate_gene_values
        Also store the copy number log ratio of every gene (cells x genes) in
        `adata.layers["gene_values_{key_added}"]`, as :func:`~infercnvpy.tl.infercnv` does. Gene values come from the
        clones' segment means at gene resolution, not from the bins.
    bin_size
        Size of the genomic bins of the output, in bases.
    chunksize
        Number of cells processed at a time.
    random_state
        Seed for subsampling and k-means.
    key_added
        Name of the results in `adata.obs`, `adata.obsm` and `adata.uns`.
    inplace
        Write the results to `adata`. If `False`, return them.
    n_jobs
        Number of threads.

    Returns
    -------
    If `inplace=False`, a tuple `(obs, cna, uns, gene_values)`: a DataFrame with the `adata.obs` columns below, the
    values of `adata.obsm["X_copy_number"]` and `adata.uns["copy_number"]`, and the gene values (`None` unless
    `calculate_gene_values=True`). Otherwise `None`, and sets the following fields
    (named after `key_added`):

    `adata.obs["copy_number"]` : :class:`pandas.Categorical`
        `"aneuploid"` (posterior probability above 0.5), `"diploid"`, `"not.defined"` for cells removed by the
        filters, or `"undetermined"` for every cell when the data hold too little information (see `force_calls`).
    `adata.obs["copy_number_score"]` : :class:`float`
        Posterior probability that the cell is aneuploid.
    `adata.obs["copy_number_llr"]` : :class:`float`
        Log likelihood ratio of the best aneuploid clone against the diploid clone (unscaled, without priors); a
        continuous per-cell score.
    `adata.obs["copy_number_clone"]` : :class:`pandas.Categorical`
        Clone of the cell: `"N0"` for the diploid clone, `"T1"`, `"T2"`, ... for aneuploid clones (the most probable
        aneuploid clone of an aneuploid cell), or the cell's label for cells that are `"not.defined"` or
        `"undetermined"`.
    `adata.obs["copy_number_spillover"]` : :class:`float`
        Fraction of the cell's counts that is best explained by the pooled expression of the aneuploid cells, mixed
        with its own type's diploid expression (0 to 1). High values in normal cell types point to segmentation merges
        or doublets with tumour cells; for cells of marker-rich (normal) types, up to 0.5 of this mixture is part of
        the diploid expectation, so that spillover alone does not make them aneuploid.
    `adata.obsm["X_copy_number"]` : :class:`numpy.ndarray`
        Copy number log ratios (float32), cells x genomic bins: the posterior mixture of the clones' segment means, a
        continuous score. Cells that are `"not.defined"` or `"undetermined"` have 0 (no copy number change), so that
        embeddings, smoothing and heatmaps work on all cells, as with :func:`~infercnvpy.tl.infercnv`.
    `adata.layers["gene_values_copy_number"]` : :class:`numpy.ndarray`
        Only if `calculate_gene_values=True`: log ratio per gene (float32), as `X_copy_number`; `NaN` for genes
        removed by the filters.
    `adata.uns["copy_number"]["chr_pos"]` : :class:`dict`
        Maps each chromosome to the column of `X_copy_number` where it starts.
    `adata.uns["copy_number"]["bins"]` : :class:`pandas.DataFrame`
        `chromosome`, `start` and `end` of each bin.
    `adata.uns["copy_number"]["window_start"]`, `adata.uns["copy_number"]["window_end"]` : :class:`numpy.ndarray`
        Start and end of each bin, under the keys that :func:`~infercnvpy.tl.infercnv` uses for its windows.
    `adata.uns["copy_number"]["clones"]` : :class:`pandas.DataFrame`
        Per clone: `n_cells`, `aneuploid`, `amplitude` (smooth amplitude of its profile) and `burden` (fraction of
        genes in long non-neutral segments).
    `adata.uns["copy_number"]["clone_profiles"]` : :class:`pandas.DataFrame`
        Log ratio of each clone in each bin.
    `adata.uns["copy_number"]["segments"]` : :class:`pandas.DataFrame`
        Segments of the aneuploid clones: `clone`, `chromosome`, `start`, `end`, `n_genes`, `log_ratio` with a 95%
        interval (`ci_low`, `ci_high`), `state_ratio` (copy ratio of the HMM state) and `p_neutral` (mean posterior
        probability of the neutral state).
    `adata.uns["copy_number"]["genes"]` : :class:`pandas.DataFrame`
        The genes used, in genomic order, with their `dosage` weight (1: expression follows copy number fully; the
        final likelihood uses these weights).
    `adata.uns["copy_number"]["baseline"]` : :class:`str`
        How the first diploid cells were found.
    `adata.uns["copy_number"]["seed"]` : :class:`dict`
        The baseline `method`, the `flatness_gap` between the flat and the other groups and the gap that counts as a
        clear split (`clear_gap`). Without a clear split and without marker-rich groups (`method` "flattest group"),
        a warning says that the baseline is uncertain.
    `adata.uns["copy_number"]["temperature"]` : :class:`float`
        Scale applied to the log likelihoods for the posterior.
    `adata.uns["copy_number"]["params"]` : :class:`dict`
        Parameters of this call.

    When the cells are `"undetermined"`, `uns` holds `chr_pos`, `bins`, `window_start`, `window_end`, `params` and
    `undetermined` (the number of genes passing the filters and the median counts per cell in them).
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns):
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and `end` columns in `adata.var`."
        )
    _check_args(
        window_size=window_size, min_genes=min_genes, n_iter=n_iter, n_hops=n_hops, bin_size=bin_size,
        chunksize=chunksize, n_clusters=n_clusters, max_fit_cells=max_fit_cells, min_detection=min_detection,
        segment_length=segment_length,
    )  # fmt: skip
    is_ref = _reference_cells(adata, reference_key, reference_cat)
    G = None if connectivity_key is None else _graph(adata, connectivity_key)
    if isinstance(normal_markers, Mapping):  # lists, so that the parameters can be written to h5ad/zarr
        normal_markers = {k: _as_list(v) for k, v in normal_markers.items()}
    params = _params_for_uns(
        layer=layer, reference_key=reference_key, reference_cat=reference_cat,
        exclude_chromosomes=exclude_chromosomes, exclude_genes=exclude_genes, min_genes=min_genes,
        min_detection=min_detection, window_size=window_size, n_clusters=n_clusters, segment_length=segment_length,
        min_burden=min_burden, normal_markers=normal_markers, n_iter=n_iter, max_fit_cells=max_fit_cells,
        connectivity_key=connectivity_key, n_hops=n_hops, force_calls=force_calls, bin_size=bin_size,
        random_state=random_state,
    )  # fmt: skip
    rng = np.random.default_rng(random_state)
    expr = _as_matrix(adata.X if layer is None else adata.layers[layer], lazy=True)
    if isinstance(expr, np.ndarray):  # the kernels read CSR; convert once, not per block and pass
        expr = sp.csr_matrix(expr)
    excluded = _excluded_genes(adata.var_names, exclude_genes)

    with numba_threads(n_jobs):
        source = _Rows(expr, chunksize)
        start = log.debug("copy_number: %d cells x %d genes", adata.n_obs, adata.n_vars)
        cells, lay, var, candidates = _filter_cells_genes(
            adata, source, excluded, exclude_chromosomes, min_genes, min_detection
        )
        n = len(cells)
        if is_ref is not None and not is_ref[cells].any():
            raise ValueError(f"No cell passing the filters has `{reference_key}` in {reference_cat!r}.")
        fit = np.arange(n)
        if max_fit_cells is not None and n > max_fit_cells:  # a sample, with every reference cell
            fit = np.sort(rng.choice(n, max_fit_cells, replace=False))
            if is_ref is not None:
                fit = np.union1d(fit, np.flatnonzero(is_ref[cells]))
        n_fit = len(fit)
        fit_source, fit_rows = source, cells[fit]
        if n_fit < n:  # the sample in memory, so that the passes of the fit do not read every block
            fit_source = _Rows(sp.vstack([_csr(X)[r] for X, r in source.rows_of(cells[fit])]).tocsr(), chunksize)
            fit_rows = np.arange(n_fit)
        data = _Data(fit_source, fit_rows, lay, window_size)
        depth = float(np.median(data.size))
        if not force_calls and (lay.n_genes < _MIN_PANEL_GENES or depth < _MIN_MEDIAN_COUNTS):
            log.warning(
                "copy_number: %d genes pass the filters, with a median of %d counts per cell; reliable calls need at "
                "least %d genes and %d counts per cell, so %d cells are labelled %r. For small panels, use "
                "`tl.infercnv` with reference cells, smoothed with `tl.smooth_cnv`; `force_calls=True` calls the "
                "cells anyway.",
                lay.n_genes, depth, _MIN_PANEL_GENES, _MIN_MEDIAN_COUNTS, n, UNDETERMINED,
            )  # fmt: skip
            bins, chr_pos, _ = lay.bins(bin_size)
            uns = {**_bin_uns(bins, chr_pos), "undetermined": {"n_genes": lay.n_genes, "median_counts": depth}}
            uns["params"] = params
            gene_values = None
            if calculate_gene_values:
                gene_values = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)
                gene_values[:, lay.order] = 0.0
            return _outputs(
                adata, cells, key_added, inplace, UNDETERMINED, UNDETERMINED, [UNDETERMINED],
                np.zeros((adata.n_obs, len(bins)), dtype=np.float32), uns, gene_values,
            )  # fmt: skip
        hmm = _HMM(lay, segment_length)

        # expression clusters and their pooled profiles
        n_clusters = n_clusters or int(np.clip(n_fit // 50, 8, 40))
        pcs = data.expression_pcs(rng)
        clusters = _kmeans(pcs, min(n_clusters, n_fit), rng)
        Yc, Sc = data.pooled(clusters)
        gene_var = _gene_variability(Yc, Sc, lay)
        burden = _pairwise_burden(hmm, Yc, Sc, gene_var, lay)
        groups = _groups(burden, min_burden)

        gcell = groups[clusters]
        n_groups = int(groups.max()) + 1
        if is_ref is not None:
            normal = is_ref[cells[fit]]
            baseline_method, flat_gap = "reference", np.nan
        else:
            log.info(
                "copy_number: no reference cells, so the diploid baseline comes from the sample itself. Where known "
                "normal cells are annotated, give them as `reference_key` / `reference_cat` for a reliable baseline."
            )
            is_marker = np.asarray(adata.var_names.isin(_marker_list(normal_markers)))
            marker, total = _marker_counts(data, clusters, Yc, var, candidates, is_marker)
            normal, baseline_method, flat_gap = _seed(Yc, marker, total, groups, gcell, lay)

        em = _EM(data, hmm, gene_var, min_burden, clusters, Yc, Sc)
        if is_ref is None and normal_markers is not None:
            em.normal_type = _marker_types(Yc, marker / total, lay)
            log.debug("copy_number: marker-rich expression types %s", np.flatnonzero(em.normal_type))
        frac = np.bincount(gcell, weights=normal, minlength=n_groups) / np.bincount(gcell, minlength=n_groups)
        lut = np.where(frac > 0.5, 0, 1 + np.arange(n_groups))
        assign = np.where(normal, 0, lut[gcell])
        fixed = is_ref is not None
        reseeded = "markers" in baseline_method  # marker-rich groups outweigh flatness: no restart
        # passes: after each, a clearly flatter clone becomes diploid and the passes restart from it, or clones without
        # a smooth profile merge into the diploid clone; a last single-iteration pass follows the last change
        for pass_ in range(_N_PASSES + 1):
            em.set_baseline(normal)
            assign, resp, llr = em.run(assign, n_iter if pass_ < _N_PASSES else 1)
            if pass_ == _N_PASSES:
                break
            k = int(assign.max()) + 1
            target = np.arange(k)
            if not fixed and not reseeded:
                Yk = data.pooled(assign, k)[0]
                sizes = np.bincount(assign, minlength=k)
                flat = np.array([_flatness(y, lay) for y in Yk])
                small = sizes < min(_MIN_RESEED, int(sizes.max()))
                small[0] = False  # the diploid clone itself may be a rare normal population
                flat[small] = np.inf
                seed = int(np.argmin(flat))
                log.debug("copy_number: pass %d, clone flatness %s", pass_, np.round(flat, 3))
                if flat[seed] < flat[0] - _RESEED_GAP:
                    reseeded, baseline_method = True, "flattest clone"
                    target[0], target[seed] = seed, 0
                    assign = target[assign]
                    normal = assign == 0
                    continue
            amp = np.array([_smooth_amplitude(em.final[3][i], lay) for i in range(k)])
            log.debug("copy_number: pass %d, clone amplitudes %s", pass_, np.round(amp, 3))
            target[amp < _AMP_TUMOUR] = 0
            if (target == np.arange(k)).all():
                break
            assign = _compact(target[assign])
            if not fixed:
                normal = assign == 0

        if n_fit < n:  # assign every cell with the fitted model, in one pass over the counts
            resp, llr = em.assign_all(source, cells, data.project(source, cells), pcs, clusters)
        if G is not None:
            resp, llr = _smooth_posteriors(G, cells, n_hops, resp, llr)
        k = resp.shape[1]
        p_an = resp[:, 1:].sum(axis=1)
        clone = np.where(p_an > 0.5, 1 + resp[:, 1:].argmax(axis=1), 0) if k > 1 else np.zeros(n, dtype=np.int64)
        names = np.array(["N0", *(f"T{i}" for i in range(1, k))], dtype=object)
        mu_f, pn_f, st_f, lr_f, v_f = em.final
        seg_values, segments = _segments(lr_f, v_f, st_f, pn_f, lay, names, hmm.means)
        segments = segments[segments["clone"] != "N0"].reset_index(drop=True)
        seg_values[0] = 0.0
        bins, chr_pos, M = lay.bins(bin_size)
        prof = np.asarray(M @ seg_values.T).T
        resp32 = resp.astype(np.float32)
        cna = np.zeros((adata.n_obs, len(bins)), dtype=np.float32)
        K.mix_rows(resp32, prof.astype(np.float32), cells, np.arange(prof.shape[1]), cna)
        gene_values = None
        if calculate_gene_values:
            gene_values = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)
            gene_values[:, lay.order] = 0.0
            K.mix_rows(resp32, seg_values.astype(np.float32), cells, lay.order, gene_values)
        clones = pd.DataFrame(
            {
                "n_cells": np.bincount(clone, minlength=k),
                "aneuploid": names != "N0",
                "amplitude": [_smooth_amplitude(x, lay) for x in lr_f],
                "burden": [0.0] + [_burden(mu_f[i], pn_f[i], lay) for i in range(1, k)],
            },
            index=names,
        )
        log.debug("copy_number: clones\n%s", clones.round(3).to_string())

    uns = {
        **_bin_uns(bins, chr_pos),
        "clones": clones,
        "clone_profiles": pd.DataFrame(prof, index=names, columns=bins.index),
        "segments": segments,
        "genes": pd.DataFrame(
            {"chromosome": lay.chrom, "start": lay.start, "end": lay.end}
            | ({"dosage": em.dosage} if _DOSAGE == "final" else {}),
            index=np.asarray(adata.var_names)[lay.order],
        ),
        "baseline": baseline_method,
        "seed": {"method": baseline_method, "flatness_gap": float(flat_gap), "clear_gap": _FLAT_GAP},
        "temperature": em.tau,
        "params": params,
    }
    calls = np.where(p_an > 0.5, LABELS[0], LABELS[1])
    n_an = int((p_an > 0.5).sum())
    log_result(
        "copy_number",
        "%d aneuploid and %d diploid cells in %d clones; %d of %d cells not defined",
        *(n_an, n - n_an, k, adata.n_obs - n, adata.n_obs),
        dropped=adata.n_obs - n,
        total=adata.n_obs,
        remedy="lower `min_genes`",
        time=start,
    )
    if baseline_method == "flattest group":
        log.warning(
            "copy_number: the diploid baseline is uncertain. No group of cells is clearly flatter along the genome "
            "than the others (flatness gap %.3f, a clear split needs %.2f) and none is rich in normal markers, so the "
            "single flattest group was taken; in samples with few normal cells this can be a tumour group, which "
            "inverts the calls. Give known normal cells as `reference_key` / `reference_cat`.",
            flat_gap, _FLAT_GAP,
        )  # fmt: skip
    if n_an == 0:
        log.warning(
            "copy_number: no cell is called aneuploid. The diploid baseline comes from the sample itself, so a sample "
            "without normal cells (tumour only, a cell line) is called diploid; give reference cells "
            "(`reference_key`, `reference_cat`) if that is the case."
        )
    return _outputs(
        adata, cells, key_added, inplace, calls, names[clone], names, cna, uns, gene_values,
        score=p_an, llr=llr, spill=em.spill,
    )  # fmt: skip


def _outputs(adata, cells, key_added, inplace, labels, clones, clone_names, cna, uns, gene_values, **values):
    """Write the outputs to `adata` (or return them); cells removed by the filters are `"not.defined"`.

    `values` are the float columns (`score`, `llr`, `spill`) of the cells that pass the filters; missing ones are NaN.
    """

    def per_cell(x, fill):
        out = np.full(adata.n_obs, fill, dtype=object if isinstance(fill, str) else np.float64)
        out[cells] = x
        return out

    clone_cats = [*clone_names, *([LABELS[2]] if len(cells) < adata.n_obs else [])]
    obs = pd.DataFrame(
        {
            key_added: pd.Categorical(per_cell(labels, LABELS[2]), categories=LABELS),
            f"{key_added}_score": per_cell(values.get("score", np.nan), np.nan),
            f"{key_added}_llr": per_cell(values.get("llr", np.nan), np.nan),
            f"{key_added}_clone": pd.Categorical(per_cell(clones, LABELS[2]), categories=clone_cats),
            f"{key_added}_spillover": per_cell(values.get("spill", np.nan), np.nan),
        },
        index=adata.obs_names,
    )
    if not inplace:
        return obs, cna, uns, gene_values
    for c in obs.columns:
        adata.obs[c] = obs[c]
    adata.obsm[f"X_{key_added}"] = cna
    if gene_values is not None:
        adata.layers[f"gene_values_{key_added}"] = gene_values
    adata.uns[key_added] = uns
    return None


def _bin_uns(bins: pd.DataFrame, chr_pos: dict) -> dict:
    return {
        "chr_pos": chr_pos,
        "bins": bins,
        "window_start": bins["start"].to_numpy(),  # as written by tl.infercnv, for region selection
        "window_end": bins["end"].to_numpy(),
    }


def _check_args(*, n_clusters, max_fit_cells, min_detection, segment_length, **at_least_one) -> None:
    for name, value in at_least_one.items():
        if value < (0 if name == "min_genes" else 1):
            raise ValueError(f"`{name}` must be at least {0 if name == 'min_genes' else 1}, got {value}.")
    if n_clusters is not None and n_clusters < 1:
        raise ValueError(f"`n_clusters` must be at least 1, got {n_clusters}.")
    if max_fit_cells is not None and max_fit_cells < 2:
        raise ValueError(f"`max_fit_cells` must be at least 2, got {max_fit_cells}.")
    if not 0 <= min_detection <= 1:
        raise ValueError(f"`min_detection` must be between 0 and 1, got {min_detection}.")
    if segment_length <= 0:
        raise ValueError(f"`segment_length` must be positive, got {segment_length}.")


def _reference_cells(adata, reference_key, reference_cat) -> np.ndarray | None:
    """Which cells are reference (normal) cells, or `None` without a reference; checks the arguments."""
    if (reference_key is None) != (reference_cat is None):
        raise ValueError("Give both `reference_key` and `reference_cat` for reference cells, or neither.")
    if reference_key is None:
        return None
    if reference_key not in adata.obs.columns:
        raise KeyError(f"`{reference_key}` is not a column of `adata.obs`.")
    values = np.asarray(adata.obs[reference_key])
    missing = [c for c in _categories(reference_cat) if c not in set(values)]
    if missing:
        raise ValueError(f"No cell has `{reference_key}` in {missing!r}.")
    return np.isin(values, _categories(reference_cat))


def _smooth_posteriors(G, cells, n_hops, resp, llr):
    """Posteriors and llr averaged over each cell and its neighbours in `G` (as in `tl.smooth_cnv`).

    The graph is restricted to the cells that pass the filters; cells without neighbours keep their values.
    """
    G = G[cells][:, cells].tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    out = _smooth(G, np.column_stack([resp, llr]), None, n_hops, dtype=np.float64)
    return out[:, :-1], out[:, -1]


def _row_blocks(source: _Rows, rows: np.ndarray):
    """Yield `(i0, i1, X, rws)` for each block of the matrix that holds any of `rows` (sorted).

    `rows[i0:i1]` are the rows `rws` of the CSR block `X`. Blocks are read whole (in memory: no copy), the kernels
    pick the rows.
    """
    i0 = 0
    for X, rws in source.rows_of(rows):
        if len(rws):
            yield i0, i0 + len(rws), _csr(X), rws.astype(np.int64)
            i0 += len(rws)


def _filter_cells_genes(adata, source, excluded, exclude_chromosomes, min_genes, min_detection):
    """Cells and genes that pass the filters, in one pass over the counts.

    Cells have at least `min_genes` detected genes; genes are detected in at least `min_detection` of these cells.
    Genes count if they are placed (chromosome and start) on a chromosome named like "chr1" and not excluded; these
    candidates and the gene positions (a missing end is the start, as in `tl.infercnv`) are returned too.
    """
    var = pd.DataFrame({c: np.asarray(adata.var[c]) for c in ("chromosome", "start", "end")}, index=adata.var_names)
    var["end"] = var["end"].fillna(var["start"])
    chrom = var["chromosome"].astype(object)
    named = chrom.notna() & var["start"].notna() & chrom.astype(str).str.startswith("chr") & (chrom != "chrM")
    if not named.any():
        raise ValueError(
            'None of the chromosome names in `adata.var["chromosome"]` start with "chr". '
            'infercnvpy only uses chromosomes named like "chr1".'
        )
    if exclude_chromosomes is not None:
        named &= ~chrom.isin(_as_list(exclude_chromosomes))
    candidates = named.to_numpy() & ~excluded
    det = np.zeros((min(numba.get_num_threads(), 16), adata.n_vars), dtype=np.int64)
    ndet = []
    for X, rws in source.rows_of(None):  # all rows: each block's rows are contiguous
        if len(rws):
            X = _csr(X)
            nd = np.empty(len(rws), dtype=np.int64)
            K.row_stats(X.data, X.indices, X.indptr, int(rws[0]), min_genes, nd, det)
            ndet.append(nd)
    cells = np.flatnonzero(np.concatenate(ndet) >= min_genes)
    if len(cells) < 2:
        raise ValueError(f"Fewer than 2 cells have at least {min_genes} detected genes.")
    used = (det.sum(axis=0) / len(cells) >= min_detection) & candidates
    if not used.any():
        raise ValueError("No gene passes the filters.")
    lay = _Layout(var, used)
    log.debug("copy_number: %d of %d cells and %d genes pass the filters", len(cells), adata.n_obs, lay.n_genes)
    return cells, lay, var, candidates


def _marker_counts(data, clusters, Yc, var, candidates, is_marker) -> tuple[np.ndarray, np.ndarray]:
    """Counts of each cluster in the marker genes, and in the model's genes plus the markers.

    Every marker counts, also those that the detection filter removed from the model's genes, so that a rare normal
    population keeps its markers; the genomic flatness of the seed uses the model's genes only.
    """
    markers = is_marker & candidates
    total = Yc.sum(axis=1)
    if not markers.any():
        return np.zeros(len(Yc)), total
    M = data.pooled(clusters, len(Yc), _Layout(var, markers))[0]
    outside = data.lay.col_map[np.flatnonzero(markers)] < 0
    return M.sum(axis=1), total + M[:, outside].sum(axis=1)


def _as_list(x) -> list:
    """A string as a list of one, any other collection as a list."""
    return [x] if isinstance(x, str) else list(x)


class _Data:
    """Access to the counts of the cells passing the filters: windows of genes per cell, pooled gene counts."""

    def __init__(self, source: _Rows, cells: np.ndarray, lay, window_size: int):
        self.source, self.cells, self.lay = source, cells, lay
        bounds = []
        for g0, g in zip(lay.g0s, lay.gs, strict=True):
            bounds.extend(g0 + np.arange(max(1, g // window_size)) * window_size)
        self.wbounds = np.array([*bounds, lay.n_genes], dtype=np.int64)
        self.gene_win = np.repeat(np.arange(len(bounds)), np.diff(self.wbounds))
        self.win_parity = np.arange(len(bounds)) % 2
        n_win = len(self.wbounds) - 1
        Yw = []
        for i0, i1, X, rws in _row_blocks(source, cells):
            out = np.empty((i1 - i0, n_win), dtype=np.float32)
            K.window_counts(
                X.data, X.indices, X.indptr, rws, lay.col_map, self.gene_win, out, n_kernel_chunks(len(out))
            )
            Yw.append(out)
        self.Yw = np.concatenate(Yw)
        self.size = self.Yw.sum(axis=1, dtype=np.float64)

    def blocks(self, rows: np.ndarray | None = None):
        """Yield `(i0, i1, X)`: positions `i0:i1` in `rows` (default: all cells) and their counts (CSR, genomic order)."""
        rows = self.cells if rows is None else rows
        i0 = 0
        for X, rws in self.source.rows_of(rows):
            yield i0, i0 + len(rws), _csr(X)[rws][:, self.lay.order].astype(np.float64)
            i0 += len(rws)

    def pooled(self, labels: np.ndarray, k: int | None = None, lay=None):
        """Pooled gene counts (k x genes of `lay`, default: the model's) and pooled size factors of each label."""
        lay = self.lay if lay is None else lay
        k = int(labels.max()) + 1 if k is None else k
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        slabs = np.zeros((min(numba.get_num_threads(), 16), k, lay.n_genes))
        for i0, i1, X, rws in _row_blocks(self.source, self.cells):
            K.pooled_counts(X.data, X.indices, X.indptr, rws, lay.col_map, labels[i0:i1], slabs)
        return slabs.sum(axis=0), np.bincount(labels, weights=self.size, minlength=k)

    def expression_pcs(self, rng) -> np.ndarray:
        """First principal components of log-normalised expression of the most variable genes."""
        sub = _subsample(np.arange(len(self.cells)), _N_SUB, rng)
        Xs = sp.vstack([Xb for _, _, Xb in self.blocks(self.cells[sub])]).tocsr()
        Xs = sp.diags(1e4 / np.maximum(self.size[sub], 1)) @ Xs
        Xs.data = np.log1p(Xs.data)
        mean = np.asarray(Xs.mean(axis=0)).ravel()
        var = np.asarray(Xs.multiply(Xs).mean(axis=0)).ravel() - mean**2
        hvg = np.sort(np.argsort(var)[::-1][: min(_N_HVG, len(var))])
        D = Xs[:, hvg].toarray() - mean[hvg]
        n_pcs = max(1, min(_N_PCS, D.shape[0] - 1, D.shape[1]))
        proj = _top_right_singular_vectors(D, n_pcs)
        self.hvg_of_gene = np.full(self.lay.n_genes, -1, dtype=np.int64)
        self.hvg_of_gene[hvg] = np.arange(len(hvg))
        self.loadings = np.ascontiguousarray(proj)
        self.pc_offset = mean[hvg] @ proj
        return self.project(self.source, self.cells)

    def project(self, source, rows) -> np.ndarray:
        """Principal component coordinates of the cells `rows` (any cells, not only those of this object)."""
        out = []
        for i0, i1, X, rws in _row_blocks(source, rows):
            o = np.empty((i1 - i0, self.loadings.shape[1]))
            K.project_rows(
                X.data, X.indices, X.indptr, rws, self.lay.col_map, self.hvg_of_gene, self.loadings, o,
                n_kernel_chunks(i1 - i0),
            )  # fmt: skip
            out.append(o - self.pc_offset)
        return np.concatenate(out)


def _top_right_singular_vectors(D: np.ndarray, k: int) -> np.ndarray:
    """Top `k` right singular vectors of `D` (columns).

    An exact SVD for up to `_EXACT_SVD` rows, else the eigenvectors of `D.T @ D` (exact too, and fast for many rows
    and few columns).
    """
    if D.shape[0] <= _EXACT_SVD:
        return np.linalg.svd(D, full_matrices=False)[2][:k].T
    w, vec = np.linalg.eigh(D.T @ D)
    return vec[:, np.argsort(w)[::-1][:k]]


class _HMM:
    """Copy number HMM over the genes in genomic order, with transitions by genomic distance."""

    def __init__(self, lay, segment_length: float):
        centre = (lay.start + lay.end) / 2.0
        d = np.diff(centre, prepend=centre[0])
        stay = np.exp(-np.maximum(d, 0) / segment_length)
        stay[lay.g0s] = 0.0
        self.stay = stay
        self.means = np.log(np.array(STATE_RATIOS))
        prior = np.full(len(STATE_RATIOS), (1 - _P_NEUTRAL) / (len(STATE_RATIOS) - 1))
        prior[np.argmin(np.abs(self.means))] = _P_NEUTRAL
        self.means[np.argmin(np.abs(self.means))] = 0.0
        self.log_prior = np.log(prior)

    def __call__(self, y: np.ndarray, v: np.ndarray):
        y = np.ascontiguousarray(np.atleast_2d(y), dtype=np.float64)
        v = np.ascontiguousarray(np.atleast_2d(v), dtype=np.float64)
        P, G = y.shape
        pm, pn, st = np.empty((P, G)), np.empty((P, G)), np.empty((P, G), dtype=np.int64)
        K.hmm_posteriors(y, v, self.stay, self.means, self.log_prior, _NU, pm, pn, st)
        return pm, pn, st


def _log_ratio(Y, E, Yref):
    """Gene-level log ratio of pooled counts `Y` to their expectation `E`, and its Poisson variance.

    The variance includes that of the baseline counts `Yref`.
    """
    return np.log((Y + 0.5) / (E + 0.5)), 1.0 / (Y + 0.5) + 1.0 / (Yref + 0.5)


def _remove_level(lr: np.ndarray, v: np.ndarray, b: np.ndarray, lay) -> np.ndarray:
    """`lr` (profiles x genes) minus its weighted regression on gene covariates that are not copy number.

    The covariates are the gene's baseline log expression and log length, and their squares. Cells and clusters
    differ in how their counts spread over highly and lowly expressed genes (capture and saturation), and nuclei
    against whole cells in how they spread over long and short genes (intronic reads); both show up along gene-dense
    chromosomes such as chr19 and are not copy number.
    """
    z = np.log(b)
    z = (z - z.mean()) / z.std()
    cols = [np.ones_like(z), z, z**2 - 1]
    ln = np.log(np.maximum(lay.end - lay.start, 1).astype(np.float64))
    if ln.std() > 0:
        ln = (ln - ln.mean()) / ln.std()
        cols += [ln, ln**2 - 1]
    Z = np.stack(cols, axis=1)
    out = np.empty_like(lr)
    for i in range(len(lr)):
        w = 1.0 / v[i]
        r = np.clip(lr[i], -3, 3)
        beta = np.linalg.lstsq(Z.T @ (Z * w[:, None]), Z.T @ (w * r), rcond=None)[0]
        out[i] = lr[i] - Z[:, 1:] @ beta[1:]
    return out


def _expression_deviation(resid: np.ndarray, pois: np.ndarray) -> np.ndarray:
    """Gene-level expression deviation of each clone: what the smooth copy number profile leaves.

    It is shrunk towards 0 by its Poisson noise, and carries the clone's cell type, without any copy number signal.
    """
    out = np.empty_like(resid)
    for i in range(len(resid)):
        r = np.clip(resid[i], -3, 3)
        tau2 = max(float(np.median(r**2)) / 0.455 - float(np.median(pois[i])), 1e-3)
        out[i] = r * tau2 / (tau2 + pois[i])
    return out


def _robust_var(x: np.ndarray) -> float:
    d = np.diff(x)
    return float((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2 / 2)


def _scale(lr, pois, gene_var):
    """Total variance per gene: expression variance (`gene_var` scaled to the profile) plus Poisson variance."""
    resid = lr / np.sqrt(gene_var + pois)
    s2 = max(_robust_var(resid), 1e-4)
    return s2 * (gene_var + pois)


def _gene_variability(Yc, Sc, lay) -> np.ndarray:
    """Relative expression variability of each gene across the clusters, after removing smooth trends."""
    L = np.log((Yc + 0.5) / Sc[:, None])
    L -= np.average(L, axis=0, weights=Sc)
    R = np.empty_like(L)
    for g0, g in zip(lay.g0s, lay.gs, strict=True):
        R[:, g0 : g0 + g] = L[:, g0 : g0 + g] - median_filter(L[:, g0 : g0 + g], size=(1, 51), mode="nearest")
    pois = (1.0 / (Yc + 0.5)).mean(axis=0)
    v = np.maximum(np.average(R**2, axis=0, weights=Sc) - pois, 0)
    v = v + np.median(pois)
    return np.maximum(v / np.median(v), 0.1)


def _pairwise_burden(hmm, Yc, Sc, gene_var, lay) -> np.ndarray:
    """Fraction of genes in non-neutral HMM states of the difference of each pair of cluster profiles."""
    k = len(Yc)
    L = np.log((Yc + 0.5) / Sc[:, None])
    P = 1.0 / (Yc + 0.5)
    bmean = (Yc.sum(0) + 0.5) / (Yc.sum() + 0.5 * Yc.shape[1])
    burden = np.zeros((k, k))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for s in range(0, len(pairs), 200):
        batch = pairs[s : s + 200]
        y = np.array([L[i] - L[j] for i, j in batch])
        y -= np.median(y, axis=1, keepdims=True)
        v = np.array([_scale(y[t], P[i] + P[j], gene_var) for t, (i, j) in enumerate(batch)])
        y = _remove_level(y, v, bmean, lay)
        pm, pn, _ = hmm(y, v)
        for t, (i, j) in enumerate(batch):
            burden[i, j] = burden[j, i] = _burden(pm[t], pn[t], lay)
    return burden


def _marker_list(markers) -> list[str]:
    if markers is None:
        return []
    if isinstance(markers, Mapping):
        return [g for genes in markers.values() for g in _as_list(genes)]
    return _as_list(markers)


def _seed(Yc, marker, total, groups, gcell, lay) -> tuple[np.ndarray, str, float]:
    """Cells of the first baseline.

    These are the groups in the flat cluster of a 2-means split of the group flatness (if the
    split is clear, else the flattest group), plus groups that express immune, endothelial or stromal markers far
    above the marker-poor groups and are flatter than the median group.
    """
    n_groups = int(groups.max()) + 1
    sizes = np.bincount(gcell, minlength=n_groups)
    Yg = np.array([Yc[groups == g].sum(0) for g in range(n_groups)])
    flat = np.array([_flatness(y, lay) for y in Yg])
    ok = sizes >= min(_MIN_SEED, int(sizes.max()))
    seed = np.zeros(n_groups, dtype=bool)
    seed[np.flatnonzero(ok)[np.argmin(flat[ok])]] = True
    method = "flattest group"
    lo, hi = _two_means(flat[ok], sizes[ok])
    if hi - lo > _FLAT_GAP:
        seed |= ok & (flat < (lo + hi) / 2)
        method = "flat groups"
    mfrac = np.bincount(groups, weights=marker) / np.bincount(groups, weights=total)
    # lower quartile, not median: in samples where normal cells are the majority the median group is itself marker-rich
    low = np.quantile(mfrac[ok], 0.25)
    marked = ok & (mfrac > np.maximum(_MARKER_FOLD * low, _MARKER_MIN)) & (flat < np.median(flat[ok]))
    if marked.any():
        seed |= marked
        method += " + markers"
    log.debug("copy_number: group flatness %s, marker fraction (per mille) %s, sizes %s, seed %s",
              np.round(flat, 3), np.round(1000 * mfrac, 2), sizes, np.flatnonzero(seed))  # fmt: skip
    return seed[gcell], method, hi - lo


def _marker_types(Yc, mfrac, lay) -> np.ndarray:
    """Expression types that are marker-rich (same rule as the marker-rich seed groups) and flatter than the median."""
    flat = np.array([_flatness(y, lay) for y in Yc])
    low = np.quantile(mfrac, 0.25)
    return (mfrac > max(_MARKER_FOLD * low, _MARKER_MIN)) & (flat < np.median(flat))


def _two_means(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Means of the two clusters of the weighted 1-D 2-means split of `x` (the best threshold between sorted values)."""
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    o = np.argsort(x)
    x, w = x[o], w[o].astype(np.float64)
    best, out = np.inf, (float(x[0]), float(x[-1]))
    for i in range(1, len(x)):
        m1, m2 = np.average(x[:i], weights=w[:i]), np.average(x[i:], weights=w[i:])
        cost = (w[:i] * (x[:i] - m1) ** 2).sum() + (w[i:] * (x[i:] - m2) ** 2).sum()
        if cost < best:
            best, out = cost, (float(m1), float(m2))
    return out


def _burden(pm: np.ndarray, pn: np.ndarray, lay) -> float:
    """Fraction of genes in long non-neutral runs of a profile.

    A run has at least `_MIN_SEG_GENES` genes in a non-neutral state with a posterior mean log ratio of at least
    `_MIN_EFFECT`. Short runs are mostly clustered gene families of a cell type. On small panels
    (a typical chromosome with fewer than `4 _MIN_SEG_GENES` genes) the shortest run is a quarter of its genes (at
    least 4).
    """
    g = int(np.median(lay.gs))
    min_genes = _MIN_SEG_GENES if g >= 4 * _MIN_SEG_GENES else max(4, g // 4)
    m = np.concatenate([[False], (pn < 0.5) & (np.abs(pm) >= _MIN_EFFECT), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    lengths = edges[1::2] - edges[::2]
    return float(lengths[lengths >= min_genes].sum() / len(pm))


def _groups(burden: np.ndarray, cut: float) -> np.ndarray:
    """Groups of clusters (complete linkage on the burden) whose pairwise burdens are all below `cut`."""
    if len(burden) < 2:
        return np.zeros(len(burden), dtype=np.int64)
    Z = hierarchy.linkage(squareform(burden, checks=False), method="complete")
    return hierarchy.fcluster(Z, cut, criterion="distance").astype(np.int64) - 1


def _flatness(Y: np.ndarray, lay) -> float:
    """Lag-1 autocorrelation along the genome of the window medians of a pooled log profile (not centred per gene)."""
    lr = np.log(Y + 0.5)
    size = _block_size(lay)
    vals, chrom = [], []
    for c, (g0, g) in enumerate(zip(lay.g0s, lay.gs, strict=True)):
        nw = max(1, g // size)
        for w in range(nw):
            e = g0 + g if w == nw - 1 else g0 + (w + 1) * size
            vals.append(np.median(lr[g0 + w * size : e]))
            chrom.append(c)
    x = np.array(vals) - np.mean(vals)
    same = np.diff(chrom) == 0
    return float((x[1:] * x[:-1])[same].sum() / max((x**2).sum(), 1e-12))


def _segments(lr: np.ndarray, v: np.ndarray, st: np.ndarray, pn: np.ndarray, lay, names, means):
    """Segments of each profile and the gene values they give.

    A segment is a run of genes with the same most probable HMM state on one chromosome. Its value is the
    precision-weighted mean of the (clipped) gene log ratios, with a 95% interval from the summed precisions (the
    variances are scaled to the robust spread of the profile, see `_scale`). Returns the gene values (profiles x
    genes) and a table with one row per segment.
    """
    out = np.empty_like(lr)
    starts = np.zeros(lr.shape[1], dtype=bool)
    starts[lay.g0s] = True
    frames = []
    for i in range(len(lr)):
        b = starts.copy()
        b[1:] |= st[i, 1:] != st[i, :-1]
        seg = np.cumsum(b) - 1
        w = 1.0 / v[i]
        den = np.bincount(seg, weights=w)
        mean = np.bincount(seg, weights=w * np.clip(lr[i], -2, 2)) / den
        out[i] = mean[seg]
        first = np.flatnonzero(b)
        last = np.r_[first[1:], len(seg)] - 1
        half = 1.96 / np.sqrt(den)
        frames.append(
            pd.DataFrame(
                {
                    "clone": names[i],
                    "chromosome": lay.chrom[first],
                    "start": lay.start[first],
                    "end": lay.end[last],
                    "n_genes": last - first + 1,
                    "log_ratio": mean,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "state_ratio": np.exp(means[st[i, first]]),
                    "p_neutral": np.bincount(seg, weights=pn[i]) / np.bincount(seg),
                }
            )
        )
    return out, pd.concat(frames, ignore_index=True)


class _EM:
    """Copy number clones of cells with known expression types (steps 4 to 7 of `copy_number`)."""

    def __init__(self, data: _Data, hmm: _HMM, gene_var, min_burden, types, Yt, St):
        self.data, self.hmm, self.gene_var, self.min_burden = data, hmm, gene_var, min_burden
        self.types, self.Yt, self.St = np.ascontiguousarray(types, dtype=np.int64), Yt, St
        self.n_types = len(St)
        self.normal_type = np.zeros(len(St), dtype=bool)  # types whose diploid expectation keeps their smooth part
        self.dosage = np.ones(Yt.shape[1])
        self.neutral = int(np.argmin(np.abs(hmm.means)))
        par = data.win_parity[data.gene_win]
        self.masks = (par == 0, par == 1)

    def set_baseline(self, normal):
        """Baseline from the cells `normal`, and the expression deviations of the types relative to it."""
        self.Yr = self.data.pooled(normal.astype(np.int64), 2)[0][1]
        self.b = (self.Yr + 0.5) / (self.Yr + 0.5).sum()
        lr, pois = _log_ratio(self.Yt, self.St[:, None] * self.b[None, :], self.Yr[None, :])
        v = np.array([_scale(lr[i], pois[i], self.gene_var) for i in range(len(lr))])
        lr = _remove_level(lr, v, self.b, self.data.lay)
        pm = self.hmm(lr, v)[0]
        self.e = _expression_deviation(lr - pm, pois)
        self.type_smooth = pm
        lay = self.data.lay
        self.type_blocks = np.array([_block_medians(x - _mode(x, lay), lay)[0] for x in lr])

    def type_priors(self, mu):
        """Prior clone probabilities of the cells of each type (types x clones).

        The pooled profile of a type, in blocks of 25 genes, is fitted as a non-negative mixture of the aneuploid clone
        profiles (weights summing to at most 1, the rest is diploid). A type of normal cells has weights near 0 even
        when its expression differs from the baseline smoothly, a tumour type near 1, and a mixed type its tumour
        fraction. The fit uses the pooled profiles, not the current assignments, so it does not reinforce them.
        """
        k = len(mu)
        pri = np.zeros((self.n_types, k))
        pri[:, 0] = 1.0
        if k > 1:
            lay = self.data.lay
            Mb = np.array([_block_medians(m, lay)[0] for m in mu[1:]]).T
            for t in range(self.n_types):
                w = nnls(Mb, self.type_blocks[t])[0]
                w = w / max(1.0, w.sum())
                pri[t, 1:] = w
                pri[t, 0] = 1.0 - w.sum()
        pri = np.maximum(pri, _MIN_PRIOR)
        return pri / pri.sum(axis=1, keepdims=True)

    def profiles(self, assign, k):
        """Gene-level log ratios of the clones against their expectation (with the deviations of their cells' types)."""
        data = self.data
        Y, _ = data.pooled(assign, k)
        Skt = np.zeros((k, self.n_types))
        np.add.at(Skt, (assign, self.types), data.size)
        E = self.b[None, :] * (Skt @ np.exp(self.e))
        lr, pois = _log_ratio(Y, E, self.Yr[None, :])
        v = np.array([_scale(lr[i], pois[i], self.gene_var) for i in range(k)])
        lr = _remove_level(lr, v, self.b, self.data.lay)
        return lr - np.array([_mode(x, self.data.lay) for x in lr])[:, None], v

    def fit(self, lr, v, cross: bool):
        """HMM posterior mean, P(neutral) and states per clone.

        With `cross`, each gene's value comes from the fit on the genes of the other window parity.
        """
        if not cross:
            mu, pn, st = self.hmm(lr, v)
        else:
            mu, pn, st = np.empty_like(lr), np.empty_like(lr), np.empty(lr.shape, dtype=np.int64)
            for m in self.masks:
                f_mu, f_pn, f_st = self.hmm(lr, np.where(m[None, :], v, 1e6))
                mu[:, ~m], pn[:, ~m], st[:, ~m] = f_mu[:, ~m], f_pn[:, ~m], f_st[:, ~m]
        mu[0], pn[0], st[0] = 0.0, 1.0, self.neutral
        return mu, pn, st

    def loglik(self, mu, assign, windows=None, report: bool = False):
        """Negative binomial log likelihood of each cell under each clone (cells x clones), on `windows` (mask).

        With `report`, the spillover fraction is estimated for the cells of every type (for the output).
        """
        data = self.data
        wb = data.wbounds[:-1]
        share = np.empty((self.n_types, len(mu), len(wb)))
        for t in range(self.n_types):
            expo = self.dosage[None, :] * mu + self.e[t][None, :]
            if self.normal_type[t]:  # marker-rich normal type: its diploid expectation keeps its smooth deviation
                expo[0] = expo[0] + self.type_smooth[t]
            share[t] = np.add.reduceat(self.b[None, :] * np.exp(expo), wb, axis=1)
        use = np.ones(len(wb), dtype=np.bool_) if windows is None else np.asarray(windows, dtype=np.bool_)
        share = share * use
        share = np.ascontiguousarray(share / share.sum(axis=2, keepdims=True))
        size = data.size if windows is None else data.Yw @ use.astype(data.Yw.dtype)
        size = np.maximum(size.astype(np.float64), 1e-3)
        n_chunks = min(numba.get_num_threads(), 16)
        num, den = np.zeros((n_chunks, len(wb))), np.zeros((n_chunks, len(wb)))
        assign = np.ascontiguousarray(assign, dtype=np.int64)
        K.nb_moments(data.Yw, size, self.types, assign, share, use, num, den)
        phi = 1.0 / np.clip(num.sum(0) / np.maximum(den.sum(0), 1e-12), 1e-3, 1e2)
        # spillover: the pooled window shares of the aneuploid cells, mixed into the diploid expectation of cells of
        # marker-rich types (segmentation merges with tumour cells in imaging data)
        tumour = (assign > 0).astype(data.Yw.dtype)
        g = (tumour @ data.Yw).astype(np.float64) * use if tumour.any() else np.zeros(len(wb))
        g = np.ascontiguousarray(g / max(g.sum(), 1e-12))
        spill_ok = self.normal_type & bool(tumour.any())
        with np.errstate(divide="ignore"):
            log_share = np.log(share)
        ll = np.empty((len(assign), len(mu)))
        self.spill = np.empty(len(assign))
        K.nb_loglik_types(
            data.Yw, size, self.types, share, log_share, phi, use, g, spill_ok, _MAX_SPILL, report, ll, self.spill
        )
        self.model = (share, log_share, phi, use, g, spill_ok)
        return ll

    def assign_all(self, source, cells, pcs_all, pcs_fit, types_fit):
        """Posterior clone probabilities and log likelihood ratios of all `cells` under the fitted model.

        A cell's type is the nearest k-means centre (fitted cells) in principal component space; its likelihood comes
        from one fused pass over its counts (`assign_loglik`), with the temperature and type priors of the fit.
        """
        n_types = self.n_types
        centres = np.zeros((n_types, pcs_fit.shape[1]))
        np.add.at(centres, types_fit, pcs_fit)
        centres /= np.maximum(np.bincount(types_fit, minlength=n_types), 1)[:, None]
        types = _nearest(pcs_all, centres).astype(np.int64)
        share = self.model[0]
        ll = np.empty((len(cells), share.shape[1]))
        self.spill = np.empty(len(cells))
        for i0, i1, X, rws in _row_blocks(source, cells):
            K.assign_loglik(
                X.data, X.indices, X.indptr, rws, self.data.lay.col_map, self.data.gene_win, types[i0:i1],
                *self.model, _MAX_SPILL, ll[i0:i1], self.spill[i0:i1], n_kernel_chunks(i1 - i0),
            )  # fmt: skip
        return self.posterior(ll, types)

    def posterior(self, ll, types):
        """Posterior clone probabilities (temperature-scaled likelihoods, type priors) and the unscaled llr."""
        llr = ll[:, 1:].max(axis=1) - ll[:, 0] if ll.shape[1] > 1 else np.zeros(len(ll))
        ll = self.tau * ll + np.log(self.priors)[types]
        resp = np.exp(ll - ll.max(axis=1, keepdims=True))
        return resp / resp.sum(axis=1, keepdims=True), llr

    def temperature(self, mu, assign) -> float:
        """Scale of the log likelihoods that calibrates them.

        It is the slope of a logistic regression (no intercept) of the sign of each cell's aneuploid-vs-diploid log
        likelihood ratio on one half of the windows on the ratio from the other half. The window likelihoods are not
        independent (expression programs, dispersion misfit), so the raw ratios are overconfident.
        """
        if len(mu) < 2:
            return 1.0
        par = self.data.win_parity
        half = [self.loglik(mu, assign, par == p) for p in (0, 1)]
        r = [h[:, 1:].max(axis=1) - h[:, 0] for h in half]
        x = np.concatenate([r[0], r[1]])
        y = np.concatenate([r[1] > 0, r[0] > 0]).astype(np.float64)
        tau = 0.01
        for _ in range(50):  # Newton steps for the 1-parameter logistic regression
            p = 1.0 / (1.0 + np.exp(-np.clip(tau * x, -50, 50)))
            grad = ((y - p) * x).sum()
            hess = (p * (1 - p) * x * x).sum() + 1e-12
            step = grad / hess
            tau = max(tau + step, 1e-6)
            if abs(step) < 1e-9 * max(tau, 1e-6):
                break
        return float(min(tau, 1.0))

    def merge_targets(self, lr, v):
        """Clone each clone merges into: aneuploid clones without differences merge into the earlier one."""
        k = len(lr)
        target = np.arange(k)
        for i in range(2, k):
            for j in range(1, i):
                if target[j] == j:
                    d = lr[i] - lr[j]
                    pmd, pnd, _ = self.hmm(d - np.median(d), v[i] + v[j])
                    if _burden(pmd[0], pnd[0], self.data.lay) < self.min_burden:
                        target[i] = j
                        break
        return target

    def run(self, assign, n_iter):
        """EM of the clone assignments; returns the assignments, the posteriors and the llr of the cells."""
        n = len(assign)
        assign = _compact(assign)
        for it in range(n_iter):
            k = int(assign.max()) + 1
            lr, v = self.profiles(assign, k)
            target = self.merge_targets(lr, v)
            if (target != np.arange(k)).any():
                log.debug("copy_number: iteration %d, clones -> %s", it, target)
                assign = _compact(target[assign])
                continue
            mu = self.fit(lr, v, cross=True)[0]
            mu_whole, _, st_whole = self.fit(lr, v, cross=False)
            priors = self.type_priors(mu_whole)
            ll = self.loglik(mu, assign)
            # drop the aneuploid clone whose likelihood gain does not pay for its segments (BIC), counted on the
            # profile fitted on all genes (the cross-fitted state paths switch at every parity boundary)
            gain = np.full(k, np.inf)
            n_seg = 1 + (np.diff(st_whole, axis=1) != 0).sum(axis=1)
            for c in range(1, k):
                mine = assign == c
                rest = np.delete(ll[mine], c, axis=1).max(axis=1)
                gain[c] = (ll[mine, c] - rest).sum() - n_seg[c] * np.log(n)
            worst = int(np.argmin(gain))
            if gain[worst] < 0:
                ll[:, worst] = -np.inf
                log.debug("copy_number: iteration %d dropped clone %d (margin %.1f)", it, worst, gain[worst])
                assign = _compact(np.argmax(ll, axis=1))
                continue
            ll += np.log(priors)[self.types]
            new = np.argmax(ll, axis=1)
            small = np.bincount(new, minlength=k) < _MIN_CLONE
            small[0] = False
            if small.any():
                ll[:, small] = -np.inf
                new = np.argmax(ll, axis=1)
            changed = float((new != assign).mean())
            log.debug("copy_number: iteration %d, %d clones, %.4f of cells changed", it, k, changed)
            assign = _compact(new)
            if changed < 2e-3:
                break
        k = int(assign.max()) + 1
        lr, v = self.profiles(assign, k)
        mu = self.fit(lr, v, cross=True)[0]
        mu_whole, pn_whole, st_whole = self.fit(lr, v, cross=False)
        self.priors = self.type_priors(mu_whole)
        if _DOSAGE == "final":  # dosage weights of this fit, before the final likelihood
            self.dosage = _dosage_weights(mu_whole, lr, v, pn_whole)
        ll = self.loglik(mu, assign, report=True)
        model, spill = self.model, self.spill
        self.tau = self.temperature(mu, assign)
        self.model, self.spill = model, spill
        log.debug("copy_number: likelihood temperature %.4g", self.tau)
        self.final = (mu_whole, pn_whole, st_whole, lr, v)
        resp, llr = self.posterior(ll, self.types)
        return assign, resp, llr


def _dosage_weights(mu, lr, v, pn, strength: float = _DOSAGE_PRIOR) -> np.ndarray:
    """Dosage sensitivity of each gene from the clone contrasts.

    Where an aneuploid clone's HMM profile is non-neutral, a gene's log ratio is regressed on the profile through the
    origin (precision-weighted, pooled over clones). The slope is shrunk towards 1 with the weight of `strength`
    segments' worth of evidence: genes that follow copy number fully get about 1, dosage-compensated genes less.
    """
    if len(mu) < 2:
        return np.ones(lr.shape[1])
    m, y, w = mu[1:], np.clip(lr[1:], -2, 2), 1.0 / v[1:]
    w = w * (pn[1:] < 0.5)
    num = (w * m * y).sum(axis=0)
    den = (w * m * m).sum(axis=0)
    prior = strength * np.median(den[den > 0]) if (den > 0).any() else 1.0
    return np.clip((num + prior) / (den + prior), 0.0, 2.0)


def _compact(assign):
    """Relabel to 0..k-1 without empty clones, keeping clone 0 (diploid) as 0 even when it is empty."""
    used = np.unique(assign)
    used = np.concatenate([[0], used[used != 0]])
    lut = np.full(int(assign.max()) + 1, -1)
    lut[used] = np.arange(len(used))
    return lut[assign]


def _block_size(lay) -> int:
    """Genes per block of the genomic statistics.

    `_BLOCK`, unless a typical chromosome has fewer than two blocks of `_BLOCK` genes (small panels): then an eighth
    of its genes (at least 4), so that blocks along a chromosome can still be compared.
    """
    g = int(np.median(lay.gs))
    return _BLOCK if g >= 2 * _BLOCK else max(4, g // 8)


def _block_medians(lr: np.ndarray, lay, size: int | None = None):
    size = _block_size(lay) if size is None else size
    vals, ch = [], []
    for c, (g0, g) in enumerate(zip(lay.g0s, lay.gs, strict=True)):
        nb = max(1, g // size)
        for part in np.array_split(np.arange(g0, g0 + g), nb):
            vals.append(np.median(lr[part]))
            ch.append(c)
    return np.array(vals), np.array(ch)


def _mode(lr: np.ndarray, lay) -> float:
    """Mode of the block medians of a profile (Gaussian kernel, bandwidth 0.05): the copy-neutral level.

    The library size normalisation moves it away from 0 when much of the genome is gained or lost.
    """
    x, _ = _block_medians(lr, lay)
    grid = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 401)
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / 0.05) ** 2).sum(axis=1)
    return float(grid[np.argmax(dens)])


def _smooth_amplitude(lr: np.ndarray, lay, size: int | None = None) -> float:
    """Amplitude of the genomically smooth part of a relative profile.

    It is the square root of the lag-1 autocovariance along the genome of the medians of blocks of `size` genes.
    Gene-level noise and expression differences between cell
    types are mostly independent between blocks and do not contribute; copy number changes span many blocks.
    """
    x, ch = _block_medians(lr, lay, size)
    x = x - np.median(x)
    same = np.diff(ch) == 0
    if not same.any():
        return 0.0
    return float(np.sqrt(max((x[1:] * x[:-1])[same].mean(), 0.0)))


# ---- gene layout and helpers (genomic order, bins, k-means) --------------------------------------------------


def _excluded_genes(var_names: pd.Index, exclude_genes: Sequence[str] | None) -> np.ndarray:
    if exclude_genes is None:
        names = var_names.astype(str)
        return names.str.startswith("HLA-") | names.isin(CELL_CYCLE_GENES)
    return var_names.isin(_as_list(exclude_genes))


class _Layout:
    """Genes in genomic order and chromosomes, and the genomic bins of the outputs."""

    def __init__(self, var: pd.DataFrame, used: np.ndarray):
        sub = var.iloc[np.flatnonzero(used)]
        chromosomes = _natural_sort(list(sub["chromosome"].unique()))
        order, g0s, gs = [], [], []
        n = 0
        for c in chromosomes:
            idx = np.flatnonzero(used & (var["chromosome"] == c).to_numpy())
            idx = idx[np.argsort(var["start"].to_numpy()[idx], kind="stable")]
            order.append(idx)
            g0s.append(n)
            gs.append(len(idx))
            n += len(idx)
        self.chromosomes = chromosomes
        self.order = np.concatenate(order)
        self.col_map = np.full(len(var), -1, dtype=np.int64)
        self.col_map[self.order] = np.arange(n)
        self.g0s = np.array(g0s, dtype=np.int64)
        self.gs = np.array(gs, dtype=np.int64)
        self.n_genes = n
        self.chrom = var["chromosome"].to_numpy()[self.order].astype(str)
        self.start = var["start"].to_numpy()[self.order].astype(np.int64)
        self.end = var["end"].to_numpy()[self.order].astype(np.int64)

    def bins(self, bin_size: int) -> tuple[pd.DataFrame, dict[str, int], sp.csr_matrix]:
        """Genomic bins, the first bin of each chromosome, and the (bins x genes) matrix that averages each bin's genes.

        A bin without genes takes the value of the gene whose centre is nearest to the bin centre. Chromosomes extend
        to their GRCh38 length (so that the bins are the same for every dataset) unless a gene lies beyond the GRCh38
        length of its chromosome or a chromosome is not one of GRCh38's: then the genome is not GRCh38, and every
        chromosome ends at its last gene.
        """
        centre = (self.start + self.end) // 2
        last = {c: int(self.end[g0 + g - 1]) for c, g0, g in zip(self.chromosomes, self.g0s, self.gs, strict=True)}
        hg38 = all(c in HG38_LENGTHS and e <= HG38_LENGTHS[c] for c, e in last.items())
        rows, cols, vals, frames = [], [], [], []
        chr_pos: dict[str, int] = {}
        b0 = 0
        for c, g0, g in zip(self.chromosomes, self.g0s, self.gs, strict=True):
            cen, genes = centre[g0 : g0 + g], np.arange(g0, g0 + g)
            length = HG38_LENGTHS[c] if hg38 else last[c]
            nb = -(-length // bin_size)
            chr_pos[c] = b0
            starts = np.arange(nb, dtype=np.int64) * bin_size
            frames.append(
                pd.DataFrame({"chromosome": c, "start": starts, "end": np.minimum(starts + bin_size, length)})
            )
            gb = np.minimum(cen // bin_size, nb - 1)
            per_bin = np.bincount(gb, minlength=nb)
            rows.append(b0 + gb)
            cols.append(genes)
            vals.append(1.0 / per_bin[gb])
            empty = np.flatnonzero(per_bin == 0)
            if len(empty):
                mid = starts[empty] + bin_size // 2
                order = np.argsort(cen, kind="stable")
                pos = np.clip(np.searchsorted(cen[order], mid), 1, g - 1) if g > 1 else np.zeros(len(empty), int)
                left, right = order[np.maximum(pos - 1, 0)], order[pos]
                nearest = np.where(np.abs(cen[left] - mid) <= np.abs(cen[right] - mid), left, right)
                rows.append(b0 + empty)
                cols.append(genes[nearest])
                vals.append(np.ones(len(empty)))
            b0 += nb
        M = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(b0, self.n_genes)
        )
        bins = pd.concat(frames, ignore_index=True)
        bins.index = bins.index.astype(str)
        return bins, chr_pos, M


def _csr(X) -> sp.csr_matrix:
    return X if sp.issparse(X) else sp.csr_matrix(X)


def _subsample(idx: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    return idx if len(idx) <= n_max else np.sort(rng.choice(idx, n_max, replace=False))


def _kmeans(X: np.ndarray, k: int, rng) -> np.ndarray:
    """Cluster of each row of `X` after Lloyd's k-means fitted on at most `20 k` rows; no empty clusters."""
    sub = X[_subsample(np.arange(len(X)), 20 * k, rng)]
    centres = sub[rng.choice(len(sub), k, replace=False)]
    for _ in range(_KMEANS_ITER):
        a = _nearest(sub, centres)
        counts = np.bincount(a, minlength=k)
        new = np.zeros_like(centres)
        np.add.at(new, a, sub)
        filled = counts > 0
        centres[filled] = new[filled] / counts[filled, None]
    assign = _nearest(X, centres)
    return np.unique(assign, return_inverse=True)[1]


def _nearest(X: np.ndarray, centres: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    c2 = (centres**2).sum(axis=1)
    return np.concatenate(
        [np.argmin(c2[None, :] - 2 * X[i : i + chunk] @ centres.T, axis=1) for i in range(0, len(X), chunk)]
    )
