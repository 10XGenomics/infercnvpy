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

import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.cluster import hierarchy
from scipy.ndimage import median_filter
from scipy.optimize import nnls
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from infercnvpy._logging import log, log_result
from infercnvpy._numba import n_kernel_chunks, numba_threads
from infercnvpy._util import _as_list, _as_matrix, _check_counts, _listed, _params_for_uns, _reference_cells
from infercnvpy.tl import _copy_number_kernels as K
from infercnvpy.tl._infercnv import NO_CHR_NAMES, _check_var, _natural_sort, _on_used_chromosomes, _Rows
from infercnvpy.tl._infercnv_kernels import _group_sums_csr
from infercnvpy.tl._spatial import _drop_cross_edges, _graph

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

_KMEANS_ITER = 30  # most Lloyd iterations of k-means
_KMEANS_STARTS = 10  # random starts of k-means, run together
_KMEANS_ROWS = 8192  # rows per matrix product of k-means

LABELS = ("aneuploid", "diploid", "not.defined", "undetermined")
UNDETERMINED = LABELS[3]

#: Markers of immune, endothelial and stromal cells, which are rarely tumor cells
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
_BLOCK = 25  # genes per block of the genomic statistics (flatness, smooth amplitude, mode); fewer on small panels
_MIN_CLONE = 5
_PCA_CHUNKS = 16  # row chunks of float sums (PCA, k-means, dispersion): fixed, so that they don't depend on `n_jobs`
_GRAM_CHUNKS = 4  # row chunks (p x p slabs) of the Gram matrix from sparse rows
_GRAM_TILE = 4096  # cells per dense tile of the Gram matrix from dense rows
_SPARSE_GRAM = 0.1  # share of nonzeros in the highly variable genes below which the Gram matrix uses the sparse rows
_DENSE_EIGH = 500  # highly variable genes up to which the PCA uses a full eigendecomposition (above: Lanczos)
_DOSAGE_PRIOR = 1.0  # strength of the shrinkage of dosage weights towards 1, in median gene precisions
_MIN_PRIOR = 0.01  # smallest prior clone probability of a cell type
_NP_SWEEPS = 1000  # most updates of the spatial posteriors
_NP_TOL = 1e-5  # largest change of a posterior at convergence
_AMP_TUMOR = 0.1  # smooth amplitude (log ratio) of an aneuploid clone's profile, see `_smooth_amplitude`
_N_PASSES = 4
_RESEED_GAP = 0.05  # how much flatter than the diploid clone another clone must be to restart the passes from it
_FLAT_GAP = 0.15  # difference of the cluster means of group flatness that counts as a clear split
_MARKER_FOLD = 5.0
_MARKER_MIN = 1e-3  # share of counts in the markers above which a group can be marker-rich
_MAX_SPILL = 0.5  # largest spillover fraction of tumor transcripts in a normal cell's diploid expectation
_MIN_SEED = 3  # cells of the smallest group that can be the first diploid group (a rare normal population)
_MIN_RESEED = 10  # cells of the smallest clone that can become the diploid clone later
_MIN_SEG_GENES = 20  # genes of the shortest run that counts towards the burden; fewer on small panels
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
    max_neighbor_weight: float = 0.8,
    force_calls: bool = False,
    bin_size: int = 1_000_000,
    chunksize: int = 5000,
    n_restarts: int = 5,
    random_state: int | np.random.Generator = 0,
    key_added: str = "copy_number",
    inplace: bool = True,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> None | tuple[pd.DataFrame, np.ndarray, dict, np.ndarray | None]:
    """Infer copy number profiles, clones and aneuploid/diploid labels with a clone-level count model.

    The expected count of a gene in a cell is the cell's size factor times a per-gene baseline (pooled diploid
    cells) times the expression deviation of the cell's type times the copy ratio of the cell's clone, with negative
    binomial noise. The steps are:

    1. keep cells with at least `min_genes` detected genes and genes detected in at least `min_detection` of them
       (on chromosomes named like "chr1"); sum each cell's counts in windows of `window_size` genes;
    2. cluster cells by expression (k-means with 10 random starts on the principal components of the most variable
       genes, both on all cells); these clusters are the expression types (with `reference_cat`, the reference cells
       and the other cells are clustered separately, so that every type is either reference or not). A copy number
       HMM on the difference of the pooled profiles of every pair of types groups the types that differ in expression
       only;
    3. take the first baseline from the groups whose pooled profile is flat along the genome, and from groups that
       express `normal_markers` far above the rest (or from `reference_cat`);
    4. the expression deviation of a type is the gene-level part of its pooled profile that the HMM does not
       explain, so it holds the cell type but no copy number signal. A clone's profile is an HMM (8 copy-ratio states,
       transitions by genomic distance, `segment_length`) on the gene-level log ratio of its pooled counts to their
       expectation, corrected for the cell's capture of highly expressed genes and centered on its most common level;
    5. assign cells to clones by EM on the negative binomial likelihood of their window counts. Profiles are
       cross-fitted (windows of one parity are scored with the profile fitted on the other parity), clones without
       differences merge, and clones whose likelihood gain does not pay for their segments are dropped;
    6. repeat: a clearly flatter clone becomes diploid (unless normal markers chose the baseline), aneuploid clones
       without a smooth profile (amplitude below 0.1) merge into the diploid clone, which re-defines the baseline;
    7. learn per-gene dosage weights from the clone contrasts and per-type clone priors from a non-negative fit of
       each type's profile on the clone profiles; the per-cell posterior uses these priors and log likelihoods scaled
       by a temperature calibrated between the two halves of the windows;
    8. repeat steps 2 to 7 `n_restarts` times, each with its own k-means seed, and keep the fit whose
       aneuploid/diploid calls agree best with those of the other fits.

    The method is made for UMI counts from droplet (10x) and imaging platforms. On full-length plate data
    (Smart-seq2) its profiles are poor.

    .. warning::
        Without reference cells, the diploid baseline comes from the sample itself. When tumor cells far outnumber
        normal cells, especially at low depth (a few hundred counts per cell), the baseline can land on a tumor group,
        and then the labels are inverted: tumor cells are called diploid and normal cells aneuploid. Pass known normal
        cells with `reference_key` and `reference_cat` in such samples. `adata.uns["copy_number"]["seed"]` shows how
        the baseline was found, and a warning flags the least certain case.

    .. warning::
        With reference cells, weak clones in shallow data can be called diploid. After each pass, an aneuploid clone
        whose profile has a smooth amplitude below 0.1 merges into the diploid clone, and in shallow data a clone that
        mixes tumor and normal cells can stay below that cut. When a run with reference cells calls few or no cells,
        run without a reference as well and compare the two.

        Every normal cell type in the sample belongs in the reference, including stromal cells such as smooth muscle
        cells and pericytes. A normal cell type left out of the reference can form clones from the smooth part of its
        expression difference to the reference, and normal cells of other types can join those clones.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts and genomic positions (`chromosome`, `start` and `end` in `adata.var`).
    layer
        Layer with the raw counts. Default: `adata.X`.
    reference_key
        Column of `adata.obs` with annotations of normal cells. With `reference_cat`, these cells define the baseline.
        The reference cells get expression types of their own. These are normal types, like the marker-rich types
        without a reference (see `normal_markers`), so their diploid expectation keeps their smooth deviation from the
        baseline and the spillover term, and their prior is diploid. The reference cells stay in the diploid clone
        while the clones are fitted, so they can't pull a clone towards their own expression. Their labels come from
        the same posterior as those of the other cells. Cells outside the reference that resemble no reference type,
        e.g. normal epithelium when the reference holds immune and stromal cells, can be called aneuploid when their
        expression differs smoothly from the reference; add them to the reference if they are known to be normal.
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
        Genes per window of the per-cell likelihood. A chromosome with fewer than `2 * window_size` genes is split
        into two windows.
    n_clusters
        Number of expression clusters. Default: one per 50 cells, between 8 and 40.
    segment_length
        Expected distance between copy number changes, in bases (sets the HMM transition probabilities).
    min_burden
        Fraction of genes in non-neutral states below which two profiles count as the same copy number.
    normal_markers
        Marker genes of cell types that are rarely tumor cells (a list, or a dict of lists per cell type), matched
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
        Let each cell's spatial neighbors inform its clone, with the cells x cells graph `adata.obsp[connectivity_key]`,
        e.g. `"spatial_connectivities"` from :func:`squidpy.gr.spatial_neighbors_knn`. Any graph with non-negative
        weights works, and entries set to 0 are not edges. The prior of a cell over the clones becomes a mixture of its
        type's prior and the weighted mean of its neighbors' posteriors, the cell itself left out. The mixing weight is
        estimated from the data by maximizing the likelihood of each cell given its neighbors, and then the posteriors
        are updated until they no longer change. A cell with strong evidence of its own keeps its call, and a cell with
        little evidence follows its neighborhood. A cell without neighbors keeps its per-cell posterior. On imaging
        data, where single cells hold few counts and neighbors usually share their clone, this lowers the errors
        inside clones. About 10 neighbors, i.e. `squidpy.gr.spatial_neighbors_knn(adata, n_neighs=10)`, is a good
        start.

        With `reference_key`, edges between reference and other cells are dropped: a reference cell says nothing about
        the copy number of a tumor cell next to it, and the other way round. Reference cells take the diploid clone
        as their neighbors' state, so their probability of being aneuploid can only go down. Without a reference, use
        the same split with your own annotation by setting the graph's entries between known normal cells and the
        other cells to 0 (see :func:`~infercnvpy.tl.smooth_cnv`). `copy_number_llr` stays the cell's own log
        likelihood ratio; `copy_number_score`, the labels, clones and profiles use the spatial posteriors. With
        several `n_restarts`, the fit is chosen without the spatial prior, and the prior then applies to that fit
        only. Default `None`: no spatial information.
    max_neighbor_weight
        Largest weight that a cell's neighbors get in its prior when `connectivity_key` is given; the weight itself is
        estimated from the data, up to this value. A lower value lets a cell's own data count more, which reduces the
        spread of calls into normal cells next to a tumor; a higher value smooths more, which helps in shallow data.
        Between 0 and 1 (not included); 0 turns the spatial prior off, so the results are those of single cells.
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
    n_restarts
        Number of fits, each with its own k-means seed. The expression types change with the seed, and the clones
        and calls can change with them. Of the fits, `copy_number` keeps the one whose aneuploid/diploid calls agree
        best with those of the other fits (the highest mean share of cells with the same call), so all of its outputs,
        i.e. labels, clones, profiles and segments, come from that one fit. `copy_number_stability` holds the share of the fits
        that agree with each cell's label. The run time grows about linearly with `n_restarts`; 1 fits once.
    random_state
        Seed (or random number generator) of the k-means starts of all fits and of the sample of `max_fit_cells`.
    key_added
        Name of the results in `adata.obs`, `adata.obsm` and `adata.uns`.
    inplace
        Write the results to `adata`. If `False`, return them.
    n_jobs
        Number of threads.
    show_progress
        Show a progress bar over the restarts, with the current step (PCA, types, EM, neighbor prior). tqdm's own
        switch, the environment variable ``TQDM_DISABLE=1``, turns it off too if it is set before tqdm is imported.

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
        or doublets with tumor cells; for cells of normal types (marker-rich types, or reference types), up to 0.5 of
        this mixture is part of the diploid expectation, so that spillover alone does not make them aneuploid.
    `adata.obs["copy_number_stability"]` : :class:`float`
        Share of the `n_restarts` fits whose call for the cell (aneuploid or diploid, without the spatial prior)
        equals its label, in steps of `1 / n_restarts`. 1 means that every fit makes the same call. Lower values
        mark cells whose call depends on the k-means seed or, with `connectivity_key`, whose label the spatial
        prior changed against the fits.
    `adata.obsm["X_copy_number"]` : :class:`numpy.ndarray`
        Copy number log ratios (natural log, float32), cells x genomic bins: the posterior mixture of the clones'
        segment means, a continuous score. Cells that are `"not.defined"` or `"undetermined"` have 0 (no copy number
        change), so that embeddings, smoothing and heatmaps work on all cells, as with :func:`~infercnvpy.tl.infercnv`.
    `adata.layers["gene_values_copy_number"]` : :class:`numpy.ndarray`
        Only if `calculate_gene_values=True`: log ratio per gene (natural log, float32), as `X_copy_number`; `NaN` for
        genes removed by the filters.
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
        Log ratio (natural log) of each clone in each bin.
    `adata.uns["copy_number"]["segments"]` : :class:`pandas.DataFrame`
        Segments of the aneuploid clones: `clone`, `chromosome`, `start`, `end`, `n_genes`, `log_ratio` (natural log)
        with a 95% interval (`ci_low`, `ci_high`), `state_ratio` (copy ratio of the HMM state) and `p_neutral` (mean
        posterior probability of the neutral state).
    `adata.uns["copy_number"]["genes"]` : :class:`pandas.DataFrame`
        The genes used, in genomic order, with their `dosage` weight (1: expression follows copy number fully; the
        final likelihood uses these weights).
    `adata.uns["copy_number"]["seed"]` : :class:`dict`
        The baseline `method`, the `flatness_gap` between the flat and the other groups and the gap that counts as a
        clear split (`clear_gap`). Without a clear split and without marker-rich groups (`method` "flattest group"),
        a warning says that the baseline is uncertain.
    `adata.uns["copy_number"]["temperature"]` : :class:`float`
        Scale applied to the log likelihoods for the posterior.
    `adata.uns["copy_number"]["restarts"]` : :class:`dict`
        `chosen` (the index of the fit that `copy_number` returns), `agreement` (the share of cells with the same
        aneuploid/diploid call in every pair of fits) and `n_aneuploid` (the number of aneuploid cells in each fit).
    `adata.uns["copy_number"]["spatial"]` : :class:`dict`
        Only with `connectivity_key`: `neighbor_weight` (the estimated weight of the neighbors in each cell's prior,
        0 to `max_neighbor_weight`), `sweeps` (number of updates of the posteriors) and `converged`.
    `adata.uns["copy_number"]["params"]` : :class:`dict`
        Parameters of this call.

    When the cells are `"undetermined"`, `uns` holds `chr_pos`, `bins`, `window_start`, `window_end`, `params` and
    `undetermined` (the number of genes passing the filters and the median counts per cell in them).
    """
    _check_var(adata)
    _check_args(
        window_size=window_size, min_genes=min_genes, n_iter=n_iter, bin_size=bin_size,
        chunksize=chunksize, n_clusters=n_clusters, max_fit_cells=max_fit_cells, min_detection=min_detection,
        segment_length=segment_length, n_restarts=n_restarts,
    )  # fmt: skip
    if not 0 <= max_neighbor_weight < 1:
        raise ValueError(f"`max_neighbor_weight` must be at least 0 and below 1, got {max_neighbor_weight}.")
    reference_cat, exclude_chromosomes, exclude_genes = map(
        _listed, (reference_cat, exclude_chromosomes, exclude_genes)
    )
    is_ref = _reference_cells(adata, reference_key, reference_cat)
    G = None if connectivity_key is None else _graph(adata, connectivity_key)
    if isinstance(normal_markers, Mapping):  # lists, so that the parameters can be written to h5ad/zarr
        normal_markers = {k: _as_list(v) for k, v in normal_markers.items()}
    normal_markers = _listed(normal_markers)
    params = _params_for_uns(
        layer=layer, reference_key=reference_key, reference_cat=reference_cat,
        exclude_chromosomes=exclude_chromosomes, exclude_genes=exclude_genes, min_genes=min_genes,
        min_detection=min_detection, window_size=window_size, n_clusters=n_clusters, segment_length=segment_length,
        min_burden=min_burden, normal_markers=normal_markers, n_iter=n_iter, max_fit_cells=max_fit_cells,
        connectivity_key=connectivity_key, max_neighbor_weight=max_neighbor_weight, force_calls=force_calls,
        bin_size=bin_size, n_restarts=n_restarts, random_state=random_state,
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
                "least %d genes and %d counts per cell, so %d cells are labeled %r. For small panels, use "
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
        ref_fit = None if is_ref is None else is_ref[cells[fit]]
        ref_all = None if is_ref is None else is_ref[cells]
        if ref_fit is None:
            log.info(
                "copy_number: no reference cells, so the diploid baseline comes from the sample itself. Where known "
                "normal cells are annotated, give them as `reference_key` / `reference_cat` for a reliable baseline."
            )
        markers = np.asarray(adata.var_names.isin(_marker_list(normal_markers))) & candidates
        mlay = _Layout(var, markers) if markers.any() else None
        n_clusters = n_clusters or int(np.clip(n_fit // 50, 8, 40))

        with _progress(n_restarts, show_progress) as bar:
            # expression clusters: PCA once; k-means and everything after it once per restart
            bar.set_postfix_str("PCA")
            pcs = data.expression_pcs()
            pcs_all = data.project(source, cells) if n_fit < n else None
            ref_share = None if ref_all is None else float(ref_all.mean())
            runs = []  # (em, resp, llr, ll, types, baseline method, flatness gap) of each restart
            for r, child in enumerate(rng.spawn(n_restarts)):
                bar.set_description(f"copy_number restart {r + 1}/{n_restarts}", refresh=False)
                em, resp, llr, ll, *baseline = _fit_clones(
                    data, hmm, pcs, ref_fit, ref_share, n_clusters, child, min_burden, n_iter, mlay,
                    bar.set_postfix_str,
                )  # fmt: skip
                runs.append([em, resp, llr, ll, em.types, *baseline])
                log.debug("copy_number: restart %d, %d clones", r, resp.shape[1])
                bar.update()
            if n_fit < n:  # assign every cell with each fitted model, in one pass over the counts
                bar.set_postfix_str("assigning all cells")
                assigned = _assign_all([run[0] for run in runs], source, cells, pcs_all, pcs, ref_all)
                for run, out in zip(runs, assigned, strict=True):
                    run[1:5] = out
            # the restart whose aneuploid/diploid calls agree best with those of the others (the medoid). Agreement
            # is the share of cells with the same call, not the adjusted Rand index: a fit with an inverted baseline
            # (tumor cells diploid, normal cells aneuploid) splits the cells like a correct fit, so its ARI with it
            # is 1
            calls = np.array([run[1][:, 1:].sum(axis=1) > 0.5 for run in runs])
            agreement = np.array([[(a == b).mean() for b in calls] for a in calls])
            best = int(np.argmax(agreement.sum(axis=1)))
            log.debug("copy_number: restart %d agrees best with the others\n%s", best, np.round(agreement, 3))
            em, resp, llr, ll, types, baseline_method, flat_gap = runs[best]
            del runs
            spatial = None
            if G is not None:
                bar.set_postfix_str("neighbor prior")
                resp, spatial = _neighbor_prior(
                    G, cells, resp, em.tau * ll, np.log(em.priors)[types], ref_all, max_neighbor_weight
                )
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
            {"chromosome": lay.chrom, "start": lay.start, "end": lay.end, "dosage": em.dosage},
            index=np.asarray(adata.var_names)[lay.order],
        ),
        "seed": {"method": baseline_method, "flatness_gap": float(flat_gap), "clear_gap": _FLAT_GAP},
        "temperature": em.tau,
        "params": params,
    }
    if spatial is not None:
        uns["spatial"] = spatial
    stability = (calls == (p_an > 0.5)).mean(axis=0)
    uns["restarts"] = {"chosen": best, "agreement": agreement, "n_aneuploid": calls.sum(axis=1)}
    labels = np.where(p_an > 0.5, LABELS[0], LABELS[1])
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
            "single flattest group was taken; in samples with few normal cells this can be a tumor group, which "
            "inverts the calls. Give known normal cells as `reference_key` / `reference_cat`.",
            flat_gap, _FLAT_GAP,
        )  # fmt: skip
    if n_an == 0 and is_ref is None:
        log.warning(
            "copy_number: no cell is called aneuploid. The diploid baseline comes from the sample itself, so a sample "
            "without normal cells (tumor only, a cell line) is called diploid; give reference cells "
            "(`reference_key`, `reference_cat`) if that is the case."
        )
    elif n_an == 0:
        log.warning(
            "copy_number: no cell is called aneuploid. With reference cells, weak clones in shallow data can be "
            "called diploid; run without a reference as well and compare the two."
        )
    return _outputs(
        adata, cells, key_added, inplace, labels, names[clone], names, cna, uns, gene_values,
        score=p_an, llr=llr, spill=em.spill, stability=stability,
    )  # fmt: skip


def _fit_clones(data, hmm, pcs, ref_fit, ref_share, n_clusters, rng, min_burden, n_iter, mlay, stage):
    """One fit of the clones (steps 2 to 7 of `copy_number`), with the k-means starts drawn from `rng`.

    `ref_share` is the share of reference cells among all cells (with `ref_fit`, the reference cells of the fitted
    ones), `mlay` the layout of the normal markers (`None` without markers). Returns the expression types, the
    fitted `_EM`, the posteriors, log likelihood ratios and log likelihoods (cells x clones) of the fitted cells, the
    baseline method and the flatness gap. `stage` is called with the name of each step (for the progress bar).
    """
    lay = data.lay
    stage("types")
    if ref_fit is None:
        clusters = _kmeans(pcs, n_clusters, rng)
    else:
        clusters, ref_type = _kmeans_split(pcs, n_clusters, rng, ref_fit, ref_share)
    Yc, Sc = data.pooled(clusters)
    gene_var = _gene_variability(Yc, Sc, lay)
    burden = _pairwise_burden(hmm, Yc, Sc, gene_var, lay)
    groups = _groups(burden, min_burden)

    gcell = groups[clusters]
    n_groups = int(groups.max()) + 1
    if ref_fit is not None:
        normal = ref_fit
        baseline_method, flat_gap = "reference", np.nan
    else:
        marker, total = _marker_counts(data, clusters, Yc, mlay)
        normal, baseline_method, flat_gap = _seed(Yc, marker, total, groups, gcell, lay)

    em = _EM(data, hmm, gene_var, min_burden, clusters, Yc, Sc)
    if ref_fit is not None:
        # reference types are normal types with a diploid prior, and reference cells stay in the diploid clone
        # during the fit: with several reference cell types, a reference type whose smooth deviation from the
        # pooled reference resembles a clone's profile would otherwise join that clone at depth
        em.normal_type = em.ref_type = ref_type
        em.pinned = ref_fit
    elif mlay is not None:
        em.normal_type = _marker_rich(marker / total, np.array([_flatness(y, lay) for y in Yc]))
        log.debug("copy_number: marker-rich expression types %s", np.flatnonzero(em.normal_type))
    frac = np.bincount(gcell, weights=normal, minlength=n_groups) / np.bincount(gcell, minlength=n_groups)
    lut = np.where(frac > 0.5, 0, 1 + np.arange(n_groups))
    assign = np.where(normal, 0, lut[gcell])
    fixed = ref_fit is not None
    reseeded = "markers" in baseline_method  # marker-rich groups outweigh flatness: no restart
    # passes: after each, a clearly flatter clone becomes diploid and the passes restart from it, or clones without
    # a smooth profile merge into the diploid clone; a last single-iteration pass follows the last change
    stage("EM")
    for pass_ in range(_N_PASSES + 1):
        em.set_baseline(normal)
        assign, resp, llr, ll = em.run(assign, n_iter if pass_ < _N_PASSES else 1)
        if pass_ == _N_PASSES:
            break
        k = int(assign.max()) + 1
        target = np.arange(k)
        if not fixed and not reseeded:
            Yk = em.Y
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
        target[amp < _AMP_TUMOR] = 0
        if (target == np.arange(k)).all():
            break
        assign = _compact(target[assign])
        if not fixed:
            normal = assign == 0
    return em, resp, llr, ll, baseline_method, flat_gap


@contextmanager
def _progress(n_restarts: int, show: bool) -> Iterator[tqdm]:
    """A progress bar over the restarts, with the current step as its postfix (off unless `show`).

    While it shows, every handler that our log records reach (the shared scverse logger's, plain or rich, and any
    further up) writes with the bars cleared, so that a message gets a line of its own.
    """
    if not show:
        with tqdm(total=n_restarts, disable=True) as bar:
            yield bar
        return
    chain: list[logging.Logger] = []
    node: logging.Logger | None = log.logger
    while node is not None:
        chain.append(node)
        node = node.parent if node.propagate else None
    try:
        for lg in chain:
            lg.handlers = [_ClearBars(h) for h in lg.handlers]
        with tqdm(total=n_restarts, desc="copy_number", unit="restart") as bar:
            yield bar
    finally:  # unwrap rather than restore, so that runs in other threads that overlap this one don't leave wrappers
        for lg in chain:
            lg.handlers = [h.handler if isinstance(h, _ClearBars) else h for h in lg.handlers]


class _ClearBars(logging.Handler):
    """Passes records to `handler` with the progress bars on the terminal cleared, and redraws them after."""

    def __init__(self, handler: logging.Handler):
        super().__init__()
        self.handler = handler

    def handle(self, record: logging.LogRecord) -> bool:
        if record.levelno >= self.handler.level and self.filter(record):
            with tqdm.external_write_mode(file=sys.stderr):
                self.handler.handle(record)
        return True


def _outputs(adata, cells, key_added, inplace, labels, clones, clone_names, cna, uns, gene_values, **values):
    """Write the outputs to `adata` (or return them); cells removed by the filters are `"not.defined"`.

    `values` are the float columns (`score`, `llr`, `spill`, `stability`) of the cells that pass the filters;
    missing ones are NaN.
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
            f"{key_added}_stability": per_cell(values.get("stability", np.nan), np.nan),
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


def _neighbor_prior(G, cells, resp, lt, log_prior, ref, max_weight):
    """Posteriors with a spatial prior: each cell's prior mixes its type's prior with its neighbors' posteriors.

    For cell i with tempered likelihoods `L_i(c) = exp(lt_i(c))` and type prior `pi_i(c)`, the posterior is
    `q_i(c) ~ L_i(c) * ((1 - lam) * pi_i(c) + lam * m_i(c))`, where `m_i` is the weighted mean of the neighbors'
    posteriors `q_j` (i itself excluded). `lam` maximizes the pseudo-likelihood `sum_i log sum_c L_i(c) * prior_i(c)`
    with `m_i` from the per-cell posteriors (concave in `lam`); then damped synchronous updates run to a fixed point.
    With `ref`, edges between reference and other cells are dropped, and reference cells use the diploid clone as
    `m_i`. `lam` is at most `max_weight`; with 0 or without neighbors, cells keep their per-cell posterior `resp`.
    Returns the posteriors and a summary dict.
    """
    if max_weight <= 0 or resp.shape[1] < 2:
        return resp, {"neighbor_weight": 0.0, "sweeps": 0, "converged": True}
    n, k = lt.shape
    G = G[cells][:, cells].tocsr()
    if ref is not None:
        _drop_cross_edges(G, ref)
    G.eliminate_zeros()
    lik = np.exp(lt - lt.max(axis=1, keepdims=True))
    base = np.exp(log_prior)
    base /= base.sum(axis=1, keepdims=True)
    q = resp.copy()  # = lik * base, normalised
    deg = np.asarray(G.sum(axis=1)).ravel()
    fixed = deg <= 0 if ref is None else (deg <= 0) | ref
    lam = _neighbor_weight(G, deg, lik, base, q, ~fixed, max_weight)
    push = np.zeros(n, dtype=bool) if ref is None or lam == 0 else ref & (deg > 0)
    if push.any():  # reference cells with neighbors are declared diploid: their neighbors' state is the diploid clone
        pr = (1.0 - lam) * base[push]
        pr[:, 0] += lam
        q[push] = lik[push] * pr / (lik[push] * pr).sum(axis=1, keepdims=True)
    out, change, w = np.empty_like(q), np.empty(n), G.data.astype(np.float64)
    converged, sweep = lam == 0.0, 0
    while not converged and sweep < _NP_SWEEPS:
        K.neighbor_prior_sweep(G.indptr, G.indices, w, deg, lik, base, lam, fixed, q, out, change)
        q, out = out, q
        sweep += 1
        converged = change.max() < _NP_TOL
    if not converged:
        log.warning("copy_number: the spatial posteriors did not converge in %d sweeps", sweep)
    log.debug("copy_number: neighbor weight %.3f, %d sweeps", lam, sweep)
    return q, {"neighbor_weight": float(lam), "sweeps": sweep, "converged": bool(converged)}


def _neighbor_weight(G, deg, lik, base, q, free, max_weight) -> float:
    """Weight `lam` of the neighbors in the prior that maximizes `sum_i log sum_c L_i(c) ((1-lam) pi_i + lam m_i)`.

    The objective is a sum of logs of functions linear in `lam`, so it is concave; its derivative falls
    monotonically, and bisection finds the root in [0, `max_weight`].
    """
    if not free.any():
        return 0.0
    m = np.asarray(G[free] @ q) / deg[free, None]
    a = (lik[free] * base[free]).sum(axis=1)
    r = (lik[free] * (m - base[free])).sum(axis=1) / np.maximum(a, 1e-300)

    def slope(x):
        return float((r / (1.0 + x * r)).sum())

    if slope(0.0) <= 0:
        return 0.0
    if slope(max_weight) >= 0:
        return max_weight
    lo, hi = 0.0, max_weight
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if slope(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


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
    named = _on_used_chromosomes(var, (chrom.isna() | var["start"].isna()).to_numpy())
    if not named.any():
        raise ValueError(NO_CHR_NAMES)
    if exclude_chromosomes is not None:
        named &= ~chrom.isin(_as_list(exclude_chromosomes)).to_numpy()
    candidates = named & ~excluded
    det = np.zeros((min(numba.get_num_threads(), 16), adata.n_vars), dtype=np.int64)
    ndet = []
    for X, rws in source.rows_of(None):  # all rows: each block's rows are contiguous
        if len(rws):
            X = _csr(X)
            if not ndet:  # the first block only, so that lazy input gets no extra pass
                _check_counts(X.data[X.indptr[rws[0]] : X.indptr[rws[-1] + 1]], "tl.copy_number")
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


def _marker_counts(data, clusters, Yc, mlay) -> tuple[np.ndarray, np.ndarray]:
    """Counts of each cluster in the marker genes (layout `mlay`), and in the model's genes plus the markers.

    Every marker counts, also those that the detection filter removed from the model's genes, so that a rare normal
    population keeps its markers; the genomic flatness of the seed uses the model's genes only.
    """
    total = Yc.sum(axis=1)
    if mlay is None:
        return np.zeros(len(Yc)), total
    M = data.pooled(clusters, len(Yc), mlay)[0]  # columns in the markers' genomic order
    outside = data.lay.col_map[mlay.order] < 0
    return M.sum(axis=1), total + M[:, outside].sum(axis=1)


class _Data:
    """Access to the counts of the cells passing the filters: windows of genes per cell, pooled gene counts."""

    def __init__(self, source: _Rows, cells: np.ndarray, lay, window_size: int):
        self.source, self.cells, self.lay = source, cells, lay
        bounds = []
        for g0, g in zip(lay.g0s, lay.gs, strict=True):
            # at least two windows, so that the cross-fitted profile of a short chromosome comes from its own genes
            step = min(window_size, max(1, g // 2))
            bounds.extend(g0 + np.arange(max(1, g // step)) * step)
        self.wbounds = np.array([*bounds, lay.n_genes], dtype=np.int64)
        self.gene_win = np.repeat(np.arange(len(bounds)), np.diff(self.wbounds))
        self.win_parity = np.arange(len(bounds)) % 2
        self.Yw = np.empty((len(cells), len(bounds)), dtype=np.float32)
        for i0, i1, X, rws in _row_blocks(source, cells):
            out = self.Yw[i0:i1]
            K.window_counts(X.data, X.indices, X.indptr, rws, lay.col_map, self.gene_win, out, n_kernel_chunks(i1 - i0))
        self.size = self.Yw.sum(axis=1, dtype=np.float64)

    def pooled(self, labels: np.ndarray, k: int | None = None, lay=None):
        """Pooled gene counts (k x genes of `lay`, default: the model's) and pooled size factors of each label."""
        lay = self.lay if lay is None else lay
        k = int(labels.max()) + 1 if k is None else k
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        slabs = np.zeros((min(numba.get_num_threads(), 16), k, lay.n_genes))
        for i0, i1, X, rws in _row_blocks(self.source, self.cells):
            _group_sums_csr(X.data, X.indices, X.indptr, rws, labels[i0:i1], False, slabs, lay.col_map)
        return slabs.sum(axis=0), np.bincount(labels, weights=self.size, minlength=k)

    def expression_pcs(self) -> np.ndarray:
        """First principal components of log-normalised expression of the most variable genes, from all cells.

        Two passes over the counts: per-gene moments pick the highly variable genes, then their Gram matrix gives the
        covariance (`G / n - mean mean^T`), so the centered matrix is never built. The Gram matrix is summed from the
        sparse rows when they are sparse, else from dense blocks of rows with BLAS.
        """
        lay, n = self.lay, len(self.cells)
        s = np.zeros((3, _PCA_CHUNKS, lay.n_genes))
        for i0, _, X, rws in _row_blocks(self.source, self.cells):
            K.log_moments(X.data, X.indices, X.indptr, rws, lay.col_map, i0, n, s[0], s[1], s[2])
        det, s1, s2 = s.sum(axis=1)
        mean = s1 / n
        var = s2 / n - mean**2
        hvg = np.sort(np.argsort(-var, kind="stable")[: min(_N_HVG, len(var))])
        self.hvg_of_gene = np.full(lay.n_genes, -1, dtype=np.int64)
        self.hvg_of_gene[hvg] = np.arange(len(hvg))
        p = len(hvg)
        if det[hvg].sum() < _SPARSE_GRAM * n * p:
            slabs = np.zeros((_GRAM_CHUNKS, p, p))
            for i0, _, X, rws in _row_blocks(self.source, self.cells):
                K.gram_sparse(X.data, X.indices, X.indptr, rws, lay.col_map, self.hvg_of_gene, i0, n, slabs)
            G = np.triu(slabs.sum(axis=0))
            G = G + np.triu(G, 1).T
        else:  # dense tiles of `_GRAM_TILE` cells, filled across blocks, so that the sum doesn't depend on them
            G, tile, t = np.zeros((p, p)), np.empty((min(_GRAM_TILE, n), p)), 0
            for _, _, X, rws in _row_blocks(self.source, self.cells):
                B = np.empty((len(rws), p))
                K.hvg_rows(X.data, X.indices, X.indptr, rws, lay.col_map, self.hvg_of_gene, B, n_kernel_chunks(len(B)))
                j = 0
                while j < len(B):
                    m = min(len(tile) - t, len(B) - j)
                    tile[t : t + m] = B[j : j + m]
                    t, j = t + m, j + m
                    if t == len(tile):
                        G += tile.T @ tile
                        t = 0
            if t:
                G += tile[:t].T @ tile[:t]
        mu = mean[hvg]
        proj = _top_eigenvectors(G / n - np.outer(mu, mu), max(1, min(_N_PCS, n - 1, p)))
        self.loadings = np.ascontiguousarray(proj)
        self.pc_offset = mu @ proj
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


def _top_eigenvectors(C: np.ndarray, k: int) -> np.ndarray:
    """Eigenvectors of the `k` largest eigenvalues of the symmetric `C` (columns, largest first).

    Lanczos (`eigsh`, from a fixed start vector) for large `C`, else a full eigendecomposition; both are exact to
    rounding. Each vector's largest entry is positive, so the result doesn't depend on the solver's signs.
    """
    p = len(C)
    if p <= _DENSE_EIGH or k >= p - 1:
        w, vec = np.linalg.eigh(C)
    else:
        w, vec = eigsh(C, k=k, which="LA", v0=np.full(p, 1 / np.sqrt(p)))
    vec = vec[:, np.argsort(-w, kind="stable")[:k]]
    peak = vec[np.abs(vec).argmax(axis=0), np.arange(vec.shape[1])]
    return vec * np.where(peak < 0, -1.0, 1.0)


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


def _robust_var(x: np.ndarray) -> np.ndarray:
    """Noise variance along each profile (rows), from the median absolute deviation of the differences of neighbors."""
    d = np.diff(x, axis=-1)
    return (1.4826 * np.median(np.abs(d - np.median(d, axis=-1, keepdims=True)), axis=-1)) ** 2 / 2


def _scale(lr, pois, gene_var):
    """Total variance per gene of each profile (rows): the expression variance `gene_var`, scaled, plus Poisson."""
    resid = lr / np.sqrt(gene_var + pois)
    return np.maximum(_robust_var(resid), 1e-4)[:, None] * (gene_var + pois)


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
        v = _scale(y, np.array([P[i] + P[j] for i, j in batch]), gene_var)
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
    marked = ok.copy()
    marked[ok] = _marker_rich(mfrac[ok], flat[ok])
    if marked.any():
        seed |= marked
        method += " + markers"
    log.debug("copy_number: group flatness %s, marker fraction (per mille) %s, sizes %s, seed %s",
              np.round(flat, 3), np.round(1000 * mfrac, 2), sizes, np.flatnonzero(seed))  # fmt: skip
    return seed[gcell], method, hi - lo


def _marker_rich(mfrac: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Marker share far above the lower quartile and flatter than the median.

    The lower quartile, not the median: in samples where normal cells are the majority, the median is marker-rich.
    """
    return (mfrac > max(_MARKER_FOLD * np.quantile(mfrac, 0.25), _MARKER_MIN)) & (flat < np.median(flat))


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
    """Lag-1 autocorrelation along the genome of the window medians of a pooled log profile (not centered per gene)."""
    bounds, chrom = lay.blocks(_block_size(lay), last_takes_rest=True)
    vals = np.empty(len(chrom))
    K.block_medians(np.log(Y + 0.5), bounds, vals)
    x = vals - np.mean(vals)
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
        self.ref_type = np.zeros(len(St), dtype=bool)  # types of reference cells: diploid prior, not fitted
        self.pinned = np.zeros(len(types), dtype=bool)  # cells held in the diploid clone during the fit
        self.dosage = np.ones(Yt.shape[1])
        self.neutral = int(np.argmin(np.abs(hmm.means)))
        par = data.win_parity[data.gene_win]
        self.masks = (par == 0, par == 1)

    def set_baseline(self, normal):
        """Baseline from the cells `normal`, and the expression deviations of the types relative to it."""
        self.Yr = self.data.pooled(normal.astype(np.int64), 2)[0][1]
        self.b = (self.Yr + 0.5) / (self.Yr + 0.5).sum()
        lr, pois = _log_ratio(self.Yt, self.St[:, None] * self.b[None, :], self.Yr[None, :])
        v = _scale(lr, pois, self.gene_var)
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
        when its expression differs from the baseline smoothly, a tumor type near 1, and a mixed type its tumor
        fraction. The fit uses the pooled profiles, not the current assignments, so it does not reinforce them.
        Types of reference cells are known to be diploid and keep the diploid prior: a fit of their nearly flat profile
        on a nearly flat clone can give that clone all the weight.
        """
        k = len(mu)
        pri = np.zeros((self.n_types, k))
        pri[:, 0] = 1.0
        if k > 1:
            lay = self.data.lay
            Mb = np.array([_block_medians(m, lay)[0] for m in mu[1:]]).T
            for t in np.flatnonzero(~self.ref_type):
                w = nnls(Mb, self.type_blocks[t])[0]
                w = w / max(1.0, w.sum())
                pri[t, 1:] = w
                pri[t, 0] = 1.0 - w.sum()
        pri = np.maximum(pri, _MIN_PRIOR)
        return pri / pri.sum(axis=1, keepdims=True)

    def profiles(self, assign, k):
        """Gene-level log ratios of the clones against their expectation (with the deviations of their cells' types)."""
        data = self.data
        self.Y = Y = data.pooled(assign, k)[0]  # the last one is that of the returned clones
        Skt = np.zeros((k, self.n_types))
        np.add.at(Skt, (assign, self.types), data.size)
        E = self.b[None, :] * (Skt @ np.exp(self.e))
        lr, pois = _log_ratio(Y, E, self.Yr[None, :])
        v = _scale(lr, pois, self.gene_var)
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

        With `report`, the spillover fraction is estimated for the cells of every type, and it and the model are kept
        (`spill`, `model`) for the output and for `_assign_all`.
        """
        data = self.data
        wb = data.wbounds[:-1]
        share = np.empty((self.n_types, len(mu), len(wb)))
        for t in range(self.n_types):
            expo = self.dosage[None, :] * mu + self.e[t][None, :]
            if self.normal_type[t]:  # normal type: its diploid expectation keeps its smooth deviation
                expo[0] = expo[0] + self.type_smooth[t]
            share[t] = np.add.reduceat(self.b[None, :] * np.exp(expo), wb, axis=1)
        use = np.ones(len(wb), dtype=np.bool_) if windows is None else np.asarray(windows, dtype=np.bool_)
        share = share * use
        share = np.ascontiguousarray(share / share.sum(axis=2, keepdims=True))
        size = data.size if windows is None else data.Yw @ use.astype(data.Yw.dtype)
        size = np.maximum(size.astype(np.float64), 1e-3)
        num, den = np.zeros((_PCA_CHUNKS, len(wb))), np.zeros((_PCA_CHUNKS, len(wb)))  # sums independent of n_jobs
        assign = np.ascontiguousarray(assign, dtype=np.int64)
        K.nb_moments(data.Yw, size, self.types, assign, share, use, num, den)
        phi = 1.0 / np.clip(num.sum(0) / np.maximum(den.sum(0), 1e-12), 1e-3, 1e2)
        # spillover: the pooled window shares of the aneuploid cells, mixed into the diploid expectation of cells of
        # normal types (marker-rich or reference types; segmentation merges with tumor cells in imaging data)
        tumor = (assign > 0).astype(data.Yw.dtype)
        g = (tumor @ data.Yw).astype(np.float64) * use if tumor.any() else np.zeros(len(wb))
        g = np.ascontiguousarray(g / max(g.sum(), 1e-12))
        spill_ok = self.normal_type & bool(tumor.any())
        with np.errstate(divide="ignore"):
            log_share = np.log(share)
        ll, spill = np.empty((len(assign), len(mu))), np.empty(len(assign))
        K.nb_loglik_types(
            data.Yw, size, self.types, share, log_share, phi, use, g, spill_ok, _MAX_SPILL, report, ll, spill
        )
        if report:
            self.model, self.spill = (share, log_share, phi, use, g, spill_ok), spill
        return ll

    def types_of(self, pcs_all, pcs_fit, ref):
        """Expression type of each cell: the nearest centre of the fitted cells' types in principal component space.

        With `ref`, reference cells take the nearest reference type, the other cells the nearest other type.
        """
        n_types = self.n_types
        centres = np.zeros((n_types, pcs_fit.shape[1]))
        np.add.at(centres, self.types, pcs_fit)
        centres /= np.maximum(np.bincount(self.types, minlength=n_types), 1)[:, None]
        if ref is None:
            return _nearest(pcs_all, centres).astype(np.int64)
        types = np.empty(len(pcs_all), dtype=np.int64)
        for mask, kinds in ((ref, self.ref_type), (~ref, ~self.ref_type)):
            ids = np.flatnonzero(kinds) if kinds.any() else np.arange(n_types)
            if mask.any():
                types[mask] = ids[_nearest(pcs_all[mask], centres[ids])]
        return types

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
        """EM of the clone assignments; returns the assignments, the posteriors, the llr and the log likelihoods."""
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
            ll[self.pinned, 1:] = -np.inf  # pinned cells stay in the diploid clone
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
        self.dosage = _dosage_weights(mu_whole, lr, v, pn_whole)  # of this fit, before the final likelihood
        ll = self.loglik(mu, assign, report=True)
        self.tau = self.temperature(mu, assign)
        log.debug("copy_number: likelihood temperature %.4g", self.tau)
        self.final = (mu_whole, pn_whole, st_whole, lr, v)
        resp, llr = self.posterior(ll, self.types)
        return assign, resp, llr, ll


def _assign_all(ems, source, cells, pcs_all, pcs_fit, ref):
    """Posteriors, llr, log likelihoods and types of all `cells` under each fitted model, in one pass over the counts.

    Each cell's type comes from `_EM.types_of`; its likelihood under each model comes from its window counts, read
    once per block, with the temperature and type priors of that fit. Sets each model's `spill`.
    """
    types = [em.types_of(pcs_all, pcs_fit, ref) for em in ems]
    ll = [np.empty((len(cells), em.model[0].shape[1])) for em in ems]
    spill = [np.empty(len(cells)) for _ in ems]
    data = ems[0].data
    for i0, i1, X, rws in _row_blocks(source, cells):
        Yw = np.empty((i1 - i0, data.Yw.shape[1]), dtype=np.float32)
        K.window_counts(X.data, X.indices, X.indptr, rws, data.lay.col_map, data.gene_win, Yw, n_kernel_chunks(i1 - i0))
        size = np.maximum(Yw.sum(axis=1, dtype=np.float64), 1e-3)
        for em, t, l, sp_ in zip(ems, types, ll, spill, strict=True):
            K.nb_loglik_types(Yw, size, t[i0:i1], *em.model, _MAX_SPILL, True, l[i0:i1], sp_[i0:i1])
    out = []
    for em, t, l, sp_ in zip(ems, types, ll, spill, strict=True):
        em.spill = sp_
        out.append((*em.posterior(l, t), l, t))
    return out


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
    """Median of the profile `lr` in each block of genes (see `_Layout.blocks`), and the chromosome of each block."""
    bounds, ch = lay.blocks(_block_size(lay) if size is None else size)
    out = np.empty(len(ch))
    K.block_medians(np.ascontiguousarray(lr, dtype=np.float64), bounds, out)
    return out, ch


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
        self._blocks: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]] = {}

    def blocks(self, size: int, last_takes_rest: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Bounds of the blocks of about `size` genes along each chromosome, and the chromosome of each block.

        A chromosome of `g` genes has `max(1, g // size)` blocks, split as `np.array_split` does, or with
        `last_takes_rest`, blocks of `size` genes and a last block that takes the rest. Computed once per rule and size.
        """
        if (size, last_takes_rest) not in self._blocks:
            bounds, ch = [0], []
            for c, g in enumerate(self.gs):
                g, nb = int(g), max(1, int(g) // size)
                if last_takes_rest:
                    sizes = np.full(nb, size)
                    sizes[-1] = g - size * (nb - 1)
                else:
                    sizes = np.full(nb, g // nb)
                    sizes[: g % nb] += 1
                bounds.extend(bounds[-1] + np.cumsum(sizes))
                ch.extend([c] * nb)
            self._blocks[size, last_takes_rest] = (np.array(bounds, dtype=np.int64), np.array(ch, dtype=np.int64))
        return self._blocks[size, last_takes_rest]

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


def _kmeans(X: np.ndarray, k: int, rng, n_starts: int = _KMEANS_STARTS) -> np.ndarray:
    """Cluster of each row of `X`: Lloyd's k-means on all rows, the best of `n_starts` random starts; no empty clusters.

    Each start begins at `k` random rows, and the starts run together: per iteration, one matrix product gives the
    distances of the rows to the centres of every start, and one pass finds the nearest centres and sums the rows. A
    cluster that empties restarts at the row farthest from its centre. The iterations stop when no row changes its
    cluster (at most `_KMEANS_ITER`), and the start with the smallest sum of squared distances is kept.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, d = X.shape
    k = min(k, n)
    C = np.concatenate([X[rng.choice(n, k, replace=False)] for _ in range(n_starts)])
    labels = np.full((n, n_starts), -1, dtype=np.int64)
    dist = np.empty((n, n_starts))
    for it in range(_KMEANS_ITER + 1):
        previous = labels.copy()
        sums, counts = np.zeros((_PCA_CHUNKS, len(C), d)), np.zeros((_PCA_CHUNKS, len(C)))
        c2 = (C**2).sum(axis=1)
        for i0 in range(0, n, _KMEANS_ROWS):
            rows = slice(i0, min(n, i0 + _KMEANS_ROWS))
            K.kmeans_assign(X[rows] @ C.T, X[rows], c2, k, labels[rows], dist[rows], sums, counts)
        size, total = counts.sum(axis=0), sums.sum(axis=0)
        empty = size == 0
        if it == _KMEANS_ITER or (not empty.any() and (labels == previous).all()):
            break
        C = np.where(empty[:, None], C, total / np.maximum(size, 1)[:, None])
        for j in np.flatnonzero(empty):
            far = int(np.argmax(dist[:, j // k]))
            C[j] = X[far]
            dist[far, j // k] = -1.0
    best = int(np.argmin(dist.sum(axis=0)))
    return np.unique(labels[:, best], return_inverse=True)[1]


def _kmeans_split(X: np.ndarray, k: int, rng, ref: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
    """Expression types fitted separately on the reference cells (`ref`) and on the other cells.

    The `k` types are shared out by the share of reference cells among all cells (`share`; the rows can be a sample
    that holds every reference cell), with at least one each; reference types come first. Returns the type of each
    row and which types are reference types.
    """
    n_other = len(ref) - int(ref.sum())
    k_ref = k if n_other == 0 else min(max(1, round(k * share)), max(1, k - 1))
    types = np.empty(len(ref), dtype=np.int64)
    types[ref] = _kmeans(X[ref], k_ref, rng)
    k1 = int(types[ref].max()) + 1
    if n_other:
        types[~ref] = k1 + _kmeans(X[~ref], max(1, k - k_ref), rng)
    return types, np.arange(int(types.max()) + 1) < k1


def _nearest(X: np.ndarray, centres: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    c2 = (centres**2).sum(axis=1)
    return np.concatenate(
        [np.argmin(c2[None, :] - 2 * X[i : i + chunk] @ centres.T, axis=1) for i in range(0, len(X), chunk)]
    )
