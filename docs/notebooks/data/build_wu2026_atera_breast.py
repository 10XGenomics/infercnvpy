r"""Build the example dataset `infercnvpy.datasets.wu2026_atera_breast`.

The script documents how the file on exampledata.scverse.org was made. It is not run by CI, and it needs
intermediate files that are not public:

`--source`
    The 10x Atera preview section of FFPE breast cancer as one h5ad (spatialdata-db entry jCLxdY1tWKmbfWOB0001),
    holding only the cells with a cell type (the "cells with an annotation only" in `uns["source"]`), with the vendor segmentation counts in `layers["counts"]`, gene positions in `var`, cell centroids in
    `obsm["spatial"]` (µm), and the columns `cell_type` (expression clustering) and `pred_state` (epithelial states
    transferred by logistic regression from the annotated demo cells of Wu et al. 2026) in `obs`.
`--raw`
    The raw counts of the catalog export of the same section, to check that the counts are unchanged.
`--regions`
    Per cell and dissected region of Wu et al. 2026: the signed distance (µm) to the region outline and the share of
    perturbed placements of the outlines that put the cell inside. The outlines of Fig. 3g were traced from the
    preprint and placed in Atera coordinates with the authors' dashed box and the scale bar of Fig. 1.
`--arms-raw`, `--arms-centred`
    Arm-level log2 copy ratios per WGS run, from the WGS reads of SRA PRJNA1509685: GRCh38 primary assembly,
    bwa-mem2, duplicates marked and skipped, CNVkit 0.9.13 in WGS mode against a flat reference, 5 kb bins, CBS,
    log2 clipped to [-2, 2], and the length-weighted mean of the segments per arm. The centered table subtracts the
    mean of the runs of regions 1 and 2.

The 10x data are licensed under CC BY 4.0. The arm table and the labels added here are by the infercnvpy authors, also
under CC BY 4.0. The arm table comes from the WGS reads of SRA PRJNA1509685, and the region labels from outlines
traced from a figure of Wu et al. 2026.

Usage::

    python build_wu2026_atera_breast.py --source atera.h5ad --raw atera_raw.h5ad --regions region_cells.tsv.gz \
        --arms-raw qc_arms.cnvkit.raw.tsv --arms-centred qc_arms.cnvkit.centred.tsv --out wu2026_atera_breast.h5ad
"""

import argparse
import shutil
import subprocess

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import sparse_dataset

# clone groups of the dissected regions, from the region colors of Fig. 3g of Wu et al. 2026
GROUPS = {"A": ["10"], "C": ["3", "5", "6", "7", "8", "9", "15", "16"], "N": ["1", "2"]}
REGION_GROUP = {**dict.fromkeys(GROUPS["C"], "C"), "10": "A", "1,2": "N"}  # regions 1 and 2 share one outline

DESCRIPTION = (
    "Arm-level copy number of the laser-dissected DCIS regions of Wu et al. 2026 (WGS, SRA PRJNA1509685). "
    "Pipeline: GRCh38 primary assembly, bwa-mem2, duplicates marked and skipped, CNVkit 0.9.13 in WGS mode, "
    "tumour-only with a flat reference, 5 kb bins, CBS, log2 clipped to [-2, 2]. Arm value = length-weighted mean "
    "segment log2 per arm (centromeres GRCh38). 'log2' is relative to the flat reference; 'log2_centred_on_N' "
    "subtracts the mean of runs cap_1_DL and cap_2_DL (regions 1 and 2), as the paper's tree script does. "
    "Clone groups follow the region colours of Fig. 3g: A = region 10, C = regions 3, 5, 6, 7, 8, 9, 15, 16, "
    "N = regions 1 and 2 (one shared outline). "
    "Region mapping: the Fig. 3g outlines were digitised from the bioRxiv figure and placed in Atera coordinates "
    "with the authors' dashed box and the Fig. 1 scale bar. The placement error is about 0.5 to 1 mm, while regions "
    "are 0.6 to 1.5 mm across, so the mapping is approximate and only usable at clone-group level, not per region. "
    "obs['wgs_region'] is the region whose outline contains the cell under the best placement; "
    "obs['wgs_region_ensemble_frac'] is the share of 108 perturbed placements that also put the cell inside. "
    "Regions 1 and 2 (group N) are contaminated: their WGS shows +0.23 on 1q and +0.15 on 17q, and the Atera cells "
    "inside the 1,2 outline show the tumour profile (12q14-15 and 17q gains), so group N is not a clean normal "
    "control. Clone group D (regions 11 to 14) lies mostly outside the Atera field of view and is not included."
)  # kept as in the uploaded file, which uses British spelling

ATTRIBUTION = (
    "Preview Data: Atera In Situ Gene Expression, FFPE Human Breast Cancer. Atera Onboard Analysis "
    "(development version), 10x Genomics (2026, June 2). Licensed under CC BY 4.0. "
    "https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer"
)


def main() -> None:
    """Write the h5ad."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["--source", "--raw", "--regions", "--arms-raw", "--arms-centred", "--out"]:
        parser.add_argument(name, required=True)
    parser.add_argument("--window", nargs=4, type=float, default=[0, 3750, 750, 7000], metavar=("X0", "X1", "Y0", "Y1"))
    args = parser.parse_args()
    x0, x1, y0, y1 = args.window

    # cells in the window, and their raw counts as integers
    with h5py.File(args.source, "r") as f:
        obs_all = ad.io.read_elem(f["obs"])
        xy = f["obsm/spatial"][:]
        sel = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1))
        counts = sp.csr_matrix(sparse_dataset(f["layers/counts"])[sel])
        var = ad.io.read_elem(f["var"])
    assert np.all(counts.data == np.round(counts.data)), "counts are not integers"
    counts = sp.csr_matrix((counts.data.astype(np.int32), counts.indices, counts.indptr), shape=counts.shape)
    with h5py.File(args.raw, "r") as f:  # the counts must equal those of the catalog export
        assert ad.io.read_elem(f["obs"]).index.equals(obs_all.index)
        check = sp.csr_matrix(sparse_dataset(f["X"])[sel[:500]])
    assert abs(check - counts[:500]).sum() == 0, "counts differ from the raw export"

    obs = obs_all.iloc[sel][["cell_type", "pred_state"]].copy()
    for column in obs:
        obs[column] = obs[column].cat.remove_unused_categories()

    # region and clone group of each cell: the region whose outline contains it under the best placement
    regions = pd.read_csv(args.regions, sep="\t", dtype={"region": str})
    regions = regions[regions["cell_id"].isin(obs.index)]
    inside = regions[regions["signed_dist_um"] >= 0].sort_values("signed_dist_um", ascending=False)
    inside = inside.drop_duplicates("cell_id").set_index("cell_id")
    obs["wgs_region"] = pd.Categorical(inside["region"].reindex(obs.index))
    obs["wgs_clone_group"] = pd.Categorical(obs["wgs_region"].astype(object).map(REGION_GROUP), categories=list(GROUPS))
    obs["wgs_region_ensemble_frac"] = inside["ensemble_frac"].reindex(obs.index).astype("float32")

    var = var[["gene_ids", "chromosome", "start", "end"]].copy()
    var["chromosome"] = var["chromosome"].astype(str)

    # WGS arm table per clone group, the mean over the runs of its regions
    raw = pd.read_csv(args.arms_raw, sep="\t", index_col=0)
    centred = pd.read_csv(args.arms_centred, sep="\t", index_col=0)
    rows = []
    for group, group_regions in GROUPS.items():
        runs = [f"cap_{r}_DL" for r in group_regions]
        for arm in raw.columns:
            rows.append(
                {
                    "clone_group": group,
                    "arm": arm,
                    "kind": "focal" if "_" in arm or "-" in arm else "arm",
                    "log2": float(raw.loc[runs, arm].mean()),
                    "log2_centred_on_N": float(centred.loc[runs, arm].mean()),
                    "log2_min_region": float(raw.loc[runs, arm].min()),
                    "log2_max_region": float(raw.loc[runs, arm].max()),
                    "n_regions": len(runs),
                    "wgs_runs": ",".join(runs),
                }
            )
    arms = pd.DataFrame(rows)
    by_run = raw.reset_index().melt(id_vars=raw.index.name or "index", var_name="arm", value_name="log2")
    by_run.columns = ["wgs_run", "arm", "log2"]

    in_window = obs["wgs_region"].value_counts()
    uns = {
        "wgs_truth": {
            "arms": arms,
            "arms_by_wgs_run": by_run,
            "description": DESCRIPTION,
            "source": "atera-dcis-gt pipeline, results/qc_arms.cnvkit.raw.tsv and qc_arms.cnvkit.centred.tsv (2026-11-15)",
            "pipeline": "GRCh38 (no alt contigs), bwa-mem2, samtools markdup, CNVkit 0.9.13 (WGS, flat reference, CBS)",
            "region_mapping": "digitised from Wu et al. 2026 Fig. 3g; approximate, clone-group level only",
            "clone_groups": GROUPS,
            "regions_in_window": {str(k): int(v) for k, v in in_window.items() if v > 0},
        },
        "source": {
            "dataset": {
                "title": "Preview Data: Atera In Situ Gene Expression, FFPE Human Breast Cancer",
                "provider": "10x Genomics",
                "url": "https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer",
                "licence": "CC BY 4.0",
                "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": ATTRIBUTION,
                "citation_guidelines": "https://www.10xgenomics.com/support/software/cell-ranger/latest/miscellaneous/cr-citations",
                "preview_notice": (
                    "This preview dataset was generated using development versions of the pre-designed panel, user "
                    "guides, and analysis software. It may be replaced with data generated using the final workflow "
                    "in the future."
                ),
                "tissue": "FFPE breast cancer, DCIS, G3 (T1c N0 M0), from BioIVT",
                "full_dataset": "18,028 genes; 170,057 cells; median 2,116 transcripts per cell",
                "spatialdata_db_uid": "jCLxdY1tWKmbfWOB0001",
                "spatialdata_db_key": "10x:atera-wta-ffpe-human-breast-cancer",
                "changes": (
                    f"Subset to a window of the section (x {x0:.0f} to {x1:.0f} um, y from {y0:.0f} um), cells with "
                    "an annotation only; vendor segmentation counts kept as integers; cell types from expression "
                    "clustering (infercnvpy authors); pred_state transferred from the Wu et al. 2026 demo cells by "
                    "logistic regression; WGS region labels and arm table added."
                ),
            },
            "preprint": {
                "citation": "Wu et al. 2026, bioRxiv",
                "doi": "10.64898/2026.08.03.741871",
                "url": "https://doi.org/10.64898/2026.08.03.741871",
            },
            "wgs": {
                "accession": "PRJNA1509685",
                "url": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1509685",
                "runs": "SRR40081004 to SRR40081019 (cap_1_DL to cap_16_DL)",
            },
        },
        "window_um": {"x": [x0, x1], "y": [y0, float(xy[sel, 1].max())]},
    }
    adata = ad.AnnData(X=counts, obs=obs, var=var, obsm={"spatial": xy[sel].astype(np.float32)}, uns=uns)
    adata.obs_names.name = None
    adata.write_h5ad(args.out, compression="gzip", compression_opts=9)

    # the shuffle filter makes the column indices compress about twice as well; gzip and shuffle are in every HDF5
    with h5py.File(args.out, "r+") as f:
        for key in ["data", "indices", "indptr"]:
            values = f["X"][key][:]
            del f["X"][key]
            chunks = (min(len(values), 1 << 20),)
            f["X"].create_dataset(key, data=values, chunks=chunks, compression="gzip", compression_opts=9, shuffle=True)
    if shutil.which("h5repack"):  # reclaims the space of the deleted datasets
        subprocess.run(["h5repack", args.out, f"{args.out}.repack"], check=True)
        shutil.move(f"{args.out}.repack", args.out)
    print(adata)


if __name__ == "__main__":
    main()
