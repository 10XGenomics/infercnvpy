"""Reading the CNV matrix for :func:`infercnvpy.pl.chromosome_heatmap`, one pixel row per run of cells.

The cells are sorted by group and every group is split into contiguous runs of cells, one per pixel row
(`pixel_rows`). A pixel row then shows either one cell of its run (`sampled_rows`: the cell in the middle, which is
what matplotlib's nearest-neighbor downscaling shows when every cell is drawn) or the mean of the run (`binned_sums`:
`_bin_sums_csr` / `_bin_sums_dense` add up the rows of each run straight from the matrix, CSR without densifying).
A figure is ~1000 pixels tall, so the result is a small dense matrix whatever the number of cells.

The matrix is read in row blocks (`infercnvpy.tl._infercnv._Rows`): an in-memory matrix as one block, a dask array
block by block and a backed matrix `CHUNKSIZE` rows at a time. `binned_sums` and `sampled_rows` pass every block also
to `also`, readers such as `ColumnRanges` and `Gram`, so that their statistics take no pass of their own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp

from infercnvpy._numba import njit_parallel, prange
from infercnvpy.tl._infercnv import _Rows

CHUNKSIZE = 10_000  # rows per block of a backed matrix


@njit_parallel
def _bin_sums_csr(indptr, indices, data, order, bounds, bins, sums):  # pragma: no cover
    """Add rows `order[bounds[j]:bounds[j + 1]]` to `sums[bins[j]]`."""
    for j in prange(len(bounds) - 1):
        row = sums[bins[j]]
        for k in range(bounds[j], bounds[j + 1]):
            i = order[k]
            for p in range(indptr[i], indptr[i + 1]):
                row[indices[p]] += data[p]


@njit_parallel
def _bin_sums_dense(X, order, bounds, bins, sums):  # pragma: no cover
    """Like `_bin_sums_csr` for a dense matrix."""
    n_cols = X.shape[1]
    for j in prange(len(bounds) - 1):
        row = sums[bins[j]]
        for k in range(bounds[j], bounds[j + 1]):
            x = X[order[k]]
            for c in range(n_cols):
                row[c] += x[c]


def _rows_per_group(counts: np.ndarray, n_rows: int) -> np.ndarray:
    """Split `n_rows` pixel rows over groups in proportion to `counts` (largest remainder).

    A non-empty group gets at least one row, and no group more rows than cells.
    """
    exact = counts * (n_rows / counts.sum())
    rows = np.floor(exact).astype(np.int64)
    rest = n_rows - rows.sum()
    rows[np.argsort(rows - exact, kind="stable")[:rest]] += 1
    return np.clip(rows, counts > 0, counts)


def pixel_rows(codes: np.ndarray, n_groups: int, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order the cells by group (`codes`, -1 is left out) and cut each group into runs of cells, one per pixel row.

    Within a group, the cells keep their order in `codes`. (:func:`scanpy.pl.heatmap` sorts with pandas' unstable
    default sort, so its order within a group is arbitrary.) Returns the cell order, the start of each run in that order (plus the end), and the
    number of rows per group.
    """
    keep = np.flatnonzero(codes >= 0)
    order = keep[codes[keep].argsort(kind="stable")]
    counts = np.bincount(codes[order], minlength=n_groups)
    rows = _rows_per_group(counts, min(n_rows, len(order)))
    starts = np.concatenate([[0], np.cumsum(counts)])
    bounds = [starts[g] + np.arange(rows[g]) * counts[g] // rows[g] for g in range(n_groups) if rows[g]]
    return order, np.concatenate([*bounds, [len(order)]]).astype(np.int64), rows


def _blocks(X: Any) -> _Rows:
    return _Rows(X, max(X.shape[0], 1) if isinstance(X, np.ndarray) or sp.issparse(X) else CHUNKSIZE)


def binned_sums(X: Any, order: np.ndarray, bounds: np.ndarray, *also: Callable[..., None]) -> np.ndarray:
    """Sum of the rows of `X` in each run `order[bounds[b]:bounds[b + 1]]` (float64, runs x columns).

    `X` is a numpy array, scipy sparse matrix, dask array or backed matrix.
    """
    n_bins = len(bounds) - 1
    sums = np.zeros((n_bins, X.shape[1]), dtype=np.float64)
    row_bin = np.full(X.shape[0], -1, dtype=np.int64)
    row_bin[order] = np.repeat(np.arange(n_bins), np.diff(bounds))
    for Xb, row0, s, e in _blocks(X).read_ahead():
        for read in also:
            read(Xb, row0, s, e)
        local = np.flatnonzero(row_bin[s:e] >= 0)
        b = row_bin[s + local]
        by_bin = np.argsort(b, kind="stable")
        bins, first = np.unique(b[by_bin], return_index=True)
        o = (row0 + local[by_bin]).astype(np.int64)
        bnd = np.append(first, len(o)).astype(np.int64)
        if sp.issparse(Xb):
            _bin_sums_csr(Xb.indptr, Xb.indices, Xb.data, o, bnd, bins, sums)
        else:
            _bin_sums_dense(Xb, o, bnd, bins, sums)
    return sums


def _slices(Xb: Any, row0: int, s: int, e: int):
    """The rows of a block, `CHUNKSIZE` at a time (none for an empty block)."""
    for r in range(row0, row0 + e - s, CHUNKSIZE):
        yield Xb[r : min(r + CHUNKSIZE, row0 + e - s)]


class ColumnRanges:
    """Minimum (`lo`) and maximum (`hi`) of each column over the rows read (with the zeros of a sparse matrix)."""

    def __init__(self, n_cols: int):
        self.lo, self.hi = np.full(n_cols, np.inf), np.full(n_cols, -np.inf)

    def __call__(self, Xb: Any, row0: int, s: int, e: int) -> None:
        for rows in _slices(Xb, row0, s, e):  # NaNs are ignored
            if sp.issparse(rows):
                b_lo, b_hi = rows.nanmin(axis=0).toarray().ravel(), rows.nanmax(axis=0).toarray().ravel()
            else:
                b_lo, b_hi = np.fmin.reduce(rows, axis=0), np.fmax.reduce(rows, axis=0)
            np.fmin(self.lo, b_lo, out=self.lo)
            np.fmax(self.hi, b_hi, out=self.hi)


class Gram:
    """`X.T @ X` (`G`) and the column sums (`sums`) of the rows read, in float64, with BLAS on dense slices."""

    def __init__(self, n_cols: int):
        self.G, self.sums = np.zeros((n_cols, n_cols)), np.zeros(n_cols)

    def __call__(self, Xb: Any, row0: int, s: int, e: int) -> None:
        for rows in _slices(Xb, row0, s, e):
            rows = (rows.toarray() if sp.issparse(rows) else np.asarray(rows)).astype(np.float64, copy=False)
            self.G += rows.T @ rows
            self.sums += rows.sum(axis=0)


def sampled_rows(X: Any, picks: np.ndarray, *also: Callable[..., None]) -> tuple[np.ndarray, float, float]:
    """Rows `picks` of `X` as a dense float64 array, and the min and max of all values of `X`.

    For sparse input, the min and max are those of the stored values (as before 0.7.0). NaNs are ignored.
    """
    out = np.empty((len(picks), X.shape[1]), dtype=np.float64)
    sorted_picks = np.argsort(picks, kind="stable")
    ranges = []
    for Xb, row0, s, e in _blocks(X).read_ahead():
        for read in also:
            read(Xb, row0, s, e)
        values = Xb.data if sp.issparse(Xb) else Xb[row0 : row0 + e - s]
        if values.size:
            ranges.append((np.nanmin(values), np.nanmax(values)))
        lo, hi = np.searchsorted(picks[sorted_picks], [s, e])
        dest = sorted_picks[lo:hi]
        rows = Xb[row0 + picks[dest] - s]
        out[dest] = rows.toarray() if sp.issparse(rows) else rows
    if not ranges:
        return out, np.nan, np.nan
    return out, float(min(r[0] for r in ranges)), float(max(r[1] for r in ranges))
