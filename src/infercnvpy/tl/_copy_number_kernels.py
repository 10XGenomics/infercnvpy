"""numba kernels of :func:`infercnvpy.tl.copy_number`.

Cell kernels work on one row block of the count matrix (CSR) at a time and map input columns to genes in genomic
order through `col_map` (-1 for dropped genes). Profile kernels work on pooled profiles (clones or clusters x genes).
"""

from __future__ import annotations

import math

import numpy as np

from infercnvpy._numba import njit, njit_parallel, prange


@njit_parallel
def window_counts(data, indices, indptr, rws, col_map, gene_win, out, n_chunks):
    """Counts of the rows `rws` summed per window of genes: `out[r, w]`; dropped genes are skipped."""
    n_rows = out.shape[0]
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(ch * step, min(n_rows, (ch + 1) * step)):
            for w in range(out.shape[1]):
                out[r, w] = 0.0
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                g = col_map[indices[k]]
                if g >= 0:
                    out[r, gene_win[g]] += data[k]


@njit
def _t_logpdf(y, m, v, nu):
    z = (y - m) * (y - m) / (nu * v)
    return -0.5 * math.log(v) - 0.5 * (nu + 1.0) * math.log1p(z)


@njit_parallel
def hmm_posteriors(Y, V, stay, means, log_prior, nu, post_mean, post_neutral, state):
    """Forward-backward of a copy number HMM on each row of `Y` (profiles x genes, genomic order).

    Emissions are Student t with `nu` degrees of freedom, location `means[s]` and scale `V[p, i]` (variance). Between
    gene `i - 1` and gene `i` the chain keeps its state with probability `stay[i]`, and otherwise draws a new state from
    the prior (`exp(log_prior)`); `stay[i] = 0` starts a chromosome. Writes the posterior mean of the state means, the
    posterior probability of the neutral state (`means == 0`) and the most probable state per gene.
    """
    P, G = Y.shape
    S = means.shape[0]
    prior = np.exp(log_prior)
    neutral = 0
    for s in range(S):
        if means[s] == 0.0:
            neutral = s
    for p in prange(P):
        emis = np.empty((G, S))
        alpha = np.empty((G, S))
        beta = np.empty(S)
        c = np.empty(G)
        for i in range(G):
            mx = -np.inf
            for s in range(S):
                emis[i, s] = _t_logpdf(Y[p, i], means[s], V[p, i], nu)
                mx = max(mx, emis[i, s])
            for s in range(S):
                emis[i, s] = math.exp(emis[i, s] - mx)
            q = stay[i]
            tot = 0.0
            if i > 0:
                for s in range(S):
                    tot += alpha[i - 1, s]
            norm = 0.0
            for s in range(S):
                pred = prior[s] if i == 0 else q * alpha[i - 1, s] + (1.0 - q) * prior[s] * tot
                alpha[i, s] = emis[i, s] * pred
                norm += alpha[i, s]
            c[i] = norm
            for s in range(S):
                alpha[i, s] /= norm
        # backward pass with the same scaling; gamma = alpha * beta
        for s in range(S):
            beta[s] = 1.0
        for i in range(G - 1, -1, -1):
            pm = 0.0
            pn = 0.0
            best = 0
            bmax = -1.0
            tot = 0.0
            for s in range(S):
                gm = alpha[i, s] * beta[s]
                tot += gm
            for s in range(S):
                gm = alpha[i, s] * beta[s] / tot
                pm += gm * means[s]
                if s == neutral:
                    pn = gm
                if gm > bmax:
                    bmax = gm
                    best = s
            post_mean[p, i] = pm
            post_neutral[p, i] = pn
            state[p, i] = best
            if i > 0:
                q = stay[i]
                mix = 0.0
                for s in range(S):
                    mix += prior[s] * emis[i, s] * beta[s]
                for s in range(S):
                    beta[s] = (q * emis[i, s] * beta[s] + (1.0 - q) * mix) / c[i]


@njit_parallel
def pooled_counts(data, indices, indptr, rws, col_map, labels, out):
    """Add the counts of each cell (row) to the row of its label: `out[chunk, label, gene]`, one slab per chunk.

    Each chunk of cells writes its own slab, so there are no write conflicts; the caller sums the slabs.
    """
    n_chunks = out.shape[0]
    n_rows = labels.shape[0]
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(ch * step, min(n_rows, (ch + 1) * step)):
            lab = labels[r]
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                g = col_map[indices[k]]
                if g >= 0:
                    out[ch, lab, g] += data[k]


@njit
def _spillover(y, a, g, use):
    """Spillover fraction of one cell.

    The fraction `s` in [0, 1] of a cell's counts `y` that maximises the Poisson likelihood of the window shares
    `(1 - s) a + s g` (`a`: the cell's own diploid shares, `g`: the pooled shares of aneuploid cells). The likelihood
    is concave in `s`, so its derivative has at most one root: Newton steps kept inside a bracket that bisection
    narrows when a step leaves it.
    """
    lo, hi, s = 0.0, 1.0, 0.0
    for it in range(60):
        f1, f2 = 0.0, 0.0
        for w in range(len(y)):
            if use[w] and y[w] > 0:
                d = g[w] - a[w]
                q = a[w] + s * d
                f1 += y[w] * d / q
                f2 -= y[w] * d * d / (q * q)
        if it == 0 and f1 <= 0.0:
            return 0.0
        if f1 > 0.0:
            lo = s
        else:
            hi = s
        new = s - f1 / f2 if f2 < 0.0 else -1.0
        if not lo < new < hi:
            new = 0.5 * (lo + hi)
        if abs(new - s) < 1e-6 or hi - lo < 1e-6:
            return new
        s = new
    return s


@njit
def _cell_loglik(y, size, t, share, log_share, phi, use, g, spill_ok, smax, report, out_row):
    """Log likelihoods of one cell under each clone; returns the cell's spillover fraction (0 if not estimated).

    For a cell of a type with `spill_ok`, the diploid expectation (clone 0) mixes its own shares with the shares of
    the aneuploid cells, with the cell's spillover fraction capped at `smax`. With `report`, the fraction is estimated
    for cells of every type (and used only for those with `spill_ok`).
    """
    K, W = share.shape[1], share.shape[2]
    s_raw = _spillover(y, share[t, 0], g, use) if (spill_ok[t] or report) else 0.0
    s = min(s_raw, smax) if spill_ok[t] else 0.0
    log_size = math.log(size)
    for k in range(K):
        mixed = k == 0 and s > 0.0
        acc = 0.0
        for w in range(W):
            if use[w]:
                p = share[t, k, w]
                if mixed:
                    p = (1.0 - s) * p + s * g[w]
                if y[w] > 0:
                    acc += y[w] * (log_size + (math.log(p) if mixed else log_share[t, k, w]))
                acc -= (y[w] + phi[w]) * math.log(size * p + phi[w])
        out_row[k] = acc
    return s_raw


@njit_parallel
def nb_loglik_types(Y, size, types, share, log_share, phi, use, g, spill_ok, smax, report, out, spill):
    """Negative binomial log likelihood (up to terms that do not depend on the clone) of each cell under each clone.

    `Y` is cells x windows (counts) and `size` each cell's count in the windows marked by `use`. `share` (and its
    logarithm `log_share`) is types x clones x windows, the expected share of the cell's counts in each window (summing
    to 1 over the used windows), and `phi` the dispersion of each window (variance `m + m^2 / phi`). Writes
    `out[cell, clone]`. Cells of types with `spill_ok` have a diploid expectation that mixes in the shares `g` of the
    aneuploid cells (see `_cell_loglik`); `spill` receives each cell's spillover fraction.
    """
    n = Y.shape[0]
    for r in prange(n):
        spill[r] = _cell_loglik(Y[r], size[r], types[r], share, log_share, phi, use, g, spill_ok, smax, report, out[r])


@njit_parallel
def nb_moments(Y, size, types, assign, share, use, num, den):
    """Per window, sums of `(y - m)^2 - m` and `m^2` over cells (`m` from the cell's type and clone).

    They give the method-of-moments dispersion; one slab per chunk of cells (`num`, `den`: chunks x windows).
    """
    n, W = Y.shape
    n_chunks = num.shape[0]
    step = (n + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(ch * step, min(n, (ch + 1) * step)):
            t, k = types[r], assign[r]
            for w in range(W):
                if use[w]:
                    m = size[r] * share[t, k, w]
                    d = Y[r, w] - m
                    num[ch, w] += d * d - m
                    den[ch, w] += m * m


@njit_parallel
def row_stats(data, indices, indptr, row0, min_genes, ndet, det):
    """Detected genes of each cell (`ndet`), and per input column the cells that detect it (`det`).

    `det` counts only the cells with at least `min_genes` detected genes (chunks x columns, one slab per chunk).
    """
    n_rows = ndet.shape[0]
    n_chunks = det.shape[0]
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(ch * step, min(n_rows, (ch + 1) * step)):
            k0, k1 = indptr[row0 + r], indptr[row0 + r + 1]
            cnt = 0
            for k in range(k0, k1):
                if data[k] > 0:
                    cnt += 1
            ndet[r] = cnt
            if cnt >= min_genes:
                for k in range(k0, k1):
                    if data[k] > 0:
                        det[ch, indices[k]] += 1


@njit_parallel
def project_rows(data, indices, indptr, rws, col_map, hvg_of_gene, loadings, out, n_chunks):
    """`out[r] = sum over the cell's highly variable genes of log1p(1e4 x count / size) x loadings[hvg]`.

    `size` is the cell's count over the used genes (`col_map >= 0`).
    """
    n_rows, P = out.shape
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(ch * step, min(n_rows, (ch + 1) * step)):
            for p in range(P):
                out[r, p] = 0.0
            size = 0.0
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                if col_map[indices[k]] >= 0:
                    size += data[k]
            f = 1e4 / max(size, 1.0)
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                g = col_map[indices[k]]
                if g >= 0:
                    h = hvg_of_gene[g]
                    if h >= 0:
                        val = math.log1p(data[k] * f)
                        for p in range(P):
                            out[r, p] += val * loadings[h, p]


@njit_parallel
def assign_loglik(
    data,
    indices,
    indptr,
    rws,
    col_map,
    gene_win,
    types,
    share,
    log_share,
    phi,
    use,
    g,
    spill_ok,
    smax,
    out,
    spill,
    n_chunks,
):
    """Negative binomial log likelihood of each cell under each clone, straight from the CSR counts.

    The cell's counts are summed per window of genes in a scratch row (one per chunk of cells), so no cells x windows
    matrix is stored. `spill` receives each cell's spillover fraction, estimated for every cell (as in
    `nb_loglik_types` with `report`).
    """
    n_rows = out.shape[0]
    W = share.shape[2]
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        buf = np.zeros(W)
        for r in range(ch * step, min(n_rows, (ch + 1) * step)):
            for w in range(W):
                buf[w] = 0.0
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                gi = col_map[indices[k]]
                if gi >= 0:
                    buf[gene_win[gi]] += data[k]
            s = 0.0
            for w in range(W):
                if use[w]:
                    s += buf[w]
            spill[r] = _cell_loglik(
                buf, max(s, 1e-3), types[r], share, log_share, phi, use, g, spill_ok, smax, True, out[r]
            )


@njit_parallel
def mix_rows(resp, prof, rows, cols, out):
    """`out[rows[r], cols[j]] = sum_k resp[r, k] * prof[k, j]`: each cell's posterior mixture of the clone profiles."""
    n, K = resp.shape
    J = prof.shape[1]
    for r in prange(n):
        i = rows[r]
        for j in range(J):
            acc = 0.0
            for k in range(K):
                acc += resp[r, k] * prof[k, j]
            out[i, cols[j]] = acc
