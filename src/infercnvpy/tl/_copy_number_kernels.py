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
def _spillover(y, a, g):
    """Spillover fraction of one cell.

    The fraction `s` in [0, 1] of a cell's counts `y` that maximises the Poisson likelihood of the window shares
    `(1 - s) a + s g` (`a`: the cell's own diploid shares, `g`: the pooled shares of aneuploid cells). The likelihood
    is concave in `s`, so its derivative has at most one root: Newton steps kept inside a bracket that bisection
    narrows when a step leaves it.
    """
    lo, hi, s = 0.0, 1.0, 0.0
    for it in range(60):
        f1, f2 = 0.0, 0.0
        for w in range(y.shape[0]):
            if y[w] > 0:
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


# coefficients of fdlibm's log (e_log.c)
_LN2_HI, _LN2_LO = 6.93147180369123816490e-01, 1.90821492927058770002e-10
_LG1, _LG2, _LG3, _LG4 = (
    6.666666666666735130e-01,
    3.999999999940941908e-01,
    2.857142874366239149e-01,
    2.222219843214978396e-01,
)
_LG5, _LG6, _LG7 = 1.818357216161805012e-01, 1.531383769920937332e-01, 1.479819860511658591e-01


@njit
def _log_rows(x, out, expo, mant):
    """`out = log(x)`, as fdlibm computes it (within 1 ulp of `math.log`).

    The loops have no branches, so the compiler runs them on several values at once (SIMD); `math.log` in a loop
    doesn't allow that. `expo` (int64) and `mant` (float64) are scratch rows. Values that are not positive normal
    numbers (0, subnormal, infinite, NaN) go through `math.log`.
    """
    n = x.shape[0]
    xi = x.view(np.int64)
    mi = mant.view(np.int64)
    odd = 0
    for i in range(n):
        hx = xi[i]
        mbits = hx & 0x000FFFFFFFFFFFFF
        up = (mbits + 0x95F6198CC4333) >> 52  # 1 if the mantissa is at least sqrt(2): halve it, raise the exponent
        expo[i] = ((hx >> 52) & 0x7FF) - 1023 + up
        mi[i] = mbits | ((1023 - up) << 52)
        odd |= (hx - 0x0010000000000000) | (0x7FEFFFFFFFFFFFFF - hx)  # negative for 0, subnormal, negative, inf, NaN
    for i in range(n):
        f = mant[i] - 1.0
        k = float(expo[i])
        s = f / (2.0 + f)
        z = s * s
        w = z * z
        r = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7))) + w * (_LG2 + w * (_LG4 + w * _LG6))
        hfsq = 0.5 * f * f
        out[i] = k * _LN2_HI - ((hfsq - (s * (hfsq + r) + k * _LN2_LO)) - f)
    if odd < 0:
        for i in range(n):
            if not (2.2250738585072014e-308 <= x[i] < np.inf):
                out[i] = math.log(x[i]) if x[i] >= 0 else np.nan


@njit
def _cell_loglik(y, size, t, share, log_share, phi, g, spill_ok, smax, report, out_row, buf):
    """Log likelihoods of one cell under each clone; returns the cell's spillover fraction (0 if not estimated).

    `y`, `share`, `log_share`, `phi` and `g` hold the used windows only (see `_used_windows`). For a cell of a type
    with `spill_ok`, the diploid expectation (clone 0) mixes its own shares with the shares of the aneuploid cells,
    with the cell's spillover fraction capped at `smax`. With `report`, the fraction is estimated for cells of every
    type (and used only for those with `spill_ok`).

    Per clone, the arguments of the logarithms are written to a row and their logarithms taken in one pass
    (`_log_rows`); the sum follows the windows in order. `buf` (6 rows) is scratch.
    """
    K, W = share.shape[1], share.shape[2]
    xb, pb, mant, lx, lp = buf[0], buf[1], buf[2], buf[3], buf[4]
    expo = buf[5].view(np.int64)
    s_raw = _spillover(y, share[t, 0], g) if (spill_ok[t] or report) else 0.0
    s = min(s_raw, smax) if spill_ok[t] else 0.0
    log_size = math.log(size)
    for k in range(K):
        mixed = k == 0 and s > 0.0
        st = share[t, k]
        if mixed:
            for w in range(W):
                p = (1.0 - s) * st[w] + s * g[w]
                pb[w] = p
                xb[w] = size * p + phi[w]
            _log_rows(pb, lp, expo, mant)
        else:
            for w in range(W):
                xb[w] = size * st[w] + phi[w]
        _log_rows(xb, lx, expo, mant)
        lst = log_share[t, k]
        acc = 0.0
        for w in range(W):
            if y[w] > 0:
                acc += y[w] * (log_size + (lp[w] if mixed else lst[w]))
            acc -= (y[w] + phi[w]) * lx[w]
        out_row[k] = acc
    return s_raw


@njit
def _used_windows(use, share, log_share, phi, g):
    """The windows marked by `use`, and `share`, `log_share`, `phi` and `g` on those windows only (contiguous).

    The cell loops then read every array in order. `share` and `log_share` are types x clones x windows.
    """
    wins = np.flatnonzero(use)
    return (
        wins,
        np.ascontiguousarray(share[:, :, wins]),
        np.ascontiguousarray(log_share[:, :, wins]),
        phi[wins],
        g[wins],
    )


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
    wins, sh, ls, ph, gu = _used_windows(use, share, log_share, phi, g)
    nu = max(len(wins), 1)
    n_chunks = min(n, 256)
    step = (n + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        scr, yc = np.empty((6, nu)), np.empty(len(wins))
        for r in range(ch * step, min(n, (ch + 1) * step)):
            for j in range(len(wins)):
                yc[j] = Y[r, wins[j]]
            spill[r] = _cell_loglik(yc, size[r], types[r], sh, ls, ph, gu, spill_ok, smax, report, out[r], scr)


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


@njit
def _row_scale(data, indices, indptr, row, col_map):
    """`1e4 / size` of a cell, `size` being its count over the used genes (`col_map >= 0`), at least 1."""
    size = 0.0
    for k in range(indptr[row], indptr[row + 1]):
        if col_map[indices[k]] >= 0:
            size += data[k]
    return 1e4 / max(size, 1.0)


@njit_parallel
def log_moments(data, indices, indptr, rws, col_map, row0, n_total, s0, s1, s2):
    """Per gene, the cells that detect it (`s0`) and the sums of `v` (`s1`) and `v^2` (`s2`) over cells.

    `v = log1p(1e4 x count / size)` as in `project_rows`; zeros add nothing. The rows `rws` of this block are the
    cells `row0` onwards of `n_total`. Each slab (chunks x genes) sums a fixed range of all cells, so the sums depend
    neither on the number of threads nor on the blocks.
    """
    n_chunks = s1.shape[0]
    n_rows = rws.shape[0]
    step = (n_total + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(max(0, ch * step - row0), min(n_rows, (ch + 1) * step - row0)):
            f = _row_scale(data, indices, indptr, rws[r], col_map)
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                g = col_map[indices[k]]
                if g >= 0 and data[k] > 0:
                    v = math.log1p(data[k] * f)
                    s0[ch, g] += 1.0
                    s1[ch, g] += v
                    s2[ch, g] += v * v


@njit_parallel
def gram_sparse(data, indices, indptr, rws, col_map, hvg_of_gene, row0, n_total, out):
    """Add `x x^T` of each cell's highly variable genes to the upper triangle of `out[chunk]` (chunks x p x p).

    `x` holds `log1p(1e4 x count / size)` as in `project_rows`. The work per cell is its number of nonzero highly
    variable genes squared, so this suits sparse rows. As in `log_moments`, each slab sums a fixed range of the
    `n_total` cells (this block starts at cell `row0`); the caller sums the slabs.
    """
    n_chunks = out.shape[0]
    n_rows = rws.shape[0]
    step = (n_total + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        hs = np.empty(out.shape[1], dtype=np.int64)
        vs = np.empty(out.shape[1])
        for r in range(max(0, ch * step - row0), min(n_rows, (ch + 1) * step - row0)):
            f = _row_scale(data, indices, indptr, rws[r], col_map)
            m = 0
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                g = col_map[indices[k]]
                if g >= 0 and data[k] > 0:
                    h = hvg_of_gene[g]
                    if h >= 0:
                        hs[m] = h
                        vs[m] = math.log1p(data[k] * f)
                        m += 1
            for a in range(m):
                ha, va = hs[a], vs[a]
                for b in range(a, m):
                    hb = hs[b]
                    if ha <= hb:
                        out[ch, ha, hb] += va * vs[b]
                    else:
                        out[ch, hb, ha] += va * vs[b]


@njit_parallel
def hvg_rows(data, indices, indptr, rws, col_map, hvg_of_gene, out, n_chunks):
    """`out[r, h] = log1p(1e4 x count / size)` of the cell's highly variable gene `h`, 0 elsewhere (dense rows)."""
    n_rows = out.shape[0]
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for r in range(ch * step, min(n_rows, (ch + 1) * step)):
            for h in range(out.shape[1]):
                out[r, h] = 0.0
            f = _row_scale(data, indices, indptr, rws[r], col_map)
            for k in range(indptr[rws[r]], indptr[rws[r] + 1]):
                g = col_map[indices[k]]
                if g >= 0:
                    h = hvg_of_gene[g]
                    if h >= 0:
                        out[r, h] = math.log1p(data[k] * f)


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
            f = _row_scale(data, indices, indptr, rws[r], col_map)
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
    wins, sh, ls, ph, gu = _used_windows(use, share, log_share, phi, g)
    nu = max(len(wins), 1)
    step = (n_rows + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        buf = np.zeros(W)
        scr, yc = np.empty((6, nu)), np.empty(len(wins))
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
            for j in range(len(wins)):
                yc[j] = buf[wins[j]]
            spill[r] = _cell_loglik(yc, max(s, 1e-3), types[r], sh, ls, ph, gu, spill_ok, smax, True, out[r], scr)


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


@njit_parallel
def neighbor_prior_sweep(indptr, indices, w, deg, lik, base, lam, fixed, q, out, change):
    """One damped synchronous update of the spatial posteriors: `out_i = (q_i + p_i) / 2`.

    `p_i ~ lik_i ((1 - lam) base_i + lam m_i)`, where `m_i` is the mean of the posteriors `q_j` of i's neighbors in
    the CSR graph (`indptr`, `indices`, weights `w`, no diagonal, row sums `deg`), weighted by `w`. The damping keeps
    the fixed point and stops neighbors from oscillating. Rows with `fixed` keep `q_i`. `change[i]` receives the
    largest change of row i.
    """
    n, k = q.shape
    for i in prange(n):
        if fixed[i]:
            for c in range(k):
                out[i, c] = q[i, c]
            change[i] = 0.0
            continue
        d = deg[i]
        tot = 0.0
        for c in range(k):
            m = 0.0
            for e in range(indptr[i], indptr[i + 1]):
                m += w[e] * q[indices[e], c]
            p = lik[i, c] * ((1.0 - lam) * base[i, c] + lam * m / d)
            out[i, c] = p
            tot += p
        most = 0.0
        for c in range(k):
            new = 0.5 * (q[i, c] + out[i, c] / tot)
            most = max(most, abs(new - q[i, c]))
            out[i, c] = new
        change[i] = most


@njit_parallel
def kmeans_assign(XC, X, c2, k, labels, dist, sums, counts):
    """Nearest centre of each row for several k-means starts at once, and the sums of the rows per centre.

    The centres of all starts are stacked (start `s` owns centres `s k` to `s k + k - 1`); `XC = X @ C^T` comes from
    one matrix product for all of them and `c2` holds their squared norms. Writes each row's cluster for each start
    (`labels`, rows x starts) and its squared distance to that centre (`dist`), and adds the row to its centres' sums
    and counts (`sums`: chunks x centres x dims, `counts`: chunks x centres; one slab per chunk of rows, summed by the
    caller).
    """
    n, d = X.shape
    n_starts = labels.shape[1]
    n_chunks = sums.shape[0]
    step = (n + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        for i in range(ch * step, min(n, (ch + 1) * step)):
            x2 = 0.0
            for t in range(d):
                x2 += X[i, t] * X[i, t]
            for s in range(n_starts):
                best = s * k
                bd = np.inf
                for j in range(s * k, (s + 1) * k):
                    dd = c2[j] - 2.0 * XC[i, j]
                    if dd < bd:
                        bd = dd
                        best = j
                labels[i, s] = best - s * k
                dist[i, s] = max(bd + x2, 0.0)
                counts[ch, best] += 1.0
                for t in range(d):
                    sums[ch, best, t] += X[i, t]


@njit
def block_medians(x, bounds, out):
    """Median of the profile `x` (genomic order) in each block of genes `bounds[b]:bounds[b + 1]`."""
    for b in range(out.shape[0]):
        out[b] = np.median(x[bounds[b] : bounds[b + 1]])
