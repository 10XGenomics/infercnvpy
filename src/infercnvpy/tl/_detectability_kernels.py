"""numba kernels of :func:`infercnvpy.tl.detectability`, one pass over the CSR counts each.

A cell of share group `c` expects `m = s_a p[c, g]` counts of gene `g` on arm `a`, with `s_a` the cell's leave-arm-out
size factor (`_sizes`). For a reference cell, the expectation leaves the cell itself out of its group's shares, so that
a cell's own counts don't pull its expectation towards them (which would shrink the dispersion and the null variance
of small groups). With `Ns[c]` the counts of the group and `N` those of the cell, a gene without counts in the cell
then expects `m0 = s p Ns / (Ns - N)`, and a gene with `x` counts `m1 = m0 - s x / (Ns - N)`.

The sums are per chunk of cells (`sums[t]`), so that the `prange` threads never write to the same memory, and the
scratch rows are allocated outside the `prange` loop, as in `_infercnv_kernels`.
"""

from __future__ import annotations

import numpy as np

from infercnvpy._numba import njit, njit_parallel, prange


@njit
def _sizes(data, indices, lo, hi, gene_arm, P, scale, out):
    """Leave-arm-out size factor of one cell (values `data[lo:hi]`) for every arm, into `out`; returns the total.

    The counts off arm `a` divided by the share of those genes, `(N - O_a) / (1 - P[a])`, so that an arm's own counts
    (and a copy number change of it) don't enter its expectation. An arm that holds almost all of the panel's counts
    has no counts to measure the size with, and gets size 0 (no information).
    """
    out[:] = 0.0
    total = 0.0
    for k in range(lo, hi):
        v = np.float64(data[k])
        total += v
        a = gene_arm[indices[k]]
        if a >= 0:
            out[a] += v
    for a in range(out.size):
        rest = 1.0 - P[a]
        out[a] = scale * (total - out[a]) / rest if rest > 1e-3 else 0.0
    return total


@njit
def _left_out(is_ref, Ns, total):
    """`Ns / (Ns - N)` of a reference cell, 1 for other cells, and 0 for a reference cell that is all of its group."""
    if not is_ref:
        return 1.0
    rest = Ns - total
    return Ns / rest if rest > 0 else 0.0


@njit_parallel
def _nb_moments(data, indices, indptr, rows, code, P, p, Ns, gene_arm, xx, xm, dm, dm2, s1, s2, buf):
    """Sums over the reference cells `rows` for the NB moments of each gene, with leave-one-out expectations.

    Per gene `x^2`, `x m`, and the corrections `m1 - m0` and `m1^2 - m0^2` of the genes with counts; per share group
    and arm `s'` and `s'^2` with `s' = s Ns / (Ns - N)`, from which `sum m0` and `sum m0^2` follow. `xx`, `xm`, `dm`,
    `dm2` are (chunks, genes), `s1`, `s2` (chunks, share groups, arms) and `buf` (chunks, arms) is scratch.
    """
    n_chunks = xx.shape[0]
    n = rows.size
    for t in prange(n_chunks):
        sz = buf[t]
        for i in range(t * n // n_chunks, (t + 1) * n // n_chunks):
            r = rows[i]
            c = code[r]
            total = _sizes(data, indices, indptr[r], indptr[r + 1], gene_arm, P[c], 1.0, sz)
            kap = _left_out(True, Ns[c], total)
            if kap == 0:
                continue
            for a in range(sz.size):
                s1[t, c, a] += sz[a] * kap
                s2[t, c, a] += (sz[a] * kap) ** 2
            for k in range(indptr[r], indptr[r + 1]):
                g = indices[k]
                a = gene_arm[g]
                if a >= 0:
                    v = np.float64(data[k])
                    m0 = sz[a] * kap * p[c, g]
                    m1 = m0 - sz[a] * v / (Ns[c] - total)
                    xx[t, g] += v * v
                    xm[t, g] += v * m1
                    dm[t, g] += m1 - m0
                    dm2[t, g] += m1 * m1 - m0 * m0


@njit_parallel
def _arm_information(
    data, indices, indptr, code, group, is_ref, P, P2, ps, phis, pos, Ns, gene_arm, bounds, scale, sums, null,
    cell_info, buf,
):  # fmt: skip
    """Per-cell Fisher information about the copy ratio of each arm, summed by group, and the null of the score.

    With `w = 1 / (1 + m phi)`, a cell's information on arm `a` is `I = sum m w`, and the variance of its efficient
    score `U = sum (x - m) w` at copy ratio `r` is `r A + r^2 B`, with `A = sum m w^2` and `B = sum m^2 phi w^2` (so
    `A + B = I`). `sums` (chunks, groups, 3, arms) gets `I`, `A`, `B`.

    For the reference cells (`is_ref`), `null` (chunks, 4, arms) gets `U^2`, `I`, `I^2` and `I^2 v_s`, the part of
    `U^2` that comes from the noise of the size factor: `v_s = 1 / (N (1 - P_a)) + (P2_all - P2_a) / (1 - P_a)^2` is
    the relative variance of the counts off the arm, with `P2` the sums of `p^2 phi` per arm and over all genes (last
    column). `cell_info` gets the information of each cell summed over arms.

    The genes of arm `a` are `bounds[a]:bounds[a + 1]` of the arm-ordered `ps` (share groups x used genes) and
    `phis`, and `pos[g]` is the place of gene `g` in that order. `scale` multiplies the expected counts (a deeper
    experiment), which makes `U` meaningless: use `null[0]` only from a call with `scale = 1`. `buf` (chunks, 2, arms)
    is scratch.
    """
    n_chunks = sums.shape[0]
    n_arms = bounds.size - 1
    n = indptr.size - 1
    for t in prange(n_chunks):
        sz = buf[t, 0]
        u = buf[t, 1]
        for r in range(t * n // n_chunks, (t + 1) * n // n_chunks):
            c = code[r]
            total = _sizes(data, indices, indptr[r], indptr[r + 1], gene_arm, P[c], scale, sz)
            kap = _left_out(is_ref[r], Ns[c], total)
            ref = kap > 0 and is_ref[r]
            if kap == 0:
                kap = 1.0
            u[:] = 0.0
            if ref:
                for k in range(indptr[r], indptr[r + 1]):
                    g = indices[k]
                    a = gene_arm[g]
                    if a >= 0:
                        v = np.float64(data[k])
                        m0 = sz[a] * kap * ps[c, pos[g]]
                        m1 = m0 - sz[a] * v / (Ns[c] - total)
                        u[a] += (v - m1) / (1.0 + m1 * phis[pos[g]]) + m0 / (1.0 + m0 * phis[pos[g]])
            cell = 0.0
            for a in range(n_arms):
                info = 0.0
                va = 0.0
                vb = 0.0
                f = sz[a] * kap
                for j in range(bounds[a], bounds[a + 1]):
                    share = ps[c, j]
                    if share == 0.0:
                        continue
                    m = f * share
                    mp = m * phis[j]
                    w = 1.0 / (1.0 + mp)
                    info += m * w
                    va += m * w * w
                    vb += m * mp * w * w
                sums[t, group[r], 0, a] += info
                sums[t, group[r], 1, a] += va
                sums[t, group[r], 2, a] += vb
                cell += info
                if ref and info > 0:
                    d = u[a] - info
                    rest = 1.0 - P[c, a]
                    v_s = 1.0 / (scale * total * rest) + (P2[c, n_arms] - P2[c, a]) / (rest * rest)
                    null[t, 0, a] += d * d
                    null[t, 1, a] += info
                    null[t, 2, a] += info * info
                    null[t, 3, a] += info * info * v_s
            cell_info[r] = cell
