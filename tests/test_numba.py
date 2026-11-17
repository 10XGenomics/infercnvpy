"""The parallel and serial twins of `njit_parallel`, and running without numba's JIT."""

import numpy as np

from infercnvpy._numba import njit_parallel, prange
from tests.conftest import run_python


@njit_parallel
def _inverse(x, out):
    for i in prange(x.size):
        out[i] = 1.0 / x[i]


def test_twins_divide_by_zero_alike():
    """Both twins use numpy's error model: a division by zero gives inf, not a ZeroDivisionError in one of them."""
    x = np.array([0.0, 2.0])
    for twin in (_inverse.parallel, _inverse.serial):
        out = np.empty(2)
        twin(x, out)
        np.testing.assert_array_equal(out, [np.inf, 0.5])


def test_without_jit():
    """With `NUMBA_DISABLE_JIT=1` (e.g. to debug or measure coverage), the kernels run as Python."""
    script = """
    import numpy as np, pandas as pd, scipy.sparse as sp
    from anndata import AnnData
    import infercnvpy as cnv

    var = pd.DataFrame({"chromosome": ["chr1"] * 12, "start": np.arange(12), "end": np.arange(12) + 1})
    X = sp.random(6, 12, density=0.5, format="csr", random_state=0, dtype=np.float32)
    adata = AnnData(X=X, var=var.set_index(var.index.astype(str)))
    cnv.tl.infercnv(adata, window_size=4, step=2)
    assert adata.obsm["X_cnv"].shape == (6, 5)
    """
    run_python(script, NUMBA_DISABLE_JIT="1")
