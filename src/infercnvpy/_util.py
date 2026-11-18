import inspect
import re
import sys
import warnings
from collections.abc import Callable, Iterator
from functools import wraps
from textwrap import dedent, indent
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from infercnvpy._logging import SKIP_OWN_FRAMES


def _as_matrix(x, what: str = "the expression matrix", *, lazy: bool = False):
    """Narrow an AnnData array to an in-memory numpy array or scipy sparse matrix.

    With `lazy`, also accept a dask array or a backed (on-disk) matrix, see `_is_dask` and `_is_backed`. Without it,
    such a matrix raises an error that tells the user how to load it. A DataFrame (as obsm allows) gives its values.
    float16 (which scipy.sparse and the kernels reject) becomes float32.
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if isinstance(x, np.ndarray) or (lazy and _is_dask(x)):
        return x.astype(np.float32) if x.dtype == np.float16 else x
    if sp.issparse(x) or (lazy and _is_backed(x)):
        return x
    if _is_dask(x) or _is_backed(x):
        kind = "a dask array" if _is_dask(x) else "a backed (on-disk) matrix"
        raise TypeError(
            f"This function needs {what} in memory, got {kind}. Load it first: call `.compute()` on a dask array "
            "and store the result back in adata (e.g. `adata.obsm['X_cnv'] = adata.obsm['X_cnv'].compute()`), or "
            "use `adata = adata.to_memory()` for a backed AnnData."
        )
    kinds = "a numpy array or scipy sparse matrix in memory"
    if lazy:
        kinds = "a numpy array, scipy sparse matrix, dask array or backed matrix"
    raise TypeError(f"infercnvpy needs {what} as {kinds}, got {type(x).__name__}.")


def _check_counts(values: np.ndarray, tool: str) -> None:
    """Raise unless `values` are raw counts (non-negative integers)."""
    if values.size and (values.min() < 0 or np.any(values != np.round(values))):
        raise ValueError(
            f"`{tool}` needs raw counts (non-negative integers); pass the layer with the counts as `layer`."
        )


def _is_dask(x) -> bool:
    """Whether `x` is a dask array (without importing dask)."""
    da = sys.modules.get("dask.array")
    return da is not None and isinstance(x, da.Array)


def _is_backed(x) -> bool:
    """Whether `x` is the matrix of a backed AnnData (an h5py dataset or an anndata sparse dataset)."""
    import h5py
    from anndata.abc import CSCDataset, CSRDataset

    return isinstance(x, h5py.Dataset | CSRDataset | CSCDataset)


def _as_df(df) -> pd.DataFrame:
    """Narrow `obs`/`var` to a pandas DataFrame (AnnData also allows lazy `Dataset2D`)."""
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError(f"infercnvpy needs `obs` and `var` as pandas DataFrames, got {type(df).__name__}.")


def _as_list(x) -> list:
    """A string as a list of one, any other collection as a list."""
    return [x] if isinstance(x, str) else list(x)


def _obs_column(adata, key: str) -> pd.Series:
    """`adata.obs[key]` as a pandas Series (the `obs` of `anndata.experimental.read_lazy` is a lazy `Dataset2D`)."""
    obs = adata.obs
    return obs[key] if isinstance(obs, pd.DataFrame) else obs[[key]].to_memory()[key]


def _reference_cells(adata, reference_key, reference_cat) -> np.ndarray | None:
    """Which cells are reference (normal) cells, or `None` without a reference; checks the arguments.

    Cells with `NaN` in the column are not reference cells.
    """
    if (reference_key is None) != (reference_cat is None):
        raise ValueError("Give both `reference_key` and `reference_cat` for reference cells, or neither.")
    if reference_key is None:
        return None
    if reference_key not in adata.obs.columns:
        raise KeyError(f"`{reference_key}` is not a column of `adata.obs`.")
    values = np.asarray(_obs_column(adata, reference_key))
    present = set(pd.unique(values[pd.notna(values)]))
    missing = [c for c in _as_list(reference_cat) if c not in present]
    if missing:
        raise ValueError(f"No cell has `{reference_key}` in {missing!r}. The column has {sorted(map(str, present))!r}.")
    return np.isin(values, _as_list(reference_cat))


def _choose_mtx_rep(adata, use_raw=False, layer=None):
    """Get gene expression from anndata depending on use_raw and layer."""
    is_layer = layer is not None
    if use_raw and is_layer:
        raise ValueError(
            f"Cannot use expression from both layer and raw. You provided:'use_raw={use_raw}' and 'layer={layer}'"
        )
    if is_layer:
        return adata.layers[layer]
    elif use_raw:
        return adata.raw.X
    else:
        return adata.X


def _listed[T](x: T) -> T | list:
    """`x` as a list if it is an iterator (e.g. a generator or `map`), so it can be read twice."""
    return list(x) if isinstance(x, Iterator) else x


def _params_for_uns(**params) -> dict:
    """Prepare call parameters for `adata.uns[...]["params"]`: list-likes (e.g. a pandas Index) as lists.

    A random number generator can't be written to h5ad or zarr and is stored as `None`.
    """

    def writable(v):
        if isinstance(v, np.random.Generator):
            return None
        if isinstance(v, str | dict) or not pd.api.types.is_list_like(v):
            return v
        return sorted(v) if isinstance(v, set | frozenset) else list(v)

    return {k: writable(v) for k, v in params.items()}


def _splice_doc[F: Callable[..., Any]](**blocks: str) -> Callable[[F], F]:
    """Replace each line `{name}` of the docstring by the text `blocks[name]`, at the indentation of that line."""

    def decorate(fn: F) -> F:
        doc = fn.__doc__ or ""
        for name, block in blocks.items():
            match = re.search(rf"^( *)\{{{name}\}}$", doc, flags=re.MULTILINE)
            assert match is not None, f"{fn.__name__} has no line {{{name}}}"
            doc = doc.replace(match.group(0), indent(dedent(block).strip("\n"), match.group(1)))
        fn.__doc__ = doc
        return fn

    return decorate


def _deprecated_positional[F: Callable[..., Any]](*names: str, until: str = "0.8.0") -> Callable[[F], F]:
    """Accept the keyword-only arguments `names` positionally, in this order, with a FutureWarning.

    For arguments that became keyword-only: calls that pass them positionally (after the positional
    parameters of the new signature) keep working until `until`.
    """

    def decorate(fn: F) -> F:
        n_positional = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in inspect.signature(fn).parameters.values()
        )

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if len(args) > n_positional:
                extra = args[n_positional:]
                if len(extra) > len(names):
                    raise TypeError(
                        f"{fn.__name__}() takes {n_positional} positional arguments (up to {n_positional + len(names)} "
                        f"with deprecated ones), but {len(args)} were given."
                    )
                given = names[: len(extra)]
                if twice := sorted(set(given) & kwargs.keys()):
                    raise TypeError(f"{fn.__name__}() got multiple values for {', '.join(twice)}.")
                warnings.warn(
                    f"Passing {', '.join(f'`{n}`' for n in given)} to `{fn.__name__}` positionally is deprecated and "
                    f"will raise an error from infercnvpy {until}. Pass "
                    f"{', '.join(f'`{n}=...`' for n in given)} as keyword argument{'s' * (len(given) > 1)} instead.",
                    FutureWarning,
                    skip_file_prefixes=SKIP_OWN_FRAMES,
                )
                kwargs.update(zip(given, extra, strict=True))
                args = args[:n_positional]
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def _copy_for_inplace_false(func: str, inplace: bool, copy: bool, *, future: str) -> bool:
    """Whether to work on a copy of adata: `copy=True`, or the deprecated `inplace=False` (with a FutureWarning).

    `inplace=False` returned a modified copy of adata before 0.8.0; from 0.8.0 it returns the results (`future`), as in
    the other functions.
    """
    if copy and not inplace:
        raise ValueError(f"{func}: pass either `copy=True` or `inplace=False`, not both.")
    if not inplace:
        warnings.warn(
            f"`{func}(..., inplace=False)` returns a modified copy of adata. From infercnvpy 0.8.0 it will return "
            f"{future} instead, as the other functions do. Pass `copy=True` to keep getting a modified copy.",
            FutureWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )
        return True
    return copy
