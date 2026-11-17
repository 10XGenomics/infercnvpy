"""numba helpers: a parallel kernel with a serial twin, and `n_jobs` as the number of numba threads.

A ``parallel=True`` kernel only runs on the main thread and outside dask tasks (`dask_task`). Anywhere else (a dask
worker, a thread pool, the main thread of a dask worker process) the serial twin runs instead: numba's default
``workqueue`` threading layer (the only one in the macOS wheels) aborts the process on concurrent use, and a caller
that brings its own threads or processes should not get numba threads on top.

All kernels use numba's disk cache, with data files named after their cache key (see `_KeyedCacheFile`), and numpy's
error model, so that both twins divide by zero alike.
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from functools import update_wrapper, wraps
from types import FunctionType
from typing import TYPE_CHECKING, Any

import numba
from numba.core.caching import FunctionCache, IndexDataCacheFile

if TYPE_CHECKING:
    from builtins import range as prange  # numba's prange iterates like range; its own type says nothing about that
    from collections.abc import Callable, Generator
else:
    from numba import prange

__all__ = ["dask_task", "n_kernel_chunks", "njit", "njit_parallel", "numba_threads", "prange", "runs_serially"]


class _KeyedCacheFile(IndexDataCacheFile):
    """numba's cache index and data files, with each data file named after its key (signature, CPU, bytecode).

    numba numbers the data files of a function (``.1.nbc``, ``.2.nbc``, ...) by the free slots in the index as the
    process read it. Processes that compile different signatures of one function at the same time (pytest-xdist
    workers, a pool of jobs on one environment) read the same index and pick the same number. The index can then end
    up mapping one signature to another signature's machine code, and every later call with it fails with "can't
    unbox array from PyObject" until the cache is deleted. With a name per key, a data file only ever holds the code
    for its own key. Concurrent index writes can still drop an entry, which only costs one recompilation.
    """

    def save(self, key, data):
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        data_name = self._data_name_pattern.replace("{number:d}", digest)
        self._save_data(data_name, data)  # before the index, so that no index entry points to a missing file
        overloads = self._load_index()
        if overloads.get(key) != data_name:
            overloads[key] = data_name
            self._save_index(overloads)


class _KeyedFunctionCache(FunctionCache):
    """numba's function cache with the `njit` options in the key.

    numba's key has the kernel's bytecode but not its compile options, which are set here, in another module.
    """

    def __init__(self, py_func, options: dict[str, Any]):
        super().__init__(py_func)
        self._options = repr(sorted(options.items()))
        self._cache_file = _KeyedCacheFile(
            self._cache_path, self._impl.filename_base, self._impl.locator.get_source_stamp()
        )

    def _index_key(self, sig, codegen):
        return (*super()._index_key(sig, codegen), self._options)


def njit(fn: Callable[..., Any], /, **options: Any) -> Any:
    """`numba.njit` with ``nogil=True``, numpy's error model and the disk cache of `_KeyedCacheFile`."""
    options = {"nogil": True, "error_model": "numpy", **options}
    dispatcher = numba.njit(fn, **options)
    if hasattr(dispatcher, "py_func"):  # with NUMBA_DISABLE_JIT=1, `njit` returns `fn` itself
        # what `cache=True` sets
        dispatcher._cache = _KeyedFunctionCache(dispatcher.py_func, options)  # type: ignore[attr-defined]
    return dispatcher


class njit_parallel:
    """Compile a function with `njit` twice, ``parallel=True`` and serial.

    Calling it runs the serial version where `runs_serially` says so, otherwise the parallel one. The two versions are
    distinct function objects with distinct names, so they don't share a cache entry.
    """

    def __init__(self, fn: Callable[..., Any], /):
        assert isinstance(fn, FunctionType)
        self.parallel = njit(_renamed(fn, "parallel"), parallel=True)
        self.serial = njit(_renamed(fn, "serial"))
        update_wrapper(self, fn)

    def __call__(self, *args):
        return (self.serial if runs_serially() else self.parallel)(*args)


_scope = threading.local()


def runs_serially() -> bool:
    """Whether kernels called here run serially: off the main thread, or inside a dask task (see `dask_task`)."""
    return threading.current_thread() is not threading.main_thread() or getattr(_scope, "in_dask_task", False)


def dask_task[F: Callable[..., Any]](fn: F) -> F:
    """Mark `fn` as a dask task: the kernels it calls run serially, whatever thread or process the scheduler uses."""

    @wraps(fn)
    def task(*args, **kwargs):
        outer = getattr(_scope, "in_dask_task", False)
        _scope.in_dask_task = True
        try:
            return fn(*args, **kwargs)
        finally:
            _scope.in_dask_task = outer

    return task  # type: ignore[return-value]


def _renamed(fn: FunctionType, suffix: str) -> FunctionType:
    new = FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    update_wrapper(new, fn)
    new.__qualname__ = f"{fn.__qualname__}-{suffix}"
    return new


def n_kernel_chunks(n_rows: int, *, per_thread: int = 4) -> int:
    """Number of row chunks for a `prange` kernel: `per_thread` per numba thread, or 1 if it runs serially."""
    if runs_serially():
        return 1
    return max(1, min(n_rows, per_thread * numba.get_num_threads()))


@contextmanager
def numba_threads(n_jobs: int | None) -> Generator[None]:
    """Run the block with `n_jobs` numba threads and restore the previous number afterwards.

    `None` keeps numba's setting (all cores unless ``NUMBA_NUM_THREADS`` says otherwise), `-1` means all
    ``NUMBA_NUM_THREADS``, and larger values are capped at that.
    """
    if n_jobs is None:
        yield
        return
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"`n_jobs` must be a positive number of threads, -1 (all) or None, got {n_jobs}.")
    n_max = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
    previous = numba.get_num_threads()
    numba.set_num_threads(n_max if n_jobs == -1 else min(n_jobs, n_max))
    try:
        yield
    finally:
        numba.set_num_threads(previous)
