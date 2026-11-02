"""Per-well process pool for the exec()-style step scripts.

The step scripts run their analysis code with exec() into the script's globals, so the per-well functions cannot
be pickled and sent to worker processes. Instead, every worker builds its own namespace exactly the way a step
script does (config + the same lib/*.py files), then receives only the function NAME and plain arguments
(numpy arrays, tuples, file paths). Results come back in task order, so tables are row-for-row identical to a
serial run, and all random draws stay in the main process.

Workers use the 'spawn' start method on every platform (Windows has no other), so behaviour is the same on the
analysis PC and elsewhere. While the pool starts, __main__.__file__ is hidden: otherwise 'spawn' would re-run the
whole step script in every worker, because the step scripts have no `if __name__ == '__main__'` guard.

Worker start-up errors are sent back and raised in the main process (instead of the pool silently restarting
workers). Workers keep the default BLAS threading, i.e. the same numerical settings as a serial run.

Number of workers: environment variable SPHEROID_WORKERS (default: CPU count - 1, at most 8).
SPHEROID_WORKERS=1 runs everything serially in the calling process (the original code path).
"""
import os, sys, contextlib

_NS = None          # worker-side namespace
_INIT_ERR = None    # exception raised while building it (re-raised on the first task)


def n_workers():
    env = os.environ.get('SPHEROID_WORKERS', '').strip()
    if env:
        return max(1, int(env))
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _worker_init(here, libs, shared):
    global _INIT_ERR
    try:
        _build_ns(here, libs, shared)
    except BaseException as e:      # do not let the worker die: Pool would restart it forever
        _INIT_ERR = e


def _build_ns(here, libs, shared):
    global _NS
    for d in (os.path.join(here, 'lib'), here):
        if d not in sys.path: sys.path.insert(0, d)
    import warnings; warnings.filterwarnings('ignore')
    ns = {'__name__': '__spheroid_worker__', 'HERE': here}
    exec('from config import *', ns)
    ns['LIB'] = lambda name: os.path.join(here, 'lib', name)
    ns['run_lib'] = lambda name: exec(open(ns['LIB'](name), encoding='utf-8').read(), ns)
    for name in libs:
        exec(open(ns['LIB'](name), encoding='utf-8').read(), ns)
    ns.update(shared or {})
    _NS = ns


def _worker_call(task):
    if _INIT_ERR is not None:
        raise RuntimeError(f'worker start-up failed: {_INIT_ERR!r}') from _INIT_ERR
    fname, args = task
    return _NS[fname](*args)


@contextlib.contextmanager
def _spawn_env():
    """Hide the main script from 'spawn' for as long as the pool lives (workers can be (re)started at any time)."""
    main = sys.modules.get('__main__')
    saved_file = main.__dict__.pop('__file__', None) if main is not None else None
    saved_spec = getattr(main, '__spec__', None)
    if main is not None: main.__spec__ = None
    try:
        yield
    finally:
        if main is not None:
            main.__spec__ = saved_spec
            if saved_file is not None: main.__file__ = saved_file


class WellPool:
    """with WellPool(HERE, ['common.py', ...], shared={...}) as pool:  results = pool.map('func_name', arg_tuples)

    `fallback_ns` (usually globals() of the step script) is used when only one worker is requested: the function
    is then looked up there and called directly, i.e. exactly the original serial code."""

    def __init__(self, here, libs, shared=None, fallback_ns=None, workers=None):
        self.here, self.libs, self.shared, self.fallback_ns = here, list(libs), dict(shared or {}), fallback_ns
        self.workers = workers or n_workers()
        self.pool = None; self._env = None

    def __enter__(self):
        if self.workers > 1:
            import multiprocessing as mp
            self._env = _spawn_env(); self._env.__enter__()
            try:
                self.pool = mp.get_context('spawn').Pool(self.workers, initializer=_worker_init,
                                                         initargs=(self.here, self.libs, self.shared))
            except BaseException:
                self._env.__exit__(None, None, None); raise
        return self

    def __exit__(self, *exc):
        try:
            if self.pool is not None:
                self.pool.close() if exc[0] is None else self.pool.terminate()
                self.pool.join()
        finally:
            if self._env is not None: self._env.__exit__(None, None, None)
        return False

    def imap(self, fname, arglist, chunksize=1):
        """Yields results in task order."""
        if self.pool is None:
            ns = self.fallback_ns
            if ns is None:
                _build_ns(self.here, self.libs, self.shared); ns = _NS
            else:
                ns.update(self.shared)
            f = ns[fname]
            for a in arglist:
                yield f(*a)
        else:
            yield from self.pool.imap(_worker_call, ((fname, tuple(a)) for a in arglist), chunksize)

    def map(self, fname, arglist, chunksize=1):
        return list(self.imap(fname, arglist, chunksize))
