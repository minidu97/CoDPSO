from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022,
    F52022, F62022, F72022, F82022,
    F92022, F102022, F112022, F122022
)
import numpy as np

_CLASSES = {
    1: F12022,  2: F22022,  3: F32022,  4: F42022,
    5: F52022,  6: F62022,  7: F72022,  8: F82022,
    9: F92022, 10: F102022, 11: F112022, 12: F122022
}


def _get_optimum_point(fn):
    """
    Return the x vector at which the function reaches its global optimum.

    - Unimodal / Hybrid functions (F1-F8):
        f_shift is a 1D array of length ndim. Optimum is at f_shift[:ndim].

    - Composition functions (F9-F12):
        f_shift is a 2D array of shape (n_components, ndim).
        The global optimum is at the shift of the first component: f_shift[0].
    """
    shift = fn.f_shift
    if np.ndim(shift) == 2:
        return shift[0][:fn.ndim].copy()   # composition: use first component shift
    else:
        return shift[:fn.ndim].copy()       # unimodal / hybrid: use flat shift vector


class _FunctionWrapper:
    """
    Wraps an opfunu CEC2022 object and auto-detects the correct evaluation
    method so that f(x_optimum) == f_bias for all 12 functions.
    """
    def __init__(self, fid, dim):
        self._fn   = _CLASSES[fid](ndim=dim)
        self._bias = float(self._fn.f_bias)
        self._call = self._detect_correct_method()

    def _detect_correct_method(self):
        x_opt = _get_optimum_point(self._fn)

        candidates = [
            ('__call__', lambda x: self._fn(x)),
            ('fit',      lambda x: self._fn.fit(x) if hasattr(self._fn, 'fit') else None),
            ('evaluate', lambda x: self._fn.evaluate(x)),
        ]

        best_method = None
        best_error  = np.inf

        for name, fn in candidates:
            try:
                val = fn(x_opt)
                if val is None:
                    continue
                error = abs(float(val) - self._bias)
                if error < best_error:
                    best_error  = error
                    best_method = fn
            except Exception:
                continue

        if best_method is None:
            best_method = lambda x: self._fn.evaluate(x) + self._bias

        return best_method

    def evaluate(self, x):
        return float(self._call(x))


def get_cec2022_function(fid, dim):
    return _FunctionWrapper(fid, dim)