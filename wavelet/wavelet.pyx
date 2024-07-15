# cython: profile=False
from libc.math cimport sqrt

from .evenodd cimport (
    __alloc_even_odd,
    __alloc_inv_even_odd,
    __fast_even_odd,
    __fast_inv_even_odd,
)

import numpy as np


cdef int NONE, ORTH, FULL

NONE = 0
ORTH = 1
FULL = 2


cdef void __dwt_alloc(size_t n, double *x, double *y, int mode) noexcept nogil:
    """Apply the wavelet transform to x."""
    cdef:
        size_t m, i
        double scale

    m = n
    scale = 2 ** (-mode / 2.0)
    while n > 1:
        for i in range(0, n, 2):
            x[i], x[i | 1] = (
                scale * (x[i] + x[i | 1]),
                scale * (x[i] - x[i | 1]),
            )
        __alloc_even_odd(n, x, y)
        # y now holds the result
        if n < m - n:
            for i in range(n):
                x[i] = y[i]
        else:
            for i in range(n, m):
                y[i] = x[i]
            x, y = y, x
        n >>= 1


def dwt_alloc(x: np.ndarray, mode: str = "backward") -> np.ndarray:
    """Wrapper for __dwt_alloc."""
    cdef double[::1] y, z
    y = x
    z = np.empty_like(x)
    cdef int norm = {
        "backward": NONE,
        "orthogonal": ORTH,
        "forward": FULL,
    }[mode]
    __dwt_alloc(y.shape[0], &y[0], &z[0], norm)
    return x


cdef void __dwt(size_t n, double *x, int mode) noexcept nogil:
    """Apply the wavelet transform to x."""
    cdef:
        size_t i
        double scale

    scale = 2 ** (-mode / 2.0)
    while n > 1:
        for i in range(0, n, 2):
            x[i], x[i | 1] = (
                scale * (x[i] + x[i | 1]),
                scale * (x[i] - x[i | 1]),
            )
        __fast_even_odd(n, x)
        n >>= 1


def dwt(x: np.ndarray, mode: str = "backward") -> np.ndarray:
    """Wrapper for __dwt."""
    cdef double[::1] y = x
    cdef int norm = {
        "backward": NONE,
        "orthogonal": ORTH,
        "forward": FULL,
    }[mode]
    __dwt(y.shape[0], &y[0], norm)
    return x


cdef void __iwt_alloc(size_t n, double *x, double *y, int mode) noexcept nogil:
    """Apply the inverse wavelet transform to x."""
    cdef:
        size_t m, i
        double scale

    m = 2
    scale = 2 ** (-mode / 2.0)
    while m <= n:
        __alloc_inv_even_odd(m, x, y)
        # y now holds the result
        if m < n - m or n == 2:
            for i in range(m):
                x[i] = y[i]
        else:
            for i in range(m, n):
                y[i] = x[i]
            x, y = y, x
        for i in range(0, m, 2):
            x[i], x[i | 1] = (
                scale * (x[i] + x[i | 1]),
                scale * (x[i] - x[i | 1]),
            )
        m <<= 1


def iwt_alloc(x: np.ndarray, mode: str = "backward") -> np.ndarray:
    """Wrapper for __iwt_alloc."""
    cdef double[::1] y, z
    y = x
    z = np.empty_like(x)
    cdef int norm = {
        "backward": FULL,
        "orthogonal": ORTH,
        "forward": NONE,
    }[mode]
    __iwt_alloc(y.shape[0], &y[0], &z[0], norm)
    return x


cdef void __iwt(size_t n, double *x, int mode) noexcept nogil:
    """Apply the inverse wavelet transform to x."""
    cdef:
        size_t m, i
        double scale

    m = 2
    scale = 2 ** (-mode / 2.0)
    while m <= n:
        __fast_inv_even_odd(m, x)
        for i in range(0, m, 2):
            x[i], x[i | 1] = (
                scale * (x[i] + x[i | 1]),
                scale * (x[i] - x[i | 1]),
            )
        m <<= 1


def iwt(x: np.ndarray, mode: str = "backward") -> np.ndarray:
    """Wrapper for __iwt."""
    cdef double[::1] y = x
    cdef int norm = {
        "backward": FULL,
        "orthogonal": ORTH,
        "forward": NONE,
    }[mode]
    __iwt(y.shape[0], &y[0], norm)
    return x
