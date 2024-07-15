# cython: profile=False
cimport numpy as np

import numpy as np


# https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
cdef extern int __builtin_ctzll(unsigned long long) nogil


cdef void __alloc_even_odd(size_t n, double *x, double *y) noexcept nogil:
    """Apply the even-odd permutation to x."""
    cdef size_t i

    for i in range(n >> 1):
        y[i] = x[i << 1]
        y[i | (n >> 1)] = x[(i << 1) | 1]


def alloc_even_odd(double[::1] x) -> np.ndarray:
    """Wrapper for __alloc_even_odd."""
    y = np.empty_like(x)
    cdef double[::1] z = y
    __alloc_even_odd(x.shape[0], &x[0], &z[0])
    return y


cdef void __alloc_inv_even_odd(size_t n, double *x, double *y) noexcept nogil:
    """Apply the inverse even-odd permutation to x."""
    cdef size_t i

    for i in range(n >> 1):
        y[i << 1] = x[i]
        y[(i << 1) | 1] = x[i | (n >> 1)]


def alloc_inv_even_odd(double[::1] x) -> np.ndarray:
    """Wrapper for __alloc_inv_even_odd."""
    y = np.empty_like(x)
    cdef double[::1] z = y
    __alloc_inv_even_odd(x.shape[0], &x[0], &z[0])
    return y


cdef void __quicksort_even_odd(size_t n, double *x) noexcept nogil:
    """Apply the even-odd permutation to x."""
    cdef size_t half, i, j

    # base case
    if n <= 2:
        return
    # pivot
    half = n >> 1
    for i in range(1, half + 1, 2):
        j = i + half - 1
        x[i], x[j] = x[j], x[i]
    # recur on both halves
    __quicksort_even_odd(half, x)
    __quicksort_even_odd(half, x + half)


def quicksort_even_odd(x: np.ndarray) -> np.ndarray:
    """Wrapper for __quicksort_even_odd."""
    cdef double[::1] y = x
    __quicksort_even_odd(y.shape[0], &y[0])
    return x


cdef void __cycle_even_odd(size_t n, double *x) noexcept nogil:
    """Apply the even-odd permutation to x."""
    cdef size_t m, d, i, j, t

    m = n - 1
    d = __builtin_ctzll(n) - 1
    for i in range(1, n >> 1, 2):
        j = ((i << 1) | (i >> d)) & m
        while j > i:
            j = ((j << 1) | (j >> d)) & m
        if j == i:
            # i is minimal
            j = ((j << 1) | (j >> d)) & m
            while j > i:
                t = ((j << 1) | (j >> d)) & m
                x[j], x[t] = x[t], x[j]
                j = t


def cycle_even_odd(x: np.ndarray) -> np.ndarray:
    """Wrapper for __cycle_even_odd."""
    cdef double[::1] y = x
    __cycle_even_odd(y.shape[0], &y[0])
    return x


cdef inline size_t tile(size_t i, size_t j, size_t k) noexcept nogil:
    """Tile the top j bits of i into k bits."""
    cdef size_t top, z, y

    top = i >> (k - j)
    z = (top << k) - top
    y = z // ((1 << j) - 1)
    return y + (y << j != y + z)


cdef void __fast_even_odd(size_t n, double *x) noexcept nogil:
    """Apply the even-odd permutation to x."""
    cdef size_t m, k, d, i, j, t, lsb

    if n == 1:
        return
    m = n - 1
    k = __builtin_ctzll(n)
    d = k - 1
    i = 1
    while i < m:
        j = ((i << 1) | (i >> d)) & m
        while j > i:
            t = ((j << 1) | (j >> d)) & m
            x[j], x[t] = x[t], x[j]
            j = t
        i += 1
        lsb = k - __builtin_ctzll(i)
        t = tile(i, lsb, k) if i >= (<size_t> 1 << lsb) else i | 1
        while t & 1 == 0:
            lsb = k - __builtin_ctzll(t)
            t = tile(t, lsb, k) if t >= (<size_t> 1 << lsb) else t | 1
        i = t


def fast_even_odd(x: np.ndarray) -> np.ndarray:
    """Wrapper for __fast_even_odd."""
    cdef double[::1] y = x
    __fast_even_odd(y.shape[0], &y[0])
    return x


cdef void __fast_inv_even_odd(size_t n, double *x) noexcept nogil:
    """Apply the inverse even-odd permutation to x."""
    cdef size_t m, k, d, i, j, t, lsb

    if n == 1:
        return
    m = n - 1
    k = __builtin_ctzll(n)
    d = k - 1
    i = 1
    while i < m:
        j = ((i << 1) | (i >> d)) & m
        while j > i:
            x[j], x[i] = x[i], x[j]
            j = ((j << 1) | (j >> d)) & m
        i += 1
        lsb = k - __builtin_ctzll(i)
        t = tile(i, lsb, k) if i >= (<size_t> 1 << lsb) else i | 1
        while t & 1 == 0:
            lsb = k - __builtin_ctzll(t)
            t = tile(t, lsb, k) if t >= (<size_t> 1 << lsb) else t | 1
        i = t


def fast_inv_even_odd(x: np.ndarray) -> np.ndarray:
    """Wrapper for __fast_inv_even_odd."""
    cdef double[::1] y = x
    __fast_inv_even_odd(y.shape[0], &y[0])
    return x
