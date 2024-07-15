from dataclasses import dataclass

import evenodd


def __builtin_ctz(v: int) -> int:
    """Return the number of number of trailing zeros."""
    return (v & -v).bit_length() - 1


@dataclass
class Opaque:
    """An uninspectable filler object."""

    value: int


def even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    return x[::2] + x[1::2]


def naive_even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    n = len(x)
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    m = n - 1
    p = list(x)
    for i in range(n >> 1):
        p[i] = x[i << 1]
    for i in range(n >> 1, n):
        p[i] = x[((i << 1) | 1) & m]
    return p


def naive_bit_even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    n = len(x)
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    m = n - 1
    p = list(x)
    j = 0
    for i in range(n >> 1):
        p[i] = x[j]
        j += 2
    j = min(1, m)
    for i in range(n >> 1, n):
        p[i] = x[j]
        j += 2
    return p


def insertion_even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    n = len(x)
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    for i in range(1, n >> 1):
        for j in range(i << 1, i, -1):
            x[j], x[j - 1] = x[j - 1], x[j]
    return x


def quicksort_even_odd(
    x: list[Opaque], n: int | None = None, start: int = 0
) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    if n is None:
        n = len(x)
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    # base case
    if n <= 2:
        return x
    # pivot
    half = n >> 1
    for i in range(start + 1, start + 1 + half, 2):
        j = i + half - 1
        x[i], x[j] = x[j], x[i]
    # recur on both halves
    quicksort_even_odd(x, half, start)
    quicksort_even_odd(x, half, start + half)
    return x


def unravel_mutual(x: list[Opaque], n: int, start: int) -> list[Opaque]:
    """Unravel the result of selection sort."""
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    # base case
    if n <= 2:
        return x
    unravel_even_odd(x, n, start)
    unravel_mutual(x, n >> 1, start)
    return x


def unravel_recur(x: list[Opaque], n: int, start: int) -> list[Opaque]:
    """Unravel the result of selection sort."""
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    # base case
    if n <= 2:
        return x
    # pivot
    half = n >> 1
    for i in range(start + 1, start + half):
        j = (i << 1) - start
        x[i], x[j] = x[j], x[i]
    # recur on both halves
    unravel_recur(x, half, start)
    unravel_recur(x, half, start + half)
    return x


def unravel_even_odd(
    x: list[Opaque], n: int | None = None, start: int = 0
) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    if n is None:
        n = len(x)
    assert n == 1 << __builtin_ctz(n), f"{n} is not a power of 2."
    # base case
    if n <= 2:
        return x
    # pivot
    half = n >> 1
    for i in range(start + 1, start + half):
        k = (i << 1) - start
        x[i], x[k] = x[k], x[i]
    # recur
    return unravel_mutual(x, half, start + half)


def cycle_even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    n = len(x)
    k = __builtin_ctz(n)
    assert n == 1 << k, f"{n} is not a power of 2."
    m = n - 1
    d = k - 1
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
    return x


def fast_even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    n = len(x)
    k = __builtin_ctz(n)
    assert n == 1 << k, f"{n} is not a power of 2."
    m = n - 1
    d = k - 1
    i = 1
    while i < m:
        j = ((i << 1) | (i >> d)) & m
        while j > i:
            t = ((j << 1) | (j >> d)) & m
            x[j], x[t] = x[t], x[j]
            j = t
        i = evenodd.skip(i + 1, k)
    return x


def tile(i: int, j: int, k: int) -> int:
    """Tile the top j bits of i into k bits."""
    top = i >> (k - j)
    z = (top << k) - top
    y = z // ((1 << j) - 1)
    return y + (y << j != y + z)


def inline_even_odd(x: list[Opaque]) -> list[Opaque]:
    """Apply the even-odd permutation to x."""
    n = len(x)
    k = __builtin_ctz(n)
    assert n == 1 << k, f"{n} is not a power of 2."
    m = n - 1
    d = k - 1
    i = 1
    while i < m:
        j = ((i << 1) | (i >> d)) & m
        while j > i:
            t = ((j << 1) | (j >> d)) & m
            x[j], x[t] = x[t], x[j]
            j = t
        i += 1
        lsb = k - __builtin_ctz(i)
        t = tile(i, lsb, k) if i >= (1 << lsb) else i | 1
        while t & 1 == 0:
            lsb = k - __builtin_ctz(t)
            t = tile(t, lsb, k) if t >= (1 << lsb) else t | 1
        i = t
    return x


if __name__ == "__main__":
    K = 20

    methods = [
        naive_even_odd,
        naive_bit_even_odd,
        # insertion_even_odd,  # much slower than the rest
        quicksort_even_odd,
        unravel_even_odd,
        cycle_even_odd,
        fast_even_odd,
        inline_even_odd,
    ]
    for k in range(K):
        n = 1 << k
        x = [Opaque(i) for i in range(n)]
        ans = even_odd(list(x))
        for method in methods:
            assert method(list(x)) == ans
    # for n in range(1 << K):
    #     x = [Opaque(i) for i in range(n)]
    #     print(even_odd(x) == naive_even_odd(x))
