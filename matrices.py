from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, jit, lax

jnp.set_printoptions(precision=3, suppress=True)
jax.config.update("jax_enable_x64", True)


class Norm(Enum):
    NONE = 0
    ORTH = 1
    FULL = 2


def to_tex(m: Array, precision: int = 3) -> str:
    """Convert a matrix to LaTeX."""
    endl = " \\\\\n"
    # -{digits}.{precision}
    length = max(map(lambda x: len(f"{x:.{precision}f}"), m.flatten()))
    fmt = lambda x: f"{x:{length}.{precision}f}"  # noqa: E731
    return f"""\
\\begin{{pmatrix}}
{endl.join(f"  {' & '.join(map(fmt, row))}" for row in m)} \\\\
\\end{{pmatrix}}"""


@partial(jit, static_argnums=(1, 2))
def normalize(m: Array, axis: int = 0, norm: Norm = Norm.ORTH) -> Array:
    """Normalize the rows or columns of m."""
    f = {
        Norm.NONE: lambda _: 1,
        Norm.ORTH: jnp.sqrt,
        Norm.FULL: lambda x: x,
    }[norm]
    return m / f(jnp.sum(m * m, axis=axis, keepdims=True))


@partial(jit, static_argnums=0)
def haar_matrix(k: int) -> Array:
    """Return the 2^k x 2^k Haar basis."""
    n = 1 << k
    m = jnp.zeros((n, n))
    # special case of first row all 1's
    m = m.at[0].set(1)
    i = 1
    for level in range(k, 0, -1):
        half = 1 << (level - 1)
        ones = jnp.ones((1, half))

        def body_fun(t: int, m: Array) -> Array:
            """Update the matrix."""
            j = t << level
            m = lax.dynamic_update_slice(m, ones, (i + t, j))
            m = lax.dynamic_update_slice(m, -ones, (i + t, j + half))
            return m

        m = lax.fori_loop(0, 1 << (k - level), body_fun, m)
        i += 1 << (k - level)
    return m


@partial(jit, static_argnums=0)
def even_odd_permutation(k: int) -> Array:
    """Return the even-odd permutation matrix."""
    n = 1 << k
    m = jnp.zeros((n, n))
    half = n >> 1
    return lax.fori_loop(
        0,
        half,
        lambda i, m: m.at[i, i << 1].set(1).at[i | half, (i << 1) | 1].set(1),
        m,
    )


@partial(jit, static_argnums=0)
def diagonal_normalization(k: int) -> Array:
    """Return the diagonal normalization."""
    n = 1 << k
    x = jnp.zeros(n).at[0].set(jnp.sqrt(n))
    i = 1
    for level in range(k):
        size = 1 << (k - level)
        x = x.at[i : i + (1 << level)].set(jnp.sqrt(size))
        i += 1 << level
    return jnp.diag(jnp.reciprocal(x))


@partial(jit, static_argnums=0)
def diagonal_haar(k: int) -> Array:
    """Return the 2x2 block diagonal Haar operations."""
    n = 1 << k
    m = jnp.zeros((n, n))
    d = jnp.array([[1, 1], [1, -1]], dtype=m.dtype)
    return lax.fori_loop(
        0,
        n >> 1,
        lambda i, m: lax.dynamic_update_slice(m, d, (i << 1, i << 1)),
        m,
    )


if __name__ == "__main__":
    k = 3
    n = 1 << k
    m = 1 << (k - 1)

    H_dwt = haar_matrix(k)
    H = normalize(H_dwt, axis=1, norm=Norm.ORTH)
    assert jnp.allclose(H @ H.T, jnp.identity(n)), "not orthogonal"
    T = diagonal_haar(k)
    P = even_odd_permutation(k)
    R = jnp.block(
        [
            [haar_matrix(k - 1), jnp.zeros((m, m))],
            [jnp.zeros((m, m)), jnp.identity(m)],
        ]
    )
    D = diagonal_normalization(k)
    # print(T)
    # print(P)
    # print(R)
    # print(D)
    # print(P @ T)
    # print(R @ P @ T)
    # print(D @ R @ P @ T)
    assert jnp.allclose(D @ R @ P @ T, H), "factorization wrong"

    H_iwt = H_dwt.T
    assert jnp.allclose(H_iwt, T @ P.T @ R.T), "factorization wrong"
    assert jnp.allclose(H_iwt @ D @ D @ H_dwt, jnp.identity(n)), "not inverted"
    assert jnp.allclose(H_dwt @ H_iwt @ D @ D, jnp.identity(n)), "not inverted"
    assert jnp.allclose(D @ H_dwt @ H_iwt @ D, jnp.identity(n)), "not inverted"

    dwt_fwrd = D @ D @ H_dwt
    dwt_orth = D @ H_dwt
    dwt_back = H_dwt
    iwt_fwrd = H_iwt
    iwt_orth = H_iwt @ D
    iwt_back = H_iwt @ D @ D
    assert jnp.allclose(dwt_fwrd @ iwt_fwrd, jnp.identity(n)), "not inverted"
    assert jnp.allclose(dwt_orth @ iwt_orth, jnp.identity(n)), "not inverted"
    assert jnp.allclose(dwt_back @ iwt_back, jnp.identity(n)), "not inverted"
