import time
from collections.abc import Callable
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from jax import Array, jit, random

from wavelet import dwt, iwt
from wavelet.wavelet import dwt_alloc, iwt_alloc

jax.config.update("jax_enable_x64", True)
rng = random.key(0)

KeyArray = Array
Function = Callable[[Array], Array]


class Transform(Protocol):
    """Protocol for type hinting."""

    def __call__(self, x: Array, mode: str = "backward") -> Array:
        """Transform x according to the normalization mode."""
        ...


figures = Path("figures")
figures.mkdir(parents=True, exist_ok=True)


class Norm(Enum):
    NONE = 0
    ORTH = 1
    FULL = 2


@partial(jit, static_argnums=0)
def scale(n: int, x: Array, power: float) -> Array:
    """Normalize the result of the wavelet transform."""
    k = n.bit_length() - 1
    power = jnp.float64(power)
    x = x.at[0].multiply(jnp.power(n, -power))
    i = 1
    for level in range(k):
        size = 1 << (k - level)
        x = x.at[i : i + (1 << level)].multiply(jnp.power(size, -power))
        i += 1 << level
    return x


def normalize(is_dwt: bool, norm: Norm) -> Callable[[Function], Transform]:
    """Decorator for wavelet transforms."""

    def decorator(f: Function) -> Transform:
        """Inner decorator."""

        def transform(x: Array, mode: str = "backward") -> Array:
            """Transform x according to the normalization mode."""
            # fmt: off
            modes = {
                "backward": Norm.NONE,
                "orthogonal": Norm.ORTH,
                "forward": Norm.FULL,
            } if is_dwt else {
                "backward": Norm.FULL,
                "orthogonal": Norm.ORTH,
                "forward": Norm.NONE,
            }
            # fmt: on
            if mode not in modes:
                raise ValueError(f"Invalid mode {mode}.")
            power = (modes[mode].value - norm.value) / 2
            x = scale(x.shape[0], x, power) if not is_dwt else x
            return scale(x.shape[0], f(x), power) if is_dwt else f(x)

        return transform

    return decorator


@partial(jit, static_argnums=1)
@normalize(is_dwt=True, norm=Norm.NONE)
def jax_dwt(x: Array) -> Array:
    """Apply the wavelet transform to x."""
    n = x.shape[0]
    while n > 1:
        even = x[:n:2]
        odd = x[1:n:2]
        x = x.at[0 : n >> 1].set(even + odd)
        x = x.at[n >> 1 : n].set(even - odd)
        n >>= 1
    return x


@partial(jit, static_argnums=1)
@normalize(is_dwt=False, norm=Norm.NONE)
def jax_iwt(x: Array) -> Array:
    """Apply the inverse wavelet transform to x."""
    n = x.shape[0]
    m = 2
    while m <= n:
        even = x[: m >> 1]
        odd = x[m >> 1 : m]
        x = x.at[0:m:2].set(even + odd)
        x = x.at[1:m:2].set(even - odd)
        m <<= 1
    return x


@normalize(is_dwt=True, norm=Norm.ORTH)
def pywt_dwt(x: Array) -> Array:
    """Apply the wavelet transform to x."""
    return jnp.concatenate(pywt.wavedec(np.array(x), "haar"))


def split_coeffs(x: Array) -> list[np.ndarray]:
    """Split x into coefficients as expected by pywavelets."""
    n = x.shape[0]
    coeffs = [np.array(x[0:1])]
    i = 1
    for level in range(n.bit_length() - 1):
        coeffs.append(np.array(x[i : i + (1 << level)]))
        i += 1 << level
    return coeffs


@normalize(is_dwt=False, norm=Norm.ORTH)
def pywt_iwt(x: Array) -> Array:
    """Apply the inverse wavelet transform to x."""
    return jnp.array(pywt.waverec(split_coeffs(x), "haar"))


def generate_testcases(rng: KeyArray, n: int, trials: int) -> list[Array]:
    """Generate testcases of size n."""
    subkeys = random.split(rng, num=trials)
    return [random.uniform(subkey, (n,)) for subkey in subkeys]


if __name__ == "__main__":
    K = 25
    trials = 20

    data = {"n": [], "algorithm": [], "time": [], "mode": [], "direction": []}
    methods = [
        (jax_dwt, True, True, "jax"),
        (pywt_dwt, True, True, "pywt"),
        (dwt_alloc, False, True, "alloc"),
        (dwt, False, True, "fast"),
        (jax_iwt, True, False, "jax"),
        (pywt_iwt, True, False, "pywt"),
        (iwt_alloc, False, False, "alloc"),
        (iwt, False, False, "fast"),
    ]
    for k in range(K):
        n = 1 << k
        rng, subkey = random.split(rng)
        testcases = generate_testcases(subkey, n, trials) + [
            jnp.arange(n, dtype=jnp.float64),
        ]
        for mode in ["backward", "orthogonal", "forward"]:
            for x in testcases:
                ans_dwt = jax_dwt(x, mode)
                ans_iwt = jax_iwt(x, mode)
                assert jnp.allclose(jax_iwt(ans_dwt, mode), x), "not inverse"
                assert jnp.allclose(jax_dwt(ans_iwt, mode), x), "not inverse"
                for i, (method, is_jax, forward, name) in enumerate(methods):
                    ans = ans_dwt if forward else ans_iwt
                    if is_jax:
                        # compile
                        method(jnp.copy(x)).block_until_ready()
                        y = jnp.copy(x)
                        start = time.time()
                        z = method(y, mode).block_until_ready()
                        elapsed = time.time() - start
                    else:
                        y = np.copy(np.array(x))
                        start = time.time()
                        z = method(y, mode)
                        elapsed = time.time() - start
                    data["n"].append(n)
                    data["algorithm"].append(name)
                    data["time"].append(elapsed)
                    data["mode"].append(mode)
                    data["direction"].append("fwd" if forward else "inv")
                    assert jnp.allclose(z, ans), f"{name} incorrect on {mode}."

    data = pd.DataFrame(data)
    sns.relplot(
        data=data,
        x="n",
        y="time",
        hue="algorithm",
        row="direction",
        col="mode",
        kind="line",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(figures / "wavelet_time.png")
    # plt.clf()
