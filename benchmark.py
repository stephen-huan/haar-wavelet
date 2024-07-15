import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import Array, jit, random

from wavelet import evenodd  # pyright: ignore

jax.config.update("jax_enable_x64", True)
rng = random.key(0)

KeyArray = Array

figures = Path("figures")
figures.mkdir(parents=True, exist_ok=True)


@jit
def even_odd(x: Array) -> Array:
    """Apply the even-odd permutation to x."""
    return jnp.concatenate((x[::2], x[1::2]))


@jit
def inv_even_odd(x: Array) -> Array:
    """Apply the inverse even-odd permutation to x."""
    y = jnp.zeros_like(x)
    half = x.shape[0] >> 1
    return y.at[::2].set(x[:half]).at[1::2].set(x[half:]) if half > 0 else x


def generate_testcases(rng: KeyArray, n: int, trials: int) -> list[Array]:
    """Generate testcases of size n."""
    subkeys = random.split(rng, num=trials)
    return [random.uniform(subkey, (n,)) for subkey in subkeys]


if __name__ == "__main__":
    K = 25
    trials = 20

    data = {"n": [], "algorithm": [], "time": []}
    methods = [
        (even_odd, True, "jax naive"),
        (evenodd.alloc_even_odd, False, "alloc"),
        (evenodd.quicksort_even_odd, False, "quicksort"),
        (evenodd.cycle_even_odd, False, "cycle"),
        (evenodd.fast_even_odd, False, "fast"),
    ]
    inverse_methods = [
        (inv_even_odd, True, "jax naive"),
        (evenodd.alloc_inv_even_odd, False, "alloc"),
        (evenodd.fast_inv_even_odd, False, "fast"),
    ]
    for k in range(K):
        n = 1 << k
        rng, subkey = random.split(rng)
        testcases = generate_testcases(subkey, n, trials) + [
            jnp.arange(n, dtype=jnp.float64),
        ]
        for x in testcases:
            ans = even_odd(jnp.copy(x))
            for inverse_method, is_jax, name in inverse_methods:
                y = jnp.copy(ans) if is_jax else np.copy(np.array(ans))
                assert (inverse_method(y) == x).all(), f"{name} not inverse."
            for i, (method, is_jax, name) in enumerate(methods):
                if is_jax:
                    # compile
                    method(jnp.copy(x)).block_until_ready()
                    y = jnp.copy(x)
                    start = time.time()
                    z = method(y).block_until_ready()
                    elapsed = time.time() - start
                else:
                    y = np.copy(np.array(x))
                    start = time.time()
                    z = method(y)
                    elapsed = time.time() - start
                data["n"].append(n)
                data["algorithm"].append(name)
                data["time"].append(elapsed)
                assert (z == ans).all(), f"{name} incorrect."

    data = pd.DataFrame(data)
    sns.lineplot(data=data, x="n", y="time", hue="algorithm")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(figures / "evenodd_time.png")
    plt.clf()
