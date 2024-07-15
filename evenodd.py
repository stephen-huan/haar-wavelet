from math import ceil

counter = 0


def __builtin_ctz(v: int) -> int:
    """Return the number of number of trailing zeros."""
    return (v & -v).bit_length() - 1


def left_rotate(i: int, j: int, n: int) -> int:
    """Return i left shifted j times."""
    k = __builtin_ctz(n)
    return ((i << j) | (i >> (k - j))) & (n - 1)


def is_minimum(i: int, n: int) -> bool:
    """Whether the i-th index is minimal in its chain."""
    m = n - 1
    return i == min((i << j) % m for j in range(n.bit_length() - 1)) or i == m


def biggest_skip(i: int, n: int) -> int:
    """Return the biggest possible skip."""
    for j in range(i + 1, n):
        if is_minimum(j, n):
            return j
    return n - 1


def tile_div(i: int, j: int, k: int) -> int:
    """Tile the top j bits of i into k bits."""
    top = i >> (k - j)
    z = (top << k) - top
    y = z // ((1 << j) - 1)
    return y + (y << j != y + z)


def tile(i: int, j: int, k: int) -> int:
    """Tile the top j bits of i into k bits."""
    j1s = (1 << j) - 1
    repeat = ((1 << k) - 1) // j1s
    left = k % j
    top = i >> (k - j)
    y = top * repeat
    y += top >> (j - left)
    if top != j1s:
        assert y == int(top * (1 << k) / j1s)
    y += (y & j1s) > top
    assert y == tile_div(i, j, k) == ceil(top * ((1 << k) - 1) / j1s)
    return y


def skip_slow(i: int, k: int) -> int:
    """Return the smallest minimal index >= i."""
    t = max(tile(i, j, k) for j in range(1, k))
    return skip_slow(t, k) if t > i else t


def skip(i: int, k: int) -> int:
    """Return the smallest minimal index > i."""
    lsb = k - __builtin_ctz(i)
    # equivalent to t = tile(i, lsb, k)
    t = tile(i, lsb, k) if i >= (1 << lsb) else i | 1
    # equivalent to skip(t, k) if t > i else t
    return skip(t, k) if t & 1 == 0 else t


def get_minima(k: int) -> list[int]:
    """Return all the minima from 0 to 2^k - 1."""
    m = (1 << k) - 1
    nums = []
    i = 1
    while i < m:
        nums.append(i)
        i = skip(i + 1, k)
    return nums


if __name__ == "__main__":
    k = 15
    n = 1 << k
    m = n - 1

    for j in range(1, k):
        for x in range(1 << k):
            assert (x <= left_rotate(x, j, n)) == (x >= tile(x, j, k))

    nums = []
    count = 0
    for i in range(1, n >> 1, 2):
        t = i + 2
        lsb = k - __builtin_ctz(t - 1)
        cond = t >> (k - lsb) == (1 << lsb) - 1
        if is_minimum(i, n):
            nums.append(i)
            assert is_minimum(t >> (k - lsb), 1 << lsb)
        if is_minimum(t >> (k - lsb), 1 << lsb):
            assert biggest_skip(i, n) == skip(i + 1, k), f"{i}"
            count += 1
    print(sum((1 << i) / i for i in range(1, k)), count, len(nums))
    assert nums == get_minima(k)

    counter = 0

    i = 1
    while i < m:
        j = skip(i + 1, k)
        for x in range(i + 1, j):
            assert not is_minimum(x, n)
        assert is_minimum(j, n)
        i = j
    # 15: 489 2190 0.223 / 3026 2190 1.382
    print(counter, len(nums), f"{counter / len(nums):.3f}")
