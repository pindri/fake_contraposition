from math import log


def complexity(eps: float = 0.05, delta: float = 0.05) -> int:
    n = 0
    increment = 1000 * int((1 / eps) * (1 / delta))
    while increment >= 1:
        n += increment if union_bound_exact_inequality(n + increment, eps, delta) else 0
        increment = int(increment / 2)
    return n


def union_bound_simple_inequality(n: int, eps: float = 0.05, delta: float = 0.05) -> bool:
    """
    obtained via growth function for VC dim 2, upper-bounding by n^2, and using the union bound.
    """
    return 2 * log(n) + n * log(1 - eps) > log(delta)


def union_bound_exact_inequality(n: int, eps: float = 0.05, delta: float = 0.05) -> bool:
    """
    obtained via growth function for VC dim 2 exactly and using the union bound.
    """
    return log((n * (n-1) + 2*n + 2)/2) + n * log(1 - eps) > log(delta)


if __name__ == '__main__':
    print(complexity(0.0001, 0.01))
