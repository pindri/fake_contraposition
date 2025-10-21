import math


def complexity(eps: float = 0.001, delta: float = 0.005, d: int = 2) -> int:
    """
    Eps-net Sample complexity, adapted from Mitzenbacher & Upfal: Probability and Computing Theorem 14.8 TODO: check
    """

    def f(s: int) -> float:
        expr = 1 - math.exp(-s * eps / 8)
        if expr <= 0:
            return float("inf")
        # all logs are natural logs
        return (2 / (math.log(2) * eps)) * (
                math.log(1 / delta)
                + d * math.log(2 * s)
                - math.log(expr)
        )

    low, high = 1, int(1e8)
    ans = None
    while low <= high:
        mid = (low + high) // 2
        if mid >= f(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1

    if ans is None:
        raise ValueError("No solution found in [1, 1e8]")
    return ans


if __name__ == '__main__':
    print(complexity(0.0001, 0.005))
