import sys


def solve() -> None:
    input = sys.stdin.readline
    n = int(input())
    sizes = list(map(int, input().split()))

    def feasible(k: int) -> bool:
        bottom_start = n - k
        return all(2 * sizes[i] <= sizes[bottom_start + i] for i in range(k))

    low, high = 0, n // 2 + 1
    while high - low > 1:
        middle = (low + high) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle

    print(low)


if __name__ == "__main__":
    solve()
