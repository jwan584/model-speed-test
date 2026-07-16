import sys
from array import array
from bisect import bisect_left


def main() -> None:
    input = sys.stdin.readline
    n = int(input())
    a = list(map(int, input().split()))

    # need[i] is the first possible lower-piece index for upper piece i.
    # Pairing k smallest with k largest is possible iff
    # max(need[i]-i, i in [l,l+k)) <= length-k.
    base = array("i", (bisect_left(a, 2 * x) - i for i, x in enumerate(a)))
    table = [base]
    length = 2
    while length <= n:
        prev = table[-1]
        half = length >> 1
        size = n - length + 1
        table.append(array("i", (max(prev[i], prev[i + half]) for i in range(size))))
        length <<= 1

    logs = [0] * (n + 1)
    for i in range(2, n + 1):
        logs[i] = logs[i >> 1] + 1

    def range_max(left: int, right: int) -> int:
        size = right - left
        level = logs[size]
        width = 1 << level
        row = table[level]
        return max(row[left], row[right - width])

    out = []
    for _ in range(int(input())):
        left, right = map(int, input().split())
        left -= 1
        length = right - left
        low, high = 0, length // 2 + 1
        while high - low > 1:
            k = (low + high) // 2
            if range_max(left, left + k) <= length - k:
                low = k
            else:
                high = k
        out.append(str(low))
    print("\n".join(out))


if __name__ == "__main__":
    main()
