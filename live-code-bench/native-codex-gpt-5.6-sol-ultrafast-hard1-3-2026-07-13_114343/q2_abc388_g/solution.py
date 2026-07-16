import sys


def solve() -> None:
    input = sys.stdin.buffer.readline

    n = int(input())
    a = list(map(int, input().split()))

    # first[i] is the first index whose cake can be below cake i.
    # Since a is sorted, all first[i] can be found with two pointers.
    first = [n] * n
    j = 0
    for i, x in enumerate(a):
        if j < i + 1:
            j = i + 1
        while j < n and a[j] < 2 * x:
            j += 1
        first[i] = j

    # For a top at index i and a bottom at index t, compatibility is
    # first[i] <= t, or equivalently first[i] - i <= t - i.
    base = [first[i] - i for i in range(n)]

    # Sparse table for range maxima of first[i] - i.
    sparse = [base]
    width = 2
    while width <= n:
        half = width >> 1
        previous = sparse[-1]
        size = n - width + 1
        sparse.append(
            [
                previous[i]
                if previous[i] >= previous[i + half]
                else previous[i + half]
                for i in range(size)
            ]
        )
        width <<= 1

    def range_max(left: int, length: int) -> int:
        level = length.bit_length() - 1
        block = 1 << level
        row = sparse[level]
        x = row[left]
        y = row[left + length - block]
        return x if x >= y else y

    q = int(input())
    answers = []
    for _ in range(q):
        left, right = map(int, input().split())
        left -= 1
        length = right - left

        low = 0
        high = length // 2 + 1
        while high - low > 1:
            k = (low + high) // 2
            # Pair left+i with right-k+i. All k pairs work iff the
            # maximum displacement needed by a top is at most length-k.
            if range_max(left, k) <= length - k:
                low = k
            else:
                high = k
        answers.append(str(low))

    sys.stdout.write("\n".join(answers))


if __name__ == "__main__":
    solve()
