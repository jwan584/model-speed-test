import sys


def solve() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)
    n = next(it)
    a = [next(it) for _ in range(n)]

    # nxt[i] is the first position whose mochi can be below mochi i.
    # Since a is sorted, all nxt values can be found with two pointers.
    nxt = [n] * n
    j = 0
    for i, value in enumerate(a):
        if j < i + 1:
            j = i + 1
        twice = value * 2
        while j < n and a[j] < twice:
            j += 1
        nxt[i] = j

    distance = [nxt[i] - i for i in range(n)]

    # Sparse table for range maxima of distance.
    sparse = [distance]
    half = 1
    while half * 2 <= n:
        previous = sparse[-1]
        size = n - half * 2 + 1
        current = [0] * size
        for i in range(size):
            x = previous[i]
            y = previous[i + half]
            current[i] = x if x >= y else y
        sparse.append(current)
        half *= 2

    q = next(it)
    answers = []
    for _ in range(q):
        left = next(it) - 1
        right = next(it) - 1

        low = 0
        high = (right - left + 1) // 2 + 1
        while high - low > 1:
            k = (low + high) // 2

            level = k.bit_length() - 1
            block = 1 << level
            row = sparse[level]
            maximum_distance = max(
                row[left], row[left + k - block]
            )

            # Pair a[left+t] with a[right-k+1+t].  This works for
            # every t exactly when the following maximum is small enough.
            if maximum_distance <= right - k + 1 - left:
                low = k
            else:
                high = k

        answers.append(str(low))

    sys.stdout.write("\n".join(answers) + "\n")


if __name__ == "__main__":
    solve()
