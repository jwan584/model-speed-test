import sys
from collections import deque


def main() -> None:
    input = sys.stdin.readline
    n, m, a, b = map(int, input().split())
    blocked = [tuple(map(int, input().split())) for _ in range(m)]

    # State contains reachability of the last B processed squares.
    state = deque([False] * (b - 1) + [True], maxlen=b)
    position = 1

    def advance_good(steps: int) -> None:
        nonlocal state
        if steps <= 0:
            return
        if a == b:
            state.rotate(-(steps % a))
            return
        while steps and not (all(state) or not any(state)):
            reachable = any(list(state)[: b - a + 1])
            state.append(reachable)
            steps -= 1
        # Both all-true and all-false are fixed points on good squares.

    def advance_bad(steps: int) -> None:
        nonlocal state
        if steps >= b:
            state = deque([False] * b, maxlen=b)
        else:
            for _ in range(steps):
                state.append(False)

    for left, right in blocked:
        good = left - position - 1
        advance_good(good)
        position += good
        bad = right - left + 1
        advance_bad(bad)
        position += bad

    advance_good(n - position)
    print("Yes" if state[-1] else "No")


if __name__ == "__main__":
    main()
