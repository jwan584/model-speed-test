import sys


def solve() -> None:
    input = sys.stdin.readline
    n, m, a, b = map(int, input().split())
    intervals = [tuple(map(int, input().split())) for _ in range(m)]

    if a == b:
        if (n - 1) % a != 0:
            print("No")
            return

        for left, right in intervals:
            first = 1 + ((left - 1 + a - 1) // a) * a
            if first <= right:
                print("No")
                return

        print("Yes")
        return

    full_mask = (1 << b) - 1
    jump_mask = full_mask ^ ((1 << (a - 1)) - 1)
    good_limit = a * (a - 1) + 2 * b

    # Bit k records whether the square k places behind the latest processed
    # square is reachable. Initially, only square 1 is reachable.
    state = 1

    def process_good(length: int, state: int) -> int:
        steps = min(length, good_limit)
        for _ in range(steps):
            if state == 0 or state == full_mask:
                break
            reachable = 1 if state & jump_mask else 0
            state = ((state << 1) & full_mask) | reachable
        return state

    def process_bad(length: int, state: int) -> int:
        return (state << min(length, b)) & full_mask

    next_square = 2
    for left, right in intervals:
        state = process_good(left - next_square, state)
        state = process_bad(right - left + 1, state)
        if state == 0:
            print("No")
            return
        next_square = right + 1

    state = process_good(n - next_square + 1, state)
    print("Yes" if state & 1 else "No")


if __name__ == "__main__":
    solve()
