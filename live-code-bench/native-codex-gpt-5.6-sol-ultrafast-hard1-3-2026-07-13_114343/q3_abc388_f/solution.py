import sys


def solve() -> None:
    input = sys.stdin.readline
    n, m, a, b = map(int, input().split())
    blocked = [tuple(map(int, input().split())) for _ in range(m)]

    # With a fixed jump length, every visited square is 1 modulo a.
    if a == b:
        if (n - 1) % a != 0:
            print("No")
            return

        for left, right in blocked:
            first_visited = 1 + (left - 1 + a - 1) // a * a
            if first_visited <= right:
                print("No")
                return

        print("Yes")
        return

    # Bit k says whether (current_position - k) is reachable.
    full = (1 << b) - 1
    jump_sources = full ^ ((1 << (a - 1)) - 1)
    state = 1  # Square 1 is reachable.

    # Because both a and a+1 are allowed, every distance of at least
    # a*(a-1) can be formed. After this many steps plus enough room to
    # fill the b-position state window, a nonzero state becomes `full`.
    saturation_steps = a * (a - 1) + 2 * b

    def advance_good(length: int, state: int) -> int:
        steps = min(length, saturation_steps)
        for _ in range(steps):
            reachable = 1 if state & jump_sources else 0
            state = ((state << 1) & full) | reachable

        if length > steps:
            # The simulated prefix has already made every bit reachable,
            # and an all-one state stays all-one on good squares.
            state = full
        return state

    current = 1
    for left, right in blocked:
        state = advance_good(left - current - 1, state)

        bad_length = right - left + 1
        if bad_length >= b:
            print("No")
            return
        state = (state << bad_length) & full
        if state == 0:
            print("No")
            return

        current = right

    state = advance_good(n - current, state)
    print("Yes" if state & 1 else "No")


if __name__ == "__main__":
    solve()
