import sys
from collections import deque


MOD = 998244353


def solve() -> None:
    input = sys.stdin.buffer.readline
    n, m = map(int, input().split())
    parent = [x - 1 for x in map(int, input().split())]

    # Peel every non-cycle vertex, starting at leaves of the reverse graph.
    indegree = [0] * n
    for p in parent:
        indegree[p] += 1

    queue = deque(i for i, deg in enumerate(indegree) if deg == 0)

    # contribution[v][t-1] is the product of the contributions from all
    # already processed children when x_v = t.  None represents all ones.
    contribution = [None] * n

    while queue:
        v = queue.popleft()
        prefix = contribution[v]

        if prefix is None:
            # v is a leaf: its DP is 1 for every fixed value, so its prefix
            # sums are 1, 2, ..., m.
            prefix = list(range(1, m + 1))
        else:
            running = 0
            for value in range(m):
                running += prefix[value]
                if running >= MOD:
                    running -= MOD
                prefix[value] = running

        p = parent[v]
        current = contribution[p]
        if current is None:
            contribution[p] = prefix
        else:
            for value in range(m):
                current[value] = current[value] * prefix[value] % MOD

        indegree[p] -= 1
        if indegree[p] == 0:
            queue.append(p)

    # The remaining vertices are exactly the directed cycles.  All values on
    # one cycle must be equal, so multiply their attached-tree contributions
    # for each possible common value and then sum over that value.
    answer = 1
    seen = [False] * n

    for start in range(n):
        if indegree[start] == 0 or seen[start]:
            continue

        cycle_dp = None
        v = start
        while True:
            seen[v] = True
            current = contribution[v]
            if current is not None:
                if cycle_dp is None:
                    cycle_dp = current
                else:
                    for value in range(m):
                        cycle_dp[value] = cycle_dp[value] * current[value] % MOD

            v = parent[v]
            if v == start:
                break

        ways = m if cycle_dp is None else sum(cycle_dp) % MOD
        answer = answer * ways % MOD

    print(answer)


if __name__ == "__main__":
    solve()
