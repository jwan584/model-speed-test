import sys
from array import array
from collections import deque


MOD = 998244353


def solve() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    n, m = data[0], data[1]
    parent = [x - 1 for x in data[2:]]

    # Remove all vertices outside cycles.  The removal order puts every
    # vertex after all of its children (vertices pointing to it).
    indegree = [0] * n
    for v in parent:
        indegree[v] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        v = parent[u]
        indegree[v] -= 1
        if indegree[v] == 0:
            queue.append(v)

    on_cycle = [d > 0 for d in indegree]

    # dp[u][value] is the number of assignments in u's in-tree when
    # x_u equals value.  Arrays are accumulated directly into each parent.
    dp = [None] * n
    ones = array("I", [1]) * (m + 1)

    for u in order:
        current = dp[u]
        if current is None:
            current = ones

        v = parent[u]
        target = dp[v]
        if target is None:
            target = array("I", ones)

        prefix = 0
        for value in range(1, m + 1):
            prefix += current[value]
            if prefix >= MOD:
                prefix -= MOD
            target[value] = target[value] * prefix % MOD

        dp[v] = target
        dp[u] = None

    # All values around a directed cycle must be equal.  Multiply the
    # attached-tree contributions for each possible common value.
    seen = [False] * n
    answer = 1
    for start in range(n):
        if not on_cycle[start] or seen[start]:
            continue

        cycle = []
        u = start
        while not seen[u]:
            seen[u] = True
            cycle.append(u)
            u = parent[u]

        product = array("I", ones)
        for u in cycle:
            contribution = dp[u]
            if contribution is not None:
                for value in range(1, m + 1):
                    product[value] = (
                        product[value] * contribution[value] % MOD
                    )
                dp[u] = None

        component_ways = 0
        for value in range(1, m + 1):
            component_ways += product[value]
            if component_ways >= MOD:
                component_ways -= MOD
        answer = answer * component_ways % MOD

    print(answer)


if __name__ == "__main__":
    solve()
