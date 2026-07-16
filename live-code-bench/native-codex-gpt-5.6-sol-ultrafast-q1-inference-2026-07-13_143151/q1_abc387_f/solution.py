import sys
from collections import deque


MOD = 998244353


def solve() -> None:
    input = sys.stdin.buffer.readline
    n, m = map(int, input().split())
    parent = [x - 1 for x in map(int, input().split())]

    # indegree[v] is the number of vertices whose outgoing edge ends at v.
    # Repeatedly removing indegree-zero vertices leaves exactly the cycles.
    indegree = [0] * n
    for v in parent:
        indegree[v] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)

    # If acc[v] is not None, acc[v][k-1] is the product of the
    # contributions of the already processed child subtrees when x_v = k.
    acc = [None] * n

    while queue:
        v = queue.popleft()
        ways = acc[v]
        acc[v] = None

        # Convert ways for a fixed x_v into the contribution to v's parent:
        # x_v may be any value not exceeding the parent's value.
        if ways is None:
            contribution = list(range(1, m + 1))
        else:
            prefix = 0
            for k in range(m):
                prefix += ways[k]
                if prefix >= MOD:
                    prefix -= MOD
                ways[k] = prefix
            contribution = ways

        p = parent[v]
        if acc[p] is None:
            acc[p] = contribution
        else:
            parent_ways = acc[p]
            for k in range(m):
                parent_ways[k] = parent_ways[k] * contribution[k] % MOD

        indegree[p] -= 1
        if indegree[p] == 0:
            queue.append(p)

    # Every remaining component is a directed cycle. Inequalities around a
    # cycle force all its vertices to have the same value.
    answer = 1
    seen = [False] * n
    for start in range(n):
        if indegree[start] == 0 or seen[start]:
            continue

        cycle = []
        v = start
        while not seen[v]:
            seen[v] = True
            cycle.append(v)
            v = parent[v]

        cycle_ways = [1] * m
        for v in cycle:
            if acc[v] is None:
                continue
            subtree_ways = acc[v]
            for k in range(m):
                cycle_ways[k] = cycle_ways[k] * subtree_ways[k] % MOD

        answer = answer * (sum(cycle_ways) % MOD) % MOD

    print(answer)


if __name__ == "__main__":
    solve()
