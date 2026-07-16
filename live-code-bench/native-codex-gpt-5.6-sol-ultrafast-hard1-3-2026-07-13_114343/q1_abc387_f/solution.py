import sys
from collections import deque


MOD = 998244353


def solve() -> None:
    input = sys.stdin.buffer.readline
    n, m = map(int, input().split())
    parent = [x - 1 for x in map(int, input().split())]

    # Remove all vertices outside directed cycles.  A removed vertex appears
    # after all of its children, which is exactly the order needed by the DP.
    indegree = [0] * n
    for p in parent:
        indegree[p] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        p = parent[v]
        indegree[p] -= 1
        if indegree[p] == 0:
            queue.append(p)

    on_cycle = [d > 0 for d in indegree]

    # All vertices on one cycle have equal values.  Give every cycle vertex
    # the representative of its contracted component.
    representative = [-1] * n
    roots = []
    for start in range(n):
        if on_cycle[start] and representative[start] == -1:
            roots.append(start)
            v = start
            while True:
                representative[v] = start
                v = parent[v]
                if v == start:
                    break

    # pending[v][j-1] is the product of the contributions already received
    # from v's children when x_v = j.  Arrays are passed upward and discarded
    # as soon as their subtree is complete.
    pending = [None] * n

    for v in order:
        contribution = pending[v]
        if contribution is None:
            # A leaf has one assignment for each fixed value.  Its prefix
            # counts (values <= j) are therefore 1, 2, ..., M.
            contribution = list(range(1, m + 1))
        else:
            # Convert counts for x_v = j into counts for x_v <= j.
            prefix = 0
            for j in range(m):
                prefix += contribution[j]
                if prefix >= MOD:
                    prefix -= MOD
                contribution[j] = prefix
            pending[v] = None

        p = parent[v]
        target = representative[p] if on_cycle[p] else p
        current = pending[target]
        if current is None:
            pending[target] = contribution
        else:
            for j in range(m):
                current[j] = current[j] * contribution[j] % MOD

    answer = 1
    for root in roots:
        counts = pending[root]
        component_count = m if counts is None else sum(counts) % MOD
        answer = answer * component_count % MOD

    print(answer)


if __name__ == "__main__":
    solve()
