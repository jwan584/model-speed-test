import sys
from collections import deque


MOD = 998244353


def solve() -> None:
    input = sys.stdin.buffer.readline
    n, m = map(int, input().split())
    parent_original = [x - 1 for x in map(int, input().split())]

    # Remove all vertices outside directed cycles.
    indegree = [0] * n
    for p in parent_original:
        indegree[p] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)
    while queue:
        v = queue.popleft()
        p = parent_original[v]
        indegree[p] -= 1
        if indegree[p] == 0:
            queue.append(p)

    on_cycle = [degree > 0 for degree in indegree]

    # Contract every directed cycle into one representative vertex.
    representative = [-1] * n
    roots = []
    for start in range(n):
        if on_cycle[start] and representative[start] == -1:
            roots.append(start)
            v = start
            while True:
                representative[v] = start
                v = parent_original[v]
                if v == start:
                    break

    # The contracted graph is a forest, with edges from child to parent.
    children = [[] for _ in range(n)]
    parent = [-1] * n
    for v in range(n):
        if on_cycle[v]:
            continue
        p = parent_original[v]
        if on_cycle[p]:
            p = representative[p]
        parent[v] = p
        children[p].append(v)

    # Put a largest subtree first.  The DP reuses its array, keeping peak
    # memory small even for a deep tree with many side branches.
    traversal = []
    stack = roots.copy()
    while stack:
        v = stack.pop()
        traversal.append(v)
        stack.extend(children[v])

    subtree_size = [1] * n
    for v in reversed(traversal):
        for child in children[v]:
            subtree_size[v] += subtree_size[child]

    for v in traversal:
        if len(children[v]) >= 2:
            largest = max(range(len(children[v])),
                          key=lambda i: subtree_size[children[v][i]])
            children[v][0], children[v][largest] = (
                children[v][largest], children[v][0]
            )

    # dp_v[k] is the number of assignments in v's subtree when x_v = k+1.
    # A child's contribution is prefix_sum(dp_child)[k].
    accumulated = [None] * n
    answer = 1

    for root in roots:
        stack = [(root, False)]
        while stack:
            v, exiting = stack.pop()
            if not exiting:
                stack.append((v, True))
                for child in reversed(children[v]):
                    stack.append((child, False))
                continue

            current = accumulated[v]
            if current is None:
                current = [1] * m

            if v == root:
                answer = answer * (sum(current) % MOD) % MOD
                accumulated[v] = None
                continue

            p = parent[v]
            parent_dp = accumulated[p]
            prefix = 0

            if parent_dp is None:
                # The first child array can be reused for the parent.
                for value in range(m):
                    prefix += current[value]
                    if prefix >= MOD:
                        prefix -= MOD
                    current[value] = prefix
                accumulated[p] = current
            else:
                for value in range(m):
                    prefix += current[value]
                    if prefix >= MOD:
                        prefix -= MOD
                    parent_dp[value] = parent_dp[value] * prefix % MOD

            accumulated[v] = None

    print(answer)


if __name__ == "__main__":
    solve()
