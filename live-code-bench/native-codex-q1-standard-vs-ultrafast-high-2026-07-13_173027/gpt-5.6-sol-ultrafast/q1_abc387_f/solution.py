import sys
from collections import deque


MOD = 998244353


def solve() -> None:
    input = sys.stdin.buffer.readline
    n, m = map(int, input().split())
    a = [x - 1 for x in map(int, input().split())]

    # Peel off all non-cycle vertices of the functional graph.
    indegree = [0] * n
    for parent in a:
        indegree[parent] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)
    on_cycle = [True] * n
    while queue:
        v = queue.popleft()
        on_cycle[v] = False
        parent = a[v]
        indegree[parent] -= 1
        if indegree[parent] == 0:
            queue.append(parent)

    # Contract each directed cycle into a single component. Every other
    # vertex is its own component.
    component = [-1] * n
    component_count = 0
    for start in range(n):
        if on_cycle[start] and component[start] == -1:
            v = start
            while True:
                component[v] = component_count
                v = a[v]
                if v == start:
                    break
            component_count += 1

    for v in range(n):
        if not on_cycle[v]:
            component[v] = component_count
            component_count += 1

    # The contracted graph is a forest whose edges point toward its roots.
    parent_component = [-1] * component_count
    child_count = [0] * component_count
    for v in range(n):
        child = component[v]
        parent = component[a[v]]
        if child != parent:
            parent_component[child] = parent
            child_count[parent] += 1

    # dp_v[value] is the number of assignments in v's subtree when v has
    # that value. Its initial value is 1, and each child contributes the
    # prefix sum of the child's DP.
    ready = deque(c for c in range(component_count) if child_count[c] == 0)
    dp = [None] * component_count
    answer = 1
    mod = MOD

    while ready:
        v = ready.popleft()
        current = dp[v]  # None represents the all-ones array.
        parent = parent_component[v]

        if parent == -1:
            ways = m if current is None else sum(current) % mod
            answer = answer * ways % mod
            continue

        target = dp[parent]
        if target is None:
            target = [1] * m
            dp[parent] = target

        prefix = 0
        if current is None:
            for value in range(m):
                prefix += 1
                target[value] = target[value] * prefix % mod
        else:
            for value in range(m):
                prefix += current[value]
                if prefix >= mod:
                    prefix -= mod
                target[value] = target[value] * prefix % mod

        dp[v] = None
        child_count[parent] -= 1
        if child_count[parent] == 0:
            ready.append(parent)

    print(answer)


if __name__ == "__main__":
    solve()
