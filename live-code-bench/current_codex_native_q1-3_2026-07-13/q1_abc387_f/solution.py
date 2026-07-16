import sys

MOD = 998244353


def main() -> None:
    input = sys.stdin.readline
    n, m = map(int, input().split())
    parent = [x - 1 for x in map(int, input().split())]

    indeg = [0] * n
    children = [[] for _ in range(n)]
    for v, p in enumerate(parent):
        indeg[p] += 1
        children[p].append(v)

    # Peeling indegree-zero vertices leaves exactly the directed cycles.
    stack = [v for v in range(n) if indeg[v] == 0]
    in_cycle = [True] * n
    order = []
    while stack:
        v = stack.pop()
        in_cycle[v] = False
        order.append(v)
        p = parent[v]
        indeg[p] -= 1
        if indeg[p] == 0:
            stack.append(p)

    # exact[v][value-1] counts the subtree when x_v equals value.
    exact = [None] * n
    for v in order:
        ways = [1] * m
        for ch in children[v]:
            if in_cycle[ch]:
                continue
            pref = 0
            row = exact[ch]
            for value in range(m):
                pref = (pref + row[value]) % MOD
                ways[value] = ways[value] * pref % MOD
        exact[v] = ways

    seen = [False] * n
    answer = 1
    for start in range(n):
        if not in_cycle[start] or seen[start]:
            continue
        cycle = []
        v = start
        while not seen[v]:
            seen[v] = True
            cycle.append(v)
            v = parent[v]

        component = [1] * m
        for root in cycle:
            for ch in children[root]:
                if in_cycle[ch]:
                    continue
                pref = 0
                row = exact[ch]
                for value in range(m):
                    pref = (pref + row[value]) % MOD
                    component[value] = component[value] * pref % MOD
        answer = answer * (sum(component) % MOD) % MOD

    print(answer)


if __name__ == "__main__":
    main()
