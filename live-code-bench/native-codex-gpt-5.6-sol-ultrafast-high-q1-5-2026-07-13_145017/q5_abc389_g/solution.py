import sys


def build_states(k):
    """Build the DAG of possible BFS-layer prefixes.

    A state is (even vertices used, odd vertices used, last layer size,
    parity of the next layer), where parity 0 means even and 1 means odd.
    """
    start = (1, 0, 1, 1)
    reachable = {start}

    # Every transition increases e + o, so this is a topological order.
    for total in range(1, 2 * k):
        current = [s for s in reachable if s[0] + s[1] == total]
        for e, o, last, parity in current:
            room = k - (o if parity else e)
            for size in range(1, room + 1):
                if parity:
                    nxt = (e, o + size, size, 0)
                else:
                    nxt = (e + size, o, size, 1)
                reachable.add(nxt)

    states = sorted(reachable, key=lambda s: s[0] + s[1])
    index = {state: i for i, state in enumerate(states)}
    edges = [[] for _ in states]

    for i, (e, o, last, parity) in enumerate(states):
        room = k - (o if parity else e)
        for size in range(1, room + 1):
            if parity:
                nxt = (e, o + size, size, 0)
            else:
                nxt = (e + size, o, size, 1)
            edges[i].append((index[nxt], last, size))

    terminals = [
        i for i, (e, o, _last, _parity) in enumerate(states)
        if e == k and o == k
    ]
    return states, edges, index[start], terminals


def interpolate_consecutive(values, mod):
    """Return monomial coefficients from f(0), f(1), ..., f(d)."""
    degree = len(values) - 1

    # Newton coefficients for f(x) = sum delta[j] * binom(x, j).
    work = values[:]
    delta = []
    length = degree + 1
    while length:
        delta.append(work[0])
        for i in range(length - 1):
            work[i] = (work[i + 1] - work[i]) % mod
        length -= 1

    inv = [0] * (degree + 1)
    if degree >= 1:
        inv[1] = 1
        for i in range(2, degree + 1):
            inv[i] = mod - (mod // i) * inv[mod % i] % mod

    answer = [0] * (degree + 1)
    basis = [1]  # binom(x, 0)
    answer[0] = delta[0]

    for j in range(1, degree + 1):
        # binom(x,j) = binom(x,j-1) * (x-(j-1)) / j
        old = basis
        basis = [0] * (j + 1)
        shift = j - 1
        inv_j = inv[j]
        for t, value in enumerate(old):
            basis[t] = (basis[t] - shift * value) % mod
            basis[t + 1] = (basis[t + 1] + value) % mod
        for t in range(j + 1):
            basis[t] = basis[t] * inv_j % mod

        weight = delta[j]
        if weight:
            for t in range(j + 1):
                answer[t] = (answer[t] + weight * basis[t]) % mod

    return answer


def solve(n, mod):
    k = n // 2
    max_edges = n * (n - 1) // 2

    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], mod - 2, mod)
    for i in range(n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod

    states, edges, start, terminals = build_states(k)
    state_count = len(states)
    values = []

    # Evaluate the edge-count generating polynomial at enough points.
    for x in range(max_edges + 1):
        q = (x + 1) % mod

        # transition[a][b] counts the choices involving a new layer of
        # size b after a layer of size a, including the 1/b! EGF weight.
        transition = [[0] * (k + 1) for _ in range(k + 1)]
        q_power = [1] * (max_edges + 1)
        for i in range(1, max_edges + 1):
            q_power[i] = q_power[i - 1] * q % mod
        for a in range(1, k + 1):
            nonempty_neighbors = (q_power[a] - 1) % mod
            power = 1
            for b in range(1, k + 1):
                power = power * nonempty_neighbors % mod
                inside = b * (b - 1) // 2
                transition[a][b] = (
                    power * q_power[inside] % mod * inv_fact[b] % mod
                )

        dp = [0] * state_count
        dp[start] = 1
        for i in range(state_count):
            value = dp[i]
            if not value:
                continue
            for target, a, b in edges[i]:
                dp[target] = (
                    dp[target] + value * transition[a][b]
                ) % mod

        total = sum(dp[i] for i in terminals) % mod
        values.append(total * fact[n - 1] % mod)

    coefficients = interpolate_consecutive(values, mod)
    return coefficients[n - 1:]


def main():
    n, mod = map(int, sys.stdin.readline().split())
    print(*solve(n, mod))


if __name__ == "__main__":
    main()
