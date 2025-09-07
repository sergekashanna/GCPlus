def burst_patterns(K, d):
    P = []
    for i in range(0, K - d + 1):
        P.append(list(range(i, i + d)))
    return P