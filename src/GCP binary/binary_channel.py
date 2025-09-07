import numpy as np

def binary_channel(w, x, Pd, Pi, Ps):
    """
    Random edit channel - binary case (Python version of MATLAB code).
    Applies edits only on a random window of length w; rest is copied as-is.

    Args:
        w  : window length (int), w=len(x) for IID setting
        x  : list/array of 0/1 ints
        Pd : deletion probability
        Pi : insertion probability
        Ps : substitution probability

    Returns:
        y: list of 0/1 ints after edits
    """
    x = list(x)
    n = len(x)
    if n == 0:
        return []

    # Clamp w to [1, n]
    w = int(w)
    if w <= 0 or w > n:
        w = n

    # Choose start position (0-based) uniformly in [0, n-w]
    pos = np.random.randint(0, n - w + 1)

    # Prefix before the window
    y = x[:pos]

    probs = [1 - Pd - Pi - Ps, Pd, Pi, Ps]  # [no-op, deletion, insertion, substitution]
    for i in range(pos, pos + w):
        r = np.random.choice([0, 1, 2, 3], p=probs)
        if r == 0:
            # No edit
            y.append(x[i])
        elif r == 1:
            # Deletion: append nothing
            continue
        elif r == 2:
            # Insertion: random bit before original
            y.append(np.random.randint(0, 2))
            y.append(x[i])
        elif r == 3:
            # Substitution: flip bit
            y.append(1 - x[i])

    # Suffix after the window
    y.extend(x[pos + w:])

    return y
