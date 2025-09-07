import random
import numpy as np

def DNA_channel(w, x, Pd, Pi, Ps):
    n = len(x)
    if n == 0:
        return ""

    # Clamp w to [1, n]
    w = int(w)
    if w <= 0 or w > n:
        w = n

    # Choose random start position for window
    pos = np.random.randint(0, n - w + 1)

    # Prefix before window (unchanged)
    y = list(x[:pos])

    nucleotides = 'ACGT'
    probs = [1 - Pd - Pi - Ps, Pd, Pi, Ps]  # [no-op, deletion, insertion, substitution]

    for i in range(pos, pos + w):
        r = np.random.choice([0, 1, 2, 3], p=probs)
        if r == 0:
            # No error
            y.append(x[i])
        elif r == 1:
            # Deletion
            continue
        elif r == 2:
            # Insertion
            new_base = random.choice(nucleotides)
            y.append(new_base)
            y.append(x[i])
        elif r == 3:
            # Substitution
            possible_bases = [base for base in nucleotides if base != x[i]]
            new_base = random.choice(possible_bases)
            y.append(new_base)

    # Suffix after window (unchanged)
    y.extend(x[pos + w:])

    return ''.join(y)
