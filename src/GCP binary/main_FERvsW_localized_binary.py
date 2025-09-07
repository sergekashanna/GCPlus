#!/usr/bin/env python3
import os, sys, math, time
import numpy as np

# ================== Path setup (like your other scripts) ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ================== Local imports ==================
# Adjust these names if your files are named differently.
from GCP_Encode_binary import GCP_Encode_binary_localized
from GCP_Decode_binary import GCP_Decode_binary_localized
from binary_channel import binary_channel
from preCompute_Patterns import burst_patterns

# ================== Code parameters ==================
k = 140
l = int(math.floor(math.log2(k)))
len_last = (k - 1) % l + 1
K = int(math.ceil(k / l))

# ================== Simulation setup ==================
total_iter = int(1e4)
w_values = list(range(l + 1, 4 * l + 2, l))  # l+1 : l : 4l+1 (inclusive)
Pe = 0.99
Pd = Pe / 3.0
Pi = Pe / 3.0
Ps = Pe / 3.0

# ================== Initializations ==================
success   = np.zeros(len(w_values), dtype=float)
code_rate = np.zeros(len(w_values), dtype=float)
error_rate= np.zeros(len(w_values), dtype=float)
failures  = np.zeros(len(w_values), dtype=float)
errors    = np.zeros(len(w_values), dtype=float)
dec_time  = np.zeros(len(w_values), dtype=float)

# ================== Main ==================
if __name__ == "__main__":
    rng = np.random.default_rng()

    for j, w in enumerate(w_values):
        c1 = int((w - 1) // l + 1)
        c  = 2 * c1
        c2 = c - c1

        # Patterns for primary check
        P1 = burst_patterns(K, c1)

        # Counters
        countS = 0
        countF = 0
        countE = 0
        total_dec_time = 0.0

        for _ in range(total_iter):
            # Random info bits
            u = rng.integers(0, 2, size=k, dtype=int).tolist()

            # Encode (localized)
            x, n, N, K_eff, q = GCP_Encode_binary_localized(u, l, c, w)

            # Pass through channel (edit channel with window w)
            y = binary_channel(w, x, Pd, Pi, Ps)

            # Decode (localized) — handle either (uhat) or (uhat, _)
            t0 = time.perf_counter()
            uhat = GCP_Decode_binary_localized(
                y=y, n=n, k=k, l=l, N=N, K=K_eff, c1=c1, c2=c2, q=q,
                len_last=len_last, P1=P1, w=w
            )
            total_dec_time += (time.perf_counter() - t0)
            
            # Tally
            if uhat == u:
                countS += 1
            elif not uhat:
                countF += 1
            else:
                countE += 1

        # Stats
        dec_time[j]  = total_dec_time / float(total_iter)
        success[j]   = countS / float(total_iter)
        failures[j]  = countF / float(total_iter)
        errors[j]    = countE / float(total_iter)
        code_rate[j] = k / float(k + 3 * (w + 1) + (c1 + c2) * l)
        error_rate[j]= (Pe * w) / float(k + 3 * (w + 1) + (c1 + c2) * l)

        print(
            f"w = {w:2d}  Code Rate = {code_rate[j]:.2f} | "
            f"Fail = {failures[j]:.4e}  Err = {errors[j]:.4e}  "
            f"Tot = {(1.0 - success[j]):.4e}  DecTime = {dec_time[j]:.4e} s  "
            f"Avg. Edit Rate = {error_rate[j]:.4f}"
        )
