#!/usr/bin/env python3
import os, sys, math, time
import numpy as np

# ================== Path setup (like your other scripts) ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ================== Local imports ==================
# Adjust these names if your files are named differently.
from GCP_Encode_DNA import GCP_Encode_DNA_localized
from GCP_Decode_DNA import GCP_Decode_DNA_localized
from DNA_channel import DNA_channel
from preCompute_Patterns import burst_patterns

# ================== Code parameters ==================
k = 168
l = 8
len_last = (k - 1) % l + 1
K = int(math.ceil(k / l))

# ================== Simulation setup ==================
total_iter = int(1e4)
w_values = [2, 3, 4, 5] 
Pe = 0.99
Pd = Pe / 3.0
Pi = Pe / 3.0
Ps = Pe / 3.0

# ================== Initializations ==================
success   = np.zeros(len(w_values), dtype=float)
code_rate = np.zeros(len(w_values), dtype=float)
inf_density = np.zeros(len(w_values), dtype=float)
code_length_DNA = np.zeros(len(w_values), dtype=int)
error_rate= np.zeros(len(w_values), dtype=float)
failures  = np.zeros(len(w_values), dtype=float)
errors    = np.zeros(len(w_values), dtype=float)
dec_time  = np.zeros(len(w_values), dtype=float)

# ================== Main ==================
if __name__ == "__main__":
    rng = np.random.default_rng()

    for j, w in enumerate(w_values):
        c1 = 2
        c2 = 2

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
            x, n, N, K_eff, q = GCP_Encode_DNA_localized(u, l, c1+c2, w)

            # Pass through channel (edit channel with window w)
            y = DNA_channel(w, x, Pd, Pi, Ps)

            # Decode (localized) — handle either (uhat) or (uhat, _)
            t0 = time.perf_counter()
            uhat = GCP_Decode_DNA_localized(
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
        W = 3 * (2*w + 1) + 1
        dec_time[j]  = total_dec_time / float(total_iter)
        success[j]   = countS / float(total_iter)
        failures[j]  = countF / float(total_iter)
        errors[j]    = countE / float(total_iter)
        code_rate[j] = k / float(k + W + (c1 + c2) * l)
        inf_density[j] = 2*code_rate[j] 
        error_rate[j]= (Pe * w) / float((k + W + (c1 + c2) * l)/2.0)
        code_length_DNA[j] =  (k + W + (c1 + c2) * l) // 2

        print(
            f"w = {w:2d}  Inf. Density = {inf_density[j]:.2f} Length (NTs) = {code_length_DNA[j]:2d} | "
            f"Fail = {failures[j]:.4e}  Err = {errors[j]:.4e}  "
            f"Tot = {(1.0 - success[j]):.4e}  DecTime = {dec_time[j]:.4e} s  "
            f"Avg. Edit Rate = {error_rate[j]:.4f}"
        )
