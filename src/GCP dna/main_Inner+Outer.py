# -*- coding: utf-8 -*-
"""
Iteration-parallel Coarse->Bracket->Refine search for minimal C per Pe (Python 3.7)
- Parallelizes over iterations/files (CHUNK_SIZE across current_total_iter).
- Search rule: if retrieval != 0, ALWAYS move to lower rate (higher C) until zero,
  then refine between the first OK and the nearest FAIL above it.
- Windows-safe prints (ASCII-only messages).
"""

import os
import sys
import time
import math
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt  # kept as in your original (unused here but harmless)

# ===== Local imports (ensure these modules are on sys.path) =====
from GCP_Encode_DNA import GCP_Encode_DNA_brute, binary_to_decimal_blocks, binary_to_dna
from DNA_channel import DNA_channel
from GCP_Decode_DNA import GCP_Decode_DNA_brute, dna_to_binary
from preCompute_Patterns import preCompute_Patterns
from Outer_Encode import Outer_RS_Encode

# ================== Code parameters (single c1) ==================
k = 168
c1 = 6          # <-- single value of c1
c2 = 1
l_in = 8
l_out = 14
K_in = int(math.ceil(k / l_in))
K_out = k // l_out
len_last = (k - 1) % l_in + 1
M = 10**4        # number of information rows (outer code dimension)

# ================== Inner precompute ==================
lim = 5           # precompute patterns for |Delta| = 0..lim-1
lambda_depths = [0]*lim
lambda_depths[0] = 1
lambda_depths[1] = 1
lambda_depths[2] = 0

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
codebook_path = os.path.join(CURRENT_DIR, "codebook_DNA.txt")

# ================== Search / sim settings ==================
current_total_iter = 10          # fixed as requested
CHUNK_SIZE = 1                   # small tasks -> good parallelism
RATE_MIN = 0.62
RATE_MAX = 1.00
RATE_STEP = 0.005

# Nuanced search knobs (same meaning as before)
DELTA_R = 4*RATE_STEP                   # 2% in rate for the 2nd warm probe (only used if first succeeds)
REFINE_MAX_STEPS = 6             # gentle refinement (few midpoints)
MAX_PROBES = 40                  # hard cap per Pe
COARSE_STRIDE = 4 #max(1, int(round(0.02 / max(RATE_STEP, 1e-6))))  # ~2% rate stride

# ================== Pe list and starting C per Pe ==================
Pe_values = np.arange(0.001, 0.015 + 1e-9, 0.001)   # example

start_C_by_Pe = {
    0.001: 0, 0.002: 0, 0.003: 0, 0.004: 0, 
    0.005: 0,
    0.006: 0, 0.007: 0, 0.008: 0, 0.009: 0, 0.010: 0,
    0.011: 0, 0.012: 0, 0.013: 0, 0.014: 0, 0.015: 0,
}
DEFAULT_START_C = 2000

# ================== Channel split per Pe (customize if needed) ==================
def channel_split(Pe: float):
    #Pd, Pi, Ps = 0.45*Pe, 0.02*Pe, 0.53*Pe
    Pd, Pi, Ps = Pe/3, Pe/3, Pe/3
    return Pd, Pi, Ps

# ================== Helpers ==================
def rate_grid():
    g = np.arange(RATE_MIN, RATE_MAX + 1e-9, RATE_STEP)
    return np.round(g, 4)

def C_from_rate(r: float, M_int: int) -> int:
    return int(math.ceil(M_int * (1.0/r - 1.0)))

def rate_from_C(C: int, M_int: int) -> float:
    return M_int / float(M_int + C) if C >= 0 else RATE_MAX

def nearest_rate_index(rg: np.ndarray, r_target: float) -> int:
    r_target = max(RATE_MIN, min(RATE_MAX, r_target))
    return int(np.argmin(np.abs(rg - r_target)))

# ================== Iteration-parallel engine (unchanged) ==================
def process_chunk(args):
    (c1_, C, Pd, Pi, Ps, chunk_idx, chunk_size, codebook_DNA, Patterns, d_min, n_check) = args

    rng = np.random.default_rng()
    countS, countF, countE, countR = 0, 0, 0, 0
    decoding_time = 0.0
    E = np.zeros(K_out, dtype=int)

    for _ in range(chunk_size):
        # --- Outer encode (per file) ---
        u = rng.integers(0, 2, (M, k), dtype=np.int8)
        u_outer = Outer_RS_Encode(u, l_out, C)

        if c1_ < 0:
            # no inner code: directly DNA map and channel
            mapped_rows = [binary_to_dna(row) for row in u_outer]
            erroneous_rows = [DNA_channel(len(row), row, Pd, Pi, Ps) for row in mapped_rows]

            for mapped_row, received_row in zip(mapped_rows, erroneous_rows):
                if len(mapped_row) == len(received_row):
                    if np.array_equal(mapped_row, received_row):
                        countS += 1
                    else:
                        countE += 1
                        # symbol-wise error counting for RS outer
                        received_row_decimal = binary_to_decimal_blocks(dna_to_binary(received_row), l_out) 
                        mapped_row_decimal = binary_to_decimal_blocks(dna_to_binary(mapped_row), l_out)
                        for i in range(K_out):
                            if mapped_row_decimal[i] != received_row_decimal[i]:
                                E[i] += 1
                else:
                    countF += 1

        else:
            encoded_rows = []
            n = N = K = q = None
            for row in u_outer:
                x, n, N, K, q, _, _, _ = GCP_Encode_DNA_brute(row, l_in, c1_, codebook_DNA)
                encoded_rows.append(x)

            erroneous_rows = [DNA_channel(len(row), row, Pd, Pi, Ps) for row in encoded_rows]

            start_time = time.perf_counter()
            for idx, row in enumerate(erroneous_rows):
                uhat, _ = GCP_Decode_DNA_brute(
                    row, n, k, l_in, N, K, c1_, q, len_last,
                    lim, Patterns, codebook_DNA, d_min
                )
                if isinstance(uhat, (list, np.ndarray)) and len(uhat) == len(u_outer[idx]) and np.array_equal(uhat, u_outer[idx]):
                    countS += 1
                elif not uhat:
                    countF += 1
                else:
                    countE += 1
                    uhat_decimal = binary_to_decimal_blocks(uhat, l_out)
                    u_outer_decimal = binary_to_decimal_blocks(u_outer[idx], l_out)
                    for i in range(K_out):
                        if uhat_decimal[i] != u_outer_decimal[i]:
                            E[i] += 1
            decoding_time += time.perf_counter() - start_time

        # Outer retrieval success: for all info symbols i, F + 2*E[i] <= C
        if all(countF + 2 * E[i] <= C for i in range(K_out)):
            countR += 1

    return (countS, countF, countE, countR, decoding_time)

def run_simulation_for(C: int, Pd: float, Pi: float, Ps: float,
                       codebook_DNA, Patterns, d_min: int, n_check: int):
    tasks = []
    num_chunks = int(math.ceil(current_total_iter / CHUNK_SIZE))
    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        remaining = current_total_iter - start
        chunk_size = min(CHUNK_SIZE, remaining)
        if chunk_size > 0:
            tasks.append((c1, C, Pd, Pi, Ps, chunk_idx, chunk_size, codebook_DNA, Patterns, d_min, n_check))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_chunk, tasks)

    S = F = E = R = 0
    T = 0.0
    for (s, f, e, r, t) in results:
        S += s; F += f; E += e; R += r; T += t

    retrieval_error_rate = 1.0 - (R / float(current_total_iter))

    metrics = {
        "S": S, "F": F, "E": E, "R": R,
        "total": S + F + E,
        "retrieval_error_rate": retrieval_error_rate,
        "avg_decode_time_per_file_s": T / float(current_total_iter) if current_total_iter > 0 else float('nan'),
    }
    return retrieval_error_rate, metrics

# ================== Search (fixed: always move to higher C after a fail) ==================
def _test_rate_index(i, rg, Pe, M_int, codebook_DNA, Patterns, d_min, n_check, log_file):
    r = float(rg[i])
    C = C_from_rate(r, M_int)
    outer_rate = M_int / float(M_int + C)
    if c1 < 0:
        inner_rate = 1
    else:
        inner_rate = k / float(k + c1 * l_in + 2 * n_check)
    info_density = 2.0 * outer_rate * inner_rate

    Pd, Pi, Ps = channel_split(Pe)
    hdr = "[Pe={:.3f}] Try rate={:.4f} (C={}) | inner_rate={:.4f}, info_density={:.4f}".format(
        Pe, r, C, inner_rate, info_density
    )
    print(hdr); print(hdr, file=log_file, flush=True)

    rerr, metrics = run_simulation_for(C, Pd, Pi, Ps, codebook_DNA, Patterns, d_min, n_check)
    msg = ("  -> retrieval_error={:.4f} | Err={:.4f} F={:.4f} E={:.4f} R={} avg_dec={:.2f}s"
           .format(rerr,
                   1.0 - (metrics['S'] / float(current_total_iter * (M_int + C))),
                   metrics['F'] / float(current_total_iter * (M_int + C)),
                   metrics['E'] / float(current_total_iter * (M_int + C)),
                   metrics['R'],
                   metrics['avg_decode_time_per_file_s']))
    print(msg); print(msg, file=log_file, flush=True)

    ok = (rerr == 0.0)
    return ok, C, r

def _refine_between(best_ok_tuple, fail_idx, rg, Pe, codebook_DNA, Patterns, d_min, n_check, log_file, probes_used):
    """
    Midpoint refine between a known OK index (best_ok_tuple[0]) and a known FAIL index (fail_idx).
    """
    probes = probes_used
    ok_idx = best_ok_tuple[0]
    lo, hi = ok_idx, fail_idx
    if lo > hi:
        lo, hi = hi, lo
    best_local = best_ok_tuple
    steps = 0
    while (hi - lo > 1) and (steps < REFINE_MAX_STEPS) and (probes < MAX_PROBES):
        mid = (lo + hi) // 2
        ok, Cx, rx = _test_rate_index(mid, rg, Pe, M, codebook_DNA, Patterns, d_min, n_check, log_file); probes += 1
        if ok:
            best_local = (mid, Cx, rx)
            lo = mid
        else:
            hi = mid
        steps += 1
    return best_local

def find_min_C_for_Pe(Pe: float, C_start: int, codebook_DNA, Patterns, d_min, n_check, log_file):
    """
    Fixed strategy:
      1) Probe an easier point (i_lo = i0 - DELTA) first.
         - If FAIL -> keep lowering rate by COARSE_STRIDE until first PASS, then refine up to the fail boundary.
         - If PASS -> probe pivot (i0). If pivot FAIL -> refine between i_lo (ok) and i0 (fail).
         - If both PASS -> walk upward by 1 until first FAIL, then refine down to boundary.
    """
    rg = rate_grid()
    n = len(rg)
    def within(i): return (0 <= i < n)

    # start near previous Pe's C_min
    r_start = rate_from_C(C_start, M)
    i0 = nearest_rate_index(rg, r_start)

    # easier point first
    delta_idx = max(1, int(round(DELTA_R / max(RATE_STEP, 1e-6))))
    i_lo = max(0, i0 - delta_idx)

    probes = 0
    best_ok = None  # (idx, C, r)

    # Probe easier point
    ok_lo, C_lo, r_lo = _test_rate_index(i_lo, rg, Pe, M, codebook_DNA, Patterns, d_min, n_check, log_file); probes += 1

    # If FAIL at easier point -> DO NOT go harder; descend until PASS, then refine up
    if not ok_lo:
        fail_idx = i_lo
        i = i_lo
        while probes < MAX_PROBES:
            i_next = i - COARSE_STRIDE
            if not within(i_next):
                return {"Pe": Pe, "C_min": None, "rate": None, "status": "no_feasible_rate_in_grid"}
            ok, Cx, rx = _test_rate_index(i_next, rg, Pe, M, codebook_DNA, Patterns, d_min, n_check, log_file); probes += 1
            if ok:
                best_ok = (i_next, Cx, rx)
                best_ok = _refine_between(best_ok, fail_idx, rg, Pe, codebook_DNA, Patterns, d_min, n_check, log_file, probes)
                return {"Pe": Pe, "C_min": best_ok[1], "rate": best_ok[2], "status": "ok"}
            else:
                fail_idx = i_next
                i = i_next

        return {"Pe": Pe, "C_min": None, "rate": None, "status": "no_feasible_rate_in_grid"}

    # PASS at easier point -> try pivot
    best_ok = (i_lo, C_lo, r_lo)
    i_hi = i0
    ok_hi, C_hi, r_hi = _test_rate_index(i_hi, rg, Pe, M, codebook_DNA, Patterns, d_min, n_check, log_file); probes += 1

    # If FAIL at pivot -> refine between i_lo (ok) and i_hi (fail)
    if not ok_hi:
        best_ok = _refine_between(best_ok, i_hi, rg, Pe, codebook_DNA, Patterns, d_min, n_check, log_file, probes)
        return {"Pe": Pe, "C_min": best_ok[1], "rate": best_ok[2], "status": "ok"}

    # If both PASS -> walk up until first FAIL, then refine down
    best_ok = (i_hi, C_hi, r_hi) if r_hi > best_ok[2] else best_ok
    last_ok = i_hi
    while probes < MAX_PROBES:
        i_next = last_ok + 1
        if not within(i_next):
            return {"Pe": Pe, "C_min": best_ok[1], "rate": best_ok[2], "status": "ok"}
        ok, Cx, rx = _test_rate_index(i_next, rg, Pe, M, codebook_DNA, Patterns, d_min, n_check, log_file); probes += 1
        if ok:
            if rx > best_ok[2]:
                best_ok = (i_next, Cx, rx)
            last_ok = i_next
        else:
            best_ok = _refine_between(best_ok, i_next, rg, Pe, codebook_DNA, Patterns, d_min, n_check, log_file, probes)
            return {"Pe": Pe, "C_min": best_ok[1], "rate": best_ok[2], "status": "ok"}

    # Fallback (probe cap reached)
    return {"Pe": Pe, "C_min": best_ok[1], "rate": best_ok[2], "status": "ok"}

# ================== Main ==================
if __name__ == '__main__':
    whole_start = time.perf_counter()

    Patterns = preCompute_Patterns(lambda_depths, K_in, len_last, lim, c1)

    if os.path.exists(codebook_path):
        with open(codebook_path, "r") as f:
            codebook_DNA = [line.strip() for line in f if line.strip()]
    else:
        codebook_DNA = None
    d_min = 5
    n_check = len(codebook_DNA[0]) if codebook_DNA else 0

    results = []

    # Helper to read dict with float keys robustly
    def get_startC_from_dict(pe):
        key = float("{:.3f}".format(pe))
        return start_C_by_Pe.get(key, DEFAULT_START_C)

    with open('results_log.txt', 'w') as log_file:
        header = "=== Coarse->Bracket->Refine search for minimal C per Pe (iteration-parallel) ==="
        print(header); print(header, file=log_file, flush=True)

        prev_Cmin = None

        for idx, Pe in enumerate(Pe_values):
            if idx == 0:
                C_start = get_startC_from_dict(Pe)
            else:
                # use previous Pe's C_min as optimistic start; falls back if None
                C_start = prev_Cmin if (prev_Cmin is not None) else get_startC_from_dict(Pe)

            start_msg = "\n--- Pe={:.3f} | start_C={} (start_rate~{:.4f}) ---".format(
                Pe, C_start, rate_from_C(C_start, M)
            )
            print(start_msg); print(start_msg, file=log_file, flush=True)

            res = find_min_C_for_Pe(Pe, C_start, codebook_DNA, Patterns, d_min, n_check, log_file)
            results.append(res)

            prev_Cmin = res["C_min"] if res.get("status") == "ok" else None

        # Summary
        print("\n=== Summary ==="); print("\n=== Summary ===", file=log_file, flush=True)
        for r in results:
            if r["status"] == "ok":
                line = "Pe={:.3f}: C_min={} (rate={:.3f})".format(r['Pe'], r['C_min'], r['rate'])
            else:
                line = "Pe={:.3f}: No feasible rate in [{:.3f}, {:.3f}] with retrieval error 0.".format(
                    r['Pe'], RATE_MIN, RATE_MAX
                )
            print(line); print(line, file=log_file, flush=True)

        total_time_msg = "\nTotal execution: {:.2f} s".format(time.perf_counter() - whole_start)
        print(total_time_msg); print(total_time_msg, file=log_file, flush=True)
