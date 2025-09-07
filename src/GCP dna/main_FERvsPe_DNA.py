import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from multiprocessing import Pool, cpu_count, RLock
import math
import os, sys
from tqdm import tqdm

# ================== Globals / Config ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- local imports from the binary folder ---
from GCP_Encode_DNA import GCP_Encode_DNA_brute, GCP_Encode_DNA_rep
from DNA_channel import DNA_channel
from GCP_Decode_DNA import GCP_Decode_DNA_brute, GCP_Decode_DNA_rep
from preCompute_Patterns import preCompute_Patterns
from GCP_theoretical_DNA import theoretical_error_brute, theoretical_error_rep

# ---------------- Code configuration ----------------
k = 168
c1 = 8
c2 = 1
l  = 8
len_last = (k - 1) % l + 1
K = int(math.ceil(k / l))
opt = False  # optimizes decoding but can slow it down when True

# Lambda depths / window for Δ: SAME for all scenarios
lim = 5                                    # precompute patterns for |Δ| = 0..lim-1
lambda_depths = [0]*lim
lambda_depths[0] = 1
lambda_depths[1] = 1
lambda_depths[2] = 0

# Channel & sweeps (fixed)
total_iter = 1000                  # total trials per Pe (per scenario)
Pe_values = np.arange(0.001, 0.015, 0.001) # grid of Pe

# Per-task iterations
TASK_ITERS = 30


# -------------- Worker initializer & worker --------------
def init_worker(patterns, codebook, mode, split_tuple, t_rep_val):
    """Runs once per worker process to receive heavy, precomputed objects."""
    global Patterns, codebook_DNA, MODE, split, t_rep_global
    Patterns = patterns
    codebook_DNA = codebook
    MODE = mode
    split = split_tuple
    t_rep_global = t_rep_val  # used only for rep mode

def process_chunk(args):
    pe, chunk_idx, chunk_size, k, l, c1, c2, len_last, lim, d_min = args
    Pd, Pi, Ps = split[0]*pe, split[1]*pe, split[2]*pe
    rng = np.random.default_rng()
    countS = countF = countE = countP = 0
    decoding_time = 0.0

    for _ in range(chunk_size):
        u = rng.integers(0, 2, k).tolist()
            
        if MODE.lower() == "brute":
            x, n, N, K_enc, q, U, X, check_par = GCP_Encode_DNA_brute(u, l, c1, codebook_DNA)
        else:
            x, n, N, K_enc, q, U, X, check_par = GCP_Encode_DNA_rep(u, l, c1, c2, t_rep_global)

        # DNA iid edit channel (IID over whole sequence)
        y = DNA_channel(n, x, Pd, Pi, Ps)

        # decode
        t0 = time.perf_counter()
        if MODE.lower() == "brute":
            uhat, check_par_hat = GCP_Decode_DNA_brute(
                y, n, k, l, N, K_enc, c1, q, len_last, lim, Patterns, codebook_DNA, d_min, opt
            )
        else:
            uhat, check_par_hat = GCP_Decode_DNA_rep(
                y, n, k, l, N, K_enc, c1, c2, q, len_last, lim, Patterns, t_rep_global, opt
            )
        decoding_time += time.perf_counter() - t0

        if check_par != check_par_hat:
            countP += 1

        if uhat == u:
            countS += 1
        elif not uhat:
            countF += 1
        else:
            countE += 1

    return (pe, countS, countF, countE, countP, decoding_time)

# -------------- Per-scenario runner --------------
def run_case(MODE, param2, split, Patterns, codebook_DNA):
    """
    param2:
      - if MODE=='brute': param2 == d_min for brute case
      - if MODE=='rep'  : param2 == t_rep for repetition case
    """
    start_all = time.perf_counter()

    # Interpret the second parameter
    if MODE.lower() == "brute":
        d_min_case = int(param2)
        t_rep_case = None
    elif MODE.lower() == "rep":
        t_rep_case = int(param2)
        d_min_case = None  # not used; just to fill args
    else:
        raise ValueError("MODE must be 'brute' or 'rep'.")

    # Codebook stats (only for brute)
    n_check = len(codebook_DNA[0]) if (MODE.lower() == "brute" and codebook_DNA is not None) else 0

    # Code rate / case tag
    if MODE.lower() == "brute":
        code_rate = k / (k + c1*l + 2*n_check)
        CASE_TAG = f"brute (d_min={d_min_case}, n_check={n_check})"
    else:
        code_rate = k / (k + c1*l + c2*t_rep_case*l)
        CASE_TAG = f"repetition (t={t_rep_case})"

    OPT_TAG = "ON" if opt else "OFF"

    tqdm.write(
        f"=== Case: {CASE_TAG} | split Pd,Pi,Ps = "
        f"({split[0]:.2f}, {split[1]:.2f}, {split[2]:.2f}) ==="
    )
    tqdm.write(f"Coderate: {code_rate:.3f}; Inf. density (bits/NT): {2*code_rate:.3f}")

    # === Theoretical curve FIRST ===
    N_rs = K + c1
    l_theory = l // 2
    if MODE.lower() == "brute":
        total_th, cond1_th, cond2_th, parity_th = theoretical_error_brute(
            Pe_values,
            l=l_theory,
            c1=c1,
            c2=c2,
            N=N_rs,
            d_min=d_min_case,              # <-- per-case d_min
            n_check=n_check,
            lambda_depths=lambda_depths,   # SAME across scenarios
            split=split
        )
    else:
        k_rep = c2 * l
        total_th, cond1_th, cond2_th, parity_th = theoretical_error_rep(
            Pe_values,
            k=k_rep,
            l=l_theory,
            c1=c1,
            c2=c2,
            N=N_rs,
            t_rep=t_rep_case,              # <-- per-case t_rep
            lambda_depths=lambda_depths,   # SAME across scenarios
            split=split
        )

    # ----- Build tasks -----
    tasks = []
    for pe in Pe_values:
        num_chunks = math.ceil(total_iter / TASK_ITERS)
        for chunk_idx in range(num_chunks):
            current_chunk = TASK_ITERS if (chunk_idx + 1) < num_chunks else total_iter - TASK_ITERS * (num_chunks - 1)
            tasks.append((float(pe), chunk_idx, int(current_chunk), k, l, c1, c2, len_last, lim, d_min_case))

    results_dict = {float(pe): {'S':0, 'F':0, 'E':0, 'P':0, 'time':0.0} for pe in Pe_values}

    # ----- Pool per-scenario -----
    NPROCS = cpu_count()
    tqdm.set_lock(RLock())
    with Pool(
        processes=NPROCS,
        initializer=init_worker,
        initargs=(Patterns, codebook_DNA if MODE.lower()=="brute" else None, MODE, split, t_rep_case if t_rep_case is not None else 0)
    ) as pool:
        for pe, S, F, E, P, t_ in tqdm(
            pool.imap_unordered(process_chunk, tasks, chunksize=1),
            total=len(tasks), 
            desc = (f"Simulating [{CASE_TAG}, " f"split=({split[0]:.2f},{split[1]:.2f},{split[2]:.2f}), " f"opt={OPT_TAG}]"), unit="task", dynamic_ncols=True
        ):
            d = results_dict[float(pe)]
            d['S'] += S; d['F'] += F; d['E'] += E; d['P'] += P; d['time'] += t_

    # === Empirical metrics (per Pe) ===
    emp_total_errors, emp_failure_errors, emp_miscorr_errors, emp_checkpar_errors, avg_times = [], [], [], [], []
    for pe in Pe_values:
        d = results_dict[float(pe)]
        emp_total_errors.append(1 - d['S'] / total_iter)
        emp_failure_errors.append(d['F'] / total_iter)
        emp_miscorr_errors.append(d['E'] / total_iter)
        emp_checkpar_errors.append(d['P'] / total_iter)
        avg_times.append(d['time'] / total_iter)    

    emp_total_errors = np.array(emp_total_errors)
    emp_failure_errors = np.array(emp_failure_errors)
    emp_miscorr_errors = np.array(emp_miscorr_errors)
    emp_checkpar_errors = np.array(emp_checkpar_errors)
    avg_times = np.array(avg_times)

    # === Write results file ===
    os.makedirs(os.path.join(CURRENT_DIR, "Results"), exist_ok=True)

    def split_tag(s):
        return f"{s[0]:.3f}-{s[1]:.3f}-{s[2]:.3f}".replace('.', 'p')

    out_path = os.path.join(CURRENT_DIR, "Results", "results_DNA.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n\n# ===== New Simulation Run =====\n")
        f.write(f"# Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# Case: {CASE_TAG}\n")
        f.write(f"split: Pd={split[0]:.2f}, Pi={split[1]:.2f}, Ps={split[2]:.2f}\n")
        f.write(f"Coderate: {code_rate:.6f}; Inf. density (bits/NT): {2*code_rate:.6f}\n")
        f.write(f"opt={OPT_TAG}\n")
        f.write(f"k={k}, l={l}, c1={c1}, c2={c2}, K={K}\n")
        if MODE.lower() == "brute":
            f.write(f"n_check={n_check}, d_min={d_min_case}\n")
        else:
            f.write(f"t_rep={t_rep_case}\n")
        f.write(f"total_iter={total_iter}, TASK_ITERS={TASK_ITERS}\n")
        f.write(f"lambda_depths={lambda_depths}; lim={lim}\n")

        f.write("\n# Per-Pe metrics\n")
        f.write("Pe\tFail\tMiscorr\tCheckParErr\tTotalErr\tAvgDecTime[s]\n")
        for i, pe in enumerate(Pe_values):
            f.write(f"{pe:.6f}\t{emp_failure_errors[i]:.8f}\t{emp_miscorr_errors[i]:.8f}\t"
                    f"{emp_checkpar_errors[i]:.8f}\t{emp_total_errors[i]:.8f}\t{avg_times[i]:.8f}\n")

        f.write("\n# Empirical arrays\n")
        f.write(f"Pe_values = {repr(Pe_values)}\n")
        f.write(f"emp_total_errors = {repr(emp_total_errors)}\n")
        f.write(f"emp_failure_errors = {repr(emp_failure_errors)}\n")
        f.write(f"emp_miscorr_errors = {repr(emp_miscorr_errors)}\n")
        f.write(f"emp_checkpar_errors = {repr(emp_checkpar_errors)}\n")
        f.write(f"avg_decoding_times = {repr(avg_times)}\n")

        f.write("\n# Theoretical arrays\n")
        f.write(f"total_th = {repr(np.array(total_th))}\n")
        f.write(f"cond1_th = {repr(np.array(cond1_th))}\n")
        f.write(f"cond2_th = {repr(np.array(cond2_th))}\n")
        f.write(f"parity_th = {repr(np.array(parity_th))}\n")

        f.write(f"\nTotal execution: {time.perf_counter() - start_all:.2f} seconds\n")

    # === Plot ===
    plt.figure()
    label_emp = f"Empirical ({'brute' if MODE=='brute' else f'rep t={t_rep_case}'}, opt={OPT_TAG})"
    label_th  = f"Theoretical ({'brute' if MODE=='brute' else f'rep t={t_rep_case}'})"
    plt.semilogy(Pe_values, emp_total_errors, '-o', label=label_emp)
    plt.semilogy(Pe_values, total_th,        '--^', label=label_th)
    plt.xlabel('Edit Probability (Pe)')
    plt.ylabel('Decoding error probability (FER)')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    pdf_name = os.path.join(
        "Results",
        f"FER_vs_Pe_DNA_{MODE}_t{t_rep_case if MODE=='rep' else 'NA'}_split{split_tag(split)}_opt{OPT_TAG}_{ts}.pdf"
    )
    os.makedirs(os.path.join(CURRENT_DIR, "Results"), exist_ok=True)
    plt.savefig(os.path.join(CURRENT_DIR, pdf_name), bbox_inches='tight')
    plt.close()

# ------------------ main ------------------
if __name__ == '__main__':
    # Compute Patterns ONCE for all scenarios (works for fixed k, l, c1)
    Patterns_global = preCompute_Patterns(lambda_depths, K, len_last, lim, c1)

    # Load brute codebook ONCE (used only in brute scenarios)
    codebook_path = os.path.join(CURRENT_DIR, "codebook_DNA.txt")
    if os.path.exists(codebook_path):
        with open(codebook_path, "r", encoding="utf-8") as f:
            codebook_global = [line.strip() for line in f if line.strip()]
    else:
        codebook_global = None  # fine for rep-only runs

    # Define the configurations to simulate
    # NOTE: 2nd item is d_min for 'brute', t_rep for 'rep'
    CASES = [
        ("brute", 5, (0.45, 0.02, 0.53)),
        ("brute", 5, (1/3, 1/3, 1/3)),
        #("rep",   5, (0.45, 0.02, 0.53)),
        #("rep",   5, (1/3, 1/3, 1/3)),
    ]

    for MODE, param2, split in CASES:
        run_case(MODE, param2, split, Patterns_global, codebook_global)

    print("\nAll scenarios finished.")
