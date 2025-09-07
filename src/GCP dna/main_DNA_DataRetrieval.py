import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
import os, sys
import math

from GCP_Encode_DNA import GCP_Encode_DNA_brute,binary_to_decimal_blocks,binary_to_dna
from DNA_channel import DNA_channel
from GCP_Decode_DNA import GCP_Decode_DNA_brute
from preCompute_Patterns import preCompute_Patterns
from Outer_Encode import Outer_RS_Encode

# Code parameters
k = 168
c1_values = [2]
c1_global = 2 # np.arange(10, 13 , 2)
c2 = 1
#C_values = np.arange(5340,100000,step)
l_in = 8
l_out = 14
K_in = int(math.ceil(k / l_in))
K_out = k // l_out
len_last = (k - 1) % l_in + 1
M = 10**4

lim = 5       # precompute patterns for |Δ| = 0..lim-1
lambda_depths = [0]*lim
lambda_depths[0] = 1
lambda_depths[1] = 1
lambda_depths[2] = 0

# Probabilites setup
Pe = 0.01
# Pd, Pi, Ps = Pe / 3, Pe / 3, Pe / 3
Pd, Pi, Ps = 0.45*Pe, 0.02*Pe, 0.53*Pe

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
codebook_path = os.path.join(CURRENT_DIR, "codebook_DNA.txt")

# Simulation setup
current_total_iter = 10 # Start with the initial value
max_iter = 100
zero_error_detected = False
C = 10 # Initial C value
step = 1000

CHUNK_SIZE = 1#max(1, 100 // (cpu_count() * 4))

def process_chunk(args):
    c1, C, chunk_idx, chunk_size, total_iter, codebook_DNA, Patterns, d_min, n_check = args
    rng = np.random.default_rng()
    countS, countF, countE, countR = 0, 0, 0, 0
    decoding_time = 0

    # Initialize error counter array for each block position (cumulative over all rows)
    E = np.zeros(K_out, dtype=int)
    
    for _ in range(chunk_size):
        u = rng.integers(0, 2, (M, k))

        # Outer code
        u_outer = Outer_RS_Encode(u,l_out,C)

        if c1 == 0:
            mapped_rows = []
            for row in u_outer:
                x = binary_to_dna(row)
                mapped_rows.append(x)  
            mapped_file = np.array(mapped_rows)

            # Insert errors
            erroneous_rows = []
            for row in mapped_file:
                y = DNA_channel(len(row),row, Pd, Pi, Ps)
                erroneous_rows.append(y)
            received_file = np.array(erroneous_rows)

            for mapped_row, received_row in zip(mapped_file, received_file):
                if len(mapped_row) == len(received_row):
                    if np.array_equal(mapped_row, received_row):
                        countS += 1
                    else:
                        for i in range(K_out):
                            if mapped_row[i] != received_row[i]:
                                E[i] += 1 
                else:
                    countF += 1
        else:
            # Inner code
            encoded_rows = []
            for row in u_outer:
                x, n, N, K, q, _, _, _ = GCP_Encode_DNA_brute(row, l_in, c1, codebook_DNA)
                encoded_rows.append(x)  
            encoded_file = np.array(encoded_rows)

            # Insert errors
            erroneous_rows = []
            for row in encoded_file:
                y = DNA_channel(len(row),row, Pd, Pi, Ps)
                erroneous_rows.append(y)
            received_file = np.array(erroneous_rows)
            
            start_time = time.perf_counter()
            for idx, row in enumerate(received_file):
                uhat, _ = GCP_Decode_DNA_brute(row, n, k, l_in, N, K, c1, q, len_last, lim, Patterns, codebook_DNA, d_min)
                #print(len(uhat))
                if np.array_equal(uhat, u_outer[idx]):
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

        # Check retrieval condition
        if all(countF + 2 * E[i] <= C for i in range(K_out)):
            countR += 1

    return (c1, countS, countF, countE, countR, decoding_time)

if __name__ == '__main__':
    whole_start = time.perf_counter()
    
    Patterns = preCompute_Patterns(lambda_depths, K_in, len_last, lim, c1_global)
    
    if os.path.exists(codebook_path):
       with open(codebook_path, "r", encoding="utf-8") as f:
           codebook_DNA = [line.strip() for line in f if line.strip()]
    else:
        codebook_DNA = None 
    d_min = 5
    n_check = len(codebook_DNA[0])
    
    # Open a log file for writing
    with open('results_log.txt', 'w') as log_file:
        #for C in C_values:           
        while True:

            outer_rate = M / (M + C)
                       
            # Print to both terminal and file# flush ensures immediate write

            for c1 in c1_values:
                inner_rate = k / (k + c1 * l_in + 2*n_check)
                msg = f"\n C: {C}, Outer Code Rate: {outer_rate:.4f}, Inner Code Rate: {inner_rate:.4f}, Inf. Density: {2*outer_rate*inner_rate:.4f}"
                print(msg)
                print(msg, file=log_file, flush=True)  
                tasks = []
                num_chunks = (current_total_iter*5 + CHUNK_SIZE - 1) // CHUNK_SIZE
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * CHUNK_SIZE
                    remaining = current_total_iter - start
                    current_chunk = min(CHUNK_SIZE, remaining)
                    tasks.append((c1, C, chunk_idx, current_chunk, current_total_iter, codebook_DNA, Patterns, d_min, n_check))

                # Parallel execution with process pool
                with Pool(processes=cpu_count()) as pool:
                    results = pool.map(process_chunk, tasks)

                # Aggregate results
                results_dict = {c1: {'S': 0, 'F': 0, 'E': 0, 'R': 0, 'time': 0} for c1 in c1_values}
                for result in results:
                    c1, S, F, E, R, t_ = result
                    results_dict[c1]['S'] += S
                    results_dict[c1]['F'] += F
                    results_dict[c1]['E'] += E
                    results_dict[c1]['R'] += R
                    results_dict[c1]['time'] += t_

                # Calculate final metrics and print
                total = results_dict[c1]['S'] + results_dict[c1]['F'] + results_dict[c1]['E']
                total_error = 1 - results_dict[c1]['S'] / total
                error_rate = results_dict[c1]['E'] / total
                failure_rate = results_dict[c1]['F'] / total
                retrieval_error_rate = 1 - results_dict[c1]['R'] / current_total_iter
                nRet = results_dict[c1]['R']
                avg_time = results_dict[c1]['time'] / current_total_iter

                # Print to both terminal and file
                msg = (f"c1: {c1}, Inner Code Rate: {inner_rate:.2f}, Inner Total Err: {total_error: .3f}, Inner Failure: {failure_rate:.3f}, Inner Error: {error_rate:.3f}, Avg File dec. Time: {avg_time:.2f}s, Retrieval Error: {retrieval_error_rate:.4f}")
                print(msg)
                print(msg, file=log_file, flush=True)

                # Check if retrieval error rate reached 0 and adjust current_total_iter
                if retrieval_error_rate == 0:
                    zero_error_detected = True
                    break
                else:
                    zero_error_detected = False
                    C = C + step

            if zero_error_detected:
                current_total_iter *= 10
                if current_total_iter > max_iter:
                    break
                adjustment_msg = f"Retrieval error rate reached 0 for C={C}. Total iterations increased to {current_total_iter} for next simulations."
                print(adjustment_msg)
                print(adjustment_msg, file=log_file, flush=True)

        # Print total execution time
        total_time_msg = f"\nTotal execution: {time.perf_counter() - whole_start:.2f} seconds"
        print(total_time_msg)
        print(total_time_msg, file=log_file, flush=True)