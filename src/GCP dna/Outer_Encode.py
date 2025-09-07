import numpy as np
from reedsolo import RSCodec

def binary_to_decimal_blocks(binary_message, block_length):
    # Ensure the binary message is divisible by block_length
    if len(binary_message) % block_length != 0:
        raise ValueError("The length of the binary message must be divisible by the block length.")
    
    # Group the binary message into blocks of block_length
    blocks = [binary_message[i:i + block_length] for i in range(0, len(binary_message), block_length)]
    
    # Convert each binary block to decimal
    decimal_blocks = [int("".join(map(str, block)), 2) for block in blocks]
    
    return decimal_blocks

def decimal_to_binary_blocks(decimal_list, block_length):
    # Convert each decimal number to binary with fixed block length
    return [f"{x:0{block_length}b}" for x in decimal_list]

def Outer_RS_Encode(u, l, C):

    M, k = u.shape
    if k % l != 0:
        raise ValueError("Each row length must be divisible by the block length.")
        
    num_blocks = k // l  # number of blocks per row
    
    # Step 1: Convert each row from binary to a list of decimal blocks.
    D = np.zeros((M, num_blocks), dtype=int)
    for i in range(M):
        # Convert row (as list) to decimal blocks.
        D[i, :] = binary_to_decimal_blocks(u[i, :].tolist(), l)
    
    # Step 2: Transpose.
    D_transposed = D.T
    
    # Step 3: RS encode each row.
    rsEncoder = RSCodec(C, c_exp=l)
    encoded_rows = []
    for row in D_transposed:
        # RS encoding expects a list of integers.
        encoded_row = list(rsEncoder.encode(list(row)))
        encoded_rows.append(encoded_row)
    encoded_matrix = np.array(encoded_rows)
    
    # Step 4: Transpose back so that rows become columns.
    encoded_matrix_T = encoded_matrix.T
    
    # Step 5: Convert each decimal block back to binary.
    final_binary_matrix = []
    for row in encoded_matrix_T:
        # Convert row (list of decimals) to a list of binary strings (each of fixed length l)
        binary_strings = decimal_to_binary_blocks(row.tolist(), l)
        # Flatten each binary string into a list of integer bits
        binary_row = [int(bit) for bin_str in binary_strings for bit in bin_str]
        final_binary_matrix.append(binary_row)
    final_binary_matrix = np.array(final_binary_matrix)
    
    return final_binary_matrix