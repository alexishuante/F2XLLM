import cupy as cp
import numpy as np
import math

def generate_row(n, elements_per_row, seed):
    cp.random.seed(seed)
    row_elements = 0
    cols_in_row = set()
    val = []
    col = []
    
    while row_elements < elements_per_row and len(cols_in_row) < n:
        col_index = cp.random.randint(0, n)
        if col_index not in cols_in_row:
            val.append(cp.random.randint(1, 101))  # Assuming non-zero values are between 1 and 100
            col.append(col_index)
            cols_in_row.add(col_index)
            row_elements += 1
    
    return val, col

def generate_csr(n, sparsity):
    total_elements = n * n
    non_zero_elements = int((1 - sparsity) * total_elements)
    print('Total elements: ' + str(total_elements))
    print('Non-zero elements: ' + str(non_zero_elements))

    elements_per_row2 = math.ceil(non_zero_elements / n)
    elements_per_row = non_zero_elements // n
    
    print('Elements per row: ' + str(elements_per_row))
    print('Elements per row2: ' + str(elements_per_row2))

    val = []
    col = []
    rowptr = [0]

    seeds = cp.random.randint(0, 1_000_000, size=n)
    for row in range(n):
        row_val, row_col = generate_row(n, elements_per_row, seeds[row])
        val.extend(row_val)
        col.extend(row_col)
        rowptr.append(len(val))
    
    return cp.array(val), cp.array(col), cp.array(rowptr)

def save_csr_to_file(val, col_ind, row_ptr, filename="matrix_csr_test.txt"):
    """Saves the CSR representation to a text file.

    Args:
        val (cp.ndarray): The values array.
        col_ind (cp.ndarray): The column indices array.
        row_ptr (cp.ndarray): The row pointers array.
        filename (str): The name of the file to save (default: "matrix_csr.txt").
    """
    with open(filename, "w") as f:
        np.savetxt(f, cp.asnumpy(val), fmt="%f", header="Values:")
        np.savetxt(f, cp.asnumpy(col_ind), fmt="%d", header="Column Indices:")
        np.savetxt(f, cp.asnumpy(row_ptr), fmt="%d", header="Row Pointers:")

# Example usage
n = 300000  # Size of the square matrix
sparsity = .99  # 99% of the matrix elements are zeros
val, col, rowptr = generate_csr(n, sparsity)

save_csr_to_file(val, col, rowptr)
print('Number of values in val: ' + str(len(val)))

# The number of values inside the column indices array is
print('Number of values in col_ind: ' + str(len(col)))

# The number of values inside the row pointers array is
print('Number of values in row_ptr: ' + str(len(rowptr)))

print("CSR representation saved to matrix_csr_test.txt")
