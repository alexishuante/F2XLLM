import numpy as np
import random
import scipy.sparse as sp

def create_csr_matrix(n, nonzero_percentage):
    # Calculate the total number of elements and the number of nonzeros
    total_elements = n * n
    nonzeros = int((nonzero_percentage / 100.0) * total_elements)
    
    # Initialize the matrix with zeros
    matrix = np.zeros((n, n))
    
    # Fill the matrix with nonzeros at random positions
    filled_positions = set()
    while len(filled_positions) < nonzeros:
        row = random.randint(0, n-1)
        col = random.randint(0, n-1)
        if (row, col) not in filled_positions:
            matrix[row, col] = random.randint(1, 10)
            filled_positions.add((row, col))
    
    # Convert to CSR format
    csr_matrix = sp.csr_matrix(matrix)
    return csr_matrix

def write_csr_to_file(csr_matrix, filename):
    with open(filename, 'w') as f:
        # Write data and indices arrays
        for i in range(len(csr_matrix.data)):
            f.write(f"{csr_matrix.data[i]},{csr_matrix.indices[i]}\n")
        # Separately write the indptr array to distinguish it from data and indices
        f.write("indptr," + ",".join(map(str, csr_matrix.indptr)) + "\n")

# Adjust the read function if necessary based on how data is written

# Example usage
n = 5  # Matrix size
nonzero_percentage = 40  # Percentage of non-zero elements
csr_matrix = create_csr_matrix(n, nonzero_percentage)
filename = 'csr_matrix.csv'
write_csr_to_file(csr_matrix, filename)

# To read the matrix back
# Ensure the read_csr_from_file function is correctly implemented