import numpy as np
from scipy.sparse import csr_matrix

# Matrix dimensions
rows = 3
cols = 3

# Desired percentage of non-zeros
percentage_nnz = 0.5  # 50% non-zero elements

# Calculate the number of non-zero elements
nnz = int(rows * cols * percentage_nnz)

# Generate random data, indices, and indptr
data = np.random.rand(nnz)
indices = np.random.randint(0, cols, nnz)
indptr = np.zeros(rows + 1, dtype=int)

# Calculate bincounts and ensure the result fits into indptr
bincounts = np.bincount(np.random.randint(0, rows, nnz))
np.cumsum(bincounts, out=indptr[1:1 + len(bincounts)])

# Combine and sort data and indices based on row-major order
combined = np.column_stack((data, indices))
row_indices = np.repeat(np.arange(rows), np.diff(indptr))
sorted_indices = np.lexsort((combined[:, 1], row_indices))  # Sort by column, then row
sorted_combined = combined[sorted_indices]

# Update data and indices with the sorted values
data = sorted_combined[:, 0]
indices = sorted_combined[:, 1].astype(int)

# Create CSR matrix
csr_matrix = csr_matrix((data, indices, indptr), shape=(rows, cols))

# Print information
print("Percentage of non-zeros:", (nnz / (rows * cols)) * 100)
print("Data:", data)  # Print the sorted data
print("Indices:", indices) # Print the sorted indices
print("Indptr:", indptr)
