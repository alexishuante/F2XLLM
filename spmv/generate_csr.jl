using Random
using Printf
using LinearAlgebra
using Base.Threads

function generate_csr(n, sparsity)
    val = Float64[]
    col = Int[]
    rowptr = [0]
    
    total_elements = n * n
    non_zero_elements = round(Int, (1 - sparsity) * total_elements)
    println("Total elements: $total_elements")
    println("Non-zero elements: $non_zero_elements")

    elements_per_row2 = ceil(Int, non_zero_elements / n)
    elements_per_row = div(non_zero_elements, n)
    
    println("Elements per row: $elements_per_row")
    println("Elements per row2: $elements_per_row2")

    lock = ReentrantLock()  # To handle concurrent access to shared arrays
    
    Threads.@threads for row in 1:n
        row_elements = 0
        cols_in_row = Set{Int}()
        
        while row_elements < elements_per_row && length(cols_in_row) < n
            col_index = rand(1:n)
            if !(col_index in cols_in_row)
                lock(lock) do  # Lock the arrays during modification
                    push!(val, rand(1:100))  # Assuming non-zero values are between 1 and 100
                    push!(col, col_index)
                end
                push!(cols_in_row, col_index)
                row_elements += 1
            end
        end
        
        lock(lock) do  # Lock the array during modification
            push!(rowptr, length(val))
        end
    end
    
    return val, col, rowptr
end

function save_csr_to_file(val, col_ind, row_ptr, filename="matrix_csr_test.txt")
    open(filename, "w") do f
        println(f, "Values:")
        for v in val
            println(f, v)
        end
        println(f, "Column Indices:")
        for c in col_ind
            println(f, c)
        end
        println(f, "Row Pointers:")
        for r in row_ptr
            println(f, r)
        end
    end
end

# Example usage
n = 300000  # Size of the square matrix
sparsity = 0.99  # 99% of the matrix elements are zeros
val, col, rowptr = generate_csr(n, sparsity)

save_csr_to_file(val, col, rowptr)
println("Number of values in val: $(length(val))")

# The number of values inside the column indices array is
println("Number of values in col_ind: $(length(col))")

# The number of values inside the row pointers array is
println("Number of values in row_ptr: $(length(rowptr))")

println("CSR representation saved to matrix_csr_test.txt")
