<<<<<<< HEAD
subroutine spmv_parallel(n, nnz, val, row, col, x, y)
    integer, intent(in) :: n, nnz
    real, intent(in) :: val(nnz), x(n)
    integer, intent(in) :: row(n+1), col(nnz)
    real, intent(out) :: y(n)
    integer i, j
  
    !$ACC PARALLEL LOOP DEFAULT(PRESENT) COPYIN(val, row, col, x) COPYOUT(y)
    do i = 1, n
      y(i) = 0.0
      do j = row(i), row(i+1)-1
        y(i) = y(i) + val(j) * x(col(j))
      enddo
    enddo
    !$ACC END PARALLEL LOOP
  end subroutine spmv_parallel

  module matrix_operations
    implicit none
  contains
    ! Subroutine to populate a sparse matrix in CSR format with random values.
    ! Parameters:
    ! n - The dimension of the square matrix (number of rows/columns).
    ! density - The density of non-zero elements in the matrix.
    ! val - The output array to hold the non-zero values of the matrix.
    ! row - The output array to hold the row pointers for CSR format.
    ! col - The output array to hold the column indices of the non-zero values.
    subroutine populate_matrix(n, density, val, row, col)
        integer, intent(in) :: n
        real, intent(in) :: density
        real, intent(out) :: val(:)
        integer, intent(out) :: row(:), col(:)
        integer :: i, j, nnz
        real :: rand_val
  
        nnz = 0
        do i = 1, n
            do j = 1, n
                call random_number(rand_val)
                if (rand_val < density) then
                    nnz = nnz + 1
                    val(nnz) = rand_val  ! Assign random value to val array.
                    col(nnz) = j  ! Assign column index to col array.
                endif
            end do
            row(i+1) = nnz + 1  ! Update row pointer array.
        end do
    end subroutine populate_matrix
  end module matrix_operations

program main
    use matrix_operations
    implicit none
    integer, parameter :: n = 10
    real, parameter :: density = 0.5
    real :: val(n*n), x(n), y(n)
    integer :: row(n+1), col(n*n)
    integer :: i

    call populate_matrix(n, density, val, row, col)

    x = 1.0

    ! Assuming spmv_parallel is defined elsewhere and correctly
    call spmv_parallel(n, n*n, val, row, col, x, y)

    do i = 1, n
        print*, 'y(', i, ') = ', y(i)
    end do
end program main