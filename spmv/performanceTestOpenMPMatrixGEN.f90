module sparse_matrix_mod
  implicit none
  contains

  subroutine generateSparseMatrix(rows, cols, disparity, seed, row_ptr, col_ind, values)
    integer, intent(in) :: rows, cols
    real, intent(in) :: disparity
    integer, intent(in) :: seed
    integer, allocatable, intent(out) :: row_ptr(:), col_ind(:)
    real, allocatable, intent(out) :: values(:)
    integer :: total_elements, non_zero_elements, i, j, count
    real :: rnd_row, rnd_col

    call random_seed(put=seed)
    total_elements = rows * cols
    non_zero_elements = int(total_elements * disparity)
    allocate(row_ptr(rows+1))
    allocate(col_ind(non_zero_elements))
    allocate(values(non_zero_elements))
    row_ptr = 0
    count = 0

    do i = 1, non_zero_elements
      call random_number(rnd_row)
      call random_number(rnd_col)
      j = int(rnd_row * rows) + 1
      col_ind(i) = int(rnd_col * cols) + 1
      values(i) = rand()  ! Assuming non-zero elements are random floats
      row_ptr(j) = row_ptr(j) + 1
    end do

    ! Convert row_ptr to CSR format
    do i = 2, rows + 1
      row_ptr(i) = row_ptr(i) + row_ptr(i-1)
    end do
  end subroutine generateSparseMatrix

  subroutine saveSparseMatrix(filename, row_ptr, col_ind, values)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: row_ptr(:), col_ind(:)
    real, intent(in) :: values(:)
    integer :: i, unit

    open(newunit=unit, file=filename, status='replace', action='write')
    do i = 1, size(row_ptr)
      write(unit, *) row_ptr(i)
    end do
    do i = 1, size(col_ind)
      write(unit, *) col_ind(i), values(i)
    end do
    close(unit)
  end subroutine saveSparseMatrix

end module sparse_matrix_mod

program testSparseMatrix
  use sparse_matrix_mod
  implicit none
  integer, parameter :: rows = 1000000, cols = 1000000
  real, parameter :: disparity = 0.01
  integer, allocatable :: row_ptr(:), col_ind(:)
  real, allocatable :: values(:)
  integer :: seed

  seed = 42
  call generateSparseMatrix(rows, cols, disparity, seed, row_ptr, col_ind, values)
  call saveSparseMatrix('sparse_matrix.dat', row_ptr, col_ind, values)
end program testSparseMatrix