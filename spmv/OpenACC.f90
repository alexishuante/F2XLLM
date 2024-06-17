!program test_spmv_parallel
!    implicit none
!    integer, parameter :: n = 4, nnz = 8
!    real, dimension(nnz) :: val = [1.0, 7.0, 5.0, 3.0, 9.0, 2.0, 8.0, 6.0]
!    integer, dimension(n+1) :: row = [1, 3, 6, 8, 9]
!    integer, dimension(nnz) :: col = [1, 2, 1, 3, 4, 2, 3, 4]
!    real, dimension(n) :: x = [1.0, 2.0, 3.0, 4.0], y(n)
!    integer :: i
!  
!    row = [1, 3, 6, 8, 9]
!    val = [1.0, 7.0, 5.0, 3.0, 9.0, 2.0, 8.0, 6.0]
!    col = [1, 2, 1, 3, 4, 2, 3, 4]
!
!    x = [1.0, 2.0, 3.0, 4.0]
!    call spmv_parallel(n, nnz, val, row, col, x, y)
!  
!    print *, 'Resulting vector y:'
!    do i = 1, n
!      print *, y(i)
!    end do
!  end program test_spmv_parallel

program main
    implicit none
    integer, parameter :: n = 4, nnz = 8
    real :: val(nnz), x(n), y(n)
    integer :: row(n+1), col(nnz)
    integer :: i

    ! Initialize the sparse matrix and the vector


    row = [1, 3, 6, 8, 9]
    val = [1.0, 7.0, 5.0, 3.0, 9.0, 2.0, 8.0, 6.0]
    col = [1, 2, 1, 3, 4, 2, 3, 4]

    x = [1.0, 2.0, 3.0, 4.0]

    ! Call the subroutine
    call spmv_parallel(n, nnz, val, row, col, x, y)

    ! Print the result
    do i = 1, n
        print*, 'y(', i, ') = ', y(i)
    enddo
end program main

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
