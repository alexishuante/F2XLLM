program test_spmv_parallel
    use omp_lib  ! For omp_get_wtime
    implicit none
    integer, parameter :: n = 4, nnz = 4
    real, dimension(nnz) :: val = [1.0, 2.0, 3.0, 4.0]
    integer, dimension(n+1) :: row = [1, 2, 3, 4, 5]
    integer, dimension(nnz) :: col = [1, 2, 3, 4]
    real, dimension(n) :: x = [1.0, 2.0, 3.0, 4.0], y(n)
    integer :: i
    real :: startTime, endTime, totalTime, averageTime
    integer :: numCalls, callIndex

    ! Warm-up call
    call spmv_parallel(n, nnz, val, row, col, x, y)

    ! Timing and averaging
    totalTime = 0.0
    numCalls = 10
    do callIndex = 1, numCalls
        startTime = omp_get_wtime()
        call spmv_parallel(n, nnz, val, row, col, x, y)
        endTime = omp_get_wtime()
        totalTime = totalTime + (endTime - startTime)
    end do
    averageTime = totalTime / numCalls

    ! Print the average time
    print*, 'Average time of 10 calls: ', averageTime, ' seconds'

    ! Print the resulting vector y from the last call
    print *, 'Resulting vector y from the last call:'
    do i = 1, n
      print *, y(i)
    end do
end program test_spmv_parallel

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
  