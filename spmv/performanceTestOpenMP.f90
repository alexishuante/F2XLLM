program main
    use omp_lib  ! For omp_get_wtime
    implicit none
    integer, parameter :: n = 4, nnz = 8
    real :: val(nnz), x(n), y(n)
    integer :: row(n+1), col(nnz)
    integer :: i
    real :: startTime, endTime, totalTime, averageTime
    integer :: numCalls, callIndex

    ! Initialize the sparse matrix and the vector
    row = [1, 3, 6, 8, 9]
    val = [1.0, 7.0, 5.0, 3.0, 9.0, 2.0, 8.0, 6.0]
    col = [1, 2, 1, 3, 4, 2, 3, 4]
    x = [1.0, 2.0, 3.0, 4.0]

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

    ! Print the result from the last call
    do i = 1, n
        print*, 'y(', i, ') = ', y(i)
    end do
end program main

subroutine spmv_parallel(n, nnz, val, row, col, x, y)
    integer, intent(in) :: n, nnz
    real, intent(in) :: val(nnz), x(n)
    integer, intent(in) :: row(n+1), col(nnz)
    real, intent(out) :: y(n)
    integer i, j
  
    !$OMP PARALLEL DO DEFAULT(NONE) SHARED(n, nnz, val, row, col, x, y) PRIVATE(i, j)
    do i = 1, n
      y(i) = 0.0
      do j = row(i), row(i+1)-1
        y(i) = y(i) + val(j) * x(col(j))
      enddo
    enddo
    !$OMP END PARALLEL DO
  end subroutine spmv_parallel