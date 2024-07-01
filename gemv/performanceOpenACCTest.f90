program test_gemv_parallel
    implicit none
    integer, parameter :: n = 50000000
    real :: A(n, n), x(n), y(n)
    integer :: i, j
    real :: start_time, end_time, total_time, average_time
    integer :: num_calls

    ! Initialize matrix A and vector x with some values
    do i = 1, n
        x(i) = i
        do j = 1, n
            A(i, j) = i + j
        end do
    end do

    ! Warm-up call
    call gemv_parallel(n, A, x, y)

    total_time = 0.0
    num_calls = 10

    ! Measure time for 10 calls
    do i = 1, num_calls
        call cpu_time(start_time)
        call gemv_parallel(n, A, x, y)
        call cpu_time(end_time)
        total_time = total_time + (end_time - start_time)
    end do

    average_time = total_time / num_calls
    print *, 'Average time taken for 10 calls:', average_time, 'seconds'

end program test_gemv_parallel

subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: A(n, n)
    real, intent(in) :: x(n)
    real, intent(out) :: y(n)
    integer :: i, j
    real :: sum

    !$ACC PARALLEL LOOP
    do i = 1, n
        sum = 0.0
        do j = 1, n
            sum = sum + A(i, j) * x(j)
        end do
        y(i) = sum
    end do
    !$ACC END PARALLEL LOOP

end subroutine gemv_parallel