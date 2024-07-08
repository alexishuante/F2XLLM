PROGRAM main
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 300000000
    REAL, DIMENSION(n) :: x, y
    REAL :: a
    INTEGER :: i
    REAL :: start, finish, elapsed_time, average_time
    INTEGER :: num_runs, run

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Set the GPU device to device 1
    CALL acc_set_device_num(1, acc_device_nvidia)  ! Use acc_device_nvidia or acc_device_default as appropriate

    ! Warm-up call to ensure fair timing
    CALL saxpy(n, a, x, y)

    ! Reset y for the actual runs
    y = 0.0

    num_runs = 1000
    elapsed_time = 0.0

    DO run = 1, num_runs
        CALL cpu_time(start)
        CALL saxpy(n, a, x, y)
        CALL cpu_time(finish)
        elapsed_time = elapsed_time + (finish - start)
    END DO

    average_time = elapsed_time / num_runs
    PRINT *, "Average execution time over ", num_runs, " runs: ", average_time, " seconds."

    ! The results printing loop is commented out due to the large value of n

END PROGRAM main

subroutine saxpy(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a
    real, intent(in) :: x(n)
    real, intent(inout) :: y(n)
    integer :: i
    !$acc kernels
    do i = 1, n
        y(i) = a * x(i) + y(i)
    end do
    !$acc end kernels
end subroutine saxpy