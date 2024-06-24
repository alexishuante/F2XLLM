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

PROGRAM main
    IMPLICIT NONE

    ! Calculate array size to target memory usage
    INTEGER, PARAMETER :: target_memory_percent = 30 ! Adjust this (25-50)
    REAL, PARAMETER :: bytes_per_real = 4           ! Size of a real number in bytes
    INTEGER, PARAMETER :: total_memory_gb = 32        ! MI100 memory in GB
    INTEGER, PARAMETER :: target_memory_bytes = target_memory_percent * total_memory_gb * 1024**3 / 100
    INTEGER, PARAMETER :: n = target_memory_bytes / (2 * bytes_per_real) ! Divide by 2 for two arrays

    REAL, DIMENSION(n) :: x, y
    REAL :: a
    REAL :: start_time, end_time, elapsed_time, total_time
    INTEGER :: i

    ! Initialize on the CPU
    a = 2.0
    x = 3.0
    y = 0.0

    ! Warmup run
    CALL saxpy(n, a, x, y)

    ! Timed runs
    total_time = 0.0
    DO i = 1, 10
        CALL CPU_TIME(start_time)   ! Get start time

        !$acc data copyin(x, y)  ! Copy data to GPU
        CALL saxpy(n, a, x, y)
        !$acc data copyout(y)   ! Copy results back to CPU

        CALL CPU_TIME(end_time)    ! Get end time
        elapsed_time = end_time - start_time
        total_time = total_time + elapsed_time
    END DO

    ! Calculate average time
    PRINT *, "Average execution time (seconds):", total_time / 10.0

    ! Optional: Verify memory usage
    PRINT *, "Estimated memory used (GB):", (n * 2 * bytes_per_real) / 1024.0**3

END PROGRAM main



