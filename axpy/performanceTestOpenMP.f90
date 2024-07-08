PROGRAM main
    USE iso_c_binding, ONLY: C_LONG_LONG
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 1000000000
    REAL(KIND=8), ALLOCATABLE :: x(:), y(:)
    REAL(KIND=8) :: a
    INTEGER :: count_max
    INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
    REAL :: wall_time
    INTEGER :: i

    ! Allocate the vectors on the heap
    ALLOCATE(x(n), y(n))

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Warmup run
    CALL saxpy_parallel(n, a, x, y)

    y=0.0

    ! Find average per call time
    count_max = 10
    count_diff = 0
    DO i = 1, count_max
        CALL system_clock(count_rate=count_rate)
        CALL system_clock(start_count)
        CALL saxpy_parallel(n, a, x, y)
        CALL system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    END DO

    wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
    PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

    ! Deallocate the vectors
    DEALLOCATE(x, y)

END PROGRAM main

subroutine saxpy_parallel(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real(KIND=8), intent(in) :: a
    real(KIND=8), intent(in) :: x(n)
    real(KIND=8), intent(inout) :: y(n)
    integer :: i

    !$OMP PARALLEL DO
    do i = 1, n
        y(i) = a * x(i) + y(i)
    end do
    !$OMP END PARALLEL DO

end subroutine saxpy_parallel