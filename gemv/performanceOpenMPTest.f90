PROGRAM main
    USE iso_c_binding, ONLY: C_LONG_LONG
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 31622  
    REAL(KIND=8), ALLOCATABLE :: A(:,:), x(:), y(:)
    INTEGER :: i
    INTEGER :: count_max
    INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
    REAL :: wall_time

    ! Allocate the arrays dynamically
    ALLOCATE(A(n, n), x(n), y(n))
    A = 1.0
    x = 1.0
    y = 0.0

    ! Do a warmup run to ensure fair timing
    CALL gemv_parallel(n, A, x, y)

    ! Reset y for the actual runs
    y = 0.0

    ! Find average per call time
    count_max = 10
    count_diff = 0
    i = 0
    DO i = 1, count_max
        CALL system_clock(count_rate=count_rate)
        CALL system_clock(start_count)
        CALL gemv_parallel(n, A, x, y)
        CALL system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    END DO

    wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
    PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

    DEALLOCATE(A, x, y)

END PROGRAM main



subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real(KIND=8), intent(in) :: A(n, n)
    real(KIND=8), intent(in) :: x(n)
    real(KIND=8), intent(out) :: y(n)
    integer :: i, j
    real :: sum

    !$OMP PARALLEL DO PRIVATE(j, sum)
    DO i = 1, n
        sum = 0.0
        DO j = 1, n
            sum = sum + A(i, j) * x(j)
        END DO
        y(i) = sum
    END DO
    !$OMP END PARALLEL DO

end subroutine gemv_parallel