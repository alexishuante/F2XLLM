PROGRAM main
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 50000000
    REAL, DIMENSION(n, n) :: A
    REAL, DIMENSION(n) :: x, y
    INTEGER :: i, j, k
    INTEGER :: start, finish, rate
    REAL :: total_time, average_time
    total_time = 0.0

    ! Call the gemv_parallel subroutine 10 times and record the time
    DO k = 1, 10
        CALL system_clock(start)
        CALL gemv_parallel(n, A, x, y)
        CALL system_clock(finish)
        time_taken(k) = REAL(finish - start) / REAL(rate)
        total_time = total_time + time_taken(k)
    END DO

    average_time = total_time / 10.0

    ! Print the average execution time
    PRINT *, "Average execution time: ", average_time, " seconds."

END PROGRAM main



subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: A(n, n)
    real, intent(in) :: x(n)
    real, intent(out) :: y(n)
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