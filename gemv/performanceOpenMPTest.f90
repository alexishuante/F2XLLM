PROGRAM main
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 1400
    REAL, DIMENSION(n, n) :: A
    REAL, DIMENSION(n) :: x, y
    INTEGER :: i, j, k
    INTEGER :: start, finish, rate
    REAL :: total_time, average_time
    REAL :: startTime, endTime, totalTime, averageTime
    total_time = 0.0

    ! Call the gemv_parallel subroutine 10 times and record the time
    DO k = 1, 10
        ! CALL system_clock(start)
        CALL CPU_TIME(startTime)
        CALL gemv_parallel(n, A, x, y)
        CALL CPU_TIME(endTime)
        totalTime = totalTime + (endTime - startTime)
    END DO

    average_time = totalTime / 10.0

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