PROGRAM performanceTest
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 10
    REAL, DIMENSION(n) :: x, y
    REAL :: a
    INTEGER :: i
    REAL :: startTime, endTime, totalTime, averageTime
    INTEGER, PARAMETER :: numCalls = 10

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Warmup call
    CALL saxpy_parallel(n, a, x, y)

    totalTime = 0.0
    DO i = 1, numCalls
        CALL CPU_TIME(startTime)
        CALL saxpy_parallel(n, a, x, y)
        CALL CPU_TIME(endTime)
        totalTime = totalTime + (endTime - startTime)
    END DO

    averageTime = totalTime / numCalls
    PRINT *, "Average time of 10 calls: ", averageTime, " seconds"

END PROGRAM performanceTest



subroutine saxpy_parallel(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: a
    real, intent(in) :: x(n)
    real, intent(inout) :: y(n)
    integer :: i

    !$OMP PARALLEL DO
    do i = 1, n
        y(i) = a * x(i) + y(i)
    end do
    !$OMP END PARALLEL DO

end subroutine saxpy_parallel