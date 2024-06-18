PROGRAM main
    USE omp_lib  ! Include the OpenMP library
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 50000000  ! Adjusted from 10 to 50 million
    REAL, DIMENSION(n) :: x, y
    REAL :: a
    INTEGER :: i
    DOUBLE PRECISION :: start_time, end_time, elapsed_time

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Start timing
    start_time = omp_get_wtime()

    ! Call the saxpy_parallel subroutine
    CALL saxpy(n, a, x, y)

    ! End timing
    end_time = omp_get_wtime()
    elapsed_time = end_time - start_time


    ! Print the results
    ! Note: Printing 50 million results is impractical, consider summarizing
    PRINT *, "Sample y(1) = ", y(1)
    PRINT *, "Sample y(n) = ", y(n)

    ! Print the elapsed time
    PRINT *, "Elapsed time: ", elapsed_time, " seconds"


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
