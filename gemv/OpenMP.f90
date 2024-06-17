PROGRAM main
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 15
    REAL, DIMENSION(n, n) :: A
    REAL, DIMENSION(n) :: x, y
    INTEGER :: i, j

    ! Initialize the matrix and vectors
    A = 3.0
    x = 5.0
    y = 0.0

    ! Call the gemv_parallel subroutine
    CALL gemv_parallel(n, A, x, y)

    ! Print the results
    DO i = 1, n
        PRINT *, "y(", i, ") = ", y(i)
    END DO

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
do i = 1, n
    sum = 0.0
    do j = 1, n
        sum = sum + A(i, j) * x(j)
    end do
    y(i) = sum
end do
!$OMP END PARALLEL DO

end subroutine gemv_parallel


