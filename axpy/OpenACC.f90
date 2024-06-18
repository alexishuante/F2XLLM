! Test Comments test
PROGRAM main
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 500000000
    REAL, DIMENSION(n) :: x, y
    REAL :: a
    INTEGER :: i

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Call the saxpy subroutine
    CALL saxpy(n, a, x, y)

    ! Print the results
    !DO i = 1, n
    !    PRINT *, "y(", i, ") = ", y(i)
    !END DO

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
! Test Check
