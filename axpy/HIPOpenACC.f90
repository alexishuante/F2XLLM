! Test Comments test
PROGRAM main
    USE iso_c_binding, ONLY: c_int, c_float
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 5000000
    REAL, POINTER, DIMENSION(:) :: x, y
    REAL :: a
    INTEGER :: i

    ! Add an interface to the C function
    interface
        subroutine saxpy_wrapper(n, a, x, y) bind(C, name="saxpy_wrapper")
            use iso_c_binding
            integer(c_int), value :: n
            real(c_float), value :: a
            real(c_float), intent(in) :: x(*)
            real(c_float), intent(inout) :: y(*)
        end subroutine saxpy_wrapper
    end interface

    ! Allocate the arrays
    allocate(x(n), y(n))

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Call the C wrapper instead of the Fortran subroutine
    CALL saxpy_wrapper(n, a, x, y)

    ! Print the results
    !DO i = 1, n
    !    PRINT *, "y(", i, ") = ", y(i)
    !END DO

    ! Deallocate the arrays
    deallocate(x, y)

END PROGRAM main
