! Test Comments test
PROGRAM main
    USE iso_c_binding, ONLY: c_int, c_float
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 500000000
    REAL, DIMENSION(n) :: x, y
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

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Call the C wrapper instead of the Fortran subroutine
    CALL saxpy_wrapper(n, a, x, y)

! END PROGRAM main

! subroutine saxpy(n, a, x, y)
!     implicit none
!     integer, intent(in) :: n
!     real, intent(in) :: a
!     real, intent(in) :: x(n)
!     real, intent(inout) :: y(n)
!     integer :: i
!     !$acc data copyin(x, a) copy(y)
!     !$acc parallel loop
!     do i = 1, n
!         y(i) = a * x(i) + y(i)
!     end do
!     !$acc end data

! Remove the Fortran saxpy subroutine
