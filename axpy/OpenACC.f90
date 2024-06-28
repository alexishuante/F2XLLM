! PROGRAM main
!     use openacc
!     IMPLICIT NONE
!     INTEGER, PARAMETER :: n = 500000
!     REAL, DIMENSION(n) :: x, y
!     REAL :: a
!     INTEGER :: i

!     ! Initialize the scalar and vectorss
!     a = 2.0
!     x = 3.0
!     y = 0.0

!     ! Call the saxpy subroutine
!     DO i = 1, 500000
!         CALL saxpy(n, a, x, y)
!     END DO
!     CALL saxpy(n, a, x, y)

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

! end subroutine saxpy
! ! Test Comments test

program vector_add
    use openacc 
    implicit none
    integer, parameter :: N = 1000000
    real :: a(N), b(N), c(N)
  
    ! Initialize arrays
    a = 1.0
    b = 2.0
  
    !$acc parallel loop
    do i = 1, N
      c(i) = a(i) + b(i)
    end do
    !$acc end parallel loop
  end program vector_add