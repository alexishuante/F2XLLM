! ! PROGRAM main
! !     IMPLICIT NONE
! !     INTEGER, PARAMETER :: n = 300000000
! !     REAL, DIMENSION(n) :: x, y
! !     REAL :: a
! !     INTEGER :: i
! !     REAL :: start, finish, elapsed_time, average_time
! !     INTEGER :: num_runs, run

! PROGRAM main
!     USE iso_c_binding, ONLY: C_LONG_LONG
!     IMPLICIT NONE
!     INTEGER, PARAMETER :: n = 1000000000
!     REAL(KIND=8), DIMENSION(n) :: x, y
!     REAL(KIND=8) :: a
!     INTEGER :: count_max
!     INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
!     REAL :: wall_time, warmup
!     INTEGER :: i

!     ! Initialize the scalar and vectors
!     a = 2.0
!     x = 3.0
!     y = 0.0

!     ! Warm-up call to ensure fair timing
!     CALL saxpy(n, a, x, y)

!     ! Reset y for the actual runs
!     y = 0.0

!     ! num_runs = 10
!     ! elapsed_time = 0.0

!     ! DO run = 1, num_runs
!     !     CALL cpu_time(start)
!     !     CALL saxpy(n, a, x, y)
!     !     CALL cpu_time(finish)
!     !     elapsed_time = elapsed_time + (finish - start)
!     ! END DO

!     ! Find average per call time
!     count_max = 10
!     count_diff = 0
!     DO i = 1, count_max
!         CALL system_clock(count_rate=count_rate)
!         CALL system_clock(start_count)
!         ! CALL sleep(1)
!         CALL saxpy(n, a, x, y)
!         CALL system_clock(end_count)
!         count_diff = count_diff + (end_count - start_count)
!     END DO

!     wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
!     PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

!     ! The results printing loop is commented out due to the large value of n

! END PROGRAM main

! subroutine saxpy(n, a, x, y)
!     implicit none
!     integer, intent(in) :: n
!     real(KIND=8), intent(in) :: a
!     real(KIND=8), intent(in) :: x(n)
!     real(KIND=8), intent(inout) :: y(n)
!     integer :: i
!     !$acc kernels
!     do i = 1, n
!         y(i) = a * x(i) + y(i)
!     end do
!     !$acc end kernels
! end subroutine saxpy

PROGRAM main
    USE iso_c_binding, ONLY: C_LONG_LONG
    IMPLICIT NONE
    INTEGER, PARAMETER :: n = 1000000000
    REAL(KIND=8), DIMENSION(n) :: x, y
    REAL(KIND=8) :: a
    INTEGER :: count_max
    INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
    REAL :: wall_time
    INTEGER :: i

    ! Initialize the scalar and vectors
    a = 2.0
    x = 3.0
    y = 0.0

    ! Warm-up call to ensure fair timing
    CALL saxpy(n, a, x, y)

    ! Reset y for the actual runs
    y = 0.0

    count_max = 10
    count_diff = 0

    ! Transfer data to device once
    !$acc data copyin(x, a) create(y)
    DO i = 1, count_max
        CALL system_clock(count_rate=count_rate)
        CALL system_clock(start_count)
        CALL saxpy(n, a, x, y)
        CALL system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    END DO
    !$acc end data

    wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
    PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

END PROGRAM main

subroutine saxpy(n, a, x, y)
    implicit none
    integer, intent(in) :: n
    real(KIND=8), intent(in) :: a
    real(KIND=8), intent(in) :: x(n)
    real(KIND=8), intent(inout) :: y(n)
    integer :: i

    !$acc kernels
    do i = 1, n
        y(i) = a * x(i) + y(i)
    end do
    !$acc end kernels

end subroutine saxpy
