! program test_gemv_parallel
!     USE iso_c_binding, ONLY: C_LONG_LONG
!     implicit none
!     integer, parameter :: n = 31622
!     REAL(KIND=8) :: A(n, n), x(n), y(n)
!     INTEGER :: i
!     INTEGER :: count_max
!     INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
!     REAL :: wall_time

!     ! Initialize matrix A and vector x with some values
!     A = 1.0
!     x = 1.0
!     y = 0.0

!     ! Warm-up call
!     call gemv_parallel(n, A, x, y)

!     ! Reset y for the actual runs
!     y = 0.0

!     ! Find average per call time
!     count_max = 10
!     count_diff = 0
!     i = 0
!     DO i = 1, count_max
!         CALL system_clock(count_rate=count_rate)
!         CALL system_clock(start_count)
!         CALL gemv_parallel(n, A, x, y)
!         CALL system_clock(end_count)
!         count_diff = count_diff + (end_count - start_count)
!     END DO


!     wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
!     PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

! end program test_gemv_parallel

! subroutine gemv_parallel(n, A, x, y)
!     implicit none
!     integer, intent(in) :: n
!     real, intent(in) :: A(n, n)
!     real, intent(in) :: x(n)
!     real, intent(out) :: y(n)
!     integer :: i, j
!     real :: sum

!     !$ACC PARALLEL LOOP
!     do i = 1, n
!         sum = 0.0
!         do j = 1, n
!             sum = sum + A(i, j) * x(j)
!         end do
!         y(i) = sum
!     end do
!     !$ACC END PARALLEL LOOP

! end subroutine gemv_parallel

program test_gemv_parallel
    USE iso_c_binding, ONLY: C_LONG_LONG
    implicit none
    integer, parameter :: n = 43346
    REAL(KIND=8) :: A(n, n), x(n), y(n)
    INTEGER :: i
    INTEGER :: count_max
    INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
    REAL :: wall_time

    ! Initialize matrix A and vector x with some values
    A = 1.0
    x = 1.0
    y = 0.0

    ! Warm-up call
    call gemv_parallel(n, A, x, y)

    ! Reset y for the actual runs
    y = 0.0

    ! Find average per call time
    count_max = 10
    count_diff = 0

    ! Transfer data to device once
    !$acc data copyin(A, x) create(y)
    DO i = 1, count_max
        CALL system_clock(count_rate=count_rate)
        CALL system_clock(start_count)
        CALL gemv_parallel(n, A, x, y)
        CALL system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    END DO
    !$acc end data

    wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
    PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

end program test_gemv_parallel

subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real(KIND=8), intent(in) :: A(n, n)
    real(KIND=8), intent(in) :: x(n)
    real(KIND=8), intent(out) :: y(n)
    integer :: i, j
    real :: sum

    !$ACC PARALLEL LOOP
    do i = 1, n
        sum = 0.0
        do j = 1, n
            sum = sum + A(i, j) * x(j)
        end do
        y(i) = sum
    end do
    !$ACC END PARALLEL LOOP

end subroutine gemv_parallel
