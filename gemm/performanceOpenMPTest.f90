program main
  USE iso_c_binding, ONLY: C_LONG_LONG
  implicit none
  integer, parameter :: m = 25027, n = 25027, k = 25027
  integer, parameter :: lda = m, ldb = k, ldc = m
  real(kind=8), allocatable :: a(:,:), b(:,:), c(:,:)
  real(kind=8) :: alpha, beta
  INTEGER :: count_max
  INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
  REAL :: wall_time
  INTEGER :: i

  ! Allocate the matrices
  allocate(a(lda, k), b(ldb, n), c(ldc, n))

    ! Initialize alpha, beta, a, b, and c here...
    alpha = 1.0
    beta = 0.0
    a = 1.0
    b = 1.0
    c = 0.0

    ! Warmup call
    call gemm_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

    ! Find average per call time
    count_max = 10
    count_diff = 0
    i = 0
    DO i = 1, count_max
        CALL system_clock(count_rate=count_rate)
        CALL system_clock(start_count)
        call gemm_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        CALL system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    END DO

    wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
    PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

end program main

subroutine gemm_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    implicit none
    integer, intent(in) :: m, n, k, lda, ldb, ldc
    real(kind=8), intent(in) :: alpha, beta
    real(kind=8), intent(in) :: a(lda, k), b(ldb, n)
    real(kind=8), intent(inout) :: c(ldc, n)
    integer :: i, j, l
    real(kind=8) :: temp
  
    !$OMP PARALLEL DO PRIVATE(i, j, l, temp) SCHEDULE(STATIC)
    do j = 1, n
      do i = 1, m
        temp = 0.0
        do l = 1, k
          temp = temp + a(i, l) * b(l, j)
        end do
        c(i, j) = alpha * temp + beta * c(i, j)
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine gemm_parallel