program test_gemm_parallel
    USE iso_c_binding, ONLY: C_LONG_LONG
    implicit none
    integer, parameter :: m = 25027, n = 25027, k = 25027
    integer, parameter :: lda = m, ldb = k, ldc = m
    real(kind=8), dimension(lda, k) :: A
    real(kind=8), dimension(ldb, n) :: B
    real(kind=8), dimension(ldc, n) :: C
    real(kind=8) :: alpha = 1.0, beta = 0.0
    integer :: i, call
    INTEGER :: count_max
    INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
    REAL :: wall_time

    ! Initialize matrices A and B with some values
    A = 1.0
    B = 1.0

    ! Initialize matrix C with zeros
    C = 0.0

    ! Warmup call to gemm_parallel
    !$ACC DATA COPYIN(A, B, alpha, beta), CREATE(C)
    call gemm_parallel(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    !$ACC END DATA

    ! Reset C for the actual runs
    C = 0.0

    ! Find average per call time
    count_max = 10
    count_diff = 0

    !$ACC DATA COPYIN(A, B, alpha, beta), CREATE(C)
    CALL system_clock(count_rate=count_rate)
    DO i = 1, count_max
        CALL system_clock(start_count)
        call gemm_parallel(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        CALL system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    END DO
    !$ACC END DATA

    wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
    PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"

end program test_gemm_parallel

subroutine gemm_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    implicit none
    integer, intent(in) :: m, n, k, lda, ldb, ldc
    real(kind=8), intent(in) :: alpha, beta
    real(kind=8), intent(in) :: a(lda, k), b(ldb, n)
    real(kind=8), intent(inout) :: c(ldc, n)
    integer :: i, j, l
    real(kind=8) :: temp

    !$ACC PARALLEL LOOP COLLAPSE(2) PRIVATE(i, j, l, temp)
    do j = 1, n
        do i = 1, m
            temp = 0.0
            do l = 1, k
                temp = temp + a(i, l) * b(l, j)
            end do
            c(i, j) = alpha * temp + beta * c(i, j)
        end do
    end do
    !$ACC END PARALLEL LOOP
end subroutine gemm_parallel
