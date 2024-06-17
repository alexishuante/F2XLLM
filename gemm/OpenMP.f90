
program main
    implicit none
    integer, parameter :: m = 3, n = 3, k = 3, lda = 3, ldb = 3, ldc = 3
    real(kind=8), dimension(lda, k) :: a
    real(kind=8), dimension(ldb, n) :: b
    real(kind=8), dimension(ldc, n) :: c
    real(kind=8) :: alpha, beta
    integer :: i

    ! Initialize alpha, beta, a, b, and c here...
    alpha = 1.0
    beta = 0.0
    a = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [lda, k])
    b = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [ldb, n])
    c = 0.0

    call gemm_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

    ! Output the resulting matrix C
    print *, 'Resulting matrix C:'
    do i = 1, m
      print '(3F8.2)', C(i, :)
    end do
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
