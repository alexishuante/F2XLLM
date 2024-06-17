program test_gemm_parallel
    implicit none
    integer, parameter :: m = 3, n = 3, k = 3
    integer, parameter :: lda = m, ldb = k, ldc = m
    real(kind=8), dimension(lda, k) :: A
    real(kind=8), dimension(ldb, n) :: B
    real(kind=8), dimension(ldc, n) :: C
    real(kind=8) :: alpha = 1.0, beta = 0.0
    integer :: i, j
  
    ! Initialize matrices A and B with some values
    do i = 1, m
      do j = 1, k
        A(i, j) = i * 0.5 + j
      end do
    end do
  
    do i = 1, k
      do j = 1, n
        B(i, j) = j * 0.2 + i
      end do
    end do
  
    ! Initialize matrix C with zeros
    C = 0.0
  
    ! Call the gemm_parallel subroutine
    call gemm_parallel(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  
    ! Output the resulting matrix C
    print *, 'Resulting matrix C:'
    do i = 1, m
      print '(3F8.2)', C(i, :)
    end do
  
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
