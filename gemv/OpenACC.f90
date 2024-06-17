program test_gemv_parallel
    implicit none
    integer, parameter :: n = 15
    real :: A(n, n), x(n), y(n)
    integer :: i, j

    ! Initialize matrix A and vector x with some values
    ! do i = 1, n
    !    x(i) = i
    !    do j = 1, n
    !        A(i, j) = i + j
    !    end do
    ! end do
    A = 3.0
    x = 5.0
    y = 0.0
    ! Call the gemv_parallel subroutine
    call gemv_parallel(n, A, x, y)

    ! Print the result
    print *, 'Result vector y:'
    do i = 1, n
        print *, y(i)
    end do

end program test_gemv_parallel

subroutine gemv_parallel(n, A, x, y)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: A(n, n)
    real, intent(in) :: x(n)
    real, intent(out) :: y(n)
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
