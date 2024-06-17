program test_jacobi_parallel
    implicit none
    integer, parameter :: n = 10, niter = 100
    real(kind=8), allocatable :: u(:,:,:), unew(:,:,:)
  
    allocate(u(n, n, n), unew(n, n, n))
  
    ! Initialize arrays with some values
    u = 1.0
    unew = 0.0
  
    call jacobi_parallel(u, unew, n, niter)
  
    ! Optionally print some of the results to check correctness
    print *, 'Sample of updated u(5,5,5):', u(5,5,5)
  
    deallocate(u, unew)
  end program test_jacobi_parallel

subroutine jacobi_parallel(u, unew, n, niter)
    implicit none
    integer, intent(in) :: n, niter
    real(kind=8), dimension(n, n, n), intent(inout) :: u, unew
    integer :: i, j, k, iter
  
    do iter = 1, niter
      !$ACC PARALLEL LOOP COLLAPSE(3) PRIVATE(i, j, k)
      do k = 2, n - 1
        do j = 2, n - 1
          do i = 2, n - 1
            unew(i, j, k) = 0.125 * (u(i-1, j, k) + u(i+1, j, k) + u(i, j-1, k) &
              + u(i, j+1, k) + u(i, j, k-1) + u(i, j, k+1) + u(i, j, k))
          end do
        end do
      end do
      !$ACC END PARALLEL LOOP
      u(:, :, :) = unew(:, :, :)
    end do
  end subroutine jacobi_parallel
