program main
  implicit none
  integer, parameter :: n = 4, niter = 10, nthreads = 4
  real(kind=8), dimension(n, n, n) :: u, unew
  integer :: i, j, k, counter

  counter = 1

  ! Initialize the grid with specific values
  do k = 1, n
    do j = 1, n
      do i = 1, n
        u(i, j, k) = counter
        counter = counter + 1
      end do
    end do
  end do

  unew = u
  
  ! Call the Jacobi method
  call jacobi_parallel(u, unew, n, niter, nthreads)

  

  ! Print the final value of u and unew
  print *, "Final value of u:"
  print *, u

end program main

subroutine jacobi_parallel(u, unew, n, niter, nthreads)
  implicit none
  integer, intent(in) :: n, niter, nthreads
  real(kind=8), dimension(n, n, n), intent(inout) :: u, unew
  integer :: i, j, k, iter

  !$OMP PARALLEL NUM_THREADS(nthreads) PRIVATE(i, j, k, iter)
  do iter = 1, niter
    !$OMP DO SCHEDULE(STATIC)
    do k = 2, n - 1
      do j = 2, n - 1
        do i = 2, n - 1
          unew(i, j, k) = 0.125 * (u(i-1, j, k) + u(i+1, j, k) + u(i,j-1, k) &
            + u(i, j+1, k) + u(i, j, k-1) + u(i, j, k+1) + u(i, j, k))
        end do
      end do
    end do
    !$OMP END DO
    !$OMP BARRIER
    u(:, :, :) = unew(:, :, :)
  end do
  !$OMP END PARALLEL
end subroutine jacobi_parallel
