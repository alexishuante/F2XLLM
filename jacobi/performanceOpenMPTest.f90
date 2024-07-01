program main
  implicit none
  integer, parameter :: n = 4, niter = 10, nthreads = 4
  real(kind=8), dimension(n, n, n) :: u, unew
  integer :: i, j, k, counter
  real(kind=8) :: start_time, end_time, total_time, average_time
  integer :: call_counter

  ! Initialize timing and counter variables
  total_time = 0.0
  call_counter = 0

  ! Initialize the grid with specific values
  counter = 1
  do k = 1, n
    do j = 1, n
      do i = 1, n
        u(i, j, k) = counter
        counter = counter + 1
      end do
    end do
  end do
  unew = u

  ! Warmup call
  call jacobi_parallel(u, unew, n, niter, nthreads)

  ! Measure 10 calls to jacobi_parallel
  do call_counter = 1, 10
    start_time = omp_get_wtime()
    call jacobi_parallel(u, unew, n, niter, nthreads)
    end_time = omp_get_wtime()
    total_time = total_time + (end_time - start_time)
  end do

  average_time = total_time / 10.0

  ! Print the final value of u, unew, and the average time
  print *, "Final value of u:"
  print *, u
  print *, "Average time for 10 calls: ", average_time

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