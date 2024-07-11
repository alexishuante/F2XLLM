program main
  USE iso_c_binding, ONLY: C_LONG_LONG
  implicit none
  integer, parameter :: niter = 10, nthreads = 256 ! Double check nthreads
  integer :: n = 950  ! Adjusted dimension size
  real(kind=8), allocatable, dimension(:,:,:) :: u, unew
  integer :: i, j, k, iter
  INTEGER :: count_max
  REAL :: start_time, end_time, total_time
  INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
  REAL :: wall_time

  ! Allocate the 3D arrays
  allocate(u(n, n, n), unew(n, n, n))

  ! Initialize the grid
  u = 1.0
  unew = 0.0

  ! Warmup call
  call jacobi_parallel(u, unew, n, niter, nthreads)

    ! Find average per call time
  count_max = 10
  count_diff = 0
  i = 0
  DO i = 1, count_max
      CALL system_clock(count_rate=count_rate)
      CALL system_clock(start_count)
      call jacobi_parallel(u, unew, n, niter, nthreads)
      CALL system_clock(end_count)
      count_diff = count_diff + (end_count - start_count)
  END DO

  wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
  PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"
  
  ! Clean up
  deallocate(u, unew)

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