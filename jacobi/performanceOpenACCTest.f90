program test_jacobi_parallel_performance
    implicit none
    integer, parameter :: n = 10, niter = 100
    real(kind=8), allocatable :: u(:,:,:), unew(:,:,:)
    real(kind=8) :: start, finish, total_time, average_time
    integer :: i

    ! Include the module for timing
    use, intrinsic :: iso_fortran_env, only: system_clock, system_clock_count

    allocate(u(n, n, n), unew(n, n, n))

    ! Initialize arrays with some values
    u = 1.0
    unew = 0.0

    ! Warm-up call
    call jacobi_parallel(u, unew, n, niter)

    total_time = 0.0
    do i = 1, 10
        call system_clock(start)
        call jacobi_parallel(u, unew, n, niter)
        call system_clock(finish)
        total_time = total_time + real(finish - start, kind=8) / real(system_clock_count, kind=8)
    end do

    average_time = total_time / 10.0

    ! Optionally print the average time to check performance
    print *, 'Average time of 10 calls:', average_time

    deallocate(u, unew)
end program test_jacobi_parallel_performance

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