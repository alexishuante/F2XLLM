program main
    use iso_c_binding, only: C_LONG_LONG
    use openacc
    implicit none
    integer, parameter :: niter = 10
    integer :: n = 975  ! Adjusted dimension size
    real(kind=8), allocatable, dimension(:,:,:) :: u, unew
    integer :: i, j, k, iter
    integer :: count_max
    real :: start_time, end_time, total_time
    integer(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
    real :: wall_time

    ! Allocate the 3D arrays
    allocate(u(n, n, n), unew(n, n, n))

    ! Initialize the grid
    u = 1.0
    unew = 0.0

    ! Warmup call
    !$ACC DATA COPYIN(u), CREATE(unew)
    call jacobi_parallel(u, unew, n, niter)
    !$ACC END DATA

    ! Reset unew for the actual runs
    unew = 0.0

    ! Find average per call time
    count_max = 10
    count_diff = 0

    !$ACC DATA COPYIN(u), CREATE(unew)
    do i = 1, count_max
        call system_clock(count_rate=count_rate)
        call system_clock(start_count)
        call jacobi_parallel(u, unew, n, niter)
        call system_clock(end_count)
        count_diff = count_diff + (end_count - start_count)
    end do
    !$ACC END DATA

    wall_time = real(count_diff) / real(count_rate) / real(count_max)
    print *, "Average time of ", count_max, " calls: ", wall_time, " seconds"
  
    ! Clean up
    deallocate(u, unew)

end program main

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
