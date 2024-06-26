program vector_addition
    use iso_c_binding
    implicit none
    integer, parameter :: N = 1
    real, dimension(:), allocatable :: A, B, C
    integer :: i, j
    real :: start_time, end_time, total_time

    ! Interface to the C functions
    interface
        subroutine set_device(device_id) bind(C, name="set_device")
            import :: c_int
            integer(c_int), value :: device_id
        end subroutine set_device

        subroutine copy_to_device(src, dst, count) bind(C, name="copy_to_device")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: src, dst
            integer(c_size_t), value :: count
        end subroutine copy_to_device

        ! Interface to the HIP kernel launcher
        subroutine launch_vector_add_kernel(A, B, C, N) bind(C, name="launch_vector_add_kernel")
            import :: c_ptr, c_int
            type(c_ptr), value :: A, B, C
            integer(c_int), value :: N
        end subroutine launch_vector_add_kernel
    end interface

    ! Allocate and initialize arrays
    allocate(A(N), B(N), C(N))

    total_time = 0.0
    do j = 1, 11
        A = 1.0
        B = 2.0

        ! Set the device to the second GPU
        call set_device(1)

        ! Copy data from CPU to GPU before computation
        call copy_to_device(c_loc(A), c_loc(B), N * sizeof(A(1)))

        call cpu_time(start_time)
        ! Offload computation to the GPU using the HIP kernel
        call launch_vector_add_kernel(c_loc(A), c_loc(B), c_loc(C), N)
        call cpu_time(end_time)

        if (j > 1) then  ! Skip timing for the first iteration
            total_time = total_time + (end_time - start_time)
        end if

        ! Clean up
        deallocate(A, B, C)
        allocate(A(N), B(N), C(N))  ! Reallocate for the next iteration

        print *, "Iteration", j, "completed successfully."
    end do

    print *, "Average time for last 10 iterations: ", total_time / 10.0, "seconds."
end program vector_addition