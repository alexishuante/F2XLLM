program vector_addition
    use openacc
    implicit none
    integer, parameter :: N = 1000000000
    real, allocatable :: A(:), B(:), C(:)
    integer :: i
    real :: start_time, end_time, computation_time  ! Corrected declaration
    integer :: device_num, device_type


    ! Allocate arrays
    allocate(A(N), B(N), C(N))

    ! Initialize arrays
    do i = 1, N
        A(i) = i
        B(i) = 2*i
    end do

    ! Query the current device number and type
    device_num = acc_get_device_num(acc_device_not_host)
    device_type = acc_get_device_type()

    ! Print the device information
    print *, 'Using device number: ', device_num
    print *, 'Device type: ', device_type

    

    ! Start timing
    call cpu_time(start_time)

    ! Compute vector addition A + B = C on the GPU
    !$acc data copyin(A, B) copyout(C)
    !$acc parallel loop
    do i = 1, N
        C(i) = A(i) + B(i)
    end do
    !$acc end data

    ! End timing
    call cpu_time(end_time)

    ! Calculate and print computation time
    computation_time = end_time - start_time
    print *, 'Computation Time: ', computation_time, ' seconds'
    
    ! Deallocate arrays
    deallocate(A, B, C)
end program vector_addition