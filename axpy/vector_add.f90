program vector_add
    ! use openacc
    implicit none
    integer, parameter :: N = 1000000
    real :: A(N), B(N), C(N)

    ! Initialize arrays
    A = 1.0
    B = 2.0

    !$acc data copyin(A,B) copyout(C)
    !$acc parallel loop
    do i = 1, N
        C(i) = A(i) + B(i)
    end do
    !$acc end data

    ! Verify results (optional)
    ! do i = 1, N
    !     if (C(i) /= 3.0) then
    !         write(*,*) "Error at index:", i
    !         stop
    !     end if
    ! end do

    write(*,*) "Vector addition completed successfully"

end program vector_add
