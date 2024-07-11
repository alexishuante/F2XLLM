program read_csr_matrix
    implicit none
    integer, parameter :: maxnnz = 100
    real, dimension(maxnnz) :: val
    integer, dimension(maxnnz) :: col
    integer, dimension(maxnnz) :: row
    integer :: nnz, i, ios, valPos
    character(len=100) :: line
    character(len=5) :: temp
    real :: tempVal
    integer :: tempCol

    open(unit=10, file='csr_matrix.csv', status='old', action='read')
    row = 0

    nnz = 0
    do
        read(10, '(*(A))', iostat=ios) line
        if (ios /= 0) exit

        ! Check if the line is the index pointer line
        read(line, *, iostat=ios) temp
        if (ios == 0 .and. temp == "indptr") then
            read(line, *) temp, row
            exit
        else
            ! Revised approach to handle comma-separated values or other formats
            valPos = index(line, ',') ! Find the position of the comma, if present
            if (valPos > 0) then
                read(line(:valPos-1), *, iostat=ios) tempVal
                if (ios /= 0) cycle ! Skip lines that cannot be processed
                read(line(valPos+1:), *, iostat=ios) tempCol
                if (ios /= 0) cycle ! Skip lines that cannot be processed
            else
                read(line, *, iostat=ios) tempVal, tempCol
                if (ios /= 0) cycle ! Skip lines that cannot be processed
            endif
            val(nnz+1) = tempVal
            col(nnz+1) = tempCol
            nnz = nnz + 1
        endif
    end do

    close(unit=10)

    ! Print the arrays to verify
    do i = 1, nnz
        print *, "Value: ", val(i), " Column: ", col(i), ""
    end do

    do i = 1, nnz
        print *, "Row: ", row(i)
    end do

    print *, "Temp: ", temp 


end program read_csr_matrix