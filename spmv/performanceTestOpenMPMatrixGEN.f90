subroutine spmv_parallel(n, nnz, val, row, col, x, y)
    integer, intent(in) :: n, nnz
    real, intent(in) :: val(nnz), x(n)
    integer, intent(in) :: row(n+1), col(nnz)
    real, intent(out) :: y(n)
    integer i, j
  
    !$OMP PARALLEL DO DEFAULT(NONE) SHARED(n, nnz, val, row, col, x, y) PRIVATE(i, j)
    do i = 1, n
      y(i) = 0.0
      do j = row(i), row(i+1)-1
        y(i) = y(i) + val(j) * x(col(j))
      enddo
    enddo
    !$OMP END PARALLEL DO
  end subroutine spmv_parallel

program main
implicit none

real, dimension(4) :: val
integer, dimension(4) :: col
integer, dimension(6) :: row

real, dimension(5) :: x = [1.0, 1.0, 1.0, 1.0, 1.0]
real, dimension(5) :: y
integer :: i
integer :: status
character(len=100) :: line

! Read from a file and store the values in the arrays
! Open the file
open(1, file='matrix_csr.txt', status='old', action='read')

! Check if it finds the text "# Values:"

read(1, '(A)', iostat=status) line
if (status /= 0) then
    print *, "Error reading file"
    stop
end if

if (trim(line) == "# Values:") then
    print *, "Found Values!"
    ! Code to execute if the text is found
    ! Add your code here
    do i = 1, 4
        read(1, *) line
        if (trim(line) == "# Column Indices:") then
            print *, "Found Columns"
            exit ! Exit the loop if the text is found
        else
            read(line, *) val(i)
        end if
    end do
end if
print *, val

! Check if it finds the text "# Column Indices:"
read(1, '(A)', iostat=status) line
if (status /= 0) then
    print *, "Error reading file"
    stop
end if

if (trim(line) == "# Column Indices:") then
    print *, "Found Columns!"
    ! Code to execute if the text is found
    do i = 1, 4
        read(1, *) line
        if (trim(line) == "# Row Pointers:") then
            print *, "Found Row Pointers"
            exit ! Exit the loop if the text is found
        else
            read(line, *) col(i)
        end if
    end do
end if
print *, col

! Check if it finds the text "# Row Pointers:"
read(1, '(A)', iostat=status) line
if (status /= 0) then
    print *, "Error reading file"
    stop
end if

if (trim(line) == "# Row Pointers:") then
    print *, "Found Row Pointers!"
    read(1, *) row
end if
print *, row

! Increase all values inside of row by one
row = row + 1

! New row values
print *, "New row values:"
print *, row


! Close the file
close(1)

print *, "Starting parallel computation..."

! Call the subroutine
call spmv_parallel(5, 4, val, row, col, x, y)


end program main
