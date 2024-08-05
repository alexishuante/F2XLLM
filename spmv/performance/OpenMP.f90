subroutine spmv_parallel(n, nnz, val, row, col, x, y)
    integer, intent(in) :: n, nnz
    real(KIND=8), intent(in) :: val(nnz), x(n)
    integer, intent(in) :: row(n+1), col(nnz)
    real(KIND=8), intent(out) :: y(n)
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
USE iso_c_binding, ONLY: C_LONG_LONG
use omp_lib
implicit none
real(KIND=8), allocatable :: val(:)
integer, allocatable :: col(:)
integer, allocatable :: row(:)
real(KIND=8), allocatable :: x(:)
real(KIND=8), allocatable :: y(:)
integer(KIND=8) :: total_size
integer :: i, n, nnz
integer :: status
character(len=100) :: line
INTEGER :: count_max
INTEGER(C_LONG_LONG) :: start_count, end_count, count_diff, count_rate
REAL :: wall_time
INTEGER :: iter
INTEGER :: iMaxThreads
    
call    OMP_SET_NUM_THREADS(64) ! set to 64 threads
! Find the number of threads
iMaxThreads = OMP_GET_MAX_THREADS()
PRINT *, "Threads set to: ", iMaxThreads

n = 353851
! nnz = 1252105302
nnz = 1251924838
print *, "n: ", n
! print *, "total size: ", total_size
print *, "nnz: ", nnz


allocate(val(nnz))
allocate(col(nnz))
allocate(row(n + 1))
allocate(x(n))
allocate(y(n))

x = 1.0
y = 0.0


! Read from a file and store the values in the arrays
! Open the file
open(1, file='matrix_csr_test_final.txt', status='old', action='read')

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
    do i = 1, nnz
        read(1, *) line
        if (trim(line) == "# Column Indices:") then
            print *, "Found Columns"
            exit ! Exit the loop if the text is found
        else
            ! print *, line
            read(line, *) val(i)
            
        end if
    end do
end if
! print *, val

! Check if it finds the text "# Column Indices:"
read(1, '(A)', iostat=status) line
if (status /= 0) then
    print *, "Error reading file"
    stop
end if

if (trim(line) == "# Column Indices:") then
    print *, "Found Columns!"
    ! Code to execute if the text is found
    do i = 1, nnz
        read(1, *) line
        if (trim(line) == "# Row Pointers:") then
            print *, "Found Row Pointers"
            exit ! Exit the loop if the text is found
        else
            read(line, *) col(i)
        end if
    end do
end if
! print *, col

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
! print *, row

! Increase all values inside of row by one
row = row + 1

! New row values
! print *, "New row values:"
! print *, row


! Close the file
close(1)

print *, "Starting parallel computation..."

! Warmup Run
call spmv_parallel(n, nnz, val, row, col, x, y)

print *, "Warmup run done!"

! Find average per call time
count_max = 10
count_diff = 0
DO iter = 1, count_max
    CALL system_clock(count_rate=count_rate)
    CALL system_clock(start_count)
    CALL spmv_parallel(n, nnz, val, row, col, x, y)
    CALL system_clock(end_count)
    count_diff = count_diff + (end_count - start_count)
END DO

wall_time = REAL(count_diff) / REAL(count_rate) / REAL(count_max)
PRINT *, "Average time of ", count_max, " calls: ", wall_time, " seconds"



end program main
