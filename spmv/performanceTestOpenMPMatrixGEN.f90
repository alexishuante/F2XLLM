subroutine read_and_print_csr_matrix()
  implicit none
  integer :: io_status, col, n, i, count
  real :: value
  character(len=200) :: line
  integer, allocatable :: indptr(:), cols(:)
  real, allocatable :: values(:)
  integer :: unit_number
  integer :: j  ! Declare loop variable


  ! Open the file
  open(newunit=unit_number, file='csr_matrix.csv', status='old', action='read', iostat=io_status)
  if (io_status /= 0) then
      print *, 'Error opening file.'
      return
  endif

  ! First pass: count the number of value-column pairs
  count = 0
  do
      read(unit_number, '(A)', iostat=io_status) line
      if (io_status /= 0) exit  ! Exit loop if end of file or error
      if (line(1:6) == 'indptr') exit  ! Stop counting at 'indptr'
      count = count + 1
  end do

  ! Allocate arrays based on count
  allocate(values(count))
  allocate(cols(count))

  ! Rewind to read the values and columns again
  rewind(unit_number)

  ! Second pass: read values and columns into arrays
  i = 0
  do while (i < count)
      i = i + 1
      read(unit_number, '(A)', iostat=io_status) line
      if (line(1:6) == 'indptr') exit  ! Stop reading at 'indptr'
      read(line, *, iostat=io_status) values(i), cols(i)
  end do

  ! Close the file
  close(unit_number)

  ! Print the stored values and column indices
  do i = 1, count
      print *, 'Value:', values(i), 'Column:', cols(i)
  end do

  ! Open the file again to read indptr values
  open(newunit=unit_number, file='csr_matrix.csv', status='old', action='read', iostat=io_status)
  if (io_status /= 0) then
      print *, 'Error opening file.'
      return
  endif

  ! Skip lines until indptr is found
  do
      read(unit_number, '(A)', iostat=io_status) line
      if (io_status /= 0) exit  ! Exit loop if end of file or error
      if (line(1:6) == 'indptr') exit  ! Found 'indptr'
  end do

  ! Process the indptr line to count the number of elements
  count = 0
  do i = 7, len_trim(line)  ! Start from character 7 to skip 'indptr,'
      if (line(i:i) == ',' .or. i == len_trim(line)) then
          count = count + 1
      endif
  end do

  ! Allocate the indptr array
  allocate(indptr(count))
  indptr = 0  ! Initialize all elements to zero

  ! Debug print
  print *, "Reading indptr from: ", line(7:)

  ! Read the indptr values into the array
  read(line(7:), *) (indptr(j), j = 1, count)

  ! Print the indptr array
  print *, 'indptr:', indptr

  ! Close the file
  close(unit_number)
end subroutine read_and_print_csr_matrix

program main
  implicit none

  ! Call the subroutine to read and print the CSR matrix
  call read_and_print_csr_matrix

