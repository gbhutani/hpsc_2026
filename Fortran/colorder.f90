program colorder
	implicit none
	integer, parameter :: n=10000
	real(kind=8) :: arr(n,n)

	integer :: i,j

	do j=1,n
		do i=1,n
			arr(i,j) = arr(i,j) + 1
		enddo
	enddo
end program colorder