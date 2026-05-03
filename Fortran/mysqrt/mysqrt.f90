program mysqrt
	implicit none
	real(kind=8) :: x, my_sqrt, sqrt_fortran, error
        real(kind=8), external :: sqrt2
	! main code to call sqrt2(x)
	x = 2.d0
        my_sqrt = sqrt2(x)
	sqrt_fortran = sqrt(x)
	error = abs(my_sqrt - sqrt_fortran)
        print *, "x = ", x
        print *, "mysqrt = ", my_sqrt
        print *, "sqrt = ", sqrt_fortran
        print *, "error = ", error
end program mysqrt

function sqrt2(x)
        implicit none
	! intent first
	real(kind=8), intent(in) :: x
	real(kind=8) :: sqrt2

	! local variables
	real(kind=8) :: s, s0, delta_s, tol
	integer :: k, kmax

	! initialise variables
	kmax = 100
	s = 1.d0
	tol = 1.d-14

	do k=1,kmax
		s0 = s
		s = 0.5 * (s + x/s)
		delta_s = abs(s - s0)
		if (delta_s < tol) then
                        ! print *, "stopping at ", k
                        ! print *, "s = ",s
			exit
		endif
	end do
        sqrt2 = s
end function sqrt2

! Exercise: write a subroutine sqrt2_sub that returns
! two arguments - sqrt and the number of iterations taken to stop.
! The variables kmax, s_init and tol should also be passed
! to the subroutine
! Include debug flag too just like in the Python code
