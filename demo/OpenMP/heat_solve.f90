program heat_solve
    implicit none
    real(8) :: tolerance, dumax
    integer :: grid_size, max_steps, actual_steps, threads

    ! TIP: Increase grid_size to make the problem more computationally 
    ! expensive, then decrease max_steps to "force" the code to finish quickly.
    ! 'tolerance' ensures accuracy of linear solve using the iterative method.

    grid_size = 150
    tolerance = 1.d-7
    
    ! CONTROL LOOP: Set a low max_steps for a quick run, but that may result in inaccurate solve
    max_steps = 500000 

    print "(A, I4, A, I4)", "Configuring Simulation: ", grid_size, " x", grid_size
    print "(A, I6)", "Hard-cap on iterations: ", max_steps
    print *, "------------------------------------------------"

    ! Compare Serial vs Parallel within a controlled iteration count
    threads = 1
    call jacobi2d_sub(grid_size, tolerance, max_steps, actual_steps, dumax, threads)

    threads = 4
    call jacobi2d_sub(grid_size, tolerance, max_steps, actual_steps, dumax, threads)

end program heat_solve