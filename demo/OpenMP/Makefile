
FC = gfortran
FFLAGS = -fbounds-check -fopenmp
LFLAGS = $(FFLAGS)

.PHONY: plots, clean, clobber

%.o : %.f90
	$(FC) $(FFLAGS) -c $<

heat_solve.exe: heat_solve.o jacobi2d_sub.o
	$(FC) $(LFLAGS) heat_solve.o jacobi2d_sub.o -o heat_solve.exe

solution.txt: heat_solve.exe
	@echo 
	@echo Running code...
	./heat_solve.exe

plots: solution.txt
	@echo 
	@echo Plotting results...
	python plot_solution.py

clean:
	rm -f *.o *.exe

clobber: clean
	rm -f solution.txt *.png

