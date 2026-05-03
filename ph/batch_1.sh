#!/bin/sh
#SBATCH -N 1 
#SBATCH --ntasks-per-node=4 
#SBATCH --time=00:00:05
#SBATCH --job-name=omp # specifies job name
#SBATCH --error=job.%J.err_node_4 # specifies error file name
#SBATCH --output=job.%J.out_node_4 # specifies output file name
#SBATCH --partition=standard # specifies queue name
cd $SLURM_SUBMIT_DIR
# export I_MPI_FABRICS=shm:dapl # For Intel MPI versions 2019 onwards this value must be shm:ofi
# mpiexec.hydra -n $SLURM_NTASKS lammps.exe
./a.out
