#!/bin/bash -l
#SBATCH --job-name=impala_ti64
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dfrancom@lanl.gov


echo "hostname:"
hostname
printf "\n"

echo "RAM:"
free -h
printf "\n"

echo "ncores:"
grep -c ^processor /proc/cpuinfo
printf "\n"


echo start
module load anaconda/Anaconda3.2019.10
source activate impala_general
module load openmpi/2.1.3-gcc_6.4.0
mpiexec -np 16 python -m mpi4py run_ti64.py
echo end
