#!/bin/bash -l
#SBATCH --job-name=impala_ti64
#SBATCH --output=res_impala_clust_ti64.txt
#SBATCH --partition=amd-rome,amd-rome-gpu,amd-epyc,amd-epyc-gpu
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dfrancom@lanl.gov

echo start
module load anaconda/Anaconda3.2019.10
source activate impala
module load openmpi
mpiexec -np 16 python -m mpi4py run_ti64.py
echo end
