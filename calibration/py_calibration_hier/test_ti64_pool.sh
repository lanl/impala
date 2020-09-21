#!/bin/bash -l
#SBATCH --job-name=impala_ti64
#SBATCH --output=res_impala_clust_ti64.txt
#SBATCH --partition=amd-rome,amd-rome-gpu,amd-epyc,amd-epyc-gpu
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptrubey@lanl.gov

cd ~/impala/calibration/py_calibration_hier
echo start
conda init bash
conda activate impala
module load openmpi
mpiexec -np 16 python -m mpi4py test_ti64_pool.py
echo end
