#!/bin/sh -l
#SBATCH --job-name=py_calib
#SBATCH --time=04:00:00
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dfrancom@lanl.gov

module load anaconda/Anaconda3

source activate imm_env

python setup.py build_ext --inplace

hostname

conda env list

free -h

grep -c ^processor /proc/cpuinfo

echo start

python ~/git/immpala/calibration/py_calibration_hier/test_pt_ti64.py

echo end
