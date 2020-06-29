#!/bin/sh -l
#SBATCH --job-name=calib
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mem=60000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dfrancom@lanl.gov

module load gcc
module load anaconda/Anaconda3.2019.10
module load R/3.5.1


echo start

free -h

R CMD BATCH calib_one.R

echo end
