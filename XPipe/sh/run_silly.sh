#!/bin/sh
#
#SBATCH --job-name=test_slurm
#SBATCH --output=test_slurm.log
#

path=$1

srun -N6 -n6 -l --multi-prog silly.conf path