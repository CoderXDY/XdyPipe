#!/bin/sh
srun -N1 -n1 -r0 -l run.sh 0 $1
srun -N1 -n1 -r1 -l run.sh 1 $1
srun -N1 -n1 -r2 -l run.sh 2 $1
srun -N1 -n1 -r3 -l run.sh 3 $1
srun -N1 -n1 -r4 -l run.sh 4 $1
srun -N1 -n1 -r5 -l run.sh 5 $1