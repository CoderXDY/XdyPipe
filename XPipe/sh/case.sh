#!/bin/sh
case $SLURM_NODEID in
    0)  sh run.sh 0 $1
    ;;
    1)  sh run.sh 1 $1
    ;;
    2)  sh run.sh 2 $1
    ;;
    3)  sh run.sh 3 $1
    ;;
    4)  sh run.sh 4 $1
    ;;
    5)  sh run.sh 5 $1
    ;;
esac