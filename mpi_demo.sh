#! /bin/bash
#yhrun -n 2 -N 4 thpython mpi_demo.py -size 2 -path file:///HOME/sysu_wgwu_4/share_file
#echo $GLIBC_DIR
#yhrun -n 1 -N 1 python test.py
#nodelist=$(scontrol show hostname $SLURM_JOB_NODELIST)
#echo $nodelist


#source /WORK/app/toolshs/cnmodule.sh
#module add anaconda3/4.2.0
#source activate ato
#GLIBC_DIR=/HOME/sysu_wgwu_4/glibc2.17/lib
#alias thpython="$GLIBC_DIR/ld-2.17.so --library-path $GLIBC_DIR:/lib64:$LD_LIBRARY_PATH `which python`"
#thpython mpi_demo.py -size 2 -path file:///HOME/sysu_wgwu_4/share_file
shopt expand_aliases
shopt -s expand_aliases
shopt expand_aliases
alias thpython="$GLIBC_DIR/ld-2.17.so --library-path $GLIBC_DIR:/lib64:$LD_LIBRARY_PATH `which python`"
#thpython test.py
#thpython mpi_demo.py -size 2 -path file:///HOME/sysu_wgwu_4/share_file
#thpython split_model.py -size 5 -path file:///HOME/sysu_wgwu_4/share_file
thpython pipeline.py -size 5 -path file:///HOME/sysu_wgwu_4/share_file
