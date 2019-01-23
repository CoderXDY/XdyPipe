#!/bin/bash
count=0
nodelist=$(scontrol show hostname $SLURM_JOB_NODELIST)
for node in $nodelist;
do
    name[$count]=$node
    count=`expr $count + 1`
done
for ((i=0;i<$count;i++))
do

    ssh -q ${name[$i]} "nohup /HOME/sysu_wgwu_4/xpipe/XPipe/tianhe.sh $i $1> ./sc_res_pipe_$i.log 2>&1 &"
done
wait
