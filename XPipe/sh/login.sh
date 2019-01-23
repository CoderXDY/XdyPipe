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
    echo "#!/bin/sh
    shopt -s expand_aliases
    source ../../.bashrc
    thpython test_slurm.py -size 12 -path $1 -rank $i > test_$i.log
    " > ${name[$i]}'_'run.sh
    chmod +x ${name[$i]}'_'run.sh
    ssh -q ${name[$i]} "nohup ./${name[$i]}_run.sh > ./test_$i.log 2>&1 &"
done