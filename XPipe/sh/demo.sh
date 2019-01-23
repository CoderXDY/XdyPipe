#!/bin/bash
count=0
model=$1
protocol=$2
ps_num=$3
worker_num=$4
sleep_time=$5
second=3600
sleep_time=`expr $sleep_time \* $second`

nodelist=$(scontrol show hostname $SLURM_JOB_NODELIST)

for node in $nodelist;
do
        name[$count]=$node
        count=`expr $count + 1`
done

string_ps="${name[0]}:2222"
string_worker="${name[$ps_num]}:2222"

#build string_ps
for ((i=1;i<$ps_num;i++))
do
  stringtemp=",${name[$i]}:2222"
  string_ps=$string_ps$stringtemp
done

#build string_worker
for ((i=$ps_num+1;i<$[$ps_num+$worker_num];i++))
do
  stringtemp=",${name[$i]}:2222"
  string_worker=$string_worker$stringtemp
done

echo $string_ps
echo $string_worker

for ((i=0;i<$ps_num;i++))
do
  echo  "#!/bin/bash
  shopt -s expand_aliases
  source /WORK/app/modules/setmodule.sh
  module load tensorflow/1.0.1_cpu_py35
  setalias
  source activate tensorflow_2.0.1_cpu
  cd WORKSPACE/
  python $model'_'$protocol'_'CPU.py --ps_hosts=$string_ps --worker_hosts=$string_worker --job_name=ps --task_index=$i  > $model'_'$protocol'_'$3ps$4worker_ps$i.log" > $model'_'$protocol'_'run_$3ps$4worker_ps$i.sh
  chmod +x ./$model'_'$protocol'_'run_$3ps$4worker_ps$i.sh

  ssh -q ${name[$i]} 'nohup ./'"$model"'_'"$protocol"'_run_'"$ps_num"'ps'"$worker_num"'worker_ps'"$i"'.sh > '"$model"'_'"$protocol"'_output_'"$ps_num"'ps'"$worker_num"'worker_ps'"$i"'.txt 2>&1 &'
done
