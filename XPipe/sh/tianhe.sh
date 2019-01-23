#!/bin/sh
#
#SBATCH --job-name=test_slurm
#SBATCH --output=test_slurm.log
#
#SBATCH --ntasks=12
#SBATCH -N 6
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:04:00


path=$1

#srun -N1 -n2 --exclusive thpython sc_res_pipe.py -size 12 -path $1 -rank 0 -buffer_size 100 -layer_type 0 -basic True -out_plane 64 -num_block 2 -stride 1 -in_plane 1 -batch_size 32 -data_worker 1 -package_size 4 -send_num 12 &
#srun -N1 -n2 --exclusive thpython sc_res_pipe.py -size 12 -path $1 -rank 1 -buffer_size 100 -layer_type 1 -basic True -out_plane 64 -num_block 2 -stride 1 -in_plane 64 -batch_size 32 -data_worker 1 -package_size 4 -send_num 12 &
#srun -N1 -n2 --exclusive thpython sc_res_pipe.py -size 12 -path $1 -rank 2 -buffer_size 100 -layer_type 1 -basic True -out_plane 128 -num_block 2 -stride 2 -in_plane 64 -batch_size 32 -data_worker 1 -package_size 4 -send_num 12 &
#srun -N1 -n2 --exclusive thpython sc_res_pipe.py -size 12 -path $1 -rank 3 -buffer_size 100 -layer_type 1 -basic True -out_plane 256 -num_block 2 -stride 2 -in_plane 128 -batch_size 32 -data_worker 1 -package_size 4 -send_num 12 &
#srun -N1 -n2 --exclusive thpython sc_res_pipe.py -size 12 -path $1 -rank 4 -buffer_size 100 -layer_type 1 -basic True -out_plane 512 -num_block 2 -stride 2 -in_plane 256 -batch_size 32 -data_worker 1 -package_size 4 -send_num 12 &
#srun -N1 -n2 --exclusive thpython sc_res_pipe.py -size 12 -path $1 -rank 5 -buffer_size 100 -layer_type 2 -basic True -out_plane 64 -num_block 2 -stride 1 -in_plane 1 -batch_size 32 -data_worker 1 -package_size 4 -send_num 12 &
#wait

for i in {0..5}
do
    srun -N1 -n2 --exclusive run.sh $i $path &
done
wait
