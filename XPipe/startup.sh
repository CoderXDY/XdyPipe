#!/bin/sh
python sc_res_pipe.py -size 12 -path /home/xiao/project/XPipe/share -rank 0 -buffer_size 100 -layer_type 0 -basic True -out_plane 64 -num_block 2 -stride 1 -in_plance 1 batch_size 32 -data_worker 1 -package_size 4 -send_num 12
python sc_res_pipe.py -size 12 -path /home/xiao/project/XPipe/share -rank 1 -buffer_size 100 -layer_type 1 -basic True -out_plane 64 -num_block 2 -stride 1 -in_plance 64 batch_size 32 -data_worker 1 -package_size 4 -send_num 12
python sc_res_pipe.py -size 12 -path /home/xiao/project/XPipe/share -rank 2 -buffer_size 100 -layer_type 1 -basic True -out_plane 128 -num_block 2 -stride 2 -in_plance 64 batch_size 32 -data_worker 1 -package_size 4 -send_num 12
python sc_res_pipe.py -size 12 -path /home/xiao/project/XPipe/share -rank 3 -buffer_size 100 -layer_type 1 -basic True -out_plane 256 -num_block 2 -stride 2 -in_plance 128 batch_size 32 -data_worker 1 -package_size 4 -send_num 12
python sc_res_pipe.py -size 12 -path /home/xiao/project/XPipe/share -rank 4 -buffer_size 100 -layer_type 1 -basic True -out_plane 512 -num_block 2 -stride 2 -in_plance 256 batch_size 32 -data_worker 1 -package_size 4 -send_num 12
python sc_res_pipe.py -size 12 -path /home/xiao/project/XPipe/share -rank 5 -buffer_size 100 -layer_type 2 -basic True -out_plane 64 -num_block 2 -stride 1 -in_plance 1 batch_size 32 -data_worker 1 -package_size 4 -send_num 12

