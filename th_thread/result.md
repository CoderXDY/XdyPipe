tianhe
best_acc: 92.92
start_epoch: 189
each_epoch:22s

92.54
138
each_epoch:32s


### Forward-Backward Parllelism Experiments ###

### Pipeline Parllelism Experiments ###

| method | dataset | model | device | lr | batch_size | buffer_size | epoch | acc | eachtime |  
| ------ | ------ | ------ |  ------ | ------ | ------ | :------: | :------: | :------: | :------: |
| layer_dist | cifar10 | Res18 | 2080*2 | 0.01 | 64 | / | ? | ? | ? |  
| pipeline | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | ? |  
| ours | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | ? |  
| layer_dist | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | / | ? | ? | ? |  
| pipeline | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |  
| ours | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |      
| ours + grad_compress | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |

### Grad Compress Experiments ###
> buffer_size, therhold is important for speed, may buffersize 4 and queue of thread set 8
