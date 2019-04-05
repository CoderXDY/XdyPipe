tianhe
best_acc: 92.92
start_epoch: 189
each_epoch:22s

92.54
138
each_epoch:32s


### Forward-Backward Parllelism Experiments ###
| method | dataset | model | device | lr | batch_size | data_worker | buffer_size | epoch | acc | eachtime |  
| :------: | :------: | :------: |  :------: | :------: | :------: | :-----: | :------: | :------: | :------: | :------: |
| layer_dist | cifar10 | vgg16 | 2080ti*2 | 0.01 | 64 | 0 | 2 | 79 | 90.79 | 21s |  
| pipedream | cifar10 | vgg16 | 2080ti*2 | 0.01 | 64 | 0 | 2 | 75 | 90.4 | 21s |  
| ours | cifar10 | vgg16 | 2080ti*2 | 0.01 | 64 | 2 | 8/4 | 0 | ? | ? | ? |      
| 

### Pipeline Parllelism Experiments ###

| method | dataset | model | device | lr | batch_size | buffer_size | epoch | acc | eachtime |  
| ------ | ------ | ------ |  ------ | ------ | ------ | :------: | :------: | :------: | :------: |
| layer_dist | cifar10 | Res18 | 2080*2 | 0.01 | 64 | / | ? | ? | 2m36s |  
| pipeline | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | ? |  
| ours | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | 2m26s |  
| ours + grad_clip | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | 1m15s | 
| layer_dist | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | / | ? | ? | ? |  
| pipeline | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |  
| ours | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |      
| ours + grad_compress | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |

### Grad Compress Experiments ###
> buffer_size, therhold, data_worker is important for speed, may buffersize 4 and queue of thread set 8



| method | dataset | model | device | lr | batch_size | buffer_size | epoch | acc | eachtime |  
| ------ | ------ | ------ |  ------ | ------ | ------ | :------: | :------: | :------: | :------: |
| layer_dist | cifar10 | Res18 | 2080*2 | 0.01 | 64 | / | ? | ? | 2m36s |  
| pipeline | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | ? |  
| ours | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | 2m26s |  
| ours + grad_clip | cifar10 | Res18 | 2080*2 | 0.01 | 64 | ? | ? | ? | 1m15s | 
| layer_dist | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | / | ? | ? | ? |  
| pipeline | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |  
| ours | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |      
| ours + grad_compress | cifar10 | Res18 | tianhe*2 | 0.01 | 64 | ? | ? | ? | ? |


### Grad Quantize ###


| method | dataset | model | device | lr | batch_size | buffer_size | bit | epoch | acc | eachtime |  
| ------ | ------ | ------ |  ------ | ------ | ------ | :------: | :-----: | :------: | :------: | :------: |
| ours | cifar10 | Res18 | 2080*2 | 0.01 | 64 | 8/4 | 8 |183 | 91.38 | 12s |  
| ours quantied | cifar10 | vgg16 | 2080*2 | 0.01 | 64 | 8/4 | 8 |171 | 91.36 | 7s | 