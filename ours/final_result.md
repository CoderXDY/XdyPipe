# Convergence #
## 2080ti * 2 ##
* res18
* vgg16
## tianhe * 3 ##
* vgg 19
* googlenet
* res50
* tbc

# Precision #
| method | dataset | model | device | lr | batch_size | buffer_size | epoch | acc | eachtime |  
| ------ | ------ | ------ |  ------ | ------ | ------ | :------: | :------: | :------: | :------: |
| normal | cifar10 | VggNet19 | tian*3 | 0.01 | 64 | 2 | 124 | 91.01 | ~2min |  
| normal | cifar10 | ResNet101 | tianhe*3 | 0.01 | 64 | 2 | ? | ? | ? |  
| normal | cifar10 | GoogleNet | tianhe*3 | 0.01 | 64 | 2 | ? | ? | ~8min |  
| normal | cifar10 | DPN92 | tianhe*3 | 0.01 | 64 | 2 | ? | ? | ? | ? |
| fbp3 | cifar10 | VggNet19 | tianhe*3 | 0.01 | 64 | 5 | 152 | 90.6 | ~40s(no cal compress time only compress) |  