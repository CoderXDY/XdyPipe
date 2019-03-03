import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = F.relu(self.layer(x))
        return x




def train(rank, model):
    dist.init_process_group(backend='tcp', init_method="file:////home/xdy/projects/XdyPipe/th_thread/temp", world_size=2, rank=rank)
    a = torch.randn([10]).requires_grad_()
    b = model(a)

    print('rank:' + str(rank))


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = Layer(10, 2).to(torch.device('cuda:0'))
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(2):
        p = mp.Process(target=train, args=(rank, model))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


