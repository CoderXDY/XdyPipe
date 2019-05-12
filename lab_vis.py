import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', help='the filename of log')
    parser.add_argument('-count', type=int)
    parser.add_argument('-key')
    parser.add_argument('-prop', type=int)
    args = parser.parse_args()
    path_list = args.file.split()
    path_result = {}
    for path in path_list:
        points = []
        count = 0
        index = 0
        with open(path, 'r') as f:
            for line in f:
                if count == args.count:
                    break
                strs = line.strip().split(':')
                if strs[0] == args.key:
                    if index % args.prop == 0:
                        points.append(float(strs[1]))
                        count += 1
                        index += 1
                    else:
                        index += 1
                else:
                    pass
            path_result[path] = points

    x = np.arange(args.count)
    colors = ['red', 'blue', 'yellow']
    index = 0
    for name, points in path_result.items():
        plt.plot(x, np.array(points), color=colors[index], label=name)
        index += 1
    plt.title('compare')
    plt.xlabel('epochs')
    plt.ylabel('loss or acc')
    plt.legend()
    plt.show()
