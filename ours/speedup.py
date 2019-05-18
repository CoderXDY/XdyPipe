import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    normals = [97, (8*60+52), (3*60+2), (4*60+20), (9*60+13)]
    orign_gpipes = [87, (6*60+12), (2*60+12), (2*60+41), (7*60+5)]
    orign_fbps = [85,(4*60+40),(2*60+10),(2*60+35),(5*60+40)]


    normals = [val * 200 for val in normals]
    orign_gpipes = [val * 200 for val in orign_gpipes]
    orign_fbps = [val * 200 for val in orign_fbps]

    gpipes = [normals[0]/orign_gpipes[0], normals[1]/orign_gpipes[1], normals[2]/orign_gpipes[2], normals[3]/orign_gpipes[3], normals[4]/orign_gpipes[4]]
    fbps = [normals[0]/orign_fbps[0], normals[1]/orign_fbps[1], normals[2]/orign_fbps[2], normals[3]/orign_fbps[3], normals[4]/orign_fbps[4]]
    for i in range(5):
        print("gipes-" + str(i) + ": " + str(gpipes[i]))
        print("fbps-" + str(i) + ": " + str(fbps[i]))

    o = [1,1,1,1,1]
    models = ['VGGNet19', 'GoogleNet', 'ResNet18', 'ResNe34', 'ResNet50']

    x = list(range(len(o)))
    plt.ylabel('speedup over model-parallel in 3 workers')
    total_width, n = 0.8, 3
    width = total_width / n
    plt.bar(x, o, width=width, label='model-parallel', fc='g')
    for i in range(len(x)):
        x[i] = x[i] + width

    plt.bar(x, gpipes, width=width, label='GPipe', tick_label=models, fc='r')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, fbps, width=width, label='FB-P', fc='b')
    plt.legend()
    plt.show()
    
    """
    fbp = [1, 1.3, 1.6, 1.7]
    num = ['2-workers', '3-workers', '4-workers', '5-workers']
    x = list(range(len(fbp)))
    plt.ylabel('speedup over FB-P pipeline in 2 workers')
    plt.bar(x, fbp, fc='b', width=0.5, tick_label=num)
    plt.legend()
    plt.show()


