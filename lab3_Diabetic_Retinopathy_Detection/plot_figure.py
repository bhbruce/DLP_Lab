import csv
import numpy as np
import matplotlib.pyplot as plt
target_list = list()
with open('output.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        target_list.append(row)

resnet50 = np.array(target_list[0:4])
resnet50 = resnet50.astype(np.float32)
resnet18 = np.array(target_list[4:8])
resnet18 = resnet18.astype(np.float32)
title_dict = ('Train(w/ pretrianed)',
              'Test(w/ pretrianed)',
              'Train(w/o pretrianed)',
              'Test(w/o pretrianed)')


def plot_curve(x, data, title_dict, network='Resnet50'):
    figure, ax = plt.subplots()
    plots = ax.plot(x, data, label='')
    figure.set_size_inches(8, 4)
    ax.legend(plots, title_dict, loc='best',
    framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})
    ax.set_title(f'Comparision({network}) w/ & w/o pretrained')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    figure.tight_layout()
    # plt.show()
    plt.savefig(f'{network}_curve.png')


x1 = np.arange(6)
x2 = np.arange(10)
plot_curve(x1, resnet50.transpose(), title_dict, 'ResNet50')
plot_curve(x2, resnet18.transpose(), title_dict, 'ResNet18')
