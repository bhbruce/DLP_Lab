from parser import argparser
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

args = argparser()
args = vars(args)
title_dict = ('Train-ACC', 'Test-ACC')
title_dict1 = ('Train-G_loss', 'Train-D_loss')


def plot_curve(x, data, title_dict, title='', ylabel='Accuracy'):
    figure, ax = plt.subplots()
    plots = ax.plot(x, data, label='')
    figure.set_size_inches(8, 4)
    ax.legend(plots, title_dict, loc='best',
    framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    figure.tight_layout()
    # plt.show()
    plt.savefig(f'{title}_curve.png')


if os.path.isfile(args['resume']):
    checkpoint = torch.load(args['resume'])
    trian_acc_list = checkpoint['TrainAcc_list']
    test_acc_list = checkpoint['TestAcc_list']

    gloss_list = checkpoint['Gloss_list']
    dloss_list = checkpoint['Dloss_list']
    print(checkpoint['epoch'])
    print(checkpoint['best_acc'])

    trian_acc_list = np.array(trian_acc_list, dtype=np.float)
    trian_acc_list = trian_acc_list.reshape(1, -1)
    test_acc_list = np.array(test_acc_list, dtype=np.float)
    test_acc_list = test_acc_list.reshape(1, -1)

    gloss_list = np.array(gloss_list, dtype=np.float)
    gloss_list = gloss_list.reshape(1, -1)

    dloss_list = np.array(dloss_list, dtype=np.float)
    dloss_list = dloss_list.reshape(1, -1)

    x = np.arange(checkpoint['epoch'])
    print(trian_acc_list.shape)
    print(trian_acc_list.shape)
    y = np.concatenate((trian_acc_list, test_acc_list), 0)
    y = y.transpose(1, 0)
    y1 = np.concatenate((gloss_list, dloss_list))
    y1 = y1.transpose(1, 0)
    print(y.shape)
    plot_curve(x, y, title_dict, 'Accuracy')
    plot_curve(x, y1, title_dict1, 'Loss', 'Loss')
