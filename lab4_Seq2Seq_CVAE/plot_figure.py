from parser import argparser
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

args = argparser()
args = vars(args)
title_dict = ('Train-BLEU4', 'Test-BLEU4', 'Test-Gaussian')
title_dict1 = ('Train-CE_loss', 'Train-KLD_loss')


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
    CE_list = checkpoint['CE_list']
    KLD_list = checkpoint['KLD_list']
    trian_bleu4_list = checkpoint['trian_bleu4_list']
    test_bleu4_list = checkpoint['test_bleu4_list']
    test_cond_list = checkpoint['test_cond_list']
    print(checkpoint['epoch'])
    print(checkpoint['best_bleu4_acc'])
    print(checkpoint['best_cond_acc'])

    trian_bleu4_list = np.array(trian_bleu4_list, dtype=np.float)
    trian_bleu4_list = trian_bleu4_list.reshape(1, -1)
    test_bleu4_list = np.array(test_bleu4_list, dtype=np.float)
    test_bleu4_list = test_bleu4_list.reshape(1, -1)
    test_cond_list = np.array(test_cond_list, dtype=np.float)
    test_cond_list = test_cond_list.reshape(1, -1)

    KLD_list = np.array(KLD_list, dtype=np.float)
    KLD_list = KLD_list.reshape(1, -1)

    CE_list = np.array(CE_list, dtype=np.float)
    CE_list = CE_list.reshape(1, -1)

    x = np.arange(checkpoint['epoch'])
    print(trian_bleu4_list.shape)
    print(trian_bleu4_list.shape)
    y = np.concatenate((trian_bleu4_list, test_bleu4_list,
                        test_cond_list), 0)
    y = y.transpose(1, 0)
    y1 = np.concatenate((CE_list, KLD_list))
    y1 = y1.transpose(1, 0)
    print(y.shape)
    plot_curve(x, y, title_dict, 'Accuracy')
    plot_curve(x, y1, title_dict1, 'Loss', 'Loss')
