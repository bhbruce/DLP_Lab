import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, data, title_dict, network='EGG'):
    figure, ax = plt.subplots()
    plots = ax.plot(x, data, label='')
    figure.set_size_inches(8, 4)
    ax.legend(plots, title_dict, loc='best',
              framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})
    ax.set_title(f'Activation Function Comparision({network})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    figure.tight_layout()
    # plt.show()
    plt.savefig(f'{network}_curve.png')


"""
a = np.arange(300)
x = np.reshape(np.arange(900), (3, 300))
x = x.transpose()
print(x.shape)
title_dict = ('relu_train', 'relu_test', 'leaky_relu_train')

plot_curve(a, x, title_dict)
"""
