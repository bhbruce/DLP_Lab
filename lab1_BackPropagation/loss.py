import numpy as np

"""
Mean Square Error
"""
def MSE(pred, label):
    N = pred.shape[0]  # number of data
    C = pred.shape[1]  # number of class
    loss = np.sum(np.square(pred - label)) / N / C
    error = 2 * (pred - label) / N / C
    return loss, error

"""
Binary cross entropy loss
"""
def BCE(pred, label, epsilon=1e-8):
    N = pred.shape[0]
    tmp1 = 1 - label
    tmp2 = 1 - pred + epsilon
    loss = label * np.log(pred+epsilon) + tmp1 * np.log(tmp2)
    loss = -np.sum(loss) / N
    error = (-label / (pred + epsilon) + (tmp1/tmp2)) / N
    return loss, error
