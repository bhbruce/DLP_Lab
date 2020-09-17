import numpy as np
import argparse
from utils import *
from loss import MSE
from loss import BCE
parser = argparse.ArgumentParser(description='Numpy DNN Training')

parser.add_argument('--loss', default='BCE', type=str, metavar='BCE',
                    help='choose loss BCE/MSE')
parser.add_argument('--epochs', default=int(1e6), type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--seed', default=5, type=int, metavar='X',
                    help='seed for initializing training. ')
parser.add_argument('--data', default='xor', type=str, metavar='xor/linear',
                    help='choose xor/linear data ')
parser.add_argument('--target_loss', default=1e-3, type=float,
                    metavar='target loss',
                    help='loss to stop training')

args = parser.parse_args()
# print(type(vars(args)))


class Linear:
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False):

        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.normal(size=(in_features,
                                             out_features)).astype(np.float32)
        if bias:
            self.bias = np.array(out_features).astype(np.float32)
        else:
            self.bias = None

        self.grad_weight = None
        self.last_weight_vel = 0  # last velocity of weight
        self.grad_bias = None
        self.last_bias_vel = 0  # last velocity of bias

    def forward(self, input) -> np.array:
        self.input = input
        if self.bias is not None:
            return input @ self.weight + self.bias
        else:
            return input @ self.weight

    def backward(self, output_grad) -> np.array:
        self.grad_weight = self.input.transpose() @ output_grad
        if self.bias is not None:
            self.grad_bias = output_grad.mean(axis=0)
        else:
            self.grad_bias = None

        return output_grad @ self.weight.transpose()

    def update(self, lr, momentum=0.9):
        self.last_weight_vel = momentum * self.last_weight_vel\
                               + lr * self.grad_weight
        self.weight = self.weight - self.last_weight_vel

        if self.bias is not None:
            self.last_bias_vel = momentum * self.last_bias_vel\
                                 + lr * self.grad_bias
            self.bias = self.bias - self.last_bias_vel


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, input) -> np.array:
        self.out = 1 / (1+np.exp(-input))
        return self.out

    def backward(self, grad) -> np.array:
        grad_sigmoid = self.out * (1 - self.out)
        return grad * grad_sigmoid


class MLP:
    def __init__(self):
        self.fc1 = Linear(2, 8)
        self.act1 = Sigmoid()
        self.fc2 = Linear(8, 16)
        self.act2 = Sigmoid()
        self.fc3 = Linear(16, 1)
        self.act3 = Sigmoid()

    def forward(self, input) -> np.array:
        out = self.fc1.forward(input)
        out = self.act1.forward(out)
        out = self.fc2.forward(out)
        out = self.act2.forward(out)
        out = self.fc3.forward(out)
        out = self.act3.forward(out)
        return out

    def backward(self, grad: np.array) -> np.array:
        bw_grad = grad
        bw_grad = self.act3.backward(bw_grad)
        bw_grad = self.fc3.backward(bw_grad)
        bw_grad = self.act2.backward(bw_grad)
        bw_grad = self.fc2.backward(bw_grad)
        bw_grad = self.act1.backward(bw_grad)
        bw_grad = self.fc1.backward(bw_grad)
        return bw_grad

    def update(self, lr=0.001, momentum=0):
        self.fc1.update(lr, momentum)
        self.fc2.update(lr, momentum)
        self.fc3.update(lr, momentum)


def main():
    np.random.seed(args.seed)
    epochs = args.epochs
    lr = args.lr
    loss_list = []
    acc_list = []
    # choose loss function
    if args.loss == 'BCE':
        criterion = BCE
    else:
        criterion = MSE

    model = MLP()

    # prepare data
    if args.data == 'xor':
        x, y = generate_XOR_easy()
    else:
        x, y = generate_linear(n=100)

    for i in range(epochs):
        # forward path
        y_pred = model.forward(x)
        # calculate loss and error
        loss, error = criterion(y_pred, y)
        # calculate accuracy
        pred = (y_pred > 0.5)
        correct = np.sum(np.equal(pred, y))
        acc = correct * 100 / y.shape[0]
        loss_list.append(loss)
        acc_list.append(acc)

        if (i+1) % 500 == 0:  # print frequency
            print('Epoch:{:<7d} acc:{:3.2f}% loss:{:2.4f}'.format(i + 1, acc, loss))
        if loss < args.target_loss:
            print('==========================================')
            print('Achieve target loss {:.4f} in epoch {:<7d} acc:{:3.2f}% loss:{:2.4f}'.format(args.target_loss, i + 1, acc, loss))
            print('Setting: ', vars(args))
            idx = 0
            for i, j in zip(y_pred,y):
                idx = idx + 1
                print('data {:<7d} pred:{:1.4f} label:{}'.format(idx, float(i),int(j)))
            break

        # backward path
        model.backward(error)
        # update parameter: if momentum=0, SGD
        model.update(lr, args.momentum)

    show_result(x, y, pred, args.data)
    show_curve(loss_list, args.data)


if __name__ == '__main__':
    main()
