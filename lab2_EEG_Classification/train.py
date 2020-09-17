import argparse
import os

import torch.nn as nn
import torch.nn.parallel
import torch.optim

import dataloader
from model import DeepConvNet
from model import EGGNet
import numpy as np
import copy
parser = argparse.ArgumentParser(description='BCI data Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='EEG',
                    choices={'EEG', 'DCN'},
                    help='model architecture: '
                         ' | EEG or DCN(DeepConvNet)'
                         ' (default: EEG)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
best_acc1 = 0


def main():
    global best_acc1
    args = parser.parse_args()
    args = vars(args)
    torch.manual_seed(0)
    np.random.seed(0)

    # To determine if your system supports CUDA
    print("==> Check devices..")
    if (args['gpu'] is not None) and torch.cuda.is_available():
        device = 'cuda'
        print("Current device: ", device)
        print(torch.cuda.device_count(), " GPUs is available")
        print("Our selected device: ", args['gpu'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'
        print("Current device: ", device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args['gpu'] is not None:
        criterion.cuda(args['gpu'])

    # Data loading
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()

    train_data = torch.from_numpy(train_data.astype(np.float32))
    train_label = torch.from_numpy(train_label.astype(np.long))

    test_data = torch.from_numpy(test_data.astype(np.float32))
    test_label = torch.from_numpy(test_label.astype(np.long))

    if args['gpu'] is not None:
        torch.cuda.set_device(args['gpu'])
        train_data = train_data.cuda(args['gpu'])
        train_label = train_label.cuda(args['gpu'])
        test_data = test_data.cuda(args['gpu'])
        test_label = test_label.cuda(args['gpu'])
    acc_list = list()
    # train three models with different activation functions
    for i in range(1, 4):
        train_list = list()
        test_list = list()
        best_acc1 = 0
        best_epoch = 0

        # create model
        print("=> creating model '{}'".format(args['arch']))
        if i == 1:
            act = 'LeakyReLU'
        elif i == 2:
            act = 'ELU'
        else:
            act = 'ReLU'

        if args['arch'] == 'EEG':
            model = EGGNet(act)
        else:
            model = DeepConvNet(act)
        if args['gpu'] is not None:
            model = model.cuda(args['gpu'])

        optimizer = torch.optim.Adam(model.parameters(), args['lr'],
                                     weight_decay=args['weight_decay'])
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not os.path.exists(log_base_dir):
            os.mkdir(log_base_dir)

        filename = str(args['arch']) + '_' + act
        filename += '_bs' + str(args['batch_size']) + '_epoch' + str(args['epochs']) + \
                    '_lr' + str(args['lr']) + '_wd' + str(args['weight_decay'])

        log_dir = os.path.join(log_base_dir, filename)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        save_best_ckpt_file = log_dir + '/' + filename + '_best_ckpt.pth.tar'
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(args['epochs']):
            # shuffle training data
            r = torch.randperm(train_label.shape[0])
            train_data = train_data[r]
            train_label = train_label[r]
            # train for one epoch
            train_acc, train_loss = train(train_data, train_label, model,
                                          criterion, optimizer, epoch, args)
            train_list.append(train_acc)
            # evaluate on validation set
            test_acc = validate(test_data, test_label, model, criterion, args)
            test_list.append(test_acc)
            print(f'[{epoch:4}] Train Acc {train_acc:3.3f}; '
                  f'Train Loss {train_loss:.3f}; Test Acc {test_acc:3.3f} '
                  f'Best acc: {best_acc1:3.3f}',
                  end='\r' if epoch != (args['epochs']-1) else '\n')
            if(test_acc >= best_acc1):
                best_acc1 = test_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)
        torch.save({'epoch': best_epoch + 1,
                    'arch': args['arch'],
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    }, save_best_ckpt_file)

        print(f'Best ckpt({act}) test accuracy: {best_acc1:3.3f}')
        acc_list.append(train_list)
        acc_list.append(test_list)

    from utils import plot_curve
    x = np.arange(args['epochs'])
    title_dict = ('leaky_relu_train', 'leaky_relu_test',
                  'elu_train', 'elu_test',
                  'relu_train', 'relu_test')
    plot_curve(x, np.array(acc_list).transpose(), title_dict, args['arch'])


def train(train_data, train_label, model, criterion, optimizer, epoch, args):

    model.train()  # switch to train mode
    correct = 0
    avg_loss = 0.0
    for batch in range(0, len(train_label), args['batch_size']):
        # measure data loading time
        end = min(batch+args['batch_size'], len(train_label))

        images = train_data[batch:end]
        target = train_label[batch:end]
        # compute output
        output = model(images)
        loss = criterion(output, target)
        pred = torch.argmax(output, dim=1)
        correct += torch.sum(pred == target)
        avg_loss += loss * images.shape[0]

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = correct.type(torch.float) * 100 / len(train_label)
    avg_loss = avg_loss / len(train_label)
    return acc, avg_loss


def validate(test_data, test_label, model, criterion, args):
    # switch to evaluate mode
    model.eval()
    correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for batch in range(0, len(test_label), args['batch_size']):
            end = min(batch+args['batch_size'], len(test_label))
            images = test_data[batch:end]
            target = test_label[batch:end]

            output = model(images)
            loss = criterion(output, target)

            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == target)
            avg_loss += loss * images.shape[0]

    acc = correct.type(torch.float) * 100 / len(test_label)
    avg_loss = avg_loss / len(test_label)
    return acc


if __name__ == '__main__':
    main()
