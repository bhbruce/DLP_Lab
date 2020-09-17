import argparse
import os

import torch.nn as nn
import torch.nn.parallel
import torch.optim

import dataloader
from model import DeepConvNet
from model import EGGNet
import numpy as np
parser = argparse.ArgumentParser(description='BCI data Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='EEG',
                    choices={'EEG', 'DCN'},
                    help='model architecture: '
                         ' | EEG or DCN(DeepConvNet)'
                         ' (default: EEG)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--act', default=0, type=int,
                    help='model activation 0:relu 1:leaky relu 2:elu')
best_acc1 = 0


def main():
    global best_acc1
    args = parser.parse_args()
    args = vars(args)
    torch.manual_seed(0)
    np.random.seed(0)
    args['batch_size'] = 1080
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

    # create model
    print("=> creating model '{}'".format(args['arch']))
    if args['act'] == 1:
        act = 'LeakyReLU'
    elif args['act'] == 2:
        act = 'ELU'
    else:
        act = 'ReLU'

    if args['arch'] == 'EEG':
        model = EGGNet(act)
    else:
        model = DeepConvNet(act)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args['gpu'] is not None:
        criterion.cuda(args['gpu'])

    # optionally resume from a checkpoint
    if args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            if args['gpu'] is None:
                checkpoint = torch.load(args['resume'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args['gpu'])
                checkpoint = torch.load(args['resume'], map_location=loc)

            best_acc1 = checkpoint['best_acc1']
            best_epoch = checkpoint['epoch']
            print('Loading model accuracy:{:<3.2f} at epoch:{:3d}'.format(
                    best_acc1,
                    best_epoch))
            if args['gpu'] is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.cuda(args['gpu'])

            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    # Data loading
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    test_data = torch.from_numpy(test_data.astype(np.float32))
    test_label = torch.from_numpy(test_label.astype(np.long))

    if args['gpu'] is not None:

        torch.cuda.set_device(args['gpu'])
        model = model.cuda(args['gpu'])
        test_data = test_data.cuda(args['gpu'])
        test_label = test_label.cuda(args['gpu'])

    acc = validate(test_data, test_label, model, criterion, args)
    print(f'Evaluate accuracy: {acc:3.3f}')
    return


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
