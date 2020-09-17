import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from dataset import RetinopathyDataset
from torch.optim.lr_scheduler import StepLR
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diy_resnet import DIY_ResNet18, DIY_ResNet50
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gamma', type=float, metavar='M', default=0.5,
                    help='Learning rate step gamma (default: 0.5)')
parser.add_argument('--step_size', type=int, metavar='M', default=30,
                    help='Learning step_size (default: 30)')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--diy', dest='diy', action='store_true',
                    help='use diy resnet? ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--patience', default=3, type=int,
                    help='Patience for early stop')

Retinopathy_class = ['No_DR', 'Mild', 'Moderate', 'Severe', 'ProlifeDR']
best_acc1 = 0
np.set_printoptions(precision=2)


def input_transform_train():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def input_transform_validate():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    global best_acc1

    args = parser.parse_args()

    # Setup the seed
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # To determine if your system supports CUDA
    print("==> Check devices..")
    if (args.gpu is not None) and torch.cuda.is_available():
        device = 'cuda'
        print("Current device: ", device)
        print(torch.cuda.device_count(), " GPUs is available")
        print("Our selected device: ", args.gpu)
    else:
        device = 'cpu'
        print("Current device: ", device)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.diy:
            print("use diy resnet model")
            if args.arch == 'resnet18':
                model = DIY_ResNet18()
            else:
                model = DIY_ResNet50()
        else:
            model = models.__dict__[args.arch]()

    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, 5)

    # model = DIY_ResNet50()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.gpu is not None:
        criterion.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model = nn.DataParallel(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset = RetinopathyDataset(args.data,
                                       input_transform=input_transform_train,
                                       mode='train')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_dataset = RetinopathyDataset(args.data,
                                     input_transform=input_transform_validate,
                                     mode='test')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        print('resume pth: ', args.resume)
        validate(val_loader, model, criterion, args, display=True)
        return
    else:
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not os.path.exists(log_base_dir):
            os.mkdir(log_base_dir)

        filename = str(args.arch)
        if args.pretrained:
            filename += '_pretrained'
        filename += '_batch_size' + str(args.batch_size) + \
                    '_total_epochs' + str(args.epochs) + \
                    '_lr' + str(args.lr) + '_gamma' + str(args.gamma) + \
                    '_momentum' + str(args.momentum) + \
                    '_weight_decay' + str(args.weight_decay) + \
                    '_step_size' + str(args.step_size)

        log_dir = os.path.join(log_base_dir, filename)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        save_best_ckpt_file = log_dir + '/best_ckpt.pth.tar'

        early_stop_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = 0
        writer = SummaryWriter(log_dir)
        train_acc_list = list()
        test_acc_list = list()

        for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            train_acc = train(train_loader, model, criterion,
                              optimizer, epoch, args)
            # evaluate on validation set
            test_acc = validate(val_loader, model, criterion, args)

            train_acc_list.append(train_acc.cpu().numpy())
            test_acc_list.append(test_acc.cpu().numpy())

            writer.add_scalar('Training Acc', train_acc, epoch)
            writer.add_scalar('Validation Acc', test_acc, epoch)
            writer.flush()

            if(test_acc < best_acc1):  # if this update cause acc1 drop
                early_stop_counter += 1
                # resume to best parameter
                model.load_state_dict(best_model_wts)
            else:  # remember best acc@1 and save checkpoint
                best_acc1 = test_acc
                best_epoch = epoch
                early_stop_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'epoch': best_epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_acc1': best_acc1,
                           }, save_best_ckpt_file)

            if early_stop_counter >= args.patience:
                break
            scheduler.step()

        model.load_state_dict(best_model_wts)
        print('best model testing:')
        validate(val_loader, model, criterion, args, display=True)

        import csv
        with open('output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(train_acc_list)
            writer.writerow(test_acc_list)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top3],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(images.shape)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top3.update(acc3[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg


def validate(val_loader, model, criterion, args, display=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                             [batch_time, losses, top1, top3],
                             prefix='Test: ')

    nb_classes = 5

    # switch to evaluate mode
    model.eval()
    confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=torch.int32)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t][p] += 1

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top3.update(acc3[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    print((confusion_matrix.numpy()))
    if display:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        confusion_matrix = confusion_matrix.numpy().astype(np.float32)
        row_sum = confusion_matrix.sum(axis=1)
        normalize = confusion_matrix / (row_sum * np.ones(confusion_matrix.shape)).transpose()
        disp = ConfusionMatrixDisplay(normalize,
                                      display_labels=Retinopathy_class)
        disp.plot()
        plt.savefig('confusion_matrix.png')
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
