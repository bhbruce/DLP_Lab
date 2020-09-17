from argparse import ArgumentParser


def argparser():
    parser = ArgumentParser(description='PyTorch Seq2Seq-CVAE')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-kl', '--KL_annealing', metavar='KL_annealing',
                        default='cycle',
                        choices={'cycle', 'mono'},
                        help='KL_annealing mode: '
                             ' | cycle or mono(Monotonic) (default: cycle)')
    parser.add_argument('-dp', '--dropout', default=0, type=float, metavar='N',
                        help='dropout')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=874, type=int, metavar='N',
                        help='seed default:874')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    print(args)
