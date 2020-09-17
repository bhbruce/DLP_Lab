import torch
import torch.nn as nn


def activation(act='ReLU'):
    if act == 'ReLU':
        return nn.ReLU()
    elif act == 'LeakyReLU':
        return nn.LeakyReLU()
    else:
        return nn.ELU()


class EGGNet(nn.Module):
    def __init__(self, act='ReLU'):
        super(EGGNet, self).__init__()
        self.firstconv = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=(1, 51),
                                  padding=(0, 25), bias=False),
                        nn.BatchNorm2d(16)
                        )
        self.depthwiseConv = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=(2, 1),
                                  groups=16, bias=False),
                        nn.BatchNorm2d(32),
                        activation(act),
                        nn.AvgPool2d((1, 4), stride=(1, 4)),
                        nn.Dropout(p=0.25)
                        )
        self.separableConv = nn.Sequential(
                        nn.Conv2d(32, 32, kernel_size=(1, 15),
                                  padding=(0, 7), bias=False),
                        nn.BatchNorm2d(32),
                        activation(act),
                        nn.AvgPool2d((1, 8), stride=(1, 8)),
                        nn.Dropout(p=0.25),
                        nn.Flatten()
                        )
        self.classify = nn.Linear(736, 2, bias=True)

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = self.classify(out)
        return out


def basic_block(it_channels=1, out_channels=1, act='ReLU'):
    return nn.Sequential(nn.Conv2d(it_channels, out_channels,
                                   kernel_size=(1, 5), bias=True),
                         nn.BatchNorm2d(out_channels),
                         activation(act),
                         nn.MaxPool2d((1, 2)),
                         nn.Dropout(p=0.5))


class DeepConvNet(nn.Module):
    def __init__(self, act='ReLU'):
        super(DeepConvNet, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 25, kernel_size=(1, 5),
                                              bias=False),
                                    nn.Conv2d(25, 25, kernel_size=(2, 1),
                                              bias=True),
                                    nn.BatchNorm2d(25),
                                    activation(act),
                                    nn.MaxPool2d((1, 2)),
                                    nn.Dropout(p=0.5))
        self.block2 = basic_block(it_channels=25, out_channels=50, act=act)
        self.block3 = basic_block(it_channels=50, out_channels=100, act=act)
        self.block4 = basic_block(it_channels=100, out_channels=200, act=act)
        self.classify = nn.Sequential(nn.Flatten(),
                                      nn.Linear(8600, 2, bias=True))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.classify(out)
        return out


def main():
    torch.manual_seed(0)
    input = torch.randn((1080, 1, 2, 750), dtype=torch.float32)
    model = DeepConvNet()
    print(model)
    print(model(input))


if __name__ == '__main__':
    main()
