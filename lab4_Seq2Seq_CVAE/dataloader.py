from __future__ import unicode_literals, print_function, division
import string
import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd
import torch.utils.data as data

plt.switch_backend('agg')

MAX_LENGTH = 17

all_letters = string.ascii_letters[0:26]
all_letters = "SEP" + all_letters  # SOS:0, EOS:1, PAD: 2
all_letters_list = list(all_letters)
# P: pad, S: start of word, E: end of word
tense = {0: 'simple_present',
         1: 'third_person',
         2: 'present progressive',
         3: 'simple past'}

n_letters = len(all_letters)


def remove_string_tag(str_in):
    # str_in = str_in.replace("P","")
    # str_in = str_in.replace("E","")
    # str_in = str_in.replace("S","")
    str_in = str_in.strip("P")
    str_in = str_in.strip("S")
    str_in = str_in.strip("E")
    return str_in


def IndexToletter(index_tensor):
    out = ""
    for i in index_tensor:
        out += all_letters[i]
    # out = np.take(all_letters_list, index_tensor.cpu())
    return remove_string_tag(out)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


def lineToTensor(line):
    line = line
    tensor = torch.zeros(len(line), 1)
    for li, letter in enumerate(line):
        tensor[li][0] = letterToIndex(letter)
    return tensor


class TrainDataset(data.Dataset):
    def __init__(self, dir='./dataset/train.txt'):
        if(os.path.isfile(dir)):
            data = pd.read_csv(dir, delimiter=" ", header=None)
            # print(data.size)
            self.data = np.squeeze(data.values)
            self.data = self.data.reshape(-1)
            # print(self.data[0])
        else:
            print("Error! File not exist in path: ", dir)
            raise
        max = 0

        acc = np.zeros(16)
        for index in range(self.data.size):
            if max < len(self.data[index]):
                max = len(self.data[index])
            acc[len(self.data[index])] = acc[len(self.data[index])] + 1
        # print(max)
        # print(acc)

    def __getitem__(self, index):
        # voc = lineToTensor(self.data[index])
        # voc = voc.type(torch.long)
        voc = "S" + (self.data[index]) + "E"
        voc = voc + (MAX_LENGTH-len(voc)) * "P"
        cond = index % 4
        cond = torch.Tensor([cond])
        cond = cond.type(torch.long)
        # print(f"idx={index:} voc: {len(voc)} cond={cond:}")
        return voc, cond

    def __len__(self):
        return self.data.shape[0]


def collate_wrapper(batch):
    # batch: list batch[BS][0]: voc batch[BS][1]: cond
    # find max seq_len
    max = 0
    batch = list(batch)
    for i in range(len(batch)):
        letter_len = len(batch[i][0])
        if max < letter_len:
            max = letter_len
        # print(len(batch[i][0]))
    cond_list = list()
    letter_list = list()

    for i in range(len(batch)):
        batch[i] = list(batch[i])
        letter_len = len(batch[i][0])

        batch[i][0] = batch[i][0] + (max - letter_len)*"P"
        # print(len(batch[i][0]))
        letter_list.append(lineToTensor(batch[i][0]))
        cond_list.append(batch[i][1])

    # print('condlist: ', cond_list)
    # print('letter_list: ', letter_list)
    cond = torch.cat(cond_list)
    letter = torch.cat(letter_list, dim=1).type(torch.long)
    return (letter, cond)


class TestDataset(data.Dataset):
    def __init__(self, dir='./dataset/test.txt'):
        if(os.path.isfile(dir)):
            data = pd.read_csv(dir, delimiter=" ", header=None)
            # print(data.size)
            self.data = np.squeeze(data.values)
            # self.data = self.data.reshape(-1)
            # print(self.data[0])
            self.cond_list = list()

            self.cond_list.append([0, 3])
            self.cond_list.append([0, 2])
            self.cond_list.append([0, 1])
            self.cond_list.append([0, 1])
            self.cond_list.append([3, 1])

            self.cond_list.append([0, 2])
            self.cond_list.append([3, 0])
            self.cond_list.append([2, 0])
            self.cond_list.append([2, 3])
            self.cond_list.append([2, 1])

        else:
            print("Error! File not exist in path: ", dir)
            raise

    def __getitem__(self, index):
        # voc = lineToTensor(self.data[index])
        # voc = voc.type(torch.long)
        voc1 = "S"+(self.data[index][0])+"E"
        voc1 = voc1 + (MAX_LENGTH-len(voc1)) * "P"
        voc2 = "S"+(self.data[index][1])+"E"
        voc2 = voc2 + (MAX_LENGTH-len(voc2)) * "P"
        cond1 = self.cond_list[index][0]
        cond2 = self.cond_list[index][1]
        # print(f"idx={index:} cond={cond:}")
        voc1 = lineToTensor(voc1)
        voc2 = lineToTensor(voc2)
        cond1 = torch.Tensor([cond1])
        cond2 = torch.Tensor([cond2])
        return voc1.type(torch.long), voc2.type(torch.long), cond1.type(torch.long), cond2.type(torch.long)

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # Data = TrainDataset()
    # loader = data.DataLoader(Data, batch_size=4, collate_fn=collate_wrapper,
    #                     pin_memory=True)

    # for batch_ndx, sample in enumerate(loader):
    #     print(sample[0].shape)
    #     print(sample[1].shape)
    #     print("===")
    test_data = TestDataset()
    for idx in range(test_data.__len__()):
        print(idx)
        print("voc1:", test_data.__getitem__(idx)[0].shape)
        print("voc2:", test_data.__getitem__(idx)[1].shape)
        print("cond1:", test_data.__getitem__(idx)[2].shape)
        print("cond2:", test_data.__getitem__(idx)[3].shape)
        print("===")
