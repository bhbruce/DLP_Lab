import pandas as pd
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def input_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def getData(mode, datapath='./'):
    if mode == 'train':
        img = pd.read_csv(datapath + 'train_img.csv')
        label = pd.read_csv(datapath + 'train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(datapath + 'test_img.csv')
        label = pd.read_csv(datapath + 'test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyDataset(data.Dataset):
    def __init__(self, root='./', mode='test',
                 input_transform=input_transform):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): store all image names
            self.label (int or float list): store all ground truth label values
            self.mode: use to identify which file getData should load
            self.datapath: store the path of data
            Dataset_dir
            |--train_img.csv
            |--train_label.csv
            |--test_img.csv
            |--test_label.csv
            |--data/
               |--images
        """
        self.root = root
        # get image name and label
        self.img_name, self.label = getData(mode, root)
        self.mode = mode
        self.datapath = root + 'data/'
        print("> Found %d images..." % (len(self.img_name)))
        self.input_transform = input_transform

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""
        # step1. Get the image path from 'self.img_name' and load it.
        path = self.datapath + self.img_name[index] + '.jpeg'
        img = Image.open(path)  # PIL read w/ RGB format

        # step2. Get the ground truth label from self.label
        label = self.label[index]

        # step3. Transform the .jpeg rgb images
        if self.input_transform:
            img = self.input_transform()(img)

        # step4. Return processed image and label
        return img, label


if __name__ == '__main__':
    # test dataset
    train_dataset = RetinopathyDataset(root='./lab3_data/', mode='train')
    test_dataset = RetinopathyDataset(root='./lab3_data/', mode='test')
    print(train_dataset.__len__())
    print(test_dataset.__len__())
    print(train_dataset.__getitem__(0)[0].shape)
    print(train_dataset.__getitem__(0)[0][2][100:150])
