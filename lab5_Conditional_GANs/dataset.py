import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch
from torch.utils.data import DataLoader


def getLabelDict(datapath="./data/"):
    df_label = pd.read_json(datapath+"objects.json",
                            lines=True)
    label_name_dict = dict(df_label.loc[0])
    num_classes = len(label_name_dict)
    return label_name_dict, num_classes


def getData(mode='train', datapath="./data/"):
    if mode == 'train':
        df_train = pd.read_json(datapath+"train.json",
                                lines=True)
        label_list = list(df_train.values[0])  # list of list
        img_name_list = list(df_train.columns)
        return img_name_list, label_list
    else:
        df_test = pd.read_json(datapath+"test.json",
                               lines=True)
        label_list = list(df_test.values[0])  # list of list
        return None, label_list


"""
IclevrDataset Args:
    root (string): Root path of the data
    mode: Indicate procedure status(train or test)
    self.img_name (string list): store images names
    self.label (list of list): store multi-classes of images
    self.mode: use to identify which file getData should load
    self.imagepath: store the path of image
    root
    |--objects.json
    |--train.json
    |--test.json
    |--data/
       |--images
"""
class IclevrDataset(data.Dataset):
    def __init__(self, root='./data/', mode='test',
                 image_size=64):
        self.root = root
        self.img_name, self.label = getData(mode, root)
        self.mode = mode
        self.imagepath = root + 'images/'
        # print("> Found %d images..." % (len(self.label)))
        self.label_dict, self.num_classes = getLabelDict(root)

        self.input_transform = transforms.Compose(
                               [transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Get the image path from 'self.img_name' and load it.
        if self.mode == 'train':
            path = self.imagepath + self.img_name[index]
            img = Image.open(path)  # PIL read w/ RGBA format
            img = self.input_transform(img.convert('RGB'))

        # Get the ground truth labels from self.label 
        label_list = self.label[index]
        output_label = torch.zeros(self.num_classes,
                                   dtype=torch.long)
        # Change multi-classes list to output_label vector
        for i in label_list:
            label = torch.tensor(self.label_dict[i])
            label = F.one_hot(label,
                              num_classes=self.num_classes)
            output_label += label

        if self.mode == 'train':
            return img, output_label
        return output_label


if __name__ == '__main__':
    # test dataset
    train_dataset = IclevrDataset(root='./data/', mode='train')
    test_dataset = IclevrDataset(root='./data/', mode='test')
    print(train_dataset.__len__())
    print(test_dataset.__len__())
    print(train_dataset.__getitem__(5)[0].shape)
    print(train_dataset.__getitem__(5)[1])
    print(test_dataset.__getitem__(5).shape)
    dataloader = DataLoader(train_dataset,
                            batch_size=4,
                            shuffle=False)

    for i, (imgs, labels) in enumerate(dataloader):
        print(imgs.shape)
        print(labels.shape)
        break
    dataloader = DataLoader(test_dataset,
                            batch_size=4,
                            shuffle=False)

    for i, (labels) in enumerate(dataloader):
        print(labels.shape)
        break
