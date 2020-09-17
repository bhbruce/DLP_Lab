import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as nnF
import torch
from PIL import Image


def genlabels(num_samples, args, FloatTensor):
    num_class = np.random.randint(1, 3, (num_samples,))
    gen_labels = torch.zeros(num_samples, 24)
    for num, idx in enumerate(num_class):
        a = np.random.choice(args.n_classes - 1, idx, replace=False) + 1
        a = torch.tensor(a)
        a = nnF.one_hot(a, num_classes=args.n_classes)
        a = a.sum(0)
        gen_labels[num] = a

    labels = Variable(gen_labels.type(FloatTensor))
    return labels


def sample_image(n_row, batches_done, generator, args, FloatTensor):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    num_samples = n_row ** 2
    rand_list = [torch.randn((1, args.latent_dim))
                 for num_rand in range(num_samples)]
    z = torch.cat(rand_list, dim=0)
    z = z.type(FloatTensor)
    labels = genlabels(num_samples, args, FloatTensor)

    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"images/{batches_done:d}.png",
               nrow=n_row, normalize=True)


def test_img(test_dataset, generator,
             eval_model, z_test, FloatTensor, save_result=False):
    # 1.generate 32 test imgs and save them individually
    generator.eval()
    images = list()
    for i in range(test_dataset.__len__()):
        label = test_dataset.__getitem__(i).unsqueeze(0)
        label = Variable(label)
        label = label.type(FloatTensor)
        z = z_test[i]
        z = z.unsqueeze(0)
        # Sample noise and labels as generator input
        gen_img = generator(z, label)
        save_image(gen_img.data, f"images/test/{i:d}.png",
                   nrow=1, normalize=True)
        images.append(gen_img)

    # Save whole images into single image if save_result is true
    if save_result is True:
        images = torch.cat(images)
        save_image(images.data, "images/test.png",
                   nrow=4, normalize=True)

    # 2.load image one by one and test
    acc = 0
    for i in range(test_dataset.__len__()):
        label = test_dataset.__getitem__(i).unsqueeze(0)
        label = Variable(label.type(FloatTensor))
        img = Image.open(f"images/test/{i:d}.png")  # PIL read w/ RGBA format
        img = test_dataset.input_transform(img.convert('RGB'))
        img = img.unsqueeze(0)
        img = Variable(img.type(FloatTensor))
        acc += eval_model.eval(img, label)
    return acc * 100 / test_dataset.__len__()
