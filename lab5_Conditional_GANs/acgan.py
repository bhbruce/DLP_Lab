import os
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import IclevrDataset as Dataset
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms.functional as F
import torch
from parser import argparser
from PIL import Image
from evaluator import evaluation_model
from utils import sample_image, test_img, genlabels
os.makedirs("images", exist_ok=True)
os.makedirs("images/test", exist_ok=True)
os.makedirs("result", exist_ok=True)
args = argparser()
print(args)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        # self.label_emb = nn.Embedding(args.n_classes, args.latent_dim)
        self.label_emb = nn.Linear(args.n_classes, args.latent_dim, bias=True)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.init_size = args.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Linear(args.latent_dim*2, 128 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        labels_embd = self.label_emb(labels)
        # labels_embd = self.LeakyReLU(labels_embd)
        # print(noise.shape, labels_embd.shape)
        # gen_input = torch.mul(labels_embd, noise)
        gen_input = torch.cat([labels_embd, noise], dim=-1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1),
                                       nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2,
                                       args.n_classes),
                                       nn.Sigmoid())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


def train(dataloader, optimizer_G, optimizer_D,
          generator, discriminator, eval_model,
          adversarial_loss, auxiliary_loss,
          args, epoch=0, noise_ratio=0):
    generator.train()
    discriminator.train()
    total_num = 0
    avg_Acc = 0
    avg_Dloss = 0
    avg_Gloss = 0
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        total_num += batch_size
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0),
                         requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))

        rand_list = [torch.randn((1, args.latent_dim))
                     for num_rand in range(batch_size)]
        z = torch.cat(rand_list, dim=0)
        z = z.type(FloatTensor)
        # gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, batch_size)))

        gen_labels = genlabels(batch_size, args, FloatTensor)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels.type(FloatTensor))

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) +
                        auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        noise = [torch.unsqueeze(torch.randn_like(real_imgs[0]), 0)
                 for num_rand in range(real_imgs.shape[0])]
        noise = torch.cat(noise, dim=0)
        real_imgs_input = real_imgs * (1 - noise_ratio) + noise_ratio * (noise)

        real_pred, real_aux = discriminator(real_imgs_input)

        d_real_loss = (adversarial_loss(real_pred, valid) +
                       auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) +
                       auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        # pred = np.concatenate([real_aux.data.cpu().numpy(),
        #                        fake_aux.data.cpu().numpy()], axis=0)
        # gt = np.concatenate([labels.data.cpu().numpy(),
        #                      gen_labels.data.cpu().numpy()], axis=0)
        # gen_imgs = nnF.interpolate(gen_imgs, size=(64, 64))
        d_acc = eval_model.compute_acc(fake_aux, gen_labels)
        d_loss.backward()
        optimizer_D.step()

        avg_Gloss += (g_loss * batch_size)
        avg_Dloss += (d_loss * batch_size)
        avg_Acc += (d_acc * batch_size)
        print(f"[Epoch:{epoch:3d}/{args.n_epochs:}] "
              f"ACC:{avg_Acc*100/total_num:3.2f} "
              f"d_loss:{avg_Dloss/total_num:.4f} "
              f"g_loss:{avg_Gloss/total_num:.4f} "
              f"[Batch:{i:3d}/{len(dataloader)}]",
              end='\r')
        batches_done = epoch * len(dataloader) + i

        if batches_done % args.sample_interval == 0:
            sample_image(n_row=10,
                         batches_done=batches_done,
                         generator=generator,
                         args=args,
                         FloatTensor=FloatTensor)
    avg_Acc = avg_Acc*100/total_num
    avg_Dloss = avg_Dloss/total_num
    avg_Gloss = avg_Gloss/total_num
    return avg_Acc, avg_Dloss, avg_Gloss


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    rand_list = [torch.randn((1, args.latent_dim)) for num_rand in range(32)]
    z_test = torch.cat(rand_list, dim=0)
    z_test = z_test.type(FloatTensor)
    z_test.requires_grad = False

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.BCELoss()  # torch.nn.CrossEntropyLoss()

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)
    eval_model = evaluation_model()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    test_dataset = Dataset(mode='test')

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=args.lr,
                                   betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=args.lr,
                                   betas=(args.b1, args.b2))

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume,
                                    map_location=device)
            # best_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            generator.load_state_dict(checkpoint['Gen_state_dict'])
            discriminator.load_state_dict(checkpoint['Dis_state_dict'])

            # print(f"Resume ckpt best_acc:{best_acc:2.2f}")
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))
    if args.evaluate is True:
        accuracy = test_img(test_dataset, generator, eval_model,
                            z_test.data, FloatTensor, save_result=True)
        print(f'Testing accuracy is {accuracy:2.2f}')
        return

    train_dataset = Dataset(mode='train', image_size=args.img_size)
    # Configure data loader
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)

    # ----------
    #  Training
    # ----------
    noise_ratio = 0.15

    best_acc = 0
    for epoch in range(args.n_epochs):
        if (epoch+1) % 100 == 0:
            noise_ratio = noise_ratio * 0.9
        acc, Dloss, Gloss = train(dataloader,
                                  optimizer_G, optimizer_D,
                                  generator, discriminator, eval_model,
                                  adversarial_loss, auxiliary_loss,
                                  args, epoch=epoch, noise_ratio=noise_ratio)
        test_acc = test_img(test_dataset, generator,
                            eval_model, z_test, FloatTensor)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch+1,
                        'Gen_state_dict': generator.state_dict(),
                        'Dis_state_dict': discriminator.state_dict(),
                        'Gen_optimizer': optimizer_G.state_dict(),
                        'Dis_optimizer': optimizer_D.state_dict(),
                        'lr': args.lr,
                        'best_acc': best_acc
                        }, "./result/best_model.pth")

        print(f"[Epoch:{epoch:3d}/{args.n_epochs:}] "
              f"TrainACC:{acc:3.2f} "
              f"TestAcc:{test_acc:3.2f} "
              f"d_loss:{Dloss:.4f} "
              f"g_loss:{Gloss:.4f}")

        torch.save({'epoch': epoch+1,
                    'Gen_state_dict': generator.state_dict(),
                    'Dis_state_dict': discriminator.state_dict(),
                    'Gen_optimizer': optimizer_G.state_dict(),
                    'Dis_optimizer': optimizer_D.state_dict(),
                    'lr': args.lr,
                    'best_acc': test_acc
                    }, "./result/model.pth")


if __name__ == '__main__':
    main(args)
