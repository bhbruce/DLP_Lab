import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import IclevrDataset as Dataset
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms.functional as F
import torch
from parser import argparser
from evaluator import evaluation_model
from utils import sample_image, test_img, genlabels
os.makedirs("images", exist_ok=True)
os.makedirs("result", exist_ok=True)
os.makedirs("images/test", exist_ok=True)
args = argparser()
# print(args)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_shape = (args.channels, args.img_size, args.img_size)


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
        # self.l1 = nn.Linear(args.latent_dim+args.n_classes,
        #                     args.latent_dim*2)
        self.l2 = nn.Linear(args.latent_dim*2, args.latent_dim)
        self.latent_dim = args.latent_dim
        ngf = 64
        self.conv_blocks = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.latent_dim, ngf * 8, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, args.channels, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, labels):
        # noise(BS, latent_size); label(BS, n_classes=24)
        labels_embd = self.label_emb(labels)
        out = torch.cat([labels_embd, noise], dim=-1)
        # labels_embd = self.label_emb(labels)
        # gen_input = torch.cat([labels, noise], dim=-1)
        # out = self.l1(gen_input)
        out = self.l2(out)
        out = self.LeakyReLU(out)
        out = out.view(out.shape[0], self.latent_dim, 1, 1)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        # The height and width of downsampled image
        self.img_size = int(np.prod(img_shape))
        self.transforms = nn.Linear(self.img_size+args.n_classes,
                                    self.img_size)
        # Output layers
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.channels, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
        )

    def forward(self, img, labels):
        # img(bs,3, img_size=64, img_size=64)
        # label(bs,n_classes=24)
        img = img.view(img.shape[0], -1)
        tmp = torch.cat([img, labels], -1)
        tmp = self.transforms(tmp)
        tmp = tmp.view(-1, *img_shape)
        out = self.main(tmp)
        out = out.view(-1, 1)
        return out


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) *
                    fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0),
                    requires_grad=False)
    # Get gradient w.r.t. interpolates

    gradients = torch.autograd.grad(outputs=d_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=fake,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(dataloader, optimizer_G, optimizer_D,
          generator, discriminator, eval_model,
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

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        rand_list = [torch.randn((1, args.latent_dim))
                     for num_rand in range(batch_size)]
        z = torch.cat(rand_list, dim=0)
        z = z.type(FloatTensor)

        # Generate a batch of images
        fake_imgs = generator(z, labels)

        optimizer_D.zero_grad()

        # Loss for real images
        # noise = torch.randn_like(real_imgs)
        # real_imgs = real_imgs*(1-noise_ratio) + noise_ratio * noise
        real_pred = discriminator(real_imgs, labels)

        # Loss for fake images
        fake_pred = discriminator(fake_imgs, labels)
        # Gradient penalty
        lambda_gp = 10
        gradient_penalty = compute_gradient_penalty(discriminator,
                                                    real_imgs.detach(),
                                                    fake_imgs.detach(),
                                                    labels.detach())

        # Adversarial loss
        d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + (
                 lambda_gp * gradient_penalty)
        d_loss.backward()
        optimizer_D.step()

        # Calculate discriminator accuracy
        d_acc = eval_model.eval(fake_imgs.detach(), labels.detach())

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        if i % 5 == 0:
            # -----------------
            #  Train Generator
            # -----------------
            rand_list = [torch.randn((1, args.latent_dim))
                         for num_rand in range(batch_size)]
            z = torch.cat(rand_list, dim=0)
            z = z.type(FloatTensor)
            # Generate a batch of images
            gen_labels = genlabels(batch_size, args, FloatTensor)
            gen_imgs = generator(z, gen_labels.type(FloatTensor))
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = -torch.mean(validity)
            g_loss.backward()
            optimizer_G.step()

        avg_Gloss += (g_loss.item() * batch_size)
        avg_Dloss += (d_loss.item() * batch_size)
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

    avg_Acc = avg_Acc * 100 / total_num
    avg_Dloss = avg_Dloss / total_num
    avg_Gloss = avg_Gloss / total_num
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

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)
    eval_model = evaluation_model()

    if cuda:
        generator.cuda()
        discriminator.cuda()

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
                            num_workers=args.n_cpu,
                            shuffle=True)
    #  Training
    # noise_ratio = 0.15
    best_acc = 0
    Gloss_list = list()
    Dloss_list = list()
    TestAcc_list = list()
    TrainAcc_list = list()

    for epoch in range(args.n_epochs):
        # if (epoch+1) % 100 == 0:
        #     noise_ratio = noise_ratio * 0.9
        acc, Dloss, Gloss = train(dataloader,
                                  optimizer_G, optimizer_D,
                                  generator, discriminator, eval_model,
                                  args, epoch=epoch)
        test_acc = test_img(test_dataset, generator,
                            eval_model, z_test.data, FloatTensor)
        print(f"[Epoch:{epoch:3d}/{args.n_epochs:}] "
              f"TrainACC:{acc:3.2f} "
              f"TestAcc:{test_acc:3.2f} "
              f"d_loss:{Dloss:.4f} "
              f"g_loss:{Gloss:.4f}")

        Gloss_list.append(Gloss)
        Dloss_list.append(Dloss)
        TestAcc_list.append(test_acc)
        TrainAcc_list.append(acc)
        if test_acc > best_acc:
            best_acc = test_acc

            print(f"Save model with new best accuracy{best_acc:2.2f}")
            torch.save({'epoch': epoch+1,
                        'Gen_state_dict': generator.state_dict(),
                        'Dis_state_dict': discriminator.state_dict(),
                        'Gen_optimizer': optimizer_G.state_dict(),
                        'Dis_optimizer': optimizer_D.state_dict(),
                        'lr': args.lr,
                        'best_acc': best_acc,
                        'Dloss_list': Dloss_list,
                        'Gloss_list': Gloss_list,
                        'TestAcc_list': TestAcc_list,
                        'TrainAcc_list': TrainAcc_list
                        }, "./result/best_model.pth")

        torch.save({'epoch': epoch+1,
                    'Gen_state_dict': generator.state_dict(),
                    'Dis_state_dict': discriminator.state_dict(),
                    'Gen_optimizer': optimizer_G.state_dict(),
                    'Dis_optimizer': optimizer_D.state_dict(),
                    'lr': args.lr,
                    'best_acc': test_acc,
                    'Dloss_list': Dloss_list,
                    'Gloss_list': Gloss_list,
                    'TestAcc_list': TestAcc_list,
                    'TrainAcc_list': TrainAcc_list
                    }, "./result/model.pth")


if __name__ == '__main__':
    main(args)
