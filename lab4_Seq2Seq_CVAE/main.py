import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data as data
import random

from dataloader import all_letters, TrainDataset, collate_wrapper
from dataloader import MAX_LENGTH, IndexToletter, TestDataset
from test_criterion import Gaussian_score, compute_bleu
from parser import argparser
import os

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2Seq_VAE(nn.Module):
    def __init__(self, voc_len=29, hidden_size=256, num_layers=2,
                 latent_size=29, cond_embd_size=8,
                 cond_len=4, dropout=0):
        super(Seq2Seq_VAE, self).__init__()

        self.cond_embd1 = nn.Embedding(cond_len, cond_embd_size)
        self.enoder = EncoderRNN(voc_len=voc_len, num_layers=num_layers,
                                 hidden_size=hidden_size,
                                 cond_embd_size=cond_embd_size)

        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        self.decoder = DecoderRNN(hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  latent_size=latent_size,
                                  output_size=voc_len,
                                  cond_embd_size=cond_embd_size,
                                  dropout=dropout)

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_KLD(self, mu, logvar):
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def test_Gaussian(self, MAX_LENGTH=MAX_LENGTH):
        cond = torch.arange(4, device=device)
        Batch_size = cond.shape[0]

        # z:(MAX_LENGTH, BS, latent_size)
        z = torch.randn((MAX_LENGTH, 1, self.latent_size), device=device)
        z = z.repeat(1, Batch_size, 1)
        cembd = self.cond_embd1(cond)
        hidden = self.decoder.initHidden(z, cembd)
        decoder_input = torch.tensor([[SOS_token]*Batch_size], device=device)

        outputs = list()

        for di in range(MAX_LENGTH):
            # print('decoder_input',decoder_input.shape)
            output, hidden = self.decoder(decoder_input, hidden)
            topv, topi = output.topk(1)
            outputs.append(topi)

            decoder_input = topi.squeeze().view(1, -1).detach()
        return torch.cat(outputs).squeeze()

    def forward(self, x, cond1, cond2, criterion,
                teacher_forcing_ratio=0, is_train=True):
        target = x.detach()
        batch_size = cond1.shape[0]
        # cond: (BS)-> cembd: (BS,8)
        # x: (MAX_length, BS)
        # cond1: for Rnn1, cond2: for Rnn2
        cembd1 = self.cond_embd1(cond1)  # (BS, cond_embd_size=8)

        # cembd = cembd.view(1, cembd.shape[0], cembd.shape[1])
        hid = self.enoder.initHidden(cond1.shape[0], cembd1)

        # en_output:(MAX_length, BS, hidden_size)
        en_output, hidden = self.enoder(x, hid)
        mu = self.fc21(en_output)
        logvar = self.fc22(en_output)
        z = self.reparameterize(mu, logvar)  # z:(MAX_length, BS, latent_size)
        loss_KLD = self.compute_KLD(mu, logvar)
        loss_CE = 0

        cembd2 = self.cond_embd1(cond2)  # (BS, cond_embd_size=8)

        target_length = x.shape[0]
        target = target.unsqueeze(0)
        hidden = self.decoder.initHidden(z, cembd2)
        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device)
        # print(decoder_input.shape)
        outputs = list()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if is_train:
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    # print('force:',decoder_input.shape)
                    output, hidden = self.decoder(decoder_input, hidden)
                    # print(output.shape, x[di].shape)
                    loss_CE += criterion(output.squeeze(), x[di])
                    # output = output[0]
                    # loss_CE += criterion(output, x[di])
                    topv, topi = output.topk(1)
                    outputs.append(topi)
                    if di != target_length-1:
                        decoder_input = target[:, di+1]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    # print('no force:', decoder_input.shape)
                    output, hidden = self.decoder(decoder_input, hidden)
                    # print(output.shape , x[di].shape)
                    # print(di, output.shape, x[di].shape)
                    # output = output[0]
                    loss_CE += criterion(output.squeeze(), x[di])
                    # loss_CE += criterion(output, x[di])
                    topv, topi = output.topk(1)
                    outputs.append(topi)
                    decoder_input = topi.squeeze().view(1, -1).detach()  # detach from history as input
        else:  # testing
            for di in range(target_length):
                output, hidden = self.decoder(decoder_input, hidden)
                topv, topi = output.topk(1)
                outputs.append(topi)
                decoder_input = topi.squeeze().view(1, -1).detach()  # detach from history as input
                if topi == EOS_token:
                    break

        return torch.cat(outputs).squeeze(), loss_KLD, loss_CE


# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, voc_len, num_layers=2,
                 hidden_size=256, cond_embd_size=8):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.voc_embd = nn.Embedding(voc_len, hidden_size)
        self.Rnn1 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=self.num_layers)
        self.cond_embd_size = cond_embd_size

    def forward(self, input, hidden):
        # x: (MAX_length, BS)
        # voc_embd: (MAX_length, BS, hidden_size)
        voc_embd = self.voc_embd(input)
        # output:(MAX_length, BS, hidden_size)
        output, hid = self.Rnn1(voc_embd, hidden)
        return output, hid

    def initHidden(self, Batch_size, cembd):
        # cembd:(BS, cond_embd_size=8)
        cond_embd = cembd.unsqueeze(0)  # (1,BS, cond_embd_size=8)
        cond_embd = cond_embd.repeat(self.num_layers, 1, 1)
        hid = torch.zeros(self.num_layers, Batch_size,
                          self.hidden_size-self.cond_embd_size,
                          device=device)
        hidden = torch.cat((hid, cond_embd), dim=-1)
        return (hidden, hidden)


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2,
                 latent_size=29, output_size=29,
                 cond_embd_size=8, MAX_LENGTH=MAX_LENGTH, dropout=0):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_LatToHid = nn.Embedding(latent_size, hidden_size)

        self.Rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.5)

        self.fc = nn.Linear(hidden_size, output_size)

        self.latent_to_hidden = nn.Linear(latent_size * MAX_LENGTH + cond_embd_size,
                                          hidden_size)

    def forward(self, input, hidden):
        # input:(BS, latent_size)
        # cond_embd (BS, cond_embd_size=8)
        # cat cond_embd with voc_embd-> (BS, latent+cond_embd_size)
        output = self.embedding_LatToHid(input)
        output, hid = self.Rnn2(output, hidden)
        output = self.fc(output)
        return output, hid

    def initHidden(self, zi, cond_embd):
        # cembd:(BS, cond_embd_size=8)
        # zi:(MAX_length, BS, latent_size)
        batch_size = zi.shape[1]
        cond_embd = cond_embd.unsqueeze(0)  # (1,BS, cond_embd_size=8)

        cond_embd = cond_embd.repeat(self.num_layers, 1, 1)

        zi = zi.transpose(0, 1)
        zi = zi.reshape(1, batch_size, -1)  # (1, BS, laten*MAX)
        zi = zi.repeat(self.num_layers, 1, 1)
        output = torch.cat([zi, cond_embd], dim=-1)
        hid = self.latent_to_hidden(output)
        hid = hid.cuda(device)

        # print('hid',hid.shape)
        return (hid, hid)


def train(model, dataloader, criterion,
          optimizer, epoch,
          teacher_forcing_ratio, KL_weight):

    model.train()
    acc = 0
    total_num_words = 0
    avg_KLD_loss = 0
    avg_CE_loss = 0
    for batch_ndx, (letter, cond) in enumerate(dataloader):

        cur_num_words = cond.shape[0]
        total_num_words += cur_num_words
        letter = letter.cuda(device)
        cond = cond.cuda(device)

        output, KLD_loss, CE_loss = model(letter, cond, cond,
                                          criterion,
                                          teacher_forcing_ratio)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss = CE_loss + KLD_loss * KL_weight
        avg_CE_loss += CE_loss
        avg_KLD_loss += KLD_loss

        loss.backward()
        optimizer.step()

        output = output.transpose(1, 0)
        letter = letter.transpose(1, 0)
        for pred, gt in zip(output,letter):
            pred_str = IndexToletter(pred)
            gt_str = IndexToletter(gt)
            acc += compute_bleu(pred_str, gt_str)
            # if batch_ndx == epoch or epoch == 9:
            #     print(gt_str,"\t",pred_str)

        # if batch_ndx % 500 == 0:
        #    print(f"Epoch:{epoch:} {batch_ndx:} BLEU4:{acc/total_num_words:.2f} "
        #           f"loss:{loss:.4f} KLD_loss:{KLD_loss:.4f} CE_loss:{CE_loss:.4f}")
    avg_KLD_loss = avg_KLD_loss/batch_ndx
    avg_CE_loss = avg_CE_loss/batch_ndx
    avg_bleu4 = acc/total_num_words

    print(f"Epoch:{epoch:} Training BLEU4:{avg_bleu4:2.2f} KLD_loss:{avg_KLD_loss:2.2f} CE_loss:{avg_CE_loss:2.2f}")
    return avg_bleu4, avg_CE_loss, avg_KLD_loss


def test_tense_conversion(model, test_dataset, criterion):
    acc = 0
    total_num_words = 0

    model.eval()

    for idx in range(test_dataset.__len__()):
        voc1, voc2, cond1, cond2 = test_dataset.__getitem__(idx)
        # print(voc1.shape)
        # print(voc2.shape)
        total_num_words += 1

        voc1 = voc1.cuda(device)
        cond1 = cond1.cuda(device)
        cond2 = cond2.cuda(device)

        output, KLD_loss, CE_loss = model(voc1, cond1, cond2,
                                          criterion, 0,
                                          is_train=False)
        voc1 = voc1.squeeze()
        voc2 = voc2.squeeze()
        pred_str = IndexToletter(output)
        gt_str = IndexToletter(voc2)
        in_str = IndexToletter(voc1)
        acc += compute_bleu(pred_str, gt_str)
        # print(f"\ninput: {in_str:}")
        # print(f"gt_str: {gt_str:}")
        # print(f"pred_str: {pred_str:}")
        # print(f"BLEU4: {acc/total_num_words:.2f}")

    acc = acc/total_num_words
    print(f"Testing Bleu4 acc: {acc:.2f}")
    return acc


def test_Gaussian(model, num_test):

    model.eval()
    result_list = list()
    for idx in range(num_test):
        output = model.test_Gaussian(MAX_LENGTH)
        output = output.transpose(1, 0)
        one_time_list = list()
        for i in (output):
            pred_str = IndexToletter(i)
            one_time_list.append(pred_str)
        result_list.append(one_time_list)
        # print(f"pred_str: {one_time_list:}")
    acc = Gaussian_score(result_list)
    # print(result_list)

    print(f"Testing Gaussian acc: {acc:.2f}")
    return acc


def main():
    args = argparser()
    # print(args)
    args = vars(args)
    # print('seed=',args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    lr = args['lr']
    batch_size = args['batch_size']

    Data = TrainDataset()
    train_data_loader = data.DataLoader(Data, batch_size=batch_size,
                                        collate_fn=collate_wrapper,
                                        pin_memory=True,
                                        num_workers=args['workers'],
                                        shuffle=True)
    test_data = TestDataset()

    model = Seq2Seq_VAE(dropout=args['dropout'])
    model.cuda(device)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr,
                                momentum=args['momentum'],
                                weight_decay=1e-4)

    best_bleu4_acc = 0
    best_cond_acc = 0
    CE_list = list()
    KLD_list = list()
    trian_bleu4_list = list()
    test_bleu4_list = list()
    test_cond_list = list()
    model_para = 'dropout' + str(args['dropout']) + '_lr' + str(lr) \
                 + "_bs" + str(batch_size) + "_klannel_" \
                 + str(args['KL_annealing'])

    if args['resume']:
        if os.path.isfile(args['resume']):
            print(f"resume ckpt from: {args['resume']:}")
            checkpoint = torch.load(args['resume'],
                                    map_location=device)
            best_bleu4_acc = checkpoint['best_bleu4_acc']
            best_cond_acc = checkpoint['best_cond_acc']
            model.load_state_dict(checkpoint['model_state_dict'])

            # for plot
            CE_list = checkpoint['CE_list']
            KLD_list = checkpoint['KLD_list']
            trian_bleu4_list = checkpoint['trian_bleu4_list']
            test_bleu4_list = checkpoint['test_bleu4_list']
            test_cond_list = checkpoint['test_cond_list']

            # print(f"Resume ckpt (1)Best bleu4:{best_bleu4_acc:2.2f}"
            #        f" (2)Best cond:{best_cond_acc:2.2f}")
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))

    if args['evaluate'] is True:
        model.eval()
        # print(args['seed'])
        test_bleu4 = test_tense_conversion(model, test_data, criterion)
        test_cond = test_Gaussian(model, 100)
        return 0
    best_bleu4_acc = 0
    best_cond_acc = 0
    for epoch in range(args['epochs']):
        # KL cost annealing
        if args['epochs'] == 'mono':
            KL_weight = epoch * 0.2  # Monotonic
            if KL_weight >= 1:
                KL_weight = 1
        else:  # KL cycle cost annealing
            if (epoch % 10) <= 5:
                KL_weight = epoch * 0.2
            else:
                KL_weight = 1

        if (epoch % 10) <= 5:
            teacher_forcing_ratio = 1 - epoch * 0.2
        else:
            teacher_forcing_ratio = 0

        if epoch < 10:
            teacher_forcing_ratio = 0.8

        trian_bleu4, CE_loss, KLD_loss = train(model, train_data_loader,
                                               criterion, optimizer,
                                               epoch, teacher_forcing_ratio,
                                               KL_weight)
        test_bleu4 = test_tense_conversion(model, test_data, criterion)
        test_cond = test_Gaussian(model, 100)

        CE_list.append(CE_loss)
        KLD_list.append(KLD_loss)
        trian_bleu4_list.append(trian_bleu4)
        test_bleu4_list.append(test_bleu4)
        test_cond_list.append(test_cond)

        if test_cond > best_cond_acc and test_bleu4 > best_bleu4_acc:
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'encoder_optimizer': optimizer.state_dict(),
                        'lr': lr,
                        'best_bleu4_acc': best_bleu4_acc,
                        'best_cond_acc': best_cond_acc,
                        'CE_list': CE_list,
                        'KLD_list': KLD_list,
                        'trian_bleu4_list': trian_bleu4_list,
                        'test_bleu4_list': test_bleu4_list,
                        'test_cond_list': test_cond_list
                        }, model_para+"_best_bleu4_and_gaussian_model.pth")
            print(f"Save best Gaussian model: {best_bleu4_acc:.2f} {best_cond_acc:.2f}")

        if test_bleu4 > best_bleu4_acc:
            best_bleu4_acc = test_bleu4
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'encoder_optimizer': optimizer.state_dict(),
                        'lr': lr,
                        'best_bleu4_acc': best_bleu4_acc,
                        'best_cond_acc': test_cond,
                        'CE_list': CE_list,
                        'KLD_list': KLD_list,
                        'trian_bleu4_list': trian_bleu4_list,
                        'test_bleu4_list': test_bleu4_list,
                        'test_cond_list': test_cond_list
                        }, model_para+"_best_bleu4_model.pth")
            print(f"Save best BLEU4 model: {best_bleu4_acc:.2f} {best_cond_acc:.2f}")

        if test_cond > best_cond_acc:
            best_cond_acc = test_cond
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'encoder_optimizer': optimizer.state_dict(),
                        'lr': lr,
                        'best_bleu4_acc': test_bleu4,
                        'best_cond_acc': best_cond_acc,
                        'CE_list': CE_list,
                        'KLD_list': KLD_list,
                        'trian_bleu4_list': trian_bleu4_list,
                        'test_bleu4_list': test_bleu4_list,
                        'test_cond_list': test_cond_list
                        }, model_para+"_best_gaussian_model.pth")
            print(f"Save best Gaussian model: {best_bleu4_acc:.2f} {best_cond_acc:.2f}")
    print(f"Final Result (1)Best bleu4:{best_bleu4_acc:2.2f} (2)Best cond:{best_cond_acc:2.2f}")


if __name__ == '__main__':
    main()
