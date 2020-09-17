#!/bin/bash
python test.py -a EEG --gpu 0 --act 1 --resume ./logs/EEG_LeakyReLU_bs128_epoch300_lr0.01_wd0.0005/EEG_LeakyReLU_bs128_epoch300_lr0.01_wd0.0005_best_ckpt.pth.tar
