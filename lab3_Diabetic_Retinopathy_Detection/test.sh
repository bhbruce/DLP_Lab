#!/bin/sh
python train.py -a resnet50 -j 16 --epochs 6 -b 64 --lr 4e-3 --pretrained --gpu 0 lab3_data/ -e --resume logs/resnet50_pretrained_batch_size64_total_epochs6_lr0.004_gamma0.5_momentum0.9_weight_decay0.0001_step_size30/best_ckpt.pth.tar
