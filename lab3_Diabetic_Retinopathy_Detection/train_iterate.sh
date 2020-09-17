#!/bin/sh
python train.py -a resnet50 -j 16 --epochs 6 -b 64 --lr 4e-3 --pretrained --gpu 0 --patience 5 lab3_data/ -p 100
python train.py -a resnet50 -j 16 --epochs 6 -b 64 --lr 4e-3 --gpu 0 --patience 5 lab3_data/ -p 100
python train.py -a resnet18 -j 16 --epochs 10 -b 64 --lr 4e-3 --pretrained --gpu 0 --patience 5 lab3_data/ -p 100
python train.py -a resnet18 -j 16 --epochs 10 -b 64 --lr 4e-3 --gpu 0 --patience 5 lab3_data/ -p 100
python plot_figure.py
