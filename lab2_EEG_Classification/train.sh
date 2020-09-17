#!/bin/bash

python train.py --gpu 0 --epochs 300 -b 128 --lr 1e-2 --wd 5e-4 -a DCN
python train.py --gpu 0 --epochs 300 -b 128 --lr 1e-2 --wd 5e-4 -a EEG
