#!/bin/bash
python train.py --loss MSE --lr 0.1 --data xor --momentum 0.9 --seed 1  --target_loss 0.005
python train.py --loss MSE --lr 0.1 --data linear --momentum 0.9 --seed 1  --target_loss 0.005
# python train.py --loss BCE --lr 0.1 --data xor --momentum 0.9 --seed 1  --target_loss 0.005
# python train.py --loss BCE --lr 0.1 --data linear --momentum 0.9 --seed 1  --target_loss 0.005
