#!/bin/bash

python3 train_cnn.py --seed 42 --in_channels 32 64 \
                    --num_epochs 20 --img_size 28 \
                     --use_pretrained 1 --exp_name cnn_28_hidden_dim_512 --lr 1e-3 --hidden_dim 512

python3 train_cnn.py --seed 42 --in_channels 32 64 \
                    --num_epochs 20 --img_size 28 \
                     --use_pretrained 1 --exp_name cnn_28_hidden_dim_256 --lr 1e-3 --hidden_dim 256

python3 train_cnn.py --seed 42 --in_channels 32 64 \
                    --num_epochs 20 --img_size 28 \
                     --use_pretrained 1 --exp_name cnn_28_hidden_dim_128 --lr 1e-3 --hidden_dim 128

python3 train_cnn.py --seed 42 --in_channels 32 64 \
                    --num_epochs 10 --img_size 28 \
                     --use_pretrained 1 --exp_name cnn_28_hidden_dim_1024 --lr 1e-3 --hidden_dim 1024
