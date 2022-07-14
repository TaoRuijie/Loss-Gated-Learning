#!/bin/bash

python main_train.py \
--save_path exp/baseline \
--batch_size 600 \
--lr 0.001 \
--train_list ../utils/train_list.txt \
--val_list ../utils/test_list.txt \
--train_path /data08/VoxCeleb2/wav \
--val_path /data08/VoxCeleb1/wav \
--musan_path /data08/Others/musan_split \
--rir_path /data08/Others/RIRS_NOISES/simulated_rirs \
--init_model /home/ruijie/workspace/Loss-Gated-Learning/Stage1/exp/exp1/model/model000000043.model \
--test_interval 1 \
--n_cluster 6000