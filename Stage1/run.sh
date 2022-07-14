#!/bin/bash

python main_train.py \
--save_path exp/exp1 \
--batch_size 300 \
--lr 0.001 \
--lr_decay 0.90 \
--train_list ../utils/train_list.txt \
--val_list ../utils/test_list.txt \
--train_path /data08/VoxCeleb2/wav \
--val_path /data08/VoxCeleb1/wav \
--musan_path /data08/Others/musan_split \
--test_interval 1 \