#!/bin/bash

python="/home/wuxilei/anaconda3/envs/torch/bin/python3.8"
script="/home/wuxilei/code/WatchDataProcess/UWasher/train_eval/normal_64.py"

cuda=3

CUDA_VISIBLE_DEVICES=${cuda} ${python} ${script}