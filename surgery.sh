#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=END
#SBATCH --job-name="surgery"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=8000

python surgery_flow.py \
	-f /scratch/groups/lsdavis/yixi/segnet/repcamvid/segnet_basic_train.prototxt \
	-c /scratch/groups/lsdavis/yixi/segnet/repcamvid/snapshots/bs10lr0.1_iter_4200.caffemodel \
	-t ~/segnet/segnet_basic_train_defaultconv1.prototxt \
	-o trainedrgbbs10_surg.caffemodel \
	--fromlayer='conv1' \
	--tolayer='conv1_flow' \
#python surgery_flow.py -f segnet_basic_train_rgb.prototxt -c ~/segnet/snapshots/rgblr0.1_iter_8000.caffemodel -t ~/segnet/segnet_basic_train_defaultconv1.prototxt -o trainedrgb_surg.caffemodel

