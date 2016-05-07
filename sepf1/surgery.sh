#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=END
#SBATCH --job-name="surgery"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

python /home-4/yixi@umd.edu/segnet/surgery_flow.py \
	-f /scratch/groups/lsdavis/yixi/segnet/repcamvid/segnet_basic_train.prototxt \
	-c /scratch/groups/lsdavis/yixi/segnet/repcamvid/snapshots/bs10lr0.1_iter_4200.caffemodel \
	-t ~/segnet/sepf1/segnet_basic_train_defaultconv2.prototxt \
	-o trainedrgbbs10conv2flow_surg.caffemodel \
	--fromlayer='conv2','conv_decode2','conv_decode1','conv_decode2_bn' \
	--tolayer='conv2_flow','conv_decode2_flow','conv_decode1_flow','conv_decode2_bn_flow' \
#python surgery_flow.py -f segnet_basic_train_rgb.prototxt -c ~/segnet/snapshots/rgblr0.1_iter_8000.caffemodel -t ~/segnet/segnet_basic_train_defaultconv1.prototxt -o trainedrgb_surg.caffemodel

