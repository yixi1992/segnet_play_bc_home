#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="segnetf1"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=8000

cd ~/work/yixi/segnet/segnetf1/
~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -weights ~/work/yixi/segnet/segnetf1/basic_camvid_surg.caffemodel

