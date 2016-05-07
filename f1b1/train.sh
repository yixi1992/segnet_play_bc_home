#!/bin/bash

#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f1b1"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

cd ~/work/yixi/segnet/segnetf1/

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/f1b1/segnet_basic_solver.prototxt -weights ~/segnet/f1b1/trainedf1bs10_surg.caffemodel

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/f1b1/segnet_basic_solver.prototxt -weights ~/segnet/f1b1/trainedrgbbs10_surg.caffemodel

~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/f1b1/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/f1b1/snapshots/f1b1trgbslicelr1e-3fixed_iter_15000.solverstate
