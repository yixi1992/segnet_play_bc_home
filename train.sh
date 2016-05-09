#!/bin/bash

#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f1"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

cd ~/work/yixi/segnet/segnetf1/
#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/snapshots/trgbbs10learngglr1e-3fixed_iter_10000.solverstate
#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/snapshots/trgbbs10learngglr1e-4fixed_iter_3000.solverstate


#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -weights ~/work/yixi/segnet/segnetf1/basic_camvid_surg.caffemodel


~/work/yixi/software/caffe-segnet-yixi/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -weights ~/segnet/trainedrgbbs10_surg.caffemodel
#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -weights ~/segnet/trainedrgb_surg.caffemodel

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/snapshots/learngglr1e-3adagrad_iter_11100.solverstate

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/work/yixi/segnet/segnetf1/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/snapshots/scratchgglr1e-1fixed_iter_12100.solverstate

#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/segnet_basic_solver.prototxt -snapshot /home-4/yixi@umd.edu/segnet/snapshots/rgbbs10lr0.1_iter_4000.solverstate
#~/work/yixi/software/caffe-segnet/build/tools/caffe train -gpu 0 -solver ~/segnet/segnet_basic_solver.prototxt -weights ~/work/yixi/segnet/segnetf1/basic_camvid_surg.caffemodel
