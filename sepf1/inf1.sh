#!/bin/bash

#SBATCH -t 3:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="sepf1inference"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=5000

xixi='sepf1fidltrgbbs4lr1e-3fixed'
bs=false
slice=false
fromrgb=true
fidl=true
iter_s=1000
iter_e=12000
iter_gap=1000

cur_dir=$PWD
work_dir='/scratch/groups/lsdavis/yixi/segnet/segnetf1/'

cd ${work_dir}


module load matlab
for ((n=$iter_s; n<=$iter_e; n+=${iter_gap}))
do
	caffemodel=${cur_dir}/snapshots/${xixi}_iter_${n}.caffemodel
	if [ $n = 0 ];
	then
		caffemodel=${cur_dir}/trainedf1bs10_surg.caffemodel
		if [ "$fromrgb" = true ];
		then
			caffemodel=${cur_dir}/trainedrgbbs10conv2flow_surg.caffemodel
		fi
	fi
	
	trainprototxt=${cur_dir}/segnet_basic_train.prototxt
	if [ "$bs" = true ];
	then
		trainprototxt=${cur_dir}/segnet_basic_train_batchsize.prototxt
	else

		if [ "$slice" = true ];
		then
			trainprototxt=${cur_dir}/segnet_basic_train_slice.prototxt
		else 
			if [ "$fidl" = true ];
			then
				trainprototxt=${cur_dir}/segnet_basic_train_fidl.prototxt
			fi
		fi
	fi

	inferenceprototxt=${cur_dir}/segnet_basic_inference.prototxt
	if [ "$slice" = true ];
	then
		inferenceprototxt=${cur_dir}/segnet_basic_inference_slice.prototxt
	else if [ "$fidl" = true ];
		then
			inferenceprototxt=${cur_dir}/segnet_basic_inference_fidl.prototxt
		fi
	fi
	echo $bs
	echo $slice
	echo $caffemodel
	echo $trainprototxt
	echo $inferenceprototxt

	rm -r ${cur_dir}/inference/${xixi}_iter_${n}
	mkdir ${cur_dir}/inference/${xixi}_iter_${n}
	python ${cur_dir}/../compute_bn_statistics_all.py \
		${trainprototxt} \
		${caffemodel} \
		${cur_dir}/inference/${xixi}_iter_${n}/

	rm ${cur_dir}/predictions/inf_${xixi}_iter_${n}/ -r -f
	python ${work_dir}/test_segmentation_camvid.py\
		 --model ${inferenceprototxt}\
 		 --weights ${cur_dir}/inference/${xixi}_iter_${n}/test_weights.caffemodel \
		 --iter 233 \
		 --output ${cur_dir}/predictions/inf_${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_gt.png'; predPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_pr.png'; run('${work_dir}/compute_test_results'); exit"
done


