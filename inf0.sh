#!/bin/bash

#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=yixi@cs.umd.edu
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="f1inference"
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=8000

xixi='trgbbs10learngglr1e-4fixed'
bs10=true
iter_s=4000
iter_e=10000
iter_gap=1000

cur_dir='/home-4/yixi@umd.edu/segnet/'
work_dir='/scratch/groups/lsdavis/yixi/segnet/segnetf1/'

cd ${work_dir}


module load matlab

: <<'END'
python ${work_dir}/test_segmentation_camvid.py\
		 --model ${work_dir}/segnet_basic_inference.prototxt\
		 --weights ../Models/Inference/segnet_basic_camvid.caffemodel \
		 --iter 233\
		 --output predictions/ftrgbgglr1e-4fixed_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = 'predictions/ftrgbgglr1e-4fixed_iter_${n}/*_gt.png'; predPath = 'predictions/ftrgbgglr1e-4fixed_iter_${n}/*_pr.png'; run('compute_test_results'); exit"

END

for ((n=$iter_s; n<=$iter_e; n+=${iter_gap}))
do
	caffemodel=${cur_dir}/snapshots/${xixi}_iter_${n}.caffemodel
	if [ $n = 0 ];
	then
		caffemodel=${work_dir}/trainedrgb_surg.caffemodel
		if [ $bs10 ];
		then
			caffemodel=${cur_dir}/trainedrgbbs10_surg.caffemodel
		fi
	fi
	trainprototxt=${work_dir}/segnet_basic_train.prototxt
	if [ $bs10 ];
	then
		trainprototxt=${work_dir}/segnet_basic_train_batchsize.prototxt
	fi
	rm -r ${cur_dir}/inference/${xixi}_iter_${n}
	mkdir ${cur_dir}/inference/${xixi}_iter_${n}
	python ${cur_dir}/compute_bn_statistics_lmdb.py \
		${trainprototxt} \
		${caffemodel} \
		${cur_dir}/inference/${xixi}_iter_${n}/

	rm ${cur_dir}/predictions/inf_${xixi}_iter_${n}/ -r -f
	python ${work_dir}/test_segmentation_camvid.py\
		 --model segnet_basic_inference.prototxt\
 		--weights ${cur_dir}/inference/${xixi}_iter_${n}/test_weights.caffemodel \
		 --iter 233 \
		 --output ${cur_dir}/predictions/inf_${xixi}_iter_${n}/
	matlab -nosplash -nodisplay -r "gtPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_gt.png'; predPath = '${cur_dir}/predictions/inf_${xixi}_iter_${n}/*_pr.png'; run('${work_dir}/compute_test_results'); exit"
done


