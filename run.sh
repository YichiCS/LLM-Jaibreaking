device=$1

export CUDA_VISIBLE_DEVICES=$device 

python gt_train.py --exp_name uc_advbench0010 --save_name uc_advbench0010
python gt_train.py --exp_name uc_advbench0020 --save_name uc_advbench0020
python gt_train.py --exp_name uc_advbench0030 --save_name uc_advbench0030
python gt_train.py --exp_name uc_advbench0040 --save_name uc_advbench0040
python gt_train.py --exp_name uc_advbench0050 --save_name uc_advbench0050

python gt_train.py --exp_name uc_advbench0050 --save_name uc_advbench0050_e2 --epochs 2